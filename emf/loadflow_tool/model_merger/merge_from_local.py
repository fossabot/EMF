import copy
import hashlib
import logging
import os.path
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from io import BytesIO
from time import sleep
from uuid import uuid4
from zipfile import ZipFile

import pandas
import pypowsybl.network
import triplets.rdf_parser

from aniso8601 import parse_datetime

from emf.common.integrations.elastic import Elastic
from emf.common.integrations.opdm import OPDM
from emf.loadflow_tool.scaler import query_acnp_schedules, query_hvdc_schedules, scale_balance

# from emf.loadflow_tool.scaler import query_hvdc_schedules


try:
    from emf.common.integrations import minio
except ImportError:
    from emf.common.integrations import minio_api as minio

import config
from emf.common.config_parser import parse_app_properties
from emf.common.integrations import minio_api, opdm
from emf.common.integrations.object_storage import models, file_system
from emf.common.integrations.object_storage.file_system_general import check_and_create_the_folder_path, \
    SEPARATOR_SYMBOL
from emf.common.logging.pypowsybl_logger import PyPowsyblLogGatheringHandler, PyPowsyblLogReportingPolicy
from emf.loadflow_tool import loadflow_settings
from emf.loadflow_tool.helper import load_model, load_opdm_data, create_opdm_objects, get_metadata_from_filename
from emf.loadflow_tool.model_merger import merge_functions
from emf.loadflow_tool.model_merger.model_merger import async_call, log_opdm_response
from emf.loadflow_tool.model_merger.merge_functions import run_lf, create_sv_and_updated_ssh, fix_sv_shunts, \
    fix_sv_tapsteps, export_to_cgmes_zip, filter_models, remove_small_islands
from emf.loadflow_tool.model_validator.validator import validate_model

logger = logging.getLogger(__name__)
parse_app_properties(caller_globals=globals(), path=config.paths.cgm_worker.merger)
SMALL_ISLAND_SIZE = 5
SV_INJECTION_LIMIT = 0.1


def get_opdm_data_from_models(model_data: list | pandas.DataFrame):
    """
    Check if input is already parsed to triplets. Do it otherwise
    :param model_data: input models
    :return triplets
    """
    if not isinstance(model_data, pandas.DataFrame):
        model_data = load_opdm_data(model_data)
    return model_data


def get_boundary_nodes_between_igms(model_data: list | pandas.DataFrame):
    """
    Filters out nodes that are between the igms (mentioned at least 2 igms)
    :param model_data: input models
    : return series of node ids
    """
    model_data = get_opdm_data_from_models(model_data=model_data)
    all_boundary_nodes = model_data[(model_data['KEY'] == 'TopologicalNode.boundaryPoint') &
                                    (model_data['VALUE'] == 'true')]
    # Get boundary nodes that exist in igms
    merged = pandas.merge(all_boundary_nodes,
                          model_data[(model_data['KEY'] == 'SvVoltage.TopologicalNode')],
                          left_on='ID', right_on='VALUE', suffixes=('_y', ''))
    # Get duplicates (all of them) then duplicated values. keep=False marks all duplicates True, 'first' marks first
    # occurrence to false, 'last' marks last occurrence to false. If any of them is used then in case duplicates are 2
    # then 1 is retrieved, if duplicates >3 then duplicates-1 retrieved. So, get all the duplicates and as a second
    # step, drop the duplicates
    merged = (merged[merged.duplicated(['VALUE'], keep=False)]).drop_duplicates(subset=['VALUE'])
    in_several_igms = (merged["VALUE"]).to_frame().rename(columns={'VALUE': 'ID'})
    return in_several_igms


def take_best_match_for_sv_voltage(input_data, column_name: str = 'v', to_keep: bool = True):
    """
    Returns one row for with sv voltage id for topological node
    1) Take the first
    2) If first is zero take first non-zero row if exists
    :param input_data: input dataframe
    :param column_name: name of the column
    :param to_keep: either to keep or discard a value
    """
    first_row = input_data.iloc[0]
    if to_keep:
        remaining_rows = input_data[input_data[column_name] != 0]
        if first_row[column_name] == 0 and not remaining_rows.empty:
            first_row = remaining_rows.iloc[0]
    else:
        remaining_rows = input_data[input_data[column_name] == 0]
        if first_row[column_name] != 0 and not remaining_rows.empty:
            first_row = remaining_rows.iloc[0]
    return first_row


def remove_duplicate_voltage_levels_for_topological_nodes(cgm_sv_data, original_data):
    """
    Pypowsybl 1.6.0 provides multiple sets of SvVoltage values for the topological nodes that are boundary nodes (from
    each IGM side that uses the corresponding boundary node). So this is a hack that removes one of them (preferably the
    one that is zero).
    :param cgm_sv_data: merged SV profile from where duplicate SvVoltage values are removed
    :param original_data: will be used to get boundary node ids
    :return updated merged SV profile
    """
    # Check that models are in triplets
    some_data = get_opdm_data_from_models(model_data=original_data)
    # Get ids of boundary nodes that are shared by several igms
    in_several_igms = (get_boundary_nodes_between_igms(model_data=some_data))
    # Get SvVoltage Ids corresponding to shared boundary nodes
    sv_voltage_ids = pandas.merge(cgm_sv_data[cgm_sv_data['KEY'] == 'SvVoltage.TopologicalNode'],
                                  in_several_igms.rename(columns={'ID': 'VALUE'}), on='VALUE')
    # Get SvVoltage voltage values for corresponding SvVoltage Ids
    sv_voltage_values = pandas.merge(cgm_sv_data[cgm_sv_data['KEY'] == 'SvVoltage.v'][['ID', 'VALUE']].
                                     rename(columns={'VALUE': 'SvVoltage.v'}),
                                     sv_voltage_ids[['ID', 'VALUE']].
                                     rename(columns={'VALUE': 'SvVoltage.SvTopologicalNode'}), on='ID')
    # Just in case convert the values to numeric
    sv_voltage_values[['SvVoltage.v']] = (sv_voltage_values[['SvVoltage.v']].apply(lambda x: x.apply(Decimal)))
    # Group by topological node id and by some logic take SvVoltage that will be dropped
    voltages_to_keep = (sv_voltage_values.groupby(['SvVoltage.SvTopologicalNode']).
                        apply(lambda x: take_best_match_for_sv_voltage(input_data=x,
                                                                       column_name='SvVoltage.v',
                                                                       to_keep=True), include_groups=False))
    voltages_to_discard = sv_voltage_values.merge(voltages_to_keep['ID'], on='ID', how='left', indicator=True)
    voltages_to_discard = voltages_to_discard[voltages_to_discard['_merge'] == 'left_only']
    if not voltages_to_discard.empty:
        logger.info(f"Removing {len(voltages_to_discard.index)} duplicate voltage levels from boundary nodes")
        sv_voltages_to_remove = pandas.merge(cgm_sv_data, voltages_to_discard['ID'].to_frame(), on='ID')
        cgm_sv_data = triplets.rdf_parser.remove_triplet_from_triplet(cgm_sv_data, sv_voltages_to_remove)
    return cgm_sv_data


def check_and_fix_dependencies(cgm_sv_data, cgm_ssh_data, original_data):
    """
    Seems that pypowsybl ver 1.6.0 managed to get rid of dependencies in exported file. This gathers them from
    SSH profiles and from the original models
    :param cgm_sv_data: merged SV profile that is missing the dependencies
    :param cgm_ssh_data: merged SSH profiles, will be used to get SSH dependencies
    :param original_data: original models, will be used to get TP dependencies
    :return updated merged SV profile
    """
    some_data = get_opdm_data_from_models(model_data=original_data)
    tp_file_ids = some_data[(some_data['KEY'] == 'Model.profile') & (some_data['VALUE'].str.contains('Topology'))]

    ssh_file_ids = cgm_ssh_data[(cgm_ssh_data['KEY'] == 'Model.profile') &
                                (cgm_ssh_data['VALUE'].str.contains('SteadyStateHypothesis'))]
    dependencies = pandas.concat([tp_file_ids, ssh_file_ids], ignore_index=True, sort=False)
    existing_dependencies = cgm_sv_data[cgm_sv_data['KEY'] == 'Model.DependentOn']
    dependency_difference = existing_dependencies.merge(dependencies[['ID']].rename(columns={'ID': 'VALUE'}),
                                                        on='VALUE', how='outer', indicator=True)
    if not dependency_difference.query('_merge == "right_only"').empty:
        cgm_sv_data = triplets.rdf_parser.remove_triplet_from_triplet(cgm_sv_data, existing_dependencies)
        full_model_id = cgm_sv_data[(cgm_sv_data['KEY'] == 'Type') & (cgm_sv_data['VALUE'] == 'FullModel')]
        dependencies_to_update = dependency_difference.query('_merge != "left_only"')
        logger.info(f"Mismatch of dependencies. Inserting {len(dependencies_to_update.index)} "
                    f"dependencies to SV profile")
        new_dependencies = dependencies_to_update[['VALUE']].copy().reset_index(drop=True)
        new_dependencies.loc[:, 'KEY'] = 'Model.DependentOn'
        new_dependencies.loc[:, 'ID'] = full_model_id['ID'].iloc[0]
        new_dependencies.loc[:, 'INSTANCE_ID'] = full_model_id['INSTANCE_ID'].iloc[0]
        cgm_sv_data = triplets.rdf_parser.update_triplet_from_triplet(cgm_sv_data, new_dependencies)
    return cgm_sv_data


def get_nodes_against_kirchhoff_first_law(original_models,
                                          cgm_sv_data: pandas.DataFrame = None,
                                          sv_injection_limit: float = SV_INJECTION_LIMIT,
                                          nodes_only: bool = False):
    """
    Gets dataframe of nodes in which the sum of flows exceeds the limit
    :param cgm_sv_data: merged SV profile (needed to set the flows for terminals)
    :param original_models: IGMs (triplets, dictionary)
    :param nodes_only: if true then return unique nodes only, if false then nodes with corresponding terminals
    :param sv_injection_limit: threshold for deciding whether the node is violated by sum of flows
    """
    original_models = get_opdm_data_from_models(model_data=original_models)
    if isinstance(cgm_sv_data, pandas.DataFrame):
        # Get power flow after lf
        power_flow = cgm_sv_data.type_tableview('SvPowerFlow')[['SvPowerFlow.Terminal', 'SvPowerFlow.p',
                                                                'SvPowerFlow.q']]
    else:
        power_flow = original_models.type_tableview('SvPowerFlow')[['SvPowerFlow.Terminal', 'SvPowerFlow.p',
                                                                    'SvPowerFlow.q']]
    # Get terminals
    terminals = original_models.type_tableview('Terminal').rename_axis('Terminal').reset_index()
    terminals = terminals[['Terminal', 'Terminal.ConductingEquipment', 'Terminal.TopologicalNode']]
    # Calculate summed flows per topological node
    flows_summed = ((power_flow.merge(terminals, left_on='SvPowerFlow.Terminal', right_on='Terminal', how='left')
                     .groupby('Terminal.TopologicalNode')[['SvPowerFlow.p', 'SvPowerFlow.q']]
                     .agg(lambda x: pandas.to_numeric(x, errors='coerce').sum()))
                    .rename_axis('Terminal.TopologicalNode').reset_index())
    # Get topological nodes that have mismatch
    nok_nodes = flows_summed[(abs(flows_summed['SvPowerFlow.p']) > sv_injection_limit) |
                             (abs(flows_summed['SvPowerFlow.q']) > sv_injection_limit)][['Terminal.TopologicalNode']]
    if nodes_only:
        return nok_nodes
    try:
        terminals_nodes = terminals.merge(flows_summed, on='Terminal.TopologicalNode', how='left')
        terminals_nodes = terminals_nodes.merge(nok_nodes, on='Terminal.TopologicalNode')
        return terminals_nodes
    except IndexError:
        return pandas.DataFrame()


def revert_failed_buses(cgm_sv_data,
                        original_models,
                        input_buses: pandas.DataFrame = None,
                        revert_failed_terminals: bool = True):
    """
    Reverts back flows in terminals which are located on the buses that were not calculated.
    To revert the flows back to original state following criteria has to be met:
    1) Bus state is either "FAILED" ("...considered dead and no calculation is performed...") or NO_CALCULATION
    2) For terminal there is a difference of power flows when comparing original and merged model
    3) Terminal is located in the topological node which sum of power flows differs from more than sv_injection_limit
    :param cgm_sv_data: merged SV profile (needed to set the flows for terminals)
    :param original_models: IGMs (triplets, dictionary)
    :param input_buses: dataframe of failed buses
    :param revert_failed_terminals: set it true to revert the flows for terminals
    :return updated merged SV and SSH profiles
    """
    if input_buses is None or input_buses.empty:
        return
    # Check that input is triplets
    original_models = get_opdm_data_from_models(model_data=original_models)
    powerflows = cgm_sv_data.type_tableview('SvPowerFlow').reset_index()
    terminals = original_models.type_tableview('Terminal').rename_axis('SvPowerFlow.Terminal').reset_index()
    tn_nodes = original_models.type_tableview('TopologicalNode').rename_axis('TopologicalNode').reset_index()
    voltage_levels = original_models.type_tableview('VoltageLevel').rename_axis('VoltageLevel').reset_index()
    voltage_nodes = (tn_nodes[['TopologicalNode', 'TopologicalNode.ConnectivityNodeContainer']]
                     .merge(voltage_levels[['VoltageLevel']],
                            left_on='TopologicalNode.ConnectivityNodeContainer',
                            right_on='VoltageLevel'))
    voltage_terminals = (terminals[['SvPowerFlow.Terminal',
                                    'Terminal.ConductingEquipment',
                                    'Terminal.TopologicalNode']]
                         .merge(voltage_nodes, left_on='Terminal.TopologicalNode', right_on='TopologicalNode'))
    voltage_terminals = voltage_terminals.drop(columns=['Terminal.TopologicalNode',
                                                        'TopologicalNode.ConnectivityNodeContainer'])
    failed_buses = input_buses[input_buses['status'] == 'FAILED']
    failed_terminals = voltage_terminals.merge(failed_buses[['voltage_level_id']]
                                               .drop_duplicates(keep='last'),
                                               left_on='VoltageLevel',
                                               right_on='voltage_level_id')

    new_powerflows = (cgm_sv_data.type_tableview('SvPowerFlow').reset_index()
                      .merge(failed_terminals[['SvPowerFlow.Terminal']],
                             on='SvPowerFlow.Terminal',
                             how='left', indicator=True))
    new_powerflows_both = new_powerflows[new_powerflows['_merge'] == 'both']
    fixed_powerflows = new_powerflows[new_powerflows['_merge'] == 'left_only']
    old_powerflows = (original_models.type_tableview('SvPowerFlow').reset_index()
                      .merge(failed_terminals[['SvPowerFlow.Terminal']],
                             on='SvPowerFlow.Terminal',
                             how='left', indicator=True))
    old_powerflows_both = old_powerflows[old_powerflows['_merge'] == 'both']
    fix_powerflows = new_powerflows_both.drop(columns=['SvPowerFlow.p', 'SvPowerFlow.q', '_merge'])
    fix_powerflows = fix_powerflows.merge(old_powerflows_both[['SvPowerFlow.Terminal',
                                                               'SvPowerFlow.p',
                                                               'SvPowerFlow.q']],
                                          on='SvPowerFlow.Terminal')
    if not fix_powerflows.empty:
        logger.warning(f"Found {len(fix_powerflows.index)} initialized terminals "
                       f"on {len(failed_buses.index)} failed buses, "
                       f"{len(fixed_powerflows.index)} are ok")
    if revert_failed_terminals:
        logger.warning(f"Reverting {len(fix_powerflows.index)} initialized terminals on failed buses")
        cgm_sv_data = triplets.rdf_parser.update_triplet_from_tableview(cgm_sv_data, fix_powerflows,
                                                                        update=True, add=False)
    return cgm_sv_data


def check_missing_regulation_targets(original_models):
    """
    Pre-check to see if regulation is set to true but no target is given
    Can warn one pypowsybl exception
    :param original_models: IGMs
    """
    original_models = get_opdm_data_from_models(model_data=original_models)
    try:
        injections = original_models.type_tableview('EquivalentInjection').reset_index()
        regulated_injections = injections[(injections['EquivalentInjection.regulationCapability'] == 'true') &
                                          (injections['EquivalentInjection.regulationStatus'] == 'true') &
                                          (injections['EquivalentInjection.regulationTarget'].isnull())]
        if regulated_injections.empty:
            return
        logger.error(f"Found {len(regulated_injections.index)} EquivalentInjections for which regulation is set to "
                     f"true (capability+status) but no target is provided, possible pypowsybl errors can occur")
    except KeyError as error:
        logger.error(f"{error} not present in models")


def check_energized_boundary_nodes(cgm_sv_data, cgm_ssh_data, original_models, fix_errors: bool = False):
    """
    On one case (1D RTEFrance alone on 01.08.2024 12.30Z) pypowsybl calculates the loadflow and updates
    the voltages on boundaries, however the powerflows are still copied over from the original files.
    This, therefore, joins a lot of tables and ,eventually, if voltage at some boundary node is zero and
    equivalentinjection is not then it sets this to zero
    """
    # all_boundary_nodes = original_models.type_tableview('TopologicalNode')
    original_models = get_opdm_data_from_models(model_data=original_models)
    boundary_nodes = original_models.query('KEY == "TopologicalNode.boundaryPoint" & VALUE == "true"')[['ID']]
    terminals = (original_models.type_tableview('Terminal').rename_axis('Terminal').reset_index()
                 .merge(boundary_nodes.rename(columns={'ID': 'Terminal.TopologicalNode'}),
                        on='Terminal.TopologicalNode'))[['Terminal', 'ACDCTerminal.connected',
                                                         'Terminal.ConductingEquipment', 'Terminal.TopologicalNode']]
    new_voltages = (cgm_sv_data.type_tableview('SvVoltage').rename_axis('SvVoltage').reset_index()
                    .merge(boundary_nodes.rename(columns={'ID': 'SvVoltage.TopologicalNode'}),
                           on='SvVoltage.TopologicalNode')).sort_values(by=['SvVoltage'])
    old_voltages = (original_models.type_tableview('SvVoltage').rename_axis('SvVoltage').reset_index()
                    .merge(boundary_nodes.rename(columns={'ID': 'SvVoltage.TopologicalNode'}),
                           on='SvVoltage.TopologicalNode')).sort_values(by=['SvVoltage.TopologicalNode'])
    voltage_diff = ((old_voltages[['SvVoltage.TopologicalNode', 'SvVoltage.v', 'SvVoltage.angle']]
                     .merge(new_voltages[['SvVoltage.TopologicalNode', 'SvVoltage.v', 'SvVoltage.angle']],
                            on='SvVoltage.TopologicalNode', suffixes=('_old', '_new')))
                    .sort_values(by=['SvVoltage.TopologicalNode']))
    old_powerflows = ((original_models.type_tableview('SvPowerFlow').rename_axis('SvPowerFlow').reset_index()
                       .merge(terminals.rename(columns={'Terminal': 'SvPowerFlow.Terminal'}),
                              on='SvPowerFlow.Terminal'))
                      .sort_values(by=['Terminal.TopologicalNode']))
    new_powerflows = ((cgm_sv_data.type_tableview('SvPowerFlow').rename_axis('SvPowerFlow').reset_index()
                       .merge(terminals.rename(columns={'Terminal': 'SvPowerFlow.Terminal'}),
                              on='SvPowerFlow.Terminal'))
                      .sort_values(by=['Terminal.TopologicalNode']))
    powerflow_diff = ((old_powerflows[['SvPowerFlow.Terminal', 'SvPowerFlow.p',
                                       'SvPowerFlow.q',
                                       # 'ACDCTerminal.connected'
                                       ]]
                       .merge(new_powerflows[['SvPowerFlow.Terminal',
                                              'SvPowerFlow.p',
                                              'SvPowerFlow.q',
                                              'SvPowerFlow',
                                              # 'ACDCTerminal.connected',
                                              'Terminal.ConductingEquipment', 'Terminal.TopologicalNode']],
                              on='SvPowerFlow.Terminal', suffixes=('_old', '_new')))
                      .sort_values(by=['Terminal.TopologicalNode']))
    old_injections = ((original_models.type_tableview('EquivalentInjection')
                       .rename_axis('EquivalentInjection').reset_index())
                      .merge(terminals.rename(columns={'Terminal.ConductingEquipment': 'EquivalentInjection'}),
                             on='EquivalentInjection')).sort_values(by=['Terminal.TopologicalNode'])
    new_injections = ((cgm_ssh_data.type_tableview('EquivalentInjection')
                       .rename_axis('EquivalentInjection').reset_index())
                      .merge(terminals.rename(columns={'Terminal.ConductingEquipment': 'EquivalentInjection'}),
                             on='EquivalentInjection')).sort_values(by=['Terminal.TopologicalNode'])
    injection_diff = ((old_injections[['EquivalentInjection', 'EquivalentInjection.p', 'EquivalentInjection.q']]
                       .merge(new_injections[['EquivalentInjection', 'EquivalentInjection.p', 'EquivalentInjection.q',
                                              # 'Terminal',
                                              # 'Terminal.TopologicalNode'
                                              ]], on='EquivalentInjection', suffixes=('_old', '_new')))
                      .sort_values(by=['EquivalentInjection']))
    all_together = (powerflow_diff.rename(columns={'SvPowerFlow.Terminal': 'Terminal',
                                                   'Terminal.ConductingEquipment': 'EquivalentInjection',
                                                   'Terminal.TopologicalNode': 'TopologicalNode'})
                    .merge(injection_diff, on='EquivalentInjection', how='left'))
    all_together = all_together.merge(voltage_diff.rename(columns={'SvVoltage.TopologicalNode': 'TopologicalNode'}),
                                      on='TopologicalNode', how='left').sort_values(by=['TopologicalNode'])
    # Okay it seems that voltage may be set to zero at some boundary nodes while powerflow and equivalent injection
    # are not zero.
    zero_voltages = all_together[(all_together["SvVoltage.v_new"] == 0) & (all_together["SvVoltage.angle_new"] == 0)]
    zero_voltages['Summed_flow'] = (zero_voltages[[
        # 'SvPowerFlow.p_new',
        # 'SvPowerFlow.q_new',
        'EquivalentInjection.p_new',
        'EquivalentInjection.q_new'
    ]].astype(float).abs().sum(axis=1, skipna=True))
    not_zero_flows = zero_voltages[zero_voltages['Summed_flow'] != 0]
    if not not_zero_flows.empty and fix_errors:
        logger.warning(f"{len(not_zero_flows.index)} cases voltage level is zero at boundary but injection is not")
        updated_injections = (not_zero_flows.copy(deep=True)[['EquivalentInjection']]
                              .rename(columns={'EquivalentInjection': 'ID'}))
        # Set P to 0
        updated_p_value = updated_injections[["ID"]].copy()
        updated_p_value["KEY"] = "EquivalentInjection.p"
        updated_p_value["VALUE"] = 0

        # Set Q to 0
        updated_q_value = updated_injections[["ID"]].copy()
        updated_q_value["KEY"] = "EquivalentInjection.q"
        updated_q_value["VALUE"] = 0
        cgm_ssh_data = cgm_ssh_data.update_triplet_from_triplet(pandas.concat([updated_p_value, updated_q_value],
                                                                              ignore_index=True), add=False)
    return cgm_ssh_data


def check_switch_terminals(input_data: pandas.DataFrame, column_name: str):
    """
    Checks if column of a dataframe contains only one value
    :param input_data: input data frame
    :param column_name: name of the column to check
    return True if different values are in column, false otherwise
    """
    data_slice = (input_data.reset_index())[column_name]
    return not pandas.Series(data_slice[0] == data_slice).all()


def handle_not_retained_switches_between_nodes(original_data, open_not_retained_switches: bool = False):
    """
    For the loadflow open all the non-retained switches that connect different topological nodes
    Currently it is seen to help around 9 to 10 Kirchhoff 1st law errors from 2 TSOs
    :param original_data: original models in triplets format
    :param open_not_retained_switches: if true then found switches are set to open, else it only checks and reports
    :return: updated original data
    """
    updated_switches = False
    original_models = get_opdm_data_from_models(original_data)
    not_retained_switches = original_models[(original_models['KEY'] == 'Switch.retained')
                                            & (original_models['VALUE'] == "false")][['ID']]
    closed_switches = original_models[(original_models['KEY'] == 'Switch.open')
                                      & (original_models['VALUE'] == 'false')]
    not_retained_closed = not_retained_switches.merge(closed_switches[['ID']], on='ID')
    terminals = original_models.type_tableview('Terminal').rename_axis('Terminal').reset_index()
    terminals = terminals[['Terminal',
                           # 'ACDCTerminal.connected',
                           'Terminal.ConductingEquipment',
                           'Terminal.TopologicalNode']]
    not_retained_terminals = (terminals.rename(columns={'Terminal.ConductingEquipment': 'ID'})
                              .merge(not_retained_closed, on='ID'))
    if not_retained_terminals.empty:
        return original_data, updated_switches
    between_tn = ((not_retained_terminals.groupby('ID')[['Terminal.TopologicalNode']]
                   .apply(lambda x: check_switch_terminals(x, 'Terminal.TopologicalNode')))
                  .reset_index(name='same_TN'))
    between_tn = between_tn[between_tn['same_TN']]
    if not between_tn.empty:
        updated_switches = True
        logger.warning(f"Found {len(between_tn.index)} not retained switches between topological nodes")
        if open_not_retained_switches:
            logger.warning(f"Opening not retained switches")
            open_switches = closed_switches.merge(between_tn[['ID']], on='ID')
            open_switches.loc[:, 'VALUE'] = 'true'
            original_data = triplets.rdf_parser.update_triplet_from_triplet(original_data, open_switches)
    return original_data, updated_switches


def get_str_from_enum(input_value):
    """
    Tries to parse the input to string
    :param input_value:input string
    """
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, Enum):
        return input_value.name
    try:
        return input_value.name
    except AttributeError:
        return input_value


def get_failed_buses(load_flow_results: list, network_instance: pypowsybl.network, fail_types: dict = None):
    """
    Gets dataframe of failed buses for postprocessing
    :param load_flow_results: list of load flow results for connected components
    :param network_instance: network instance to get buses
    :param fail_types: list of fail types
    :return dataframe of failed buses
    """
    all_networks = pandas.DataFrame(load_flow_results)
    all_buses = (network_instance.get_buses().reset_index()
                 .merge(all_networks.rename(columns={'connected_component_num': 'connected_component'}),
                        on='connected_component'))
    if not fail_types:
        fail_types = {
            'FAILED': get_str_from_enum(pypowsybl.loadflow.ComponentStatus.FAILED),
            'NOT CALCULATED': get_str_from_enum(pypowsybl.loadflow.ComponentStatus.NO_CALCULATION),
            'CONVERGED': get_str_from_enum(pypowsybl.loadflow.ComponentStatus.CONVERGED),
            'MAX_ITERATION_REACHED': get_str_from_enum(pypowsybl.loadflow.ComponentStatus.MAX_ITERATION_REACHED)
        }
    network_count = len(all_networks.index)
    bus_count = len(all_buses.index)
    messages = [f"Networks: {len(load_flow_results)}"]
    error_flag = False
    for status_type in fail_types:
        networks = len((all_networks[all_networks['status'] == fail_types[status_type]]).index)
        buses = len(all_buses[all_buses['status'] == fail_types[status_type]].index)
        network_count = network_count - networks
        bus_count = bus_count - buses
        if networks > 0:
            if status_type == 'MAX_ITERATION_REACHED':
                error_flag = True
            messages.append(f"{status_type}: {networks} ({buses} buses)")
    network_count = max(network_count, 0)
    bus_count = max(bus_count, 0)
    if network_count > 0:
        error_flag = True
        messages.append(f"OTHER CRITICAL ERROR:: {network_count} ({bus_count} buses)")
    troublesome_buses = pandas.DataFrame([result for result in load_flow_results
                                          if get_str_from_enum(result['status']) in fail_types.values()])
    message = '; '.join(messages)
    if error_flag:
        logger.error(message)
    else:
        logger.info(message)
    if not troublesome_buses.empty:
        troublesome_buses = (network_instance.get_buses().reset_index()
                             .merge(troublesome_buses
                                    .rename(columns={'connected_component_num': 'connected_component'}),
                                    on='connected_component'))
    return troublesome_buses


def check_for_disconnected_terminals(cgm_sv_data, original_models, fix_errors: bool = False):
    """
    Checks if disconnected terminals have powerflow different from 0
    :param cgm_sv_data: merged sv profile
    :param original_models: original profiles
    :param fix_errors: sets flows to zero
    :return (updated) sv profile
    """
    all_terminals = original_models.type_tableview('Terminal').rename_axis('SvPowerFlow.Terminal').reset_index()
    disconnected_terminals = all_terminals[all_terminals['ACDCTerminal.connected'] == 'false']
    power_flows_post = cgm_sv_data.type_tableview('SvPowerFlow').rename_axis('SvPowerFlow').reset_index()
    disconnected_powerflows = power_flows_post.merge(disconnected_terminals[['SvPowerFlow.Terminal']],
                                                     on='SvPowerFlow.Terminal')
    flows_on_powerflows = disconnected_powerflows[(abs(disconnected_powerflows['SvPowerFlow.p'].astype('float')) > 0) |
                                                  (abs(disconnected_powerflows['SvPowerFlow.q'].astype('float')) > 0)]
    if not flows_on_powerflows.empty:
        logger.error(f"Found {len(flows_on_powerflows.index)} disconnected terminals which have flows set")
        if fix_errors:
            flows_on_powerflows.loc[:, 'SvPowerFlow.p'] = 0
            flows_on_powerflows.loc[:, 'SvPowerFlow.q'] = 0
            cgm_sv_data = triplets.rdf_parser.update_triplet_from_tableview(cgm_sv_data,
                                                                            flows_on_powerflows,
                                                                            add=False,
                                                                            update=True)
    return cgm_sv_data


def check_net_interchanges(cgm_sv_data, cgm_ssh_data, original_models, fix_errors: bool = False,
                           threshold: float = None):
    """
    An attempt to calculate the net interchange 2 values and check them against those provided in ssh profiles
    :param cgm_sv_data: merged sv profile
    :param cgm_ssh_data: merged ssh profile
    :param original_models: original profiles
    :param fix_errors: injects new calculated flows into merged ssh profiles
    :param threshold: specify threshold if needed
    :return (updated) ssh profiles
    """
    original_models = get_opdm_data_from_models(model_data=original_models)
    try:
        control_areas = (original_models.type_tableview('ControlArea')
                         .rename_axis('ControlArea')
                         .reset_index())[['ControlArea', 'ControlArea.netInterchange', 'ControlArea.pTolerance',
                                          'IdentifiedObject.energyIdentCodeEic', 'IdentifiedObject.name']]
    except KeyError:
        control_areas = original_models.type_tableview('ControlArea').rename_axis('ControlArea').reset_index()
        ssh_areas = cgm_ssh_data.type_tableview('ControlArea').rename_axis('ControlArea').reset_index()
        control_areas = control_areas.merge(ssh_areas, on='ControlArea')[['ControlArea', 'ControlArea.netInterchange',
                                                                          'ControlArea.pTolerance',
                                                                          'IdentifiedObject.energyIdentCodeEic',
                                                                          'IdentifiedObject.name']]
    tie_flows = (original_models.type_tableview('TieFlow')
                 .rename_axis('TieFlow').rename(columns={'TieFlow.ControlArea': 'ControlArea',
                                                         'TieFlow.Terminal': 'Terminal'})
                 .reset_index())[['ControlArea', 'Terminal', 'TieFlow.positiveFlowIn']]
    tie_flows = tie_flows.merge(control_areas[['ControlArea']], on='ControlArea')
    try:
        terminals = (original_models.type_tableview('Terminal')
                     .rename_axis('Terminal').reset_index())[['Terminal', 'ACDCTerminal.connected']]
    except KeyError:
        terminals = (original_models.type_tableview('Terminal')
                     .rename_axis('Terminal').reset_index())[['Terminal']]
    tie_flows = tie_flows.merge(terminals, on='Terminal')
    try:
        power_flows_pre = (original_models.type_tableview('SvPowerFlow')
                           .rename(columns={'SvPowerFlow.Terminal': 'Terminal'})
                           .reset_index())[['Terminal', 'SvPowerFlow.p']]
        tie_flows = tie_flows.merge(power_flows_pre, on='Terminal', how='left')
    except Exception:
        logger.error(f"Was not able to get tie flows from original models")
    power_flows_post = (cgm_sv_data.type_tableview('SvPowerFlow')
                        .rename(columns={'SvPowerFlow.Terminal': 'Terminal'})
                        .reset_index())[['Terminal', 'SvPowerFlow.p']]

    tie_flows = tie_flows.merge(power_flows_post, on='Terminal', how='left',
                                suffixes=('_pre', '_post'))
    try:
        tie_flows_grouped = ((tie_flows.groupby('ControlArea')[['SvPowerFlow.p_pre', 'SvPowerFlow.p_post']]
                              .agg(lambda x: pandas.to_numeric(x, errors='coerce').sum()))
                             .rename_axis('ControlArea').reset_index())
    except KeyError:
        tie_flows_grouped = ((tie_flows.groupby('ControlArea')[['SvPowerFlow.p']]
                              .agg(lambda x: pandas.to_numeric(x, errors='coerce').sum()))
                             .rename_axis('ControlArea').reset_index())
        tie_flows_grouped = tie_flows_grouped.rename(columns={'SvPowerFlow.p': 'SvPowerFlow.p_post'})
    tie_flows_grouped = control_areas.merge(tie_flows_grouped, on='ControlArea')
    if threshold and threshold > 0:
        tie_flows_grouped['Exceeded'] = (abs(tie_flows_grouped['ControlArea.netInterchange']
                                             - tie_flows_grouped['SvPowerFlow.p_post']) > threshold)
    else:
        tie_flows_grouped['Exceeded'] = (abs(tie_flows_grouped['ControlArea.netInterchange']
                                             - tie_flows_grouped['SvPowerFlow.p_post']) >
                                         tie_flows_grouped['ControlArea.pTolerance'])
    net_interchange_errors = tie_flows_grouped[tie_flows_grouped.eval('Exceeded')]
    if not net_interchange_errors.empty:
        if threshold > 0:
            logger.error(f"Found {len(net_interchange_errors.index)} possible net interchange_2 problems "
                         f"over {threshold}:")
        else:
            logger.error(f"Found {len(net_interchange_errors.index)} possible net interchange_2 problems:")
        print(net_interchange_errors.to_string())
        if fix_errors:
            logger.warning(f"Updating {len(net_interchange_errors.index)} interchanges to new values")
            new_areas = cgm_ssh_data.type_tableview('ControlArea').reset_index()[['ID',
                                                                                  'ControlArea.pTolerance', 'Type']]
            new_areas = new_areas.merge(net_interchange_errors[['ControlArea', 'SvPowerFlow.p_post']]
                                        .rename(columns={'ControlArea': 'ID',
                                                         'SvPowerFlow.p_post': 'ControlArea.netInterchange'}), on='ID')
            cgm_ssh_data = triplets.rdf_parser.update_triplet_from_tableview(cgm_ssh_data, new_areas)
    return cgm_ssh_data


def check_non_boundary_equivalent_injections(cgm_sv_data, cgm_ssh_data, original_models, fix_errors: bool = False):
    """
    Checks equivalent injections that are not on boundary topological nodes
    :param cgm_sv_data: merged SV profile
    :param cgm_ssh_data: merged SSH profile
    :param original_models: igms
    :param fix_errors: if true then copies values from sv profile to ssh profile
    :return cgm_ssh_data
    """
    original_models = get_opdm_data_from_models(model_data=original_models)
    boundary_nodes = original_models.query('KEY == "TopologicalNode.boundaryPoint" & VALUE == "true"')[['ID']]
    terminals = (original_models.type_tableview('Terminal').rename_axis('SvPowerFlow.Terminal').reset_index()
                 .merge(boundary_nodes.rename(columns={'ID': 'Terminal.TopologicalNode'}),
                        on='Terminal.TopologicalNode', how='outer', indicator=True))[['SvPowerFlow.Terminal',
                                                                                      'Terminal.ConductingEquipment',
                                                                                      '_merge']]
    terminals = terminals[terminals['_merge'] == 'left_only'][['SvPowerFlow.Terminal', 'Terminal.ConductingEquipment']]
    return check_all_kind_of_injections(cgm_sv_data=cgm_sv_data,
                                        cgm_ssh_data=cgm_ssh_data,
                                        original_models=original_models,
                                        injection_name='EquivalentInjection',
                                        fields_to_check={'SvPowerFlow.p': 'EquivalentInjection.p'},
                                        terminals=terminals,
                                        fix_errors=fix_errors)


def check_all_kind_of_injections(cgm_sv_data,
                                 cgm_ssh_data,
                                 original_models,
                                 injection_name: str = 'ExternalNetworkInjection',
                                 fields_to_check: dict = None,
                                 fix_errors: bool = False,
                                 threshold: float = 0,
                                 terminals: pandas.DataFrame = None,
                                 report_sum: bool = True):
    """
    Compares the given cgm ssh injection values to the corresponding sv powerflow values in cgm sv
    :param cgm_sv_data: merged SV profile
    :param cgm_ssh_data: merged SSH profile
    :param original_models: igms
    :param injection_name: name of the injection
    :param fields_to_check: dictionary where key is the field in powerflow and value is the field in injection
    :param fix_errors: if true then copies values from sv profile to ssh profile
    :param threshold: max allowed mismatch
    :param terminals: optional, can give dataframe of terminals as input
    :param report_sum: if true prints sum of injections and powerflows to console
    :return cgm_ssh_data
    """
    if not fields_to_check:
        return cgm_ssh_data
    fixed_fields = ['ID']
    original_models = get_opdm_data_from_models(model_data=original_models)
    try:
        injections = cgm_ssh_data.type_tableview(injection_name).reset_index()
    except AttributeError:
        logger.info(f"SSH profile doesn't contain data about {injection_name}")
        return cgm_ssh_data
    injections_reduced = injections[[*fixed_fields, *fields_to_check.values()]]
    if terminals is None:
        terminals = (original_models.type_tableview('Terminal')
                     .rename_axis('SvPowerFlow.Terminal')
                     .reset_index())[['SvPowerFlow.Terminal', 'Terminal.ConductingEquipment']]
    flows = (cgm_sv_data.type_tableview('SvPowerFlow')
             .reset_index())[[*['SvPowerFlow.Terminal'], *fields_to_check.keys()]]
    terminals = terminals.merge(flows, on='SvPowerFlow.Terminal')
    terminals = terminals.merge(injections_reduced, left_on='Terminal.ConductingEquipment', right_on='ID')
    filtered_list = []
    for flow_field, injection_field in fields_to_check.items():
        filtered_list.append(terminals[abs(terminals[injection_field] - terminals[flow_field]) > threshold])
        if report_sum:
            logger.info(f"{injection_field} = {terminals[injection_field].sum()} vs "
                        f"{flow_field} = {terminals[flow_field].sum()}")
    if not filtered_list:
        return cgm_ssh_data
    filtered = pandas.concat(filtered_list).drop_duplicates().reset_index(drop=True)
    if not filtered.empty:
        logger.warning(f"Found {len(filtered.index)} mismatches between {injection_name} and flow values on terminals")
        if fix_errors:
            logger.info(f"Updating {injection_name} values from terminal flow values")
            injections_update = injections.merge(filtered[[*fixed_fields, *fields_to_check.keys()]])
            injections_update = injections_update.drop(columns=fields_to_check.values())
            injections_update = injections_update.rename(columns=fields_to_check)
            cgm_ssh_data = triplets.rdf_parser.update_triplet_from_tableview(data=cgm_ssh_data,
                                                                             tableview=injections_update,
                                                                             update=True,
                                                                             add=False)
    return cgm_ssh_data


def merge_models(list_of_models: list,
                 latest_boundary: dict,
                 time_horizon: str,
                 scenario_datetime: str,
                 merging_area: str,
                 merging_entity: str,
                 mas: str,
                 version: str = '001',
                 push_to_opdm: bool = False,
                 merge_prefix: str = 'CGM',
                 path_local_storage: str = None,
                 pre_sv_profile: list = None,
                 pre_ssh_profiles: list = None,
                 use_merged_profiles: bool = False,
                 excluded_profiles: list = None,
                 run_scaling: bool = True):
    # Load all selected models
    input_models = list_of_models + [latest_boundary]

    if use_merged_profiles:
        input_models = input_models + pre_ssh_profiles + pre_sv_profile

    parameters = {
        "iidm.import.cgmes.import-node-breaker-as-bus-breaker": 'true',
    }
    use_updated_models = False
    assembled_data = None
    try:
        assembled_data = merge_functions.load_opdm_data(input_models)
        assembled_data = triplets.cgmes_tools.update_FullModel_from_filename(assembled_data)
        if pre_sv_profile is not None and len(pre_sv_profile) > 0 and not use_merged_profiles:
            assembled_sv_data = merge_functions.load_opdm_data(pre_sv_profile)
            assembled_ssh_data = merge_functions.load_opdm_data(pre_ssh_profiles)
            try:
                get_pf_mismatch(cgm_sv_data=assembled_sv_data,
                                merged_models=assembled_data,
                                path_local_storage=path_local_storage,
                                file_name='original_model')
            except Exception:
                logger.error(f"Unable to check merged model")
            check_net_interchanges(cgm_sv_data=assembled_sv_data,
                                   cgm_ssh_data=assembled_ssh_data,
                                   original_models=assembled_data,
                                   threshold=200)
        assembled_data = merge_functions.configure_paired_boundarypoint_injections_by_nodes(assembled_data)
        escape_upper_xml = assembled_data[assembled_data['VALUE'].astype(str).str.contains('.XML')]
        if not escape_upper_xml.empty:
            escape_upper_xml['VALUE'] = escape_upper_xml['VALUE'].str.replace('.XML', '.xml')
            assembled_data = triplets.rdf_parser.update_triplet_from_triplet(assembled_data, escape_upper_xml,
                                                                             update=True,
                                                                             add=False)
        check_missing_regulation_targets(assembled_data)
        update_switches = False

        assembled_data, updated = handle_not_retained_switches_between_nodes(assembled_data,
                                                                             open_not_retained_switches=update_switches
                                                                             )

        updated_models = create_opdm_objects([merge_functions.export_to_cgmes_zip([assembled_data])])
        assembled_data
    except KeyError as error:
        updated_models = input_models
        logger.error(f"Unable to preprocess: {error}")
    merged_model = load_model(updated_models, parameters=parameters, skip_default_parameters=True)

    # TODO - run other LF if default fails
    logger.info(f"Running loadflow")

    new_settings = loadflow_settings.CGM_RELAXED_2
    pypowsybl_folder = os.path.join(path_local_storage, 'pypowsybl_logs')
    check_and_create_the_folder_path(pypowsybl_folder)
    new_settings.provider_parameters['debugDir'] = pypowsybl_folder

    solved_model = run_lf(merged_model, loadflow_settings=new_settings)

    network_itself = solved_model["network"]
    pretty_results = pandas.DataFrame(merged_model["LOADFLOW_RESULTS"])
    all_buses = (network_itself.get_buses()
                 .reset_index()
                 .merge(pretty_results.rename(columns={'connected_component_num': 'connected_component'}),
                        on='connected_component'))
    time_moment_now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    file_name_all = f"all_buses_from_{time_moment_now}.csv"
    if path_local_storage:
        check_and_create_the_folder_path(path_local_storage)
        file_name_all = path_local_storage.removesuffix('/') + '/' + file_name_all.removeprefix('/')
    all_buses.to_csv(file_name_all)

    troublesome_buses = get_failed_buses(load_flow_results=merged_model["LOADFLOW_RESULTS"],
                                         network_instance=network_itself)
    process_types_known = {'1D': 'A01', 'ID': 'A18'}
    process_type = process_types_known.get(time_horizon)
    data = None
    if run_scaling and process_type:
        if isinstance(assembled_data, pandas.DataFrame) and assembled_data is not None:
            data = assembled_data
        else:
            data = get_opdm_data_from_models(model_data=updated_models)
        elastic_client = Elastic()
        # area_eic_codes = elastic_client.get_data(query={"match_all": {}}, index='config-areas')
        area_eic_codes = elastic_client.get_docs_by_query(index='config-areas', query={"match_all": {}}, size=200)
        area_codes = area_eic_codes[['area.eic', 'area.code']].set_index('area.eic').T.to_dict('records')[0]
        date_time_value = parse_datetime(scenario_datetime)
        start_time = (date_time_value - timedelta(hours=0, minutes=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time = (date_time_value + timedelta(hours=0, minutes=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
        ac_schedules = query_acnp_schedules(process_type=process_type,
                                            utc_start=start_time,
                                            utc_end=end_time,
                                            area_eic_map=area_codes)
        dc_schedules = query_hvdc_schedules(process_type=process_type,
                                            utc_start=start_time,
                                            utc_end=end_time,
                                            area_eic_map=area_codes)
        solved_model["network"] = scale_balance(network=solved_model["network"],
                                                ac_schedules=ac_schedules,
                                                dc_schedules=dc_schedules,
                                                lf_settings=new_settings,
                                                # area_eic_map=area_eic_codes,
                                                # models=data,
                                                area_eic_map=None,
                                                models=None,
                                                abort_when_diverged=True,
                                                scale_by_regions=False,
                                                debug=True)

    logger.info(f"Loadflow done")

    # TODO - get version dynamically form ELK
    # Update time_horizon in case of generic ID process type
    new_time_horizon = None
    if time_horizon.upper() == "ID":
        # As testing model merging backwards, and models come from anywhere then for ID set the time horizon to some
        # random value: PURELY
        # time_horizon = '08'
        new_time_horizon = '01'
    # data = load_opdm_data(input_models)
    time_horizon = new_time_horizon or time_horizon
    seed_string = "baltic-rcc.eu"
    some_random_hex = hashlib.md5(seed_string.encode('utf-8')).hexdigest()
    some_random_uuid = str(uuid.UUID(hex=some_random_hex))
    model_export_parameters = {
        # CGM_EXPORT:
        # Is not yet imported to pypowsybl
        # 'iidm.export.cgmes.cgm_export': 'true',
        # ENSURE_ID_ALIAS_UNICITY: assure that all IDS imported are unique, default: false
        # "iidm.import.cgmes.ensure-id-alias-unicity": 'true',
        # true by default
        # 'iidm.export.cgmes.export-load-flow-status': 'true',
        # EXPORT_BOUNDARY_POWER_FLOWS: Export power flows at boundary odes in SV file, default is true
        # 'iidm.export.cgmes.export-boundary-power-flows': 'true',
        # EXPORT_SV_INJECTIONS_FOR_SLACKS: export total mismatch from power flow calculation as a sv injection
        # default: true
        # 'iidm.export.cgmes.export-sv-injections-for-slacks': 'true',
        # UPDATE_DEPENDENCIES:
        # 'iidm.export.cgmes.update-dependencies': 'true'
        # "iidm.export.cgmes.uuid-namespace": some_random_uuid
    }
    sv_data, ssh_data, data = create_sv_and_updated_ssh(merged_model=solved_model,
                                                        original_models=updated_models,
                                                        scenario_date=scenario_datetime,
                                                        time_horizon=time_horizon,
                                                        version=version,
                                                        merging_area=merging_area,
                                                        merging_entity=merging_entity,
                                                        mas=mas,
                                                        export_parameters=model_export_parameters)

    # Fix SV
    sv_data = fix_sv_shunts(sv_data, data)
    sv_data = fix_sv_tapsteps(sv_data, ssh_data)
    sv_data = remove_small_islands(sv_data, int(SMALL_ISLAND_SIZE))
    sv_data = check_and_fix_dependencies(cgm_sv_data=sv_data, cgm_ssh_data=ssh_data, original_data=data)
    sv_data = remove_duplicate_voltage_levels_for_topological_nodes(cgm_sv_data=sv_data, original_data=data)
    sv_data = revert_failed_buses(cgm_sv_data=sv_data,
                                  original_models=data,
                                  input_buses=troublesome_buses,
                                  revert_failed_terminals=False)
    # ssh_data= update_ssh_from_sv(merged_ssh_data=ssh_data, merged_sv_data=sv_data, original_models=data)
    ssh_data = check_all_kind_of_injections(cgm_ssh_data=ssh_data,
                                            cgm_sv_data=sv_data,
                                            original_models=data,
                                            injection_name='EnergySource',
                                            fields_to_check={'SvPowerFlow.p': 'EnergySource.activePower'},
                                            fix_errors=False)
    ssh_data = check_all_kind_of_injections(cgm_ssh_data=ssh_data,
                                            cgm_sv_data=sv_data,
                                            original_models=data,
                                            injection_name='ExternalNetworkInjection',
                                            fields_to_check={'SvPowerFlow.p': 'ExternalNetworkInjection.p'},
                                            fix_errors=False)
    ssh_data = check_all_kind_of_injections(cgm_ssh_data=ssh_data,
                                            cgm_sv_data=sv_data,
                                            original_models=data,
                                            injection_name='SynchronousMachine',
                                            fields_to_check={'SvPowerFlow.p': 'RotatingMachine.p'},
                                            fix_errors=False)
    ssh_data = check_non_boundary_equivalent_injections(cgm_sv_data=sv_data,
                                                        cgm_ssh_data=ssh_data,
                                                        original_models=data,
                                                        fix_errors=False)
    sv_data = check_for_disconnected_terminals(cgm_sv_data=sv_data, original_models=data, fix_errors=True)
    try:
        ssh_data = check_net_interchanges(cgm_sv_data=sv_data, cgm_ssh_data=ssh_data, original_models=data,
                                          fix_errors=False,
                                          threshold=200)
    except KeyError:
        logger.error(f"No fields for netInterchange")
    try:
        ssh_data = check_energized_boundary_nodes(cgm_sv_data=sv_data, cgm_ssh_data=ssh_data, original_models=data,
                                                  fix_errors=True)
    except AttributeError:
        logger.error(f"Some field was not found")
    try:
        tsos_ok, tsos_nok = get_pf_mismatch(cgm_sv_data=sv_data, merged_models=data,
                                            path_local_storage=path_local_storage)
        if not tsos_nok.empty:
            logger.warning(f"Following TSOs have Kirchhoff 1st law errors")
            print(tsos_nok.to_string())
        else:
            logger.info(f"No Kirchhoff 1st law errors detected")
    except Exception as ex_message:
        logger.error(f"Some error occurred: {ex_message}")
    # Package both input models and exported CGM profiles to in memory zip files
    serialized_data = export_to_cgmes_zip([ssh_data, sv_data])

    if push_to_opdm:
        try:
            opdm_service = opdm.OPDM()
            for item in serialized_data:
                logger.info(f"Uploading to OPDM -> {item.name}")
                async_call(function=opdm_service.publication_request, callback=log_opdm_response,
                           file_path_or_file_object=item)
                sleep(5)
        except:
            logging.error(f"""Unexpected error on uploading to OPDM:""", exc_info=True)

    # Set RMM name
    rmm_name = (f"{merge_prefix}_{time_horizon}_{version}_{parse_datetime(scenario_datetime):%Y%m%dT%H%MZ}_"
                f"{merging_area}_{uuid4()}")
    # Exclude following profiles from the IGMS
    # excluded_profiles = ['SV', 'SSH']
    if not excluded_profiles:
        excluded_profiles = []
    rmm_data = BytesIO()
    with (ZipFile(rmm_data, "w") as rmm_zip):

        # Include CGM model files
        for item in serialized_data:
            rmm_zip.writestr(item.name, item.getvalue())
        # Include original IGM files
        for object_element in input_models:
            for instance in object_element['opde:Component']:
                with ZipFile(BytesIO(instance['opdm:Profile']['DATA'])) as instance_zip:
                    zip_file_name = instance.get('opdm:Profile', {}).get('pmd:fileName')
                    zip_file_profile = instance.get('opdm:Profile', {}).get('pmd:cgmesProfile')
                    if len(instance_zip.namelist()) == 1 and zip_file_name:
                        if zip_file_profile and zip_file_profile not in excluded_profiles:
                            rmm_zip.writestr(zip_file_name, instance['opdm:Profile']['DATA'])
                    else:
                        for file_name in instance_zip.namelist():
                            logging.info(f"Adding file: {file_name}")
                            rmm_zip.writestr(file_name, instance_zip.open(file_name).read())

    # Upload to Object Storage
    rmm_object = rmm_data
    rmm_object.name = f"{rmm_name}.zip"
    # logger.info(f"Uploading RMM to MINO {OUTPUT_MINIO_BUCKET}/{rmm_object.name}")

    return rmm_object


def get_pf_mismatch(cgm_sv_data,
                    merged_models,
                    store_results: bool = True,
                    path_local_storage: str = None,
                    file_name: str = "all_lines_from",
                    use_sv_injections: bool = True):
    """
    This is for debugging Kirchhoff's 1st law. Proceed with caution, have no idea what or if something here does
    something
    :param cgm_sv_data: merged sv profile
    :param merged_models: input igms
    :param store_results: whether to store all the lines to csv
    :param path_local_storage: where to store csv
    :param file_name: name of the file where to store the results
    :param use_sv_injections:
    """
    # Get power flow after lf
    power_flow = cgm_sv_data.type_tableview('SvPowerFlow')
    # Get power flow before lf
    original_power_flow = merged_models.type_tableview('SvPowerFlow')
    # merge power flows
    power_flow_diff = (original_power_flow[['SvPowerFlow.Terminal', 'SvPowerFlow.p', 'SvPowerFlow.q']]
                       .rename(columns={'SvPowerFlow.p': 'pre.p', 'SvPowerFlow.q': 'pre.q'})
                       .merge(power_flow[['SvPowerFlow.Terminal', 'SvPowerFlow.p', 'SvPowerFlow.q']]
                              .rename(columns={'SvPowerFlow.p': 'post.p', 'SvPowerFlow.q': 'post.q'}),
                              on='SvPowerFlow.Terminal',
                              how='outer'))
    # Get terminals
    terminals = (merged_models.type_tableview('Terminal')
                 .rename_axis('Terminal').reset_index()
                 .rename(columns={'IdentifiedObject.name': 'Terminal.name'}))
    try:
        terminals = terminals[['Terminal',
                               'ACDCTerminal.connected',
                               'ACDCTerminal.sequenceNumber',
                               'Terminal.name',
                               'Terminal.ConductingEquipment',
                               'Terminal.ConnectivityNode',
                               'Terminal.TopologicalNode']]
    except Exception:
        terminals = terminals[['Terminal',
                               'ACDCTerminal.connected',
                               'ACDCTerminal.sequenceNumber',
                               'Terminal.name',
                               'Terminal.ConductingEquipment',
                               'Terminal.TopologicalNode']]
    # Get origins
    tsos = (merged_models.query('KEY == "Model.modelingAuthoritySet"')[['VALUE', 'INSTANCE_ID']]
            .rename(columns={'VALUE': 'modelingAuthoritySet'}))
    voltage_levels = (merged_models.type_tableview('VoltageLevel')
                      .rename_axis('VoltageLevel').reset_index())[['VoltageLevel', 'VoltageLevel.Substation']]
    substations = (merged_models.type_tableview('Substation')
                   .rename_axis('VoltageLevel.Substation')
                   .reset_index()
                   .rename(columns={'IdentifiedObject.name': 'Substation'}))[['VoltageLevel.Substation',
                                                                              'Substation',
                                                                              'Substation.Region']]
    regions = (merged_models.type_tableview('SubGeographicalRegion')
               .rename_axis('Substation.Region')
               .reset_index()
               .rename(columns={'IdentifiedObject.name': 'Region'}))[['Substation.Region', 'Region']]
    voltage_levels = (voltage_levels.merge(substations, on='VoltageLevel.Substation')
                      .merge(regions, on='Substation.Region'))
    voltage_levels = voltage_levels[['VoltageLevel', 'Substation', 'Region']]

    topological_node_names = ((merged_models.type_tableview('TopologicalNode')
                               .rename_axis('Terminal.TopologicalNode')
                               .reset_index())
                              .rename(columns={'IdentifiedObject.name': 'TopologicalNode.name',
                                               'TopologicalNode.EquipmentContainer': 'VoltageLevel',
                                               'IdentifiedObject.description':
                                                   'TopologicalNode.description'}))
    try:
        topological_node_names = topological_node_names[['Terminal.TopologicalNode',
                                                         'TopologicalNode.name',
                                                         'TopologicalNode.description',
                                                         'TopologicalNode.ConnectivityNodeContainer']]
    except KeyError:
        logger.error(f"TopologicalNodes are not tied to connectivityContainers")
        topological_node_names = ((topological_node_names[['Terminal.TopologicalNode',
                                                           'TopologicalNode.name',
                                                           'TopologicalNode.description',
                                                           'ConnectivityNodeContainer']])
                                  .rename(columns={'ConnectivityNodeContainer': 'TopologicalNode.'
                                                                                'ConnectivityNodeContainer'}))
    all_topological_nodes = (merged_models.query('KEY=="Type" and VALUE=="TopologicalNode"')[['ID', 'INSTANCE_ID']]
                             .rename(columns={'ID': 'Terminal.TopologicalNode'}))
    all_topological_nodes = all_topological_nodes.merge(topological_node_names, on='Terminal.TopologicalNode',
                                                        how='left')
    all_topological_nodes = (all_topological_nodes
                             .merge(tsos, on='INSTANCE_ID'))[['Terminal.TopologicalNode',
                                                              'TopologicalNode.name',
                                                              'TopologicalNode.description',
                                                              'TopologicalNode.ConnectivityNodeContainer',
                                                              'modelingAuthoritySet']]
    all_topological_nodes = all_topological_nodes.merge(voltage_levels,
                                                        left_on='TopologicalNode.ConnectivityNodeContainer',
                                                        right_on='VoltageLevel',
                                                        how='left')
    # Calculate summed flows per topological node
    flows_summed = ((power_flow_diff.merge(terminals,
                                           left_on='SvPowerFlow.Terminal',
                                           right_on='Terminal',
                                           how='left').groupby('Terminal.TopologicalNode')[['pre.p', 'pre.q',
                                                                                            'post.p', 'post.q']]
                     .agg(lambda x: pandas.to_numeric(x, errors='coerce').sum()))
                    .rename_axis('Terminal.TopologicalNode').reset_index()
                    .rename(columns={'pre.p': 'pre.p.sum', 'pre.q': 'pre.q.sum',
                                     'post.p': 'post.p.sum', 'post.q': 'post.q.sum'}))
    if use_sv_injections:
        try:
            pre_sv_injections = merged_models.type_tableview('SvInjection').rename_axis('SvInjection').reset_index()
            post_sv_injections = cgm_sv_data.type_tableview('SvInjection').rename_axis('SvInjection').reset_index()
            flows_summed = flows_summed.merge(pre_sv_injections[['SvInjection.TopologicalNode',
                                                                 'SvInjection.pInjection',
                                                                 'SvInjection.qInjection']]
                                              .rename(
                columns={'SvInjection.TopologicalNode': 'Terminal.TopologicalNode'}),
                                              on='Terminal.TopologicalNode', how='left')
            flows_summed['pre.p.sum'] = flows_summed['pre.p.sum'] + flows_summed['SvInjection.pInjection'].fillna(0)
            flows_summed['pre.q.sum'] = flows_summed['pre.q.sum'] + flows_summed['SvInjection.qInjection'].fillna(0)
            flows_summed = flows_summed.drop(columns=['SvInjection.pInjection', 'SvInjection.qInjection'])
            flows_summed = flows_summed.merge(post_sv_injections[['SvInjection.TopologicalNode',
                                                                  'SvInjection.pInjection',
                                                                  'SvInjection.qInjection']]
                                              .rename(
                columns={'SvInjection.TopologicalNode': 'Terminal.TopologicalNode'}),
                                              on='Terminal.TopologicalNode', how='left')
            flows_summed['post.p.sum'] = flows_summed['post.p.sum'] + flows_summed['SvInjection.pInjection'].fillna(0)
            flows_summed['post.q.sum'] = flows_summed['post.q.sum'] + flows_summed['SvInjection.qInjection'].fillna(0)
            flows_summed = flows_summed.drop(columns=['SvInjection.pInjection', 'SvInjection.qInjection'])
        except Exception as ex:
            logger.error(f"Cannot use svinjections: {ex}")
    flows_summed['Flow.exceeded'] = ((abs(flows_summed['post.p.sum']) > SV_INJECTION_LIMIT) |
                                     (abs(flows_summed['post.q.sum']) > SV_INJECTION_LIMIT))
    # Divide into ok and nok nodes
    nok_nodes_full = flows_summed[(abs(flows_summed['post.p.sum']) > SV_INJECTION_LIMIT) |
                                  (abs(flows_summed['post.q.sum']) > SV_INJECTION_LIMIT)]
    nok_nodes = nok_nodes_full[['Terminal.TopologicalNode']]
    ok_nodes = flows_summed.merge(nok_nodes.drop_duplicates(), on='Terminal.TopologicalNode',
                                  how='left', indicator=True)
    ok_nodes = ok_nodes[ok_nodes['_merge'] == 'left_only'][['Terminal.TopologicalNode']]
    # Merge terminals with power flows
    terminals_flows = terminals.merge(power_flow_diff.rename(columns={'SvPowerFlow.Terminal': 'Terminal'}),
                                      on='Terminal', how='left')
    # Merge terminals with summed flows at nodes
    terminals_nodes = terminals_flows.merge(flows_summed, on='Terminal.TopologicalNode', how='left')
    # Get equipment names
    all_names = ((merged_models.query('KEY == "IdentifiedObject.name"')[['ID', 'VALUE']])
                 .rename(columns={'ID': 'Terminal.ConductingEquipment', 'VALUE': 'ConductingEquipment.name'}))
    equipment_names = (merged_models.query('KEY == "Type"')[['ID', 'VALUE']]
                       .drop_duplicates().rename(columns={'ID': 'Terminal.ConductingEquipment',
                                                          'VALUE': 'ConductingEquipment.Type'}))
    switches_retain = merged_models.query('KEY == "Switch.retained"')
    switches_open = merged_models.query('KEY == "Switch.open"')
    switches_state = (switches_retain[['ID', 'VALUE']]
                      .rename(columns={'VALUE': 'retained'})
                      .merge(switches_open[['ID', 'VALUE']]
                             .rename(columns={'VALUE': 'opened'}), on='ID'))
    equipment_names = equipment_names.merge(switches_state.rename(columns={'ID': 'Terminal.ConductingEquipment'}),
                                            on='Terminal.ConductingEquipment', how='left')
    equipment_names = equipment_names.merge(all_names, on='Terminal.ConductingEquipment', how='left')
    # Merge terminals with equipment names
    terminals_equipment = terminals_nodes.merge(equipment_names, on='Terminal.ConductingEquipment', how='left')
    # Merge terminals with origins
    terminals_equipment = terminals_equipment.merge(all_topological_nodes, on='Terminal.TopologicalNode', how='left')
    terminals_equipment = terminals_equipment.sort_values(by=['Terminal.TopologicalNode'])
    # Get error nodes from report
    ok_lines = (terminals_equipment.merge(ok_nodes, on='Terminal.TopologicalNode')
                .sort_values(by=['Terminal.TopologicalNode']))
    nok_lines = (terminals_equipment.merge(nok_nodes, on='Terminal.TopologicalNode')
                 .sort_values(by=['Terminal.TopologicalNode']))
    if not nok_nodes.empty:
        logger.warning(f"Sum of flows {len(nok_nodes.index)} TN is not zero, containing {len(nok_lines.index)} devices")
    if store_results:
        time_moment_now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        file_name_all = f"{file_name}_{time_moment_now}.csv"
        if path_local_storage:
            check_and_create_the_folder_path(path_local_storage)
            file_name_all = path_local_storage.removesuffix('/') + '/' + file_name_all.removeprefix('/')
        terminals_equipment.to_csv(file_name_all)
    nok_tsos = (nok_lines.copy(deep=True)[['Terminal.TopologicalNode', 'modelingAuthoritySet']]
                .drop_duplicates(subset='Terminal.TopologicalNode', keep='last')
                .groupby('modelingAuthoritySet').size().reset_index(name='NodeCount'))
    ok_tsos = (ok_lines.copy(deep=True)[['Terminal.TopologicalNode', 'modelingAuthoritySet']]
               .drop_duplicates(subset='Terminal.TopologicalNode', keep='last')
               .groupby('modelingAuthoritySet').size().reset_index(name='NodeCount'))
    return ok_tsos, nok_tsos


def get_filename_dataframe_from_minio(minio_bucket: str,
                                      minio_client: minio.ObjectStorage = None,
                                      sub_folder: str = None):
    """
    Gets file list from minio bucket (prefix can be specified with sub folder) and converts to dataframe following
    the standard naming convention (see get_metadata_from_filename for more details)
    :param minio_client: if given
    :param minio_bucket: the name of the bucket
    :param sub_folder: prefix
    """
    minio_client = minio_client or minio.ObjectStorage()
    list_of_files = minio_client.list_objects(bucket_name=minio_bucket,
                                              prefix=sub_folder,
                                              recursive=True)
    file_name_list = []
    for file_name in list_of_files:
        try:
            # Escape prefixes
            if not file_name.object_name.endswith(SEPARATOR_SYMBOL):
                path_list = file_name.object_name.split(SEPARATOR_SYMBOL)
                file_metadata = get_metadata_from_filename(path_list[-1])
                file_metadata['full_path'] = file_name.object_name
                file_name_list.append(file_metadata)
        except ValueError:
            continue
        except Exception as ex:
            logger.warning(f"Exception when parsing the filename: {ex}")
            continue
    exploded_results = pandas.DataFrame(file_name_list)
    return exploded_results


def get_latest_missing_profile_from_dataframe(input_frame: pandas.DataFrame, scenario_date):
    """
    Gets the closest match from the input dataframe to the current scenario date considering the version number
    :param input_frame: Dataframe to filter
    :param scenario_date: given scenario date
    :return closest match(es)
    """
    if isinstance(scenario_date, str):
        try:
            scenario_date = pandas.to_datetime(scenario_date)
        except Exception:
            return pandas.DataFrame()
    past_dates = input_frame[pandas.to_datetime(input_frame["Model.scenarioTime"]) < scenario_date]
    past_date = past_dates[abs(pandas.to_datetime(past_dates['Model.scenarioTime']) - scenario_date) ==
                           min(abs(pandas.to_datetime(past_dates['Model.scenarioTime']) - scenario_date))]
    past_latest = past_date[past_date['Model.version'] == past_date['Model.version'].max()]
    return past_latest


def get_latest_set_of_full_filenames(input_dataframe,
                                     reference_dataframe: pandas.DataFrame = None,
                                     profile_types: list = None,
                                     minio_sub_folder: str = 'CGMES',
                                     minio_bucket: str = 'opdm-data',
                                     minio_client: minio.ObjectStorage = None):
    """
    Filters the input dataframe to get a necessary set of profile file names
    :param input_dataframe: Dataframe that needs to be filtered
    :param reference_dataframe: dataframe of containing all the entries
    :param profile_types: list of profiles
    :param minio_client: instance of minio where to look additional profiles
    :param minio_bucket: name of the bucket in minio where to look additional models
    :param minio_sub_folder: prefix ot ease the search (NB! searches by modeling entity)
    :return filtered dataframe of profiles
    """
    if profile_types is None:
        profile_types = ['EQ', 'SSH', 'SV', 'TP']
    missing_profiles = []
    version_list = []
    for profile_type in profile_types:
        single_profile = input_dataframe[input_dataframe['Model.messageType'] == profile_type]
        if not single_profile.empty:
            single_profile = single_profile[single_profile['Model.version'] == single_profile['Model.version'].max()]
            version_list.append(single_profile)
        else:
            missing_profiles.append(profile_type)
    # if not version_list:
    #     return pandas.DataFrame()
    max_version_number = pandas.concat(version_list)
    # max_version_number = input_dataframe[input_dataframe['Model.version'] == input_dataframe['Model.version'].max()]
    if len(max_version_number.index) == len(profile_types):
        tso_profiles = max_version_number['Model.messageType'].tolist()
        if all(item in tso_profiles for item in profile_types):
            return max_version_number
    else:
        modeling_entity = max_version_number['Model.modelingEntity'].iloc[0]
        scenario_date = pandas.to_datetime(max_version_number['Model.scenarioTime'].iloc[0])
        for missing_profile in missing_profiles:
            logger.warning(f"Missing {missing_profile} for {modeling_entity} at "
                           f"{max_version_number['Model.scenarioTime'].iloc[0]}")
            # Check if it is possible to get same profile from previous timestamp
            missing_names = pandas.DataFrame()
            if reference_dataframe is not None:
                missing_names = reference_dataframe[(reference_dataframe['Model.modelingEntity'] == modeling_entity) &
                                                    (reference_dataframe['Model.messageType'] == missing_profile)]
            if missing_names.empty:
                # Check if it is stored separately
                minio_client = minio_client or minio.ObjectStorage()
                sub_folder = minio_sub_folder + '/' + modeling_entity
                missing_names = get_filename_dataframe_from_minio(minio_bucket=minio_bucket,
                                                                  sub_folder=sub_folder,
                                                                  minio_client=minio_client)
            if not missing_names.empty:
                # Get the closest match
                past_latest = get_latest_missing_profile_from_dataframe(input_frame=missing_names,
                                                                        scenario_date=scenario_date)
                if len(past_latest.index) == 1:
                    logger.info(f"Adding closest {missing_profile} reference to {modeling_entity}")
                    max_version_number = pandas.concat([max_version_number, past_latest])
                    continue
            logger.warning(f"Skipping {modeling_entity}, unable to solve {missing_profile}")
            return pandas.DataFrame()
    return max_version_number


def get_list_of_models_from_minio_by_timestamp_time_horizon(timestamp_to_search: str = None,
                                                            bucket_name: str = 'opdm-data',
                                                            sub_folder: str = 'CGMES',
                                                            use_latest_scenario_date: bool = False,
                                                            minio_client: minio.ObjectStorage = None,
                                                            required_profiles: list = None,
                                                            time_horizon_to_search: str = None,
                                                            include_merging_agents: bool = False):
    """
    Gets the content list of minio specified by bucket and sub-folder(prefix), filters by timestamp and time horizon
    and returns dataframe consisting of paths for profiles the TSOs
    :param timestamp_to_search: scenario timestamp for the models
    :param time_horizon_to_search: time horizon for the models
    :param required_profiles: list of profiles that should be present in a single set
    :param minio_client: instance of minio where to look additional profiles
    :param use_latest_scenario_date:
    :param bucket_name: name of the bucket in minio where to look additional models
    :param sub_folder: prefix ot ease the search (NB! searches by modeling entity)
    :param include_merging_agents:
    :return filtered dataframe of profiles
    """
    minio_client = minio_client or minio.ObjectStorage()
    default_sub_folder = sub_folder
    if time_horizon_to_search:
        sub_folder = sub_folder + '/' + time_horizon_to_search
    all_filenames = get_filename_dataframe_from_minio(minio_bucket=bucket_name,
                                                      sub_folder=sub_folder,
                                                      minio_client=minio_client)
    if all_filenames.empty:
        return pandas.DataFrame()
    if timestamp_to_search:
        scenario_dates = all_filenames[all_filenames['Model.scenarioTime'] == timestamp_to_search]
    elif use_latest_scenario_date:
        scenario_dates = all_filenames[all_filenames['Model.scenarioTime'] == all_filenames['Model.scenarioTime'].max()]
    else:
        scenario_dates = all_filenames
    if not include_merging_agents:
        scenario_dates = scenario_dates[scenario_dates['Model.mergingEntity'] == '']
    # if not scenario_dates.empty and time_horizon_to_search:
    #     scenario_dates = scenario_dates[scenario_dates['Model.processType'] == time_horizon_to_search]
    latest_entries = (scenario_dates.groupby('Model.modelingEntity')
                      .apply(lambda x: get_latest_set_of_full_filenames(input_dataframe=x,
                                                                        reference_dataframe=all_filenames,
                                                                        minio_sub_folder=default_sub_folder,
                                                                        profile_types=required_profiles,
                                                                        minio_bucket=bucket_name,
                                                                        minio_client=minio_client)))
    return latest_entries


def download_latest_boundary(path_to_local_storage: str,
                             bucket_name: str = 'opdm-data',
                             sub_folder: str = 'CGMES/ENTSOE',
                             minio_client: minio.ObjectStorage = None):
    """
    Downloads the latest boundary files to the location indicated
    :param path_to_local_storage: where to store the boundary
    :param bucket_name: name of the bucket
    :param sub_folder: location in minio bucket where boundary data is
    :param minio_client: minio instance
    :return boundary profile
    """
    if not path_to_local_storage:
        return
    minio_client = minio_client or minio_api.ObjectStorage()
    boundary_file_list = get_list_of_models_from_minio_by_timestamp_time_horizon(sub_folder=sub_folder,
                                                                                 use_latest_scenario_date=True,
                                                                                 bucket_name=bucket_name,
                                                                                 minio_client=minio_client,
                                                                                 required_profiles=['EQBD', 'TPBD'])
    download_sets_from_minio(file_list=boundary_file_list,
                             column_name='full_path',
                             minio_client=minio_client,
                             bucket_name=bucket_name,
                             full_file_path=path_to_local_storage)


def download_sets_from_minio(file_list: pandas.DataFrame,
                             column_name: str = 'full_path',
                             minio_client: minio.ObjectStorage = None,
                             bucket_name: str = 'opdm-data',
                             full_file_path: str = None):
    """
    Downloads files from mino using paths from dataframe to location indicated
    :param file_list: dataframe containing profile data
    :param column_name: name of the column where paths are located
    :param minio_client: minio client if set
    :param bucket_name: name of the bucket where profiles are
    :param full_file_path: local folder path
    """
    if file_list is None or file_list.empty or not full_file_path:
        return
    minio_client = minio_client or minio.ObjectStorage()
    for single_path in file_list[column_name]:
        model_data = minio_client.download_object(bucket_name, single_path)
        if model_data:
            model_data_name = os.path.basename(single_path)
            full_file_name = full_file_path.removesuffix('/') + '/' + str(model_data_name).removeprefix('/')
            with open(full_file_name, 'wb') as profile_to_write:
                profile_to_write.write(model_data)
            logger.info(f"{single_path} is downloaded to {full_file_path}")


def download_models_from_minio_to_local_storage(path_to_local_storage: str,
                                                timestamp: str,
                                                time_horizon: str,
                                                sub_folder: str = None,
                                                bucket_name: str = 'opdm-data',
                                                minio_prefix: str = 'CGMES',
                                                minio_client: minio.ObjectStorage = None):
    """
    Main function to download models from minio to specified folder by timestamp and time horizon
    If subfolder is given then it checks if it exists and there is something, otherwise it creates the name of subfolder
    from the timestamp to the path_to_local_storage.
    If minio instance is not provided then it creates one of its own based on the default parameters minio.properties
    Note this if multiple minio instances are used
    :param path_to_local_storage: main folder where models are stored
    :param sub_folder: if specified then checks this folder otherwise it creates the name from timestamp
    :param timestamp: time stamp of the models
    :param time_horizon: time horizon of the models
    :param bucket_name: name of the bucket in minio
    :param minio_prefix: prefix (folder name) in bucket where models are located
    :param minio_client: instance of minio client
    """
    minio_client = minio_client or minio.ObjectStorage()
    if sub_folder is None:
        sub_folder = parse_datetime(timestamp)
        sub_folder = sub_folder.strftime('%Y-%m-%d-%H-%M')
        sub_folder = f"{sub_folder}Z"
        if time_horizon:
            sub_folder = f"{sub_folder}-{time_horizon}"
    full_file_path = path_to_local_storage.removesuffix('/') + '/' + sub_folder.removeprefix('/')
    check_and_create_the_folder_path(full_file_path)
    if len(os.listdir(full_file_path)) > 0:
        return os.path.normpath(full_file_path)
    file_list = get_list_of_models_from_minio_by_timestamp_time_horizon(timestamp_to_search=timestamp,
                                                                        bucket_name=bucket_name,
                                                                        sub_folder=minio_prefix,
                                                                        time_horizon_to_search=time_horizon,
                                                                        minio_client=minio_client)
    download_sets_from_minio(file_list=file_list,
                             column_name='full_path',
                             minio_client=minio_client,
                             bucket_name=bucket_name,
                             full_file_path=full_file_path)
    download_latest_boundary(path_to_local_storage=full_file_path,
                             minio_client=minio_client,
                             bucket_name=bucket_name,
                             sub_folder=minio_prefix.removesuffix('/') + '/' + 'ENTSOE')
    return os.path.normpath(full_file_path)


def run_some_function(time_stamp_name: str,
                      time_horizon_name: str = None,
                      included_models: list = None,
                      excluded_models: list = None,
                      igms_folder: str = r'C:\IGMS',
                      bucket_name: str = None,
                      minio_prefix: str = None,
                      load_in_merged_model: bool = False,
                      test_version: str = '012',
                      test_merging_entity: str = 'BALTICRSC',
                      test_merging_area: str = 'EU',
                      send_to_opdm: bool = False,
                      test_mas: str = 'http://www.baltic-rsc.eu/OperationalPlanning/CGM',
                      use_local: bool = True,
                      special_boundary_loc: str = None,
                      use_time_scenario: bool = False,
                      output_folder: str = None,
                      minio_import_models: list = None,
                      validation_needed: bool = False,
                      folder_to_study: str = None,
                      run_scaling: bool = True,
                      test_scenario_datetime_values: list = None,
                      excluded_profiles:list = None
                      ):
    if not folder_to_study:
        folder_to_study = download_models_from_minio_to_local_storage(timestamp=time_stamp_name,
                                                                      time_horizon=time_horizon_name,
                                                                      # sub_folder='202410251630',
                                                                      minio_prefix=minio_prefix,
                                                                      bucket_name=bucket_name,
                                                                      path_to_local_storage=igms_folder)
    overwrite_local_date_time = True
    if not test_scenario_datetime_values:
        test_scenario_datetime_values = ['2024-08-01T12:30:00+00:00']
    test_time_horizon = 'ID'
    examples_path = None
    for test_scenario_datetime in test_scenario_datetime_values:
        logger.info(f"Executing {test_scenario_datetime}")
        merged_ssh_profiles = []
        merged_sv_profiles = []
        time_scenario_val = test_scenario_datetime if use_time_scenario else None
        boundary_folder = special_boundary_loc or folder_to_study
        if use_local:
            loaded_boundary = file_system.get_latest_boundary(path_to_directory=boundary_folder,
                                                              local_folder_for_examples=examples_path)
            if load_in_merged_model:
                combined_models = file_system.get_latest_models_and_download(path_to_directory=folder_to_study,
                                                                             scenario_date=time_scenario_val,
                                                                             bypass_profiles=['EQ'],
                                                                             local_folder_for_examples=examples_path,
                                                                             allow_merging_entities=True)
                for model in combined_models:
                    merged_ssh_component = None
                    for component in model.get('opde:Component', {}):
                        if component.get('opdm:Profile', {}).get('pmd:mergingEntity'):
                            if component.get('opdm:Profile', {}).get('pmd:cgmesProfile') == 'SSH':
                                merged_ssh_component = component
                            if component.get('opdm:Profile', {}).get('pmd:cgmesProfile') == 'SV':
                                merged_sv_profiles.append(model)
                    if merged_ssh_component:
                        copied_model = copy.deepcopy(model)
                        model.get('opde:Component', {}).remove(merged_ssh_component)
                        copied_model['opde:Component'] = [merged_ssh_component]
                        merged_ssh_profiles.append(copied_model)
                valid_models = [model for model in combined_models if model not in merged_sv_profiles]

            else:
                valid_models = file_system.get_latest_models_and_download(path_to_directory=folder_to_study,
                                                                          scenario_date=time_scenario_val,
                                                                          local_folder_for_examples=examples_path)

            if overwrite_local_date_time:
                test_scenario_datetime = valid_models[0].get('pmd:scenarioDate', test_scenario_datetime)
                test_time_horizon = valid_models[0].get('pmd:timeHorizon', test_time_horizon)
                if test_time_horizon not in ['1D', '2D', 'YR', 'WK']:
                    test_time_horizon = 'ID'
        else:
            loaded_boundary = models.get_latest_boundary()
            # opdm_client = OPDM()
            # valid_models = opdm_client.get_latest_models_and_download(test_time_horizon, test_scenario_datetime)
            valid_models = models.get_latest_models_and_download(test_time_horizon, test_scenario_datetime, valid=True)
            validation_needed = False

        # Filter out models that are not to be used in merge
        filtered_models = filter_models(valid_models, included_models, excluded_models, filter_on='pmd:TSO')

        # Get additional models directly from Minio
        if minio_import_models:
            minio_service = minio_api.ObjectStorage()
            additional_models = minio_service.get_latest_models_and_download(time_horizon=test_time_horizon,
                                                                             scenario_datetime=test_scenario_datetime,
                                                                             model_entity=minio_import_models,
                                                                             bucket_name=INPUT_MINIO_BUCKET,
                                                                             prefix=INPUT_MINIO_FOLDER)
            test_input_models = filtered_models + additional_models
        else:
            test_input_models = filtered_models

        available_models = []
        invalid_models = []
        if validation_needed:
            for model in test_input_models:
                try:
                    response = validate_model([model, loaded_boundary])
                    model["VALIDATION_STATUS"] = response
                    if response["valid"]:
                        available_models.append(model)
                    else:
                        invalid_models.append(model)
                except Exception:
                    invalid_models.append(model)
                    logger.error("Validation failed")
            [print(dict(tso=model['pmd:TSO'], valid=model.get('VALIDATION_STATUS', {}).get('valid'),
                        duration=model.get('VALIDATION_STATUS', {}).get('validation_duration_s'))) for model in
             test_input_models]
        else:
            available_models = test_input_models
        if not available_models:
            logger.error("No models to merge :(")
            sys.exit()

        test_model = merge_models(list_of_models=available_models,
                                  latest_boundary=loaded_boundary,
                                  time_horizon=test_time_horizon,
                                  scenario_datetime=test_scenario_datetime,
                                  merging_area=test_merging_area,
                                  merging_entity=test_merging_entity,
                                  path_local_storage=output_folder,
                                  mas=test_mas,
                                  version=test_version,
                                  push_to_opdm=send_to_opdm,
                                  pre_sv_profile=merged_sv_profiles,
                                  pre_ssh_profiles=merged_ssh_profiles,
                                  use_merged_profiles=False,
                                  run_scaling=run_scaling,
                                  excluded_profiles=excluded_profiles
                                  )

        check_and_create_the_folder_path(output_folder)
        full_name = output_folder.removesuffix('/') + '/' + test_model.name.removeprefix('/')
        with open(full_name, 'wb') as write_file:
            write_file.write(test_model.getbuffer())


if __name__ == '__main__':
    import sys

    where_to_store_stuff = r'C:\where_all_output_goes'
    this_run = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    where_to_store_stuff = os.path.join(where_to_store_stuff, this_run)

    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout),
                  PyPowsyblLogGatheringHandler(topic_name='cgm_merge',
                                               send_to_elastic=False,
                                               upload_to_minio=False,
                                               logging_policy=PyPowsyblLogReportingPolicy.ALL_ENTRIES,
                                               print_to_console=False,
                                               path_to_local_folder=where_to_store_stuff,
                                               report_level=logging.ERROR)]
    )
    # Specify some timestamps, at least one has to be in order for the loop to work
    timestamps = [
        '20241122T0030Z',
        # '20241122T0130Z',
        # '20241122T0230Z',
        # '20241122T0330Z',
        # '20241122T0430Z',
        # '20241122T0530Z',
        # '20241122T0630Z',
        # '20241122T0730Z',
        # '20241122T0830Z',
        # '20241122T0930Z',
        # '20241122T1030Z',
        # '20241122T1130Z',
        # '20241122T1230Z',
        # '20241122T1330Z',
        # '20241122T1430Z',
        # '20241122T1530Z',
        # '20241122T1630Z',
        # '20241122T1730Z',
        # '20241122T1830Z',
        # '20241122T1930Z',
        # '20241122T2030Z',
        # '20241122T2130Z',
        # '20241122T2230Z',
    ]
    # either found igms are filtered by scenario date time, useful if folder consists of several timestamps
    use_scenario_for_filtering = True
    # Specify tsos to be included
    included_tsos = []
    # Specify tsos to be excluded
    excluded_tsos = []
    # Excplitly specify where igms are (if set then looks there and doesn't download if anything was found(
    igm_folder_to_study = r'C:\folder_where_input_igms_exists'
    # Bucket name in Minio
    igm_bucket_name = 'opdm-data'
    # Path in minio for models to ease the search
    igm_minio_prefix = "CGMES"
    # Either to load and use already merged model, weird stuff can happen
    use_merged_model = False
    # time horizon
    time_horizon = '1D'
    # if igm_folder_to_study is None then it parses scenario date time to year-month-date-hour-minute-time_horizon
    # searches it in this folder, checks if it has content and if not then goes to PDN and starts download everything
    # that meets the criteria
    folder_where_to_store_igms = r'C:\where_to_store_igms_from_pdn'
    # Speify version number
    version_number = '001'
    # CGM related variables
    merging_entity = 'BALTICRSC'
    mas = 'http://www.baltic-rsc.eu/OperationalPlanning/CGM'
    merging_area = 'BA'
    # Specify tsos to be downloaded from minio
    minio_models = None
    # Run through the validation pipeline
    validate_models = False
    # Apply scaling
    scale_models = False
    # Send output to opdm
    send_cgm_to_opdm = False
    # Merge files from local storage
    use_local_files = True
    # if boundary is not in same folder with igms specify the location explicitly
    boundary_location = None
    # specify igm profiles to be excluded in final cgm (for example duplicate SSH, SV(
    exclude_igm_profiles_in_cgm = []
    for time_stamp_to_study in timestamps:
        run_some_function(time_stamp_name=time_stamp_to_study,
                          time_horizon_name=time_horizon,
                          included_models=included_tsos,
                          excluded_models=excluded_tsos,
                          igms_folder=folder_where_to_store_igms,
                          bucket_name=igm_bucket_name,
                          minio_prefix=igm_minio_prefix,
                          load_in_merged_model=use_merged_model,
                          test_version=version_number,
                          test_merging_entity=merging_entity,
                          test_merging_area=merging_area,
                          test_mas=mas,
                          send_to_opdm=send_cgm_to_opdm,
                          use_local=use_local_files,
                          special_boundary_loc=boundary_location,
                          use_time_scenario=use_scenario_for_filtering,
                          test_scenario_datetime_values=[time_stamp_to_study],
                          output_folder=where_to_store_stuff,
                          minio_import_models=minio_models,
                          validation_needed=validate_models,
                          folder_to_study=igm_folder_to_study,
                          run_scaling=scale_models,
                          excluded_profiles=exclude_igm_profiles_in_cgm
                          )



