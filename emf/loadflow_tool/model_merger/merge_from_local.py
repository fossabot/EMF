import logging
import os.path
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from uuid import uuid4
from zipfile import ZipFile

import pandas
import pypowsybl.network
import triplets.rdf_parser

from aniso8601 import parse_datetime

import config
from emf.common.config_parser import parse_app_properties
from emf.common.integrations import minio_api, opdm
from emf.common.integrations.object_storage import models, file_system
from emf.common.integrations.object_storage.file_system_general import check_and_create_the_folder_path
from emf.common.logging.pypowsybl_logger import PyPowsyblLogGatheringHandler, PyPowsyblLogReportingPolicy
from emf.loadflow_tool import loadflow_settings
from emf.loadflow_tool.helper import load_model, load_opdm_data, create_opdm_objects
from emf.loadflow_tool.model_merger import merge_functions
from emf.loadflow_tool.model_merger.handlers.cgm_handler import async_call, log_opdm_response
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
    voltages_to_discard = (sv_voltage_values.groupby(['SvVoltage.SvTopologicalNode']).
                           apply(lambda x: take_best_match_for_sv_voltage(input_data=x,
                                                                          column_name='SvVoltage.v',
                                                                          to_keep=False), include_groups=False))
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
    if existing_dependencies.empty or len(existing_dependencies.index) < len(dependencies.index):
        logger.info(f"Missing dependencies. Adding {len(dependencies.index)} dependencies to SV profile")
        full_model_id = cgm_sv_data[(cgm_sv_data['KEY'] == 'Type') & (cgm_sv_data['VALUE'] == 'FullModel')]
        new_dependencies = dependencies[['ID']].copy().rename(columns={'ID': 'VALUE'}).reset_index(drop=True)
        new_dependencies.loc[:, 'KEY'] = 'Model.DependentOn'
        new_dependencies.loc[:, 'ID'] = full_model_id['ID'].iloc[0]
        new_dependencies.loc[:, 'INSTANCE_ID'] = full_model_id['INSTANCE_ID'].iloc[0]
        cgm_sv_data = triplets.rdf_parser.update_triplet_from_triplet(cgm_sv_data, new_dependencies)
    return cgm_sv_data


def get_nodes_against_kirchhoff_first_law(cgm_sv_data,
                                          original_models,
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
    # Get power flow after lf
    power_flow = cgm_sv_data.type_tableview('SvPowerFlow')[['SvPowerFlow.Terminal', 'SvPowerFlow.p', 'SvPowerFlow.q']]
    # Get terminals
    terminals = original_models.type_tableview('Terminal').rename_axis('Terminal').reset_index()
    terminals = terminals[['Terminal', 'Terminal.ConductingEquipment', 'Terminal.TopologicalNode']]
    # Calculate summed flows per topological node
    flows_summed = ((power_flow.merge(terminals, left_on='SvPowerFlow.Terminal', right_on='Terminal', how='left')
                     .groupby('Terminal.TopologicalNode')[['SvPowerFlow.p', 'SvPowerFlow.q']]
                     .sum()).rename_axis('Terminal.TopologicalNode').reset_index())
    # Get topological nodes that have mismatch
    nok_nodes = flows_summed[(abs(flows_summed['SvPowerFlow.p']) > sv_injection_limit) |
                             (abs(flows_summed['SvPowerFlow.q']) > sv_injection_limit)][['Terminal.TopologicalNode']]
    if nodes_only:
        return nok_nodes
    terminals_nodes = terminals.merge(flows_summed, on='Terminal.TopologicalNode', how='left')
    terminals_nodes = terminals_nodes.merge(nok_nodes, on='Terminal.TopologicalNode')
    return terminals_nodes


def revert_failed_buses(cgm_sv_data,
                        original_models,
                        failed_buses: pandas.DataFrame = None,
                        network_instance: pypowsybl.network = None,
                        sv_injection_limit: float = SV_INJECTION_LIMIT,
                        revert_failed_terminals: bool = False):
    """
    Reverts back flows in terminals which are located on the buses that were not calculated.
    To revert the flows back to original state following criteria has to be met:
    1) Bus state is either "FAILED" ("...considered dead and no calculation is performed...") or NO_CALCULATION
    2) For terminal there is a difference of power flows when comparing original and merged model
    3) Terminal is located in the topological node which sum of power flows differs from more than sv_injection_limit
    :param cgm_sv_data: merged SV profile (needed to set the flows for terminals)
    :param original_models: IGMs (triplets, dictionary)
    :param failed_buses: dataframe of failed buses
    :param network_instance: pypowsybl network instance
    :param sv_injection_limit: threshold for deciding whether the node is violated by sum of flows
    :param revert_failed_terminals: set it true to revert the flows for terminals
    :return updated merged SV and SSH profiles
    """
    # Check that input is triplets
    original_models = get_opdm_data_from_models(model_data=original_models)
    # Get terminals merged with nok nodes (see condition 2)
    nok_nodes = get_nodes_against_kirchhoff_first_law(cgm_sv_data=cgm_sv_data,
                                                      original_models=original_models,
                                                      sv_injection_limit=sv_injection_limit)
    # Filter buses that failed (see condition 1)
    types = {'status': [pypowsybl.loadflow.ComponentStatus.FAILED, pypowsybl.loadflow.ComponentStatus.NO_CALCULATION]}
    failed_buses = failed_buses.merge(pandas.DataFrame(types), on='status')
    if failed_buses.empty:
        return cgm_sv_data
    # Get terminals from network and merge them with failed buses
    all_terminals = network_instance.get_terminals().rename_axis('Terminal.ConductingEquipment').reset_index()
    failed_bus_terminals = all_terminals.merge(failed_buses[['id']].rename(columns={'id': 'bus_id'}), on='bus_id')
    # Filter terminals that were in buses that failed (not calculated)
    failed_terminals = (nok_nodes.rename(columns={'Terminal': 'SvPowerFlow.Terminal'})
                        .merge(failed_bus_terminals, on='Terminal.ConductingEquipment'))
    if failed_terminals.empty:
        return cgm_sv_data
    # Get power flow differences
    old_power_flows = original_models.type_tableview('SvPowerFlow')[['SvPowerFlow.Terminal',
                                                                     'SvPowerFlow.p', 'SvPowerFlow.q']]
    new_power_flows = cgm_sv_data.type_tableview('SvPowerFlow')
    power_flow_diff = old_power_flows.merge(new_power_flows[['SvPowerFlow.Terminal', 'SvPowerFlow.p', 'SvPowerFlow.q']],
                                            on='SvPowerFlow.Terminal', suffixes=('_pre', '_post'))
    power_flow_diff = power_flow_diff.merge(failed_terminals[['SvPowerFlow.Terminal', 'Terminal.TopologicalNode']],
                                            on='SvPowerFlow.Terminal')
    # Filter those terminals which had flows updated (see condition 3)
    try:
        power_flow_diff['CHANGED'] = ((abs(power_flow_diff['SvPowerFlow.p_pre']
                                           - power_flow_diff['SvPowerFlow.p_post'])
                                       + abs(power_flow_diff['SvPowerFlow.q_pre']
                                             - power_flow_diff['SvPowerFlow.q_post'])) != 0)
        power_flow_diff = power_flow_diff[power_flow_diff.eval('CHANGED')]
    except IndexError:
        return cgm_sv_data
    updated_power_flows = (new_power_flows[['SvPowerFlow.Terminal', 'Type']].reset_index()
                           .merge(power_flow_diff[['SvPowerFlow.Terminal']], on='SvPowerFlow.Terminal'))
    updated_power_flows = updated_power_flows.merge(old_power_flows, on='SvPowerFlow.Terminal')
    logger.error(f"Found flows for {len(power_flow_diff.index)} terminals "
                 f"in {len(power_flow_diff['Terminal.TopologicalNode'].unique())} TNs "
                 f"that were updated but not calculated")
    if revert_failed_terminals:
        logger.warning(f"Reverted flows for {len(power_flow_diff.index)} terminals")
        cgm_sv_data = triplets.rdf_parser.update_triplet_from_tableview(cgm_sv_data, updated_power_flows)
    return cgm_sv_data


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
        logger.warning(f"Found {len(between_tn.index)} not retained switches between topological nodes")
        if open_not_retained_switches:
            logger.warning(f"Opening not retained switches")
            open_switches = closed_switches.merge(between_tn[['ID']], on='ID')
            open_switches.loc[:, 'VALUE'] = 'true'
            original_data = triplets.rdf_parser.update_triplet_from_triplet(original_data, open_switches)
            updated_switches = True
    return original_data, updated_switches


def get_failed_buses(load_flow_results: list, network_instance: pypowsybl.network, fail_types=None):
    """
    Gets dataframe of failed buses for postprocessing
    :param load_flow_results: list of load flow results for connected components
    :param network_instance: network instance to get buses
    :param fail_types: list of fail types
    :return dataframe of failed buses
    """
    if not fail_types:
        fail_types = [pypowsybl.loadflow.ComponentStatus.FAILED,
                      pypowsybl.loadflow.ComponentStatus.NO_CALCULATION,
                      pypowsybl.loadflow.ComponentStatus.CONVERGED,
                      pypowsybl.loadflow.ComponentStatus.MAX_ITERATION_REACHED,
                      # pypowsybl.loadflow.ComponentStatus.SOLVER_FAILED
                      ]
    max_iteration = len([result for result in load_flow_results
                         if result['status'] == pypowsybl.loadflow.ComponentStatus.MAX_ITERATION_REACHED])
    successful = len([result for result in load_flow_results
                      if result['status'] == pypowsybl.loadflow.ComponentStatus.CONVERGED])
    not_calculated = len([result for result in load_flow_results
                          if result['status'] == pypowsybl.loadflow.ComponentStatus.NO_CALCULATION])
    failed = len([result for result in load_flow_results
                  if result['status'] == pypowsybl.loadflow.ComponentStatus.FAILED])
    sum_of_messages = len(load_flow_results) - successful - failed - not_calculated - max_iteration
    messages = [f"Networks: {len(load_flow_results)}"]
    if successful:
        messages.append(f"CONVERGED: {successful}")
    if failed:
        messages.append(f"INVALID: {failed}")
    if not_calculated:
        messages.append(f"NOT CALCULATED: {not_calculated}")
    if max_iteration:
        messages.append(f"MAX ITERATION: {max_iteration}")
    if sum_of_messages:
        messages.append(f"OTHER CRITICAL ERROR: {sum_of_messages}")
    message = '; '.join(messages)
    if max_iteration or sum_of_messages > 0:
        logger.error(message)
    else:
        logger.info(message)
    troublesome_buses = pandas.DataFrame([result for result in load_flow_results if result['status'] in fail_types])
    # troublesome_buses = pandas.concat([failed, not_calculated])
    if not troublesome_buses.empty:
        troublesome_buses = (network_instance.get_buses().reset_index()
                             .merge(troublesome_buses
                                    .rename(columns={'connected_component_num': 'connected_component'}),
                                    on='connected_component'))
    return troublesome_buses


def merge_models(list_of_models: list,
                 latest_boundary: dict,
                 time_horizon: str,
                 scenario_datetime: str,
                 merging_area: str,
                 merging_entity: str,
                 mas: str,
                 version: str = '001',
                 push_to_opdm: bool = False):
    # Load all selected models
    input_models = list_of_models + [latest_boundary]

    parameters = {
        "iidm.import.cgmes.import-node-breaker-as-bus-breaker": 'true',
    }
    try:
        assembled_data = merge_functions.load_opdm_data(input_models)
        assembled_data = triplets.cgmes_tools.update_FullModel_from_filename(assembled_data)
        assembled_data = merge_functions.configure_paired_boundarypoint_injections_by_nodes(assembled_data)
        escape_upper_xml = assembled_data[assembled_data['VALUE'].astype(str).str.contains('.XML')]
        if not escape_upper_xml.empty:
            escape_upper_xml['VALUE'] = escape_upper_xml['VALUE'].str.replace('.XML', '.xml')
            assembled_data = triplets.rdf_parser.update_triplet_from_triplet(assembled_data, escape_upper_xml,
                                                                             update=True,
                                                                             add=False)
        all_input_files = merge_functions.export_to_cgmes_zip([assembled_data])

        input_models = create_opdm_objects([all_input_files])
        assembled_data, updated = handle_not_retained_switches_between_nodes(assembled_data,
                                                                             open_not_retained_switches=True
                                                                             )
        # updated = False
        if updated:
            updated_input_files = merge_functions.export_to_cgmes_zip([assembled_data])
            updated_models = create_opdm_objects([updated_input_files])
        else:
            updated_models = input_models
        del assembled_data
    except KeyError as error:
        updated_models = input_models
        logger.error(f"Unable to preprocess: {error}")
    # merged_model = load_model(input_models, parameters=parameters)
    merged_model = load_model(updated_models, parameters=parameters, skip_default_parameters=True)
    # TODO - run other LF if default fails
    logger.info(f"Pre loadflow checks")

    logger.info(f"Running loadflow")
    solved_model = run_lf(merged_model, loadflow_settings=loadflow_settings.CGM_RELAXED_2)
    network_itself = solved_model["network"]

    troublesome_buses = get_failed_buses(load_flow_results=merged_model["LOADFLOW_RESULTS"],
                                         network_instance=network_itself)
    logger.info(f"Loadflow done")

    # TODO - get version dynamically form ELK
    # Update time_horizon in case of generic ID process type
    if time_horizon.upper() == "ID":
        # _task_creation_time = parse_datetime(task_creation_time, keep_timezone=False)
        # _scenario_datetime = parse_datetime(scenario_datetime, keep_timezone=False)
        #
        # time_horizon = f"{int((_scenario_datetime - _task_creation_time).seconds / 3600):02d}"
        # logger.info(f"Setting ID TimeHorizon to {time_horizon}")
        # As testing model merging backwards, and models come from anywhere then for ID set the time horizon to some
        # random value: PURELY
        time_horizon = '08'
    sv_data, ssh_data = create_sv_and_updated_ssh(solved_model, input_models, scenario_datetime,
                                                  time_horizon, version, merging_area, merging_entity, mas)
    # Fix SV
    data = merge_functions.load_opdm_data(input_models)
    sv_data = fix_sv_shunts(sv_data, data)
    sv_data = fix_sv_tapsteps(sv_data, ssh_data)
    sv_data = remove_small_islands(sv_data, int(SMALL_ISLAND_SIZE))
    sv_data = check_and_fix_dependencies(cgm_sv_data=sv_data, cgm_ssh_data=ssh_data, original_data=data)
    sv_data = remove_duplicate_voltage_levels_for_topological_nodes(cgm_sv_data=sv_data, original_data=data)
    sv_data = revert_failed_buses(cgm_sv_data=sv_data,
                                  original_models=data,
                                  network_instance=network_itself,
                                  failed_buses=troublesome_buses,
                                  revert_failed_terminals=True)

    # Package both input models and exported CGM profiles to in memory zip files
    serialized_data = export_to_cgmes_zip([ssh_data, sv_data])

    ### Upload to OPDM ###
    if push_to_opdm:
        try:
            opdm_service = opdm.OPDM()
            for item in serialized_data:
                logger.info(f"Uploading to OPDM -> {item.name}")
                async_call(function=opdm_service.publication_request, callback=log_opdm_response,
                           file_path_or_file_object=item)
        except:
            logging.error(f"""Unexpected error on uploading to OPDM:""", exc_info=True)

    # Set RMM name
    rmm_name = f"CGM_{time_horizon}_{version}_{parse_datetime(scenario_datetime):%Y%m%dT%H%MZ}_{merging_area}_{uuid4()}"
    # Exclude following profiles from the IGMS
    # excluded_profiles = ['SV', 'SSH']
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


if __name__ == '__main__':
    import sys
    # Specify the location where to store the results (makes folder with date time of the current run)
    where_to_store_stuff = r'E:\some_user_1\CGM_dump'
    this_run = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    where_to_store_stuff = os.path.join(where_to_store_stuff, this_run)

    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout),
                  # Comment this block ini if pypowsybl logs are needed
                  # PyPowsyblLogGatheringHandler(topic_name='cgm_merge',
                  #                              send_to_elastic=False,
                  #                              upload_to_minio=False,
                  #                              logging_policy=PyPowsyblLogReportingPolicy.ALL_ENTRIES,
                  #                              print_to_console=False,
                  #                              path_to_local_folder=where_to_store_stuff,
                  #                              report_level=logging.ERROR)
                  ])
    # Set this true if local files are used, if this is false then it takes the values in test_scenario_datetime_values
    # and asks them from ELK+Minio
    use_local = True
    # if use_local is true specify the path from where to get the IGMs
    folder_to_study = r'E:\some_user_2\YR_IGM\JAN_2025'
    # Set this true to if running through validator is needed
    validation_needed = False
    # If result should be sent to opdm
    send_to_opdm = False
    # Specify TSOs (as are in file names) who should be included, if empty then all are taken
    included_models = []
    # Specify TSOs who should be excluded, if empty then none is excluded
    excluded_models = []
    # if models are taken from ELK+Minio (use_local = False), specify additional TSOs that should be searched from Minio
    # directly (synchro models)
    local_import_models = []
    # Specify time_horizon and scenario date
    test_time_horizon = '1D'
    test_scenario_datetime_values = ['2024-08-01T12:30:00+00:00']
    # If use_local = True and this is True then it takes scenario date, time horizon from the file names of the IGMs
    overwrite_local_date_time = True
    # Specify some additional parameters
    test_version = '123'
    test_merging_entity = 'BALTICRSC'
    test_merging_area = 'EU'
    test_mas = 'http://www.baltic-rsc.eu/OperationalPlanning/CGM'
    # And now it starts to do something
    for test_scenario_datetime in test_scenario_datetime_values:
        logger.info(f"Executing {test_scenario_datetime}")
        if use_local:
            loaded_boundary = file_system.get_latest_boundary(path_to_directory=folder_to_study,
                                                              local_folder_for_examples=None)
            valid_models = file_system.get_latest_models_and_download(path_to_directory=folder_to_study,
                                                                      local_folder_for_examples=None)
            if overwrite_local_date_time:
                test_scenario_datetime = valid_models[0].get('pmd:scenarioDate', test_scenario_datetime)
                test_time_horizon = valid_models[0].get('pmd:timeHorizon', test_time_horizon)
                try:
                    time_horizon_value = int(test_time_horizon)
                except ValueError:
                    time_horizon_value = None
                if time_horizon_value:
                    test_time_horizon = 'ID'
        else:
            loaded_boundary = models.get_latest_boundary()
            valid_models = models.get_latest_models_and_download(test_time_horizon, test_scenario_datetime, valid=True)
            validation_needed = False

        # Filter out models that are not to be used in merge
        filtered_models = filter_models(valid_models, included_models, excluded_models, filter_on='pmd:TSO')

        # Get additional models directly from Minio
        if local_import_models and not use_local:
            minio_service = minio_api.ObjectStorage()
            additional_models = minio_service.get_latest_models_and_download(time_horizon=test_time_horizon,
                                                                             scenario_datetime=test_scenario_datetime,
                                                                             model_entity=local_import_models,
                                                                             bucket_name=INPUT_MINIO_BUCKET,
                                                                             prefix=INPUT_MINIO_FOLDER)
            test_input_models = filtered_models + additional_models
        else:
            test_input_models = filtered_models

        # SET BRELL LINE VALUES
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
                except:
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
                                  mas=test_mas,
                                  version=test_version,
                                  push_to_opdm=send_to_opdm)

        check_and_create_the_folder_path(where_to_store_stuff)
        full_name = where_to_store_stuff.removesuffix('/') + '/' + test_model.name.removeprefix('/')
        with open(full_name, 'wb') as write_file:
            write_file.write(test_model.getbuffer())
        # test_model.name = 'EMF_test_merge_find_better_place/CGM/' + test_model.name
        # save_minio_service = minio.ObjectStorage()
        # try:
        #     save_minio_service.upload_object(test_model, bucket_name=OUTPUT_MINIO_BUCKET)
        # except:
        #     logging.error(f"""Unexpected error on uploading to Object Storage:""", exc_info=True)
