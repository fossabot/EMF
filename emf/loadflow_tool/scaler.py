"""
EMF REQUIREMENTS:
1. Compare the target values for AC net positions and DC links with the values recorded
after calculating the power flow on the pan-European model.
2. The recorded flow on DC links shall be equal to the target value of the scenario.
3. The recorded AC net position shall be equal to the reference value of the scenario.
4. If discrepancy exists for one or more scheduling areas, between the two values, then a
balance adjustment by adjusting the loads has to be done.
5. The discrepancy thresholds are defined as follows:
6. Sum of AC tieline flows - AC NET Position target < 2MW
7. If the discrepancy occurs as defined in the previous step, the conforming loads of each
scheduling area are modified proportionally in order to match the netted Area AC
position, while maintaining the power factor of the loads.
8. The Jacobian is built for the new power flow iteration and new values for the AC tie line
flows are calculated, in order to check if the conforming loads in the scheduling area have
to be adjusted again.
9. If the power injection in the global slack bus exceeds a configurable threshold, this power
injection shall be redistributed on all generation units in the synchronous area
proportional to the reserve margin.
10. This loop ends:
" When all the differences between the recorded and target values of net positions of
scheduling areas are below the discrepancy thresholds, as defined previously;
" In any case after the 15th iteration16 (adjustments take place within the iterations).

TODO issues identified
1. If trying to scale model where is only one area, then flows on EQINJ does not change. Scaled difference is
distributed on generators


"""
import copy
from emf.loadflow_tool.helper import load_opdm_data
import triplets.rdf_parser
import pypowsybl as pp
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from aniso8601 import parse_datetime
from typing import Dict, List, Union
import config
from emf.common.integrations.elastic import Elastic
from emf.common.config_parser import parse_app_properties
from emf.common.decorators import performance_counter
from emf.common.integrations import elastic
from emf.loadflow_tool.helper import attr_to_dict, get_network_elements, get_slack_generators, \
    get_connected_component_counts
from emf.loadflow_tool.loadflow_settings import CGM_RELAXED_1

logger = logging.getLogger(__name__)

parse_app_properties(caller_globals=globals(), path=config.paths.cgm_worker.scaler)


def query_hvdc_schedules(process_type: str,
                         utc_start: str | datetime,
                         utc_end: str | datetime,
                         area_eic_map: Dict[str, str] | None = None) -> dict | None:
    """
    Method to get HVDC schedules (business type - B63)
    :param process_type: time horizon of schedules; A01 - Day-ahead, A18 - Intraday
    :param utc_start: start time in utc. Example: '2023-08-08T23:00:00Z'
    :param utc_end: end time in utc. Example: '2023-08-09T00:00:00Z'
    :param area_eic_map: dictionary of geographical region names and control area eic code
    :return: schedules in dict format
    """
    # Define area name to eic mapping table
    if not area_eic_map:
        # Using default mapping table from config
        import json
        with open(config.paths.cgm_worker.default_area_eic_map, "rb") as f:
            area_eic_map = json.loads(f.read())

    # Define metadata dictionary
    metadata = {
        "process.processType": process_type,
        "TimeSeries.businessType": "B63",
    }

    # Get HVDC schedules
    service = elastic.Elastic()
    schedules_df = service.query_schedules_from_elk(
        index=ELK_INDEX_PATTERN,
        utc_start=utc_start,
        utc_end=utc_end,
        metadata=metadata,
        period_overlap=True,
    )

    if schedules_df is None:
        return None

    # Map eic codes to area names
    schedules_df["in_domain"] = schedules_df["TimeSeries.in_Domain.mRID"].map(area_eic_map)
    schedules_df["out_domain"] = schedules_df["TimeSeries.out_Domain.mRID"].map(area_eic_map)

    # Filter to the latest revision number
    schedules_df = schedules_df[schedules_df.revisionNumber == schedules_df.revisionNumber.max()]

    # TODO filter out data by reason code that take only verified tada

    # Get relevant structure and convert to dictionary
    _cols = ["value", "in_domain", "out_domain", "TimeSeries.connectingLine_RegisteredResource.mRID",
             "TimeSeries.in_Domain.mRID",
             "TimeSeries.out_Domain.mRID"]
    schedules_df = schedules_df[_cols]
    schedules_df.rename(columns={"TimeSeries.connectingLine_RegisteredResource.mRID": "registered_resource"},
                        inplace=True)
    schedules_dict = schedules_df.to_dict('records')
    return schedules_dict


def query_acnp_schedules(process_type: str,
                         utc_start: str | datetime,
                         utc_end: str | datetime,
                         area_eic_map: Dict[str, str] | None = None
                         ) -> dict | None:
    """
    Method to get ACNP schedules (business type - B64)
    :param process_type: time horizon of schedules; A01 - Day-ahead, A18 - Intraday
    :param utc_start: start time in utc. Example: '2023-08-08T23:00:00Z'
    :param utc_end: end time in utc. Example: '2023-08-09T00:00:00Z'
    :param area_eic_map:
    :return:
    """
    # Define area name to eic mapping table
    if not area_eic_map:
        # Using default mapping table from config
        import json
        with open(config.paths.cgm_worker.default_area_eic_map, "rb") as f:
            area_eic_map = json.loads(f.read())

    metadata = {
        "process.processType": process_type,
        "TimeSeries.businessType": "B64",
    }

    # Get AC area schedules
    service = elastic.Elastic()
    schedules_df = service.query_schedules_from_elk(
        index=ELK_INDEX_PATTERN,
        utc_start=utc_start,
        utc_end=utc_end,
        metadata=metadata,
        period_overlap=True,
    )

    if schedules_df is None:
        return None

    # Map eic codes to area names
    schedules_df["in_domain"] = schedules_df["TimeSeries.in_Domain.mRID"].map(area_eic_map)
    schedules_df["out_domain"] = schedules_df["TimeSeries.out_Domain.mRID"].map(area_eic_map)

    # Filter to the latest revision number
    schedules_df = schedules_df[schedules_df.revisionNumber == schedules_df.revisionNumber.max()]

    # Get relevant structure and convert to dictionary
    _cols = ["value", "in_domain", "out_domain", "TimeSeries.out_Domain.mRID", "TimeSeries.in_Domain.mRID"]
    # _cols = ["value", "in_domain", "out_domain"]
    schedules_df = schedules_df[_cols]
    schedules_dict = schedules_df.to_dict('records')

    return schedules_dict


def get_areas_losses(network: pp.network.Network):
    # Calculate ACNP with losses (from cross-border lines)
    dangling_lines = get_network_elements(network, pp.network.ElementType.DANGLING_LINE, all_attributes=True)
    try:
        acnp_with_losses = dangling_lines[dangling_lines.isHvdc == ''].groupby('CGMES.regionName').p.sum()
    except AttributeError:
        acnp_with_losses = dangling_lines.groupby('CGMES.regionName').p.sum()

    # Calculate ACNP without losses (from generation and consumption)
    gens = get_network_elements(network, pp.network.ElementType.GENERATOR, all_attributes=True)
    loads = get_network_elements(network, pp.network.ElementType.LOAD, all_attributes=True)
    generation = gens.groupby('CGMES.regionName').p.sum() * -1
    consumption = loads.groupby('CGMES.regionName').p.sum()
    # Need to ensure that all series in substraction has same index values. For example when area does not have HVDC
    # connections
    # Otherwise we will get NaN values for areas without HVDC after regular substraction
    present_areas = generation.index.union(consumption.index)
    try:
        dcnp = (dangling_lines[dangling_lines.isHvdc == 'true'].groupby('CGMES.regionName')
                .p0.sum()
                .reindex(present_areas, fill_value=0))
        acnp_without_losses = generation - consumption - dcnp
    except AttributeError:
        acnp_without_losses = generation - consumption

    # Calculate losses by regions
    losses = acnp_without_losses - acnp_with_losses

    return losses


def get_scalable_hvdc(dangling_lines: pd.DataFrame,
                      target_hvdc_sp_df: pd.DataFrame,
                      regions: pd.DataFrame = None):
    names = None
    if regions is not None:
        in_regions = regions[['CGMES.regionId',
                              'IdentifiedObject.energyIdentCodeEic',
                              'IdentifiedObject.name']]
        names = target_hvdc_sp_df.merge(in_regions, left_on='TimeSeries.in_Domain.mRID',
                                        right_on='IdentifiedObject.energyIdentCodeEic', how='left',
                                        suffixes=('', '_in'))
        names = names.merge(in_regions, left_on='TimeSeries.out_Domain.mRID',
                            right_on='IdentifiedObject.energyIdentCodeEic', how='left', suffixes=('', '_out'))
    try:
        scalable_hvdc = dangling_lines[dangling_lines.isHvdc == 'true'][
            ['lineEnergyIdentificationCodeEIC', 'CGMES.regionName', 'CGMES.regionId', 'ucte-x-node-code']]
    except KeyError:
        scalable_hvdc = dangling_lines[dangling_lines.isHvdc == 'true'][
            ['lineEnergyIdentificationCodeEIC', 'CGMES.regionName', 'CGMES.regionId', 'ucte_xnode_code']]
    scalable_hvdc.reset_index(inplace=True)
    if names is not None:
        names = names.merge(scalable_hvdc[['CGMES.regionName', 'CGMES.regionId']].drop_duplicates(),
                            on='CGMES.regionId', how='left')
        names = names.merge(scalable_hvdc[['CGMES.regionName', 'CGMES.regionId']]
                            .rename(columns={'CGMES.regionId': 'CGMES.regionId_out'}).drop_duplicates(),
                            on='CGMES.regionId_out', how='left', suffixes=('', '_out'))
        names['in_domain'] = np.where(~names['CGMES.regionName'].isnull(), names['CGMES.regionName'],
                                      names['in_domain'])
        names['out_domain'] = np.where(~names['CGMES.regionName_out'].isnull(), names['CGMES.regionName_out'],
                                       names['out_domain'])
        target_hvdc_sp_df = names[['value', 'in_domain', 'out_domain', 'registered_resource',
                                   'TimeSeries.in_Domain.mRID', 'TimeSeries.out_Domain.mRID']]
    # Very special fix for terna
    if (len(scalable_hvdc[scalable_hvdc['CGMES.regionName'].str.contains('terna')].index) > 0 and
            len(scalable_hvdc[scalable_hvdc['CGMES.regionName'] == 'IT'].index) == 0):
        target_hvdc_sp_df.loc[target_hvdc_sp_df['in_domain'] == 'IT', 'in_domain'] = 'terna'
        target_hvdc_sp_df.loc[target_hvdc_sp_df['out_domain'] == 'IT', 'out_domain'] = 'terna'
    scalable_hvdc = scalable_hvdc.merge(target_hvdc_sp_df, left_on='lineEnergyIdentificationCodeEIC',
                                        right_on='registered_resource')
    mask = (scalable_hvdc['CGMES.regionName'] == scalable_hvdc['in_domain']) | (
                scalable_hvdc['CGMES.regionName'] == scalable_hvdc['out_domain'])
    scalable_hvdc = scalable_hvdc[mask]
    mask = (scalable_hvdc['CGMES.regionName'] == scalable_hvdc['in_domain']) & (scalable_hvdc['value'] > 0.0)
    scalable_hvdc['value'] = np.where(mask, scalable_hvdc['value'] * -1, scalable_hvdc['value'])
    # sorting values by abs() in descending order to be able to drop_duplicates() later
    scalable_hvdc = scalable_hvdc.loc[scalable_hvdc['value'].abs().sort_values(ascending=False).index]
    # drop duplicates by index and keep first rows (because df already sorted)
    # We should sum here TODO
    scalable_hvdc = scalable_hvdc.drop_duplicates(subset='id', keep='first')
    scalable_hvdc = scalable_hvdc.set_index('id')
    return scalable_hvdc


def get_target_ac_net_positions(ac_schedule_dict):
    # Target AC net positions mapping
    target_acnp_df = pd.DataFrame(ac_schedule_dict)
    target_acnp_df['registered_resource'] = target_acnp_df['in_domain'].where(target_acnp_df['in_domain'].notna(),
                                                                              target_acnp_df['out_domain'])
    target_acnp_df = target_acnp_df.dropna(subset='registered_resource')
    mask = (target_acnp_df['in_domain'].notna()) & (target_acnp_df['value'] > 0.0)  # value is not zero
    target_acnp_df['value'] = np.where(mask, target_acnp_df['value'] * -1, target_acnp_df['value'])
    target_acnp_df = target_acnp_df.groupby('registered_resource').agg({'value': 'sum',
                                                                        'in_domain': 'last',
                                                                        'out_domain': 'last'}).reset_index()
    return target_acnp_df


def print_loadflow_results(pf_results, island_bus_count, message: str):
    messages = []
    for result in [x for x in pf_results if x.connected_component_num in island_bus_count.keys()]:
        result_dict = attr_to_dict(result)
        result_message = (f"Network {result_dict.get('connected_component_num')}: {result_dict.get('status').name} "
                          f"({result_dict.get('distributed_active_power')})")
        messages.append(result_message)
        # logger.info(f"{message} Loadflow status: {result_dict.get('status').name}")
        # logger.debug(f"{message} Loadflow results: {result_dict}")
    logger.info(f"{message}: {(','.join(messages))}")
    if pf_results[0].status.value:
        logger.error(f"Terminating network scaling due to divergence in main island")
        return False
    return True


def update_scaling_df(network_instance, scaling_results, iteration):
    loads = get_network_elements(network_instance, pp.network.ElementType.LOAD, all_attributes=True)
    gens = get_network_elements(network_instance, pp.network.ElementType.GENERATOR, all_attributes=True)
    pre_scale_generation = gens.groupby('CGMES.regionName').p.sum() * -1
    pre_scale_consumption = loads.groupby('CGMES.regionName').p.sum()
    scaling_results.append(
        pd.concat([pre_scale_generation, pd.Series({'KEY': 'generation', 'ITER': iteration})]).to_dict())
    scaling_results.append(
        pd.concat([pre_scale_consumption, pd.Series({'KEY': 'consumption', 'ITER': iteration})]).to_dict())
    return scaling_results


def get_regions_from_models(models):
    if not isinstance(models, pd.DataFrame):
        entsoe_boundaries = load_opdm_data(models, 'EQBD')
        eq_boundaries = load_opdm_data(models, 'EQ')
        boundaries = pd.concat([entsoe_boundaries, eq_boundaries])
    else:
        boundaries = models
    regions = boundaries.type_tableview('GeographicalRegion').rename_axis('CGMES.regionId').reset_index()
    return regions


def fix_ac_net_position_names(target_acnp_df, area_eic_map, dangling_lines, regions):
    """
    Maps together schedules and regions from dangling lines using the geographical regions from the original models
    """
    if regions is None or area_eic_map is None:
        return target_acnp_df
    dangling_line_regions = dangling_lines[['CGMES.regionId', 'CGMES.regionName']].drop_duplicates()
    target_acnp_df = target_acnp_df.merge((area_eic_map[['area.eic', 'area.code']])
                                          .rename(columns={'area.code': 'registered_resource'})
                                          .drop_duplicates(),
                                          on='registered_resource', how='left')
    target_acnp_df = target_acnp_df.merge(regions[['CGMES.regionId', 'IdentifiedObject.energyIdentCodeEic']]
                                          .rename(columns={'IdentifiedObject.energyIdentCodeEic': 'area.eic'}),
                                          on='area.eic', how='left')
    # Get pre-scale HVDC setpoints
    target_acnp_df = target_acnp_df.merge(dangling_line_regions, on='CGMES.regionId', how='left')
    target_acnp_df['CGMES.regionName'] = (target_acnp_df['CGMES.regionName']
                                          .fillna(target_acnp_df['registered_resource']))
    target_acnp_df['registered_resource'] = target_acnp_df['CGMES.regionName']
    # Very special fix for terna
    if (len(dangling_line_regions[dangling_line_regions['CGMES.regionName'].str.contains('terna')].index) > 0 and
            len(dangling_line_regions[dangling_line_regions['CGMES.regionName'] == 'IT'].index) == 0):
        target_acnp_df['registered_resource'] = (target_acnp_df['registered_resource'].map({'IT': 'terna'}).fillna(
            target_acnp_df['registered_resource']))
    return target_acnp_df


@performance_counter(units='seconds')
def scale_balance(network: pp.network.Network,
                  ac_schedules: List[Dict[str, Union[str, float, None]]],
                  dc_schedules: List[Dict[str, Union[str, float, None]]],
                  lf_settings: pp.loadflow.Parameters = CGM_RELAXED_1,
                  area_eic_map: pd.DataFrame = None,
                  models: list | Dict | pd.DataFrame = None,
                  debug=bool(DEBUG),
                  abort_when_diverged: bool = True,
                  scale_by_regions: bool = True
                  ):
    """
    Main method to scale each CGM area to target balance
    :param network: pypowsybl network object
    :param ac_schedules: target AC net positions in list of dict format
    :param dc_schedules: target DC net positions in list of dict format
    :param lf_settings: loadflow settings
    :param debug: debug flag
    :param area_eic_map:
    :param models:
    :param abort_when_diverged:
    :param scale_by_regions:
    :return: scaled pypowsybl network object
    """
    island_bus_count = get_connected_component_counts(network=network, bus_count_threshold=5)
    _scaling_results = []
    _iteration = 0
    previous_network = copy.deepcopy(network)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Target HVDC setpoints
    target_hvdc_sp_df = pd.DataFrame(dc_schedules)
    regions = None
    if models is not None:
        regions = get_regions_from_models(models)

    # Target AC net positions mapping
    target_acnp_df = get_target_ac_net_positions(ac_schedule_dict=ac_schedules)
    dangling_lines = get_network_elements(network, pp.network.ElementType.DANGLING_LINE, all_attributes=True)
    target_acnp_df = fix_ac_net_position_names(target_acnp_df=target_acnp_df,
                                               area_eic_map=area_eic_map,
                                               dangling_lines=dangling_lines,
                                               regions=regions)
    target_acnp_df_reduced = target_acnp_df[['registered_resource', 'value']].drop_duplicates()
    target_acnp = target_acnp_df_reduced.set_index('registered_resource')['value']

    # Mapping HVDC schedules to network
    try:
        scalable_hvdc = get_scalable_hvdc(dangling_lines=dangling_lines,
                                          target_hvdc_sp_df=target_hvdc_sp_df,
                                          regions=regions,
                                          area_eic_map=area_eic_map
                                          )
        # Updating HVDC network elements to scheduled values
        try:
            scalable_hvdc_target = scalable_hvdc[['value', 'ucte-x-node-code']]
        except KeyError:
            scalable_hvdc_target = scalable_hvdc[['value', 'ucte_xnode_code']]
        network.update_dangling_lines(id=scalable_hvdc_target.index, p0=scalable_hvdc_target.value)
    except AttributeError as attr_error:
        logger.error(f"Cannot set HVDC lines: {attr_error}")

    # Get AC net positions scaling perimeter -> non-negative ConformLoads
    loads = get_network_elements(network, pp.network.ElementType.LOAD, all_attributes=True)
    loads = loads.merge(network.get_extensions('detail'), right_index=True, left_index=True)
    loads['power_factor'] = loads.q0 / loads.p0  # estimate the power factor of loads
    conform_loads = loads[loads['variable_p0'] > 0]

    # Solving initial loadflow
    pf_results = pp.loadflow.run_ac(network=network, parameters=lf_settings)
    pf_ok = print_loadflow_results(pf_results=pf_results, island_bus_count=island_bus_count, message=f"[INITIAL]")
    if not pf_ok and abort_when_diverged:
        previous_network.ac_scaling_results_df = pd.DataFrame(_scaling_results)
        return previous_network
    previous_network = copy.deepcopy(network)

    # Get pre-scale AC net positions for each control area
    dangling_lines = get_network_elements(network, pp.network.ElementType.DANGLING_LINE, all_attributes=True)
    try:
        prescale_acnp = dangling_lines[dangling_lines.isHvdc == ''].groupby('CGMES.regionName').p.sum()
    except AttributeError:
        prescale_acnp = dangling_lines.groupby('CGMES.regionName').p.sum()

    _scaling_results.append(pd.concat([prescale_acnp, pd.Series({'KEY': 'prescale-acnp',
                                                                 'ITER': _iteration})]).to_dict())
    logger.info(f"[ITER {_iteration}] PRE-SCALE ACNP: {prescale_acnp.round().to_dict()}")

    # Get pre-scale total network balance -> AC+DC net position
    prescale_network_np = dangling_lines.p.sum()

    _scaling_results.append({'KEY': 'prescale-network-np', 'GLOBAL': prescale_network_np, 'ITER': _iteration})
    logger.info(f"[ITER {_iteration}] PRE-SCALE NETWORK NP: {round(prescale_network_np, 2)}")

    # Get pre-scale total network balance -> AC net position
    try:
        unpaired_dangling_lines = (dangling_lines.isHvdc == '') & (dangling_lines.tie_line_id == '')
    except AttributeError:
        unpaired_dangling_lines = (dangling_lines.tie_line_id == '')
    scheduled_values = ((pd.DataFrame(target_acnp).rename_axis('CGMES.regionName').reset_index())
                        .merge(dangling_lines[['CGMES.regionName']].drop_duplicates(keep='last'),
                               on='CGMES.regionName'))

    # Validate total network AC net position from schedules to network model and scale to meet scheduled
    if scale_by_regions:
        prescale_network_acnp = dangling_lines[unpaired_dangling_lines].groupby(dangling_lines['CGMES.regionName']).agg(
            {'p': 'sum'}).reset_index()
        existing_acnp = ((dangling_lines[~unpaired_dangling_lines]).groupby(dangling_lines['CGMES.regionName']).agg(
            {'p': 'sum'})).reset_index()
        scheduled_network_acnp = scheduled_values
        _scaling_results.append({'KEY': 'prescale-network-acnp', 'GLOBAL': prescale_network_acnp.to_dict(),
                                 'ITER': _iteration})
        logger.info(f"[ITER {_iteration}] PRE-SCALE NETWORK ACNP: {round(prescale_network_acnp.p.sum(), 2)}")
        logger.info(f"[ITER {_iteration}] Scaling total network ACNP to scheduled: {round(scheduled_network_acnp, 2)}")
        dangling_lines.loc[unpaired_dangling_lines, 'participation'] = (
                    dangling_lines[unpaired_dangling_lines].p.abs() /
                    dangling_lines[unpaired_dangling_lines].p.abs().
                    groupby(dangling_lines['CGMES.regionName']).
                    transform('sum'))
        offset_network_acnp = ((prescale_network_acnp.merge(scheduled_network_acnp, on='CGMES.regionName',
                                                            how='outer', suffixes=('', '_schedule'))
                                .merge(existing_acnp, on='CGMES.regionName', how='outer', suffixes=('', '_existing')))
                               .fillna(0))
        offset_network_acnp['offset'] = offset_network_acnp['p'] - (offset_network_acnp['value'] -
                                                                    offset_network_acnp['p_existing'])
        prescale_network_acnp_diff = (dangling_lines[unpaired_dangling_lines]['CGMES.regionName']
                                      .map(dict(offset_network_acnp[['CGMES.regionName', 'offset']].values)) *
                                      dangling_lines[unpaired_dangling_lines].participation)
    else:
        prescale_network_acnp = dangling_lines[unpaired_dangling_lines].p.sum()
        existing_acnp = dangling_lines[~unpaired_dangling_lines].p.sum()
        scheduled_network_acnp = scheduled_values['value'].sum()
        _scaling_results.append({'KEY': 'prescale-network-acnp', 'GLOBAL': prescale_network_acnp, 'ITER': _iteration})
        logger.info(f"[ITER {_iteration}] PRE-SCALE NETWORK ACNP: {round(prescale_network_acnp, 2)}")
        logger.info(f"[ITER {_iteration}] Scaling total network ACNP to scheduled: {round(scheduled_network_acnp, 2)}")
        dangling_lines.loc[unpaired_dangling_lines, 'participation'] = (
                    dangling_lines[unpaired_dangling_lines].p.abs() /
                    dangling_lines[unpaired_dangling_lines].p.abs()
                    .sum())
        offset_network_acnp = prescale_network_acnp - (scheduled_network_acnp - existing_acnp)
        prescale_network_acnp_diff = offset_network_acnp * dangling_lines[unpaired_dangling_lines].participation

    # TODO discuss which one to use p or p0

    prescale_network_acnp_target = dangling_lines[unpaired_dangling_lines].p - prescale_network_acnp_diff
    prescale_network_acnp_target.dropna(inplace=True)

    network.update_dangling_lines(id=prescale_network_acnp_target.index,
                                  p0=prescale_network_acnp_target.to_list())  # TODO maintain power factor

    # Solving loadflow after aligning total network AC net position to scheduled
    pf_results = pp.loadflow.run_ac(network=network, parameters=lf_settings)
    pf_ok = print_loadflow_results(pf_results=pf_results, island_bus_count=island_bus_count,
                                   message=f"[ITER {_iteration}]")
    if not pf_ok and abort_when_diverged:
        previous_network.ac_scaling_results_df = pd.DataFrame(_scaling_results).set_index('ITER').sort_index().round(2)
        return previous_network
    previous_network = copy.deepcopy(network)

    # Validate total network AC net position alignment
    dangling_lines = get_network_elements(network, pp.network.ElementType.DANGLING_LINE, all_attributes=True)
    postscale_network_acnp = dangling_lines[unpaired_dangling_lines].p.sum()
    _scaling_results.append({'KEY': 'postscale-network-acnp', 'GLOBAL': postscale_network_acnp, 'ITER': _iteration})
    logger.info(f"[ITER {_iteration}] POST-SCALE NETWORK ACNP: {round(postscale_network_acnp, 2)}")

    # Get pre-scale generation and consumption
    if debug:
        _scaling_results = update_scaling_df(network_instance=network,
                                             scaling_results=_scaling_results,
                                             iteration=_iteration)

    # Filtering target AC net positions series by present regions in network
    target_acnp = target_acnp[target_acnp.index.isin(prescale_acnp.index)]
    _scaling_results.append(pd.concat([target_acnp, pd.Series({'KEY': 'target-acnp', 'ITER': _iteration})]).to_dict())
    logger.info(f"[ITER {_iteration}] TARGET ACNP: {target_acnp.to_dict()}")

    # Get offsets between target and pre-scale AC net positions for each control area
    offset_acnp = prescale_acnp - target_acnp[target_acnp.index.isin(prescale_acnp.index)]
    offset_acnp.dropna(inplace=True)

    _scaling_results.append(pd.concat([offset_acnp, pd.Series({'KEY': 'offset-acnp', 'ITER': _iteration})]).to_dict())
    logger.info(f"[ITER {_iteration}] PRE-SCALE ACNP offset: {offset_acnp.round().to_dict()}")

    # Perform scaling of AC part of the network with loop
    while _iteration < int(MAX_ITERATION):
        _iteration += 1

        # Get scaling area loads participation factors
        scalable_loads = get_network_elements(network, pp.network.ElementType.LOAD, all_attributes=True,
                                              id=conform_loads.index)
        scalable_loads['p_participation'] = (scalable_loads.p0 / scalable_loads.groupby('CGMES.regionName')
                                             .p0.transform('sum'))

        # Scale loads by participation factor
        scalable_loads_diff = (scalable_loads['CGMES.regionName'].map(offset_acnp) * scalable_loads.p_participation)
        scalable_loads_target = scalable_loads.p0 + scalable_loads_diff
        scalable_loads_target.dropna(inplace=True)
        scalable_loads_q = scalable_loads_target * conform_loads.merge(scalable_loads_target.rename('target'),
                                                                       left_index=True, right_index=True).power_factor
        # removing loads which target value is NaN. It can be because missing target ACNP for this area
        network.update_loads(id=scalable_loads_target.index,
                             p0=scalable_loads_target.to_list(),
                             q0=scalable_loads_q.to_list())  # maintain power factor

        # Solving post-scale loadflow
        pf_results = pp.loadflow.run_ac(network=network, parameters=lf_settings)
        pf_ok = print_loadflow_results(pf_results=pf_results, island_bus_count=island_bus_count,
                                       message=f"[ITER {_iteration}]")
        if not pf_ok and abort_when_diverged:
            previous_network.ac_scaling_results_df = pd.DataFrame(_scaling_results).set_index(
                'ITER').sort_index().round(2)
            return previous_network
        previous_network = copy.deepcopy(network)

        # Store distributed active power after AC part scaling
        distributed_power = round(pf_results[0].distributed_active_power, 2)
        _scaling_results.append({'KEY': 'distributed-power', 'GLOBAL': distributed_power, 'ITER': _iteration})

        # Get post-scale generation and consumption
        if debug:
            _scaling_results = update_scaling_df(network_instance=network,
                                                 scaling_results=_scaling_results,
                                                 iteration=_iteration)

        # Get post-scale network losses by regions
        # It is needed to estimate when loadflow engine balances entire network schedule with distributed slack enabled
        postscale_losses = get_areas_losses(network=network)
        total_network_losses = postscale_losses.sum()
        _scaling_results.append(pd.concat([postscale_losses, pd.Series({'GLOBAL': total_network_losses,
                                                                        'KEY': 'losses',
                                                                        'ITER': _iteration})]).to_dict())
        logger.debug(f"[ITER {_iteration}] POST-SCALE LOSSES: {postscale_losses.round().to_dict()}")

        # Get post-scale AC net position
        dangling_lines = get_network_elements(network, pp.network.ElementType.DANGLING_LINE, all_attributes=True)
        try:
            postscale_acnp = dangling_lines[dangling_lines.isHvdc == ''].groupby('CGMES.regionName').p.sum()
        except AttributeError:
            postscale_acnp = dangling_lines.groupby('CGMES.regionName').p.sum()

        _scaling_results.append(pd.concat([postscale_acnp, pd.Series({'KEY': 'postscale-acnp',
                                                                      'ITER': _iteration})]).to_dict())
        logger.info(f"[ITER {_iteration}] POST-SCALE ACNP: {postscale_acnp.round().to_dict()}")

        # Get post-scale total network balance
        prescale_total_np = dangling_lines.p.sum()
        logger.info(f"[ITER {_iteration}] POST-SCALE TOTAL NP: {round(prescale_total_np, 2)}")

        # Get offset between target and post-scale AC net position
        offset_acnp = postscale_acnp - target_acnp[target_acnp.index.isin(postscale_acnp.index)]
        offset_acnp.dropna(inplace=True)
        _scaling_results.append(pd.concat([offset_acnp, pd.Series({'KEY': 'offset-acnp', 'ITER': _iteration})])
                                .to_dict())
        logger.info(f"[ITER {_iteration}] POST-SCALE ACNP offsets: {offset_acnp.round().to_dict()}")

        # Breaking scaling loop if target ac net position for all areas is reached
        if all(abs(offset_acnp.values) <= int(BALANCE_THRESHOLD)):
            logger.info(f"[ITER {_iteration}] Scaling successful as ACNP offsets less than threshold: "
                        f"{int(BALANCE_THRESHOLD)} MW")
            break
    else:
        logger.warning(f"Max iteration limit reached")
        # TODO actions after scale break

    # Post-processing scaling results dataframe
    ac_scaling_results_df = pd.DataFrame(_scaling_results).set_index('ITER').sort_index().round(2)
    network.ac_scaling_results_df = ac_scaling_results_df

    return network


def hvdc_schedule_mapper(row, target_dcnp):
    """BACKLOG FUNCTION. CURRENTLY NOT USED"""
    schedules = pd.DataFrame(target_dcnp)
    eic_mask = schedules['TimeSeries.connectingLine_RegisteredResource.mRID'] == row['lineEnergyIdentificationCodeEIC']
    in_domain_mask = schedules["TimeSeries.in_Domain.regionName"] == row['CGMES.regionName']
    out_domain_mask = schedules["TimeSeries.out_Domain.regionName"] == row['CGMES.regionName']
    relevant_schedule = schedules[(eic_mask) & ((in_domain_mask) | (out_domain_mask))]

    if relevant_schedule.empty:
        logger.warning(f"No schedule available for resource: {row['lineEnergyIdentificationCodeEIC']}")
        return None

    if relevant_schedule["TimeSeries.in_Domain.regionName"].notnull().squeeze():
        return relevant_schedule["value"].squeeze() * -1
    elif relevant_schedule["TimeSeries.out_Domain.regionName"].notnull().squeeze():
        return relevant_schedule["value"].squeeze()
    else:
        logger.warning(f"Not able to define schedule direction for resource: {row['lineEnergyIdentificationCodeEIC']}")
        return None


if __name__ == "__main__":
    # Testing
    import sys

    logging.basicConfig(
        format='%(levelname) -10s %(asctime) -20s %(name) -35s %(funcName) -35s %(lineno) -5d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    model_path = r"C:\Users\martynas.karobcikas\Documents\models\rmm\test_model_ast_litgrid.zip"
    network = pp.network.load(model_path, parameters={"iidm.import.cgmes.source-for-iidm-id": "rdfID"})

    # Query target schedules
    # ac_schedules = query_acnp_schedules(process_type="A01", utc_start="2024-04-29T12:00:00Z", utc_end="2024-04-29T13:00:00Z")
    # dc_schedules = query_hvdc_schedules(process_type="A01", utc_start="2024-04-29T12:00:00Z", utc_end="2024-04-29T13:00:00Z")

    test_dc_schedules = [{'value': 350,
                          'in_domain': None,
                          'out_domain': 'LT',
                          'registered_resource': '10T-LT-SE-000013'},
                         {'value': 320,
                          'in_domain': 'LT',
                          'out_domain': None,
                          'registered_resource': '10T-LT-PL-000037'}
                         ]

    # ac_schedules.append({"value": 400, "in_domain": "LT", "out_domain": None})
    test_ac_schedules = [
        {"value": 200, "in_domain": "LT", "out_domain": None},
        {"value": 100, "in_domain": None, "out_domain": "LV"},
    ]

    network_instance = scale_balance(network=network,
                                     ac_schedules=test_ac_schedules,
                                     dc_schedules=test_dc_schedules, debug=True)
    print("Done")
    # print(network_instance.ac_scaling_results_df)

    # Set-up
    process_type = 'A01'
    elastic_client = Elastic()
    scenario_datetime = "20241212T2230Z"
    solved_model = {}
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
    solved_model["network"] = scale_balance(network=solved_model.get("network"),
                                            ac_schedules=ac_schedules,
                                            dc_schedules=dc_schedules,
                                            lf_settings=CGM_RELAXED_1,
                                            area_eic_map=area_eic_codes,
                                            models=None,
                                            abort_when_diverged=True,
                                            scale_by_regions=True,
                                            debug=True)

    # Results analysis
    # print(network.ac_scaling_results_df.query("KEY == 'generation'"))
    # print(network.ac_scaling_results_df.query("KEY == 'consumption'"))
    # print(network.ac_scaling_results_df.query("KEY == 'offset-acnp'"))

    # Other examples
    # loads = network.get_loads(id=network.get_elements_ids(element_type=pp.network.ElementType.LOAD, countries=['LT'])) 