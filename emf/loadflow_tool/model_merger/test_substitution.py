import json
import logging
import os
import re
from collections.abc import Iterable
from datetime import datetime
from io import BytesIO
from uuid import uuid4
from zipfile import ZipFile

import pandas
import requests

from emf.common.integrations.elastic import Elastic
from emf.common.integrations.object_storage.file_system import group_files_by_origin, \
    get_one_set_of_igms_from_local_storage, get_one_set_of_boundaries_from_local_storage, \
    save_merged_model_to_local_storage
from emf.common.integrations.object_storage.file_system_general import check_the_folder_path
from emf.loadflow_tool.model_merger import merge_functions
from emf.loadflow_tool.model_merger.merge_from_local import merge_models, SMALL_ISLAND_SIZE, MERGE_LOAD_FLOW_SETTINGS, \
    PY_REPORT_ELK_INDEX
from emf.task_generator.time_helper import parse_duration, convert_to_utc
import pytz
import triplets
import config
import datetime
from emf.loadflow_tool.helper import metadata_from_filename
from emf.task_generator.time_helper import parse_datetime
from emf.common.config_parser import parse_app_properties

try:
    from emf.common.integrations import minio, elastic
except ImportError:
    from emf.common.integrations import elastic, minio_api as minio
from emf.common.integrations.object_storage.models import get_latest_models_and_download, get_latest_boundary
from emf.loadflow_tool.model_merger.merge_functions import filter_models, export_to_cgmes_zip, get_opdm_data_from_models

# from rcc_common_tools import minio_api

logger = logging.getLogger(__name__)
parse_app_properties(caller_globals=globals(), path=config.paths.cgm_worker.merger)

SEPARATOR_SYMBOL = '/'
WINDOWS_SEPARATOR = '\\'

# Shift mapping logic for future scenario dates
shift_mapping = {
    # Change these pack
    # '1D': 'P0D',  # Shift for '1D' is +2 days
    # '2D': 'P1D',  # Shift for '2D' is +3 days
    '1D': 'P1D',  # Shift for '1D' is +2 days
    '2D': 'P2D',  # Shift for '2D' is +3 days
    # '1D': 'P2D',  # Shift for '1D' is +2 days
    # '2D': 'P3D',  # Shift for '2D' is +3 days
    '3D': 'P0D',  # Shift for '3D' is 0 days (today)
    '4D': 'P0D',  # Same for '4D'
    '5D': 'P0D',  # Same for '5D'
    '6D': 'P0D',  # Same for '6D'
    '7D': 'P0D',  # Same for '7D'
    '8D': 'P0D',  # Same for '8D'
    '9D': 'P0D'  # Same for '9D'
}


class NoContentFromElasticException(Exception):
    pass


def get_date_from_time_horizon(time_horizon: str):
    """
    Parses number of dates from time horizon
    :param time_horizon: input as string
    """
    if time_horizon in ['1D', '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D']:
        return time_horizon  # return the original time horizon for use in parse_duration
    elif time_horizon == 'WK':  # handle 'WK' case for a week
        return '9D'  # 7 days for the week
    raise ValueError(f"Cannot parse day from {time_horizon}")


def calculate_scenario_datetime_and_time_horizon(given_datetime: str | datetime.datetime = None,
                                                 default_time_horizon: str = '1D',
                                                 actual_offset: str = '-P1D',
                                                 imaginary_offset: str = '-P7D',
                                                 actual_time_horizons: list = None,
                                                 imaginary_time_horizons: list = None):
    """
    Gets dictionary of replacements
    :param given_datetime: starting point for generating replacements
    :param default_time_horizon: time horizon used for past scenario dates
    :param actual_offset: set this from given datetime: to get D1 and today use -P1D
    :param imaginary_offset: use this for calculating the offsets for past scenario dates (-week: -P6D)
    :param actual_time_horizons: time horizons to keep intact
    :param imaginary_time_horizons: time horizons to be substituted
    """
    timestamps = {}
    existing_timestamps = {}
    if not given_datetime:
        starting_timestamp = datetime.datetime.utcnow()
    elif isinstance(given_datetime, str):
        starting_timestamp = parse_datetime(given_datetime)
    else:
        raise ValueError(f"Undefined input: {given_datetime}")
    if starting_timestamp.tzinfo:
        starting_timestamp = starting_timestamp.astimezone(pytz.utc)
    else:
        logger.warning(f"Time zone is not defined for scenario_time variable, localizing as UTC time zone")
        starting_timestamp = starting_timestamp.tz_localize(pytz.utc)
    actual_offset_duration = parse_duration(actual_offset)
    try:
        imaginary_offset_duration = parse_duration(imaginary_offset)
    except ValueError:
        imaginary_offset_duration = None

    # Map 'WK' to 7 days for the default time horizon
    if default_time_horizon == 'WK':
        default_time_horizon = '7D'

    # Apply the new shift logic for actual time horizons
    for time_horizon in actual_time_horizons:
        time_horizon_str = get_date_from_time_horizon(time_horizon)
        shift_duration = shift_mapping.get(time_horizon_str, actual_offset_duration)

        time_delta = parse_duration(shift_duration)
        # future_scenario_date = starting_timestamp + time_delta
        future_scenario_date = starting_timestamp + time_delta + actual_offset_duration
        existing_scenario_date = future_scenario_date
        time_horizon_value = {'future_time_scenario': future_scenario_date.isoformat(),
                              'future_time_horizon': time_horizon,
                              'existing_time_scenario': existing_scenario_date.isoformat(),
                              'existing_time_horizon': time_horizon}
        existing_timestamps[existing_scenario_date.isoformat()] = time_horizon
        timestamps[time_horizon] = time_horizon_value

    # Handle imaginary horizons (same as before)
    for time_horizon in imaginary_time_horizons:
        time_horizon_str = get_date_from_time_horizon(time_horizon)
        time_delta = parse_duration(f"P{time_horizon_str}")
        # future_scenario_date = starting_timestamp + time_delta + actual_offset_duration
        future_scenario_date = starting_timestamp + time_delta
        if imaginary_offset_duration is None:
            imaginary_offset_duration = shift_mapping.get(time_horizon_str, actual_offset_duration)
            imaginary_offset_duration = parse_duration(imaginary_offset_duration)
        existing_scenario_date = future_scenario_date + actual_offset_duration + imaginary_offset_duration
        set_time_horizon = existing_timestamps.get(existing_scenario_date.isoformat(), default_time_horizon)
        time_horizon_value = {'future_time_scenario': future_scenario_date.isoformat(),
                              'future_time_horizon': time_horizon,
                              'existing_time_scenario': existing_scenario_date.isoformat(),
                              'existing_time_horizon': set_time_horizon}
        timestamps[time_horizon] = time_horizon_value
    return timestamps


def calculate_scenario_datetime_and_time_horizon_by_shifts(time_horizons: list,
                                                           shifts: dict,
                                                           given_datetime: str | datetime.datetime = None,
                                                           default_time_horizon: str = '1D',
                                                           given_time_horizons: str | list = None,
                                                           offset: str = '-P1D'):
    """
    Gets dictionary of replacements
    :param given_datetime: starting point for generating replacements
    :param default_time_horizon: time horizon used for past scenario dates
    :param given_time_horizons: time horizons that exists
    :param offset: set this from given datetime: to get D1 and today use -P1D
    :param time_horizons: time horizons to keep intact
    :param shifts: time horizons to be substituted
    """
    if given_time_horizons is None:
        given_time_horizons = ['1D', '2D']
    if isinstance(given_time_horizons, str):
        given_time_horizons = [given_time_horizons]
    timestamps = {}
    existing_timestamps = {}
    if not given_datetime:
        starting_timestamp = datetime.datetime.utcnow()
    elif isinstance(given_datetime, str):
        starting_timestamp = parse_datetime(given_datetime)
    else:
        raise ValueError(f"Undefined input: {given_datetime}")
    if starting_timestamp.tzinfo:
        starting_timestamp = starting_timestamp.astimezone(pytz.utc)
    else:
        logger.warning(f"Time zone is not defined for scenario_time variable, localizing as UTC time zone")
        starting_timestamp = starting_timestamp.tz_localize(pytz.utc)
    actual_offset_duration = parse_duration(offset)

    # Map 'WK' to 7 days for the default time horizon
    if default_time_horizon == 'WK':
        default_time_horizon = '7D'

    # Apply the new shift logic for actual time horizons
    for time_horizon in time_horizons:
        time_horizon_str = get_date_from_time_horizon(time_horizon)

        shift_duration = shifts.get(time_horizon_str, actual_offset_duration)

        time_delta_actual = parse_duration(shift_duration)
        time_delta_imaginary = parse_duration(f"P{time_horizon_str}")
        # future_scenario_date = starting_timestamp + time_delta_imaginary + actual_offset_duration
        future_scenario_date = starting_timestamp + time_delta_imaginary
        existing_scenario_date = starting_timestamp + time_delta_actual + actual_offset_duration
        existing_time_horizon = time_horizon if time_horizon in given_time_horizons else default_time_horizon
        time_horizon_value = {'future_time_scenario': future_scenario_date.isoformat(),
                              'future_time_horizon': time_horizon,
                              'existing_time_scenario': existing_scenario_date.isoformat(),
                              'existing_time_horizon': existing_time_horizon}
        existing_timestamps[existing_scenario_date.isoformat()] = time_horizon
        timestamps[time_horizon] = time_horizon_value
    return timestamps


def reduced_filename_from_metadata(metadata):
    """
    Workaround from filename_from_metadata. Escapes mergingEntity and mergingArea if not given
    :param metadata: dictionary with necessary metadata
    :return updated filename
    """
    model_part = metadata.get('pmd:modelPartReference', None)

    if model_part:
        merging_entity = metadata.get('pmd:mergingEntity')
        merging_area = metadata.get('pmd:mergingArea')
        if merging_entity and merging_area:
            model_authority = f"{metadata['pmd:mergingEntity']}-{metadata['pmd:mergingArea']}-{model_part}"
        else:
            model_authority = model_part

    else:
        model_authority = f"{metadata['pmd:mergingEntity']}-{metadata['pmd:mergingArea']}"

    # Change the time horizon in the metadata to 'WK' if it's between 1D and 9D
    if metadata['pmd:timeHorizon'] in ['1D', '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D']:
        metadata['pmd:timeHorizon'] = 'WK'

    file_name = (f"{metadata['pmd:validFrom']}_{metadata['pmd:timeHorizon']}_{model_authority}_"
                 f"{metadata['pmd:cgmesProfile']}_{metadata['pmd:versionNumber']}")
    file_name = ".".join([file_name, metadata["file_type"]])

    return file_name


def get_outage_matches(model_data: pandas.DataFrame, outages: pandas.DataFrame, outage_name_field: str = 'name'):
    """
    Some preliminary, bad and funky way to match the outage data to the mRIDs. If considering the filtered
    opc data most of the elements contained some number and letter combination (LN 321 for example) so get them
    and search equivalents from all the names in the model.
    Probably the best solution would be to use some kind of external mapping, but it is currently missing
    :param model_data: necessary IGMs (in triplets)
    :param outages: dataframe of outage data
    :param outage_name_field: name of the column in outage data to be used for mapping
    :return slice of triplets containing the IDs of the elements that are in outage data
    """
    model_data = get_opdm_data_from_models(model_data)
    object_names = model_data[model_data['KEY'] == 'IdentifiedObject.name']
    # Get word before numeric value
    word_pattern = r'\b(\w+)\b[\s\-\_]+(?=\d+)'
    # Get numeric value
    number_pattern = r'\d+'
    name_matches = []
    for index, row in outages.iterrows():
        outage_name = row[outage_name_field]
        word_matches = re.findall(word_pattern, outage_name)
        numeric_matches = re.findall(number_pattern, outage_name)
        if len(word_matches) != len(numeric_matches):
            logger.error(f"Multiple different values in opc name: {outage_name}, skipping...")
            continue
        matches = []
        for word, number in zip(word_matches, numeric_matches):
            matches.extend([word + ' ' + number, word + '-' + number, word + number])
        name_matches.append(object_names[object_names['VALUE'].str.upper().str.contains('|'.join(matches))])
    matched_lines = pandas.concat(name_matches)
    matched_lines = matched_lines.drop_duplicates(subset=['ID', 'VALUE'], keep="first")
    return matched_lines


def update_outages(model_data: pandas.DataFrame,
                   existing_outages: pandas.DataFrame,
                   outages_to_set: pandas.DataFrame,
                   eic_to_mrid_map: pandas.DataFrame,
                   outage_name_field: str = 'name',
                   outage_eic_codes_column: str = 'EIC',
                   mrid_eic_codes_column: str = 'eic',
                   mrid_mrid_codes_column: str = 'mrid'):
    """
    Links outages to devices in igm, gets difference of the outages from the old scenario date and new scenario date
    Switches on devices which outages will be expired in future and switches of devices that will go to outages
    :param model_data: necessary IGMs (in triplets)
    :param existing_outages: dataframe of outage data for the timestamp from which the model was taken
    :param outages_to_set: dataframe of outage data for a future timestamp that the model will represent
    :param outage_name_field:
    :param eic_to_mrid_map: map
    :param outage_eic_codes_column:
    :param mrid_eic_codes_column:
    :param mrid_mrid_codes_column:
    return updated model_data
    """
    model_data = get_opdm_data_from_models(model_data)
    if eic_to_mrid_map.empty:
        existing_lines = get_outage_matches(model_data=model_data,
                                            outages=existing_outages,
                                            outage_name_field=outage_name_field)
        updated_lines = get_outage_matches(model_data=model_data,
                                           outages=outages_to_set,
                                           outage_name_field=outage_name_field)
    else:
        column_headings = eic_to_mrid_map.columns.values.tolist()
        if "ID" not in column_headings and mrid_mrid_codes_column != "ID":
            eic_to_mrid_map = eic_to_mrid_map.rename(columns={mrid_mrid_codes_column: 'ID'})
        existing_lines = existing_outages[[outage_eic_codes_column]].merge(eic_to_mrid_map,
                                                                           left_on=outage_eic_codes_column,
                                                                           right_on=mrid_eic_codes_column)
        updated_lines = outages_to_set[[outage_eic_codes_column]].merge(eic_to_mrid_map,
                                                                        left_on=outage_eic_codes_column,
                                                                        right_on=mrid_eic_codes_column)
    # Tricky part: get changes
    changes = existing_lines.merge(updated_lines, on='ID', how='outer', suffixes=('_PRE', '_POST'), indicator=True)
    # Get those devices that will come to service
    set_to_service = changes[changes['_merge'] == 'left_only'][['ID']]
    # Get those devices that will go out of service
    set_out_of_service = changes[changes['_merge'] == 'right_only'][['ID']]
    statuses = model_data.type_tableview('SvStatus').reset_index()
    # Rectify "past" outages
    if not set_to_service.empty:
        logger.info(f"Devices coming to service: {len(set_to_service.index)}")
        to_service_status = statuses.merge(set_to_service.rename(columns={'ID': 'SvStatus.ConductingEquipment'}),
                                           on='SvStatus.ConductingEquipment')
        to_service_status['SvStatus.inService'] = 'true'
        model_data = triplets.rdf_parser.update_triplet_from_tableview(data=model_data,
                                                                       tableview=to_service_status,
                                                                       update=True,
                                                                       add=False)
    # Set future outages
    if not set_out_of_service.empty:
        logger.info(f"Devices going out of service: {len(set_out_of_service.index)}")
        out_service_status = statuses.merge(set_out_of_service.rename(columns={'ID': 'SvStatus.ConductingEquipment'}),
                                            on='SvStatus.ConductingEquipment')
        out_service_status['SvStatus.inService'] = 'false'
        model_data = triplets.rdf_parser.update_triplet_from_tableview(data=model_data,
                                                                       tableview=out_service_status,
                                                                       update=True,
                                                                       add=False)
    return model_data


def change_change_scenario_date_time_horizon_new(existing_models: list,
                                                 scenario_time_to_set: str,
                                                 time_horizon_to_set: str):
    """
    Changes scenario dates and time horizons in the opde models
    :param existing_models: list of opde models
    :param scenario_time_to_set: new scenario time to use
    :param time_horizon_to_set: new time horizon to use
    :return model data as triplets
    """
    parsed_scenario_time = parse_datetime(scenario_time_to_set)
    new_scenario_string = f"{parsed_scenario_time:%Y-%m-%dT%H:%M:00Z}"
    # Change to 'WK' if new_time_horizon is between 1D and 9D
    if time_horizon_to_set in ['1D', '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D']:
        time_horizon_to_set = 'WK'
    model_data = get_opdm_data_from_models(existing_models)
    # Parse file names to triplets values
    model_data = triplets.cgmes_tools.update_FullModel_from_filename(model_data)
    # Update these values
    model_data.loc[model_data.query('KEY == "Model.scenarioTime"').index, 'VALUE'] = new_scenario_string
    model_data.loc[model_data.query('KEY == "Model.processType"').index, 'VALUE'] = time_horizon_to_set
    # Update file names from updated values
    model_data = triplets.cgmes_tools.update_filename_from_FullModel(model_data)
    return model_data


def check_and_create_the_folder_path(folder_path: str):
    """
    Checks if folder path doesn't have any excessive special characters and it exists. Creates it if it does not
    :param folder_path: input given
    :return checked folder path
    """
    folder_path = check_the_folder_path(folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def save_models_to_minio(models_to_save: list,
                         retrieved_wk_files: list,
                         minio_bucket: str = 'opde-confidential-models',
                         subfolder_name: str = 'RCC_WK_IGMS',  # Set the folder name to RCC_WK_IGMS
                         save_minio_service: minio.ObjectStorage = minio.ObjectStorage()):
    """
    Saves models to minio, but only if they are not present in the list of retrieved WK files.
    All files will be saved in the 'RCC_WK_IGMS' folder.

    :param models_to_save: list of models to save
    :param retrieved_wk_files: list of already retrieved files to avoid duplication
    :param minio_bucket: name of minio bucket
    :param subfolder_name: prefix for the files if needed (set to RCC_WK_IGMS)
    :param save_minio_service: minio instance if given, created otherwise
    """
    for model in models_to_save:
        for component in model.get('opde:Component'):
            data = component.get('opdm:Profile', {}).get('DATA')
            if isinstance(data, bytes):
                data = BytesIO(data)
            file_name = component.get('opdm:Profile', {}).get('pmd:fileName')

            # Check if the file already exists in the retrieved WK files
            if any(file_name in wk_file for wk_file in retrieved_wk_files):
                logger.info(f"Skipping file {file_name} as it already exists in WK files.")
                continue
            save_model_to_minio(data=data,
                                data_file_name=file_name,
                                minio_bucket=minio_bucket,
                                subfolder_name=subfolder_name,
                                save_minio_service=save_minio_service)


def save_model_to_minio(data,
                        data_file_name,
                        minio_bucket: str = 'opde-confidential-models',
                        subfolder_name: str = 'RCC_WK_IGMS',
                        save_minio_service: minio.ObjectStorage = minio.ObjectStorage()):
    """
    Saves models to minio, but only if they are not present in the list of retrieved WK files.
    All files will be saved in the 'RCC_WK_IGMS' folder.

    :param data: data to be saved to minio
    :param data_file_name: data file name
    :param minio_bucket: minio bucket
    :param subfolder_name: prefix for the files if needed (set to RCC_WK_IGMS)
    :param save_minio_service: minio instance if given, created otherwise
    """
    # Store all files in the same folder 'RCC_WK_IGMS'
    file_loc = data_file_name
    if subfolder_name:
        file_loc = subfolder_name.removesuffix('/') + '/' + file_loc.removeprefix('/')
    logger.info(f"Uploading file to MINIO: {minio_bucket}/{file_loc}")
    data.name = file_loc
    try:
        save_minio_service.upload_object(data, bucket_name=minio_bucket)
    except Exception as e:
        logger.error(f"Error uploading {data_file_name} to Minio: {e}", exc_info=True)


def get_week_ahead_models(bucket_name, prefix, current_scenario_date):
    # Query objects from the service
    models = minio_service.query_objects(bucket_name, prefix=prefix)
    retrieved_models = []

    # Convert the current_scenario_date to a timezone-aware datetime object
    current_scenario_dt = datetime.datetime.strptime(current_scenario_date, "%Y-%m-%dT%H:%M:%S%z")

    # Calculate the cutoff date 9 days from the current_scenario_date
    cutoff_date = current_scenario_dt + datetime.timedelta(days=9)

    # Dictionary to store the latest version of each model (keyed by scenario date and TSO)
    latest_models = {}

    # Iterate over the models and filter by 'WK', TSO names (Elering, Litgrid, AST), and valid dates in the future
    for model in models:
        if "WK" in model.object_name and any(tso in model.object_name for tso in ['ELERING', 'LITGRID', 'AST']):
            # Extract the scenario date part from the object name (e.g., 20240907T0830Z)
            # Suggestion to replace
            try:
                # if date is 20240907T0830Z then corresponding regex would be:
                # fixed \d{8}T\d{4}Z or varying \d+T\d+Z
                scenario_datetime_pattern = r'(\d+T\d+Z)'
                # date_to_str = model.object_name.split('-')[3]
                date_to_str_match = re.search(scenario_datetime_pattern, model.object_name)
                if not date_to_str_match:
                    continue
                date_to_str = date_to_str_match.group(1)
                scenario_date_to = datetime.datetime.strptime(date_to_str, "%Y%m%dT%H%MZ")

                # Make scenario_date_to timezone-aware by adding UTC timezone
                scenario_date_to = scenario_date_to.replace(tzinfo=pytz.UTC)

                # Only consider files with future scenario dates (after current_scenario_date)
                if current_scenario_dt < scenario_date_to <= cutoff_date:
                    # Extract version number (e.g., from '001.zip')
                    version_match = re.search(r'-(\d{3})\.zip$', model.object_name)
                    if version_match:
                        version_number = int(version_match.group(1))  # Get version number as an integer
                    else:
                        version_number = -1  # If no version found, use a fallback

                    # Create a unique key based on scenario date and TSO name to track the latest version
                    # key = (scenario_date_to, model.object_name.split('-')[4])  # e.g., (20240907T0830Z, 'ELERING')
                    # Change this
                    entity = re.search(r'-(\w+)-\d{3}\.zip', model.object_name).group(1)
                    # key = (scenario_date_to, model.object_name.split('-')[5])
                    key = (scenario_date_to, entity)
                    # If the key doesn't exist or the current model has a higher version, update the latest model
                    if key not in latest_models or latest_models[key]['version'] < version_number:
                        latest_models[key] = {'version': version_number, 'object_name': model.object_name}
            except KeyError:
                logger.error(f"Unable to parse {model.object_name}")

    # Collect the latest models into the retrieved_models list
    for key, data in latest_models.items():
        retrieved_models.append(data['object_name'])

    # Return the list of all retrieved models (latest versions only)
    if retrieved_models:
        return retrieved_models
    else:
        print("No models found within the future date range and with valid TSO names.")
        return []


def package_wk_model_to_zip(updated_wk_model: dict):
    """
    Packages individual IGM to zip and gives it a name
    :param updated_wk_model: model in opde format
    """
    this_run = datetime.datetime.now().strftime("%Y-%m-%d")
    meta_data = metadata_from_filename(os.path.basename(updated_wk_model.get('pmd:content-reference')))
    wk_model = BytesIO()
    with (ZipFile(wk_model, "w") as wk_zip):
        for instance in updated_wk_model['opde:Component']:
            with ZipFile(BytesIO(instance['opdm:Profile']['DATA'])) as instance_zip:
                zip_file_name = instance.get('opdm:Profile', {}).get('pmd:fileName')
                if len(instance_zip.namelist()) == 1 and zip_file_name:
                    wk_zip.writestr(zip_file_name, instance['opdm:Profile']['DATA'])
                else:
                    for single_file_name in instance_zip.namelist():
                        logging.info(f"Adding file: {single_file_name}")
                        wk_zip.writestr(single_file_name, instance_zip.open(single_file_name).read())
    new_file_name = (f"{this_run}-{meta_data.get('pmd:validFrom', '')}-"
                     f"{meta_data.get('pmd:timeHorizon', '')}-"
                     f"{updated_wk_model.get('pmd:TSO', '')}-"
                     f"{meta_data.get('pmd:versionNumber', '')}.zip")
    wk_model.name = new_file_name
    return wk_model


def get_tso_name_from_string(str_value):
    """
    Tries to return a tso name from string
    :param str_value: name of the file to be parsed
    :return tso name if exists
    """
    try:
        components = os.path.basename(str_value).split('-')
        if len(components) == 7:
            return components[5]
        logger.warning(f"{str_value} is not a standard")
        regex_match = re.search(r'-(\w+)-\d{3}.', os.path.basename(str_value)).group(1)
        if regex_match:
            return regex_match
        return None
    except AttributeError:
        return None


def package_models_for_pypowsybl(list_of_zip: list):
    """
    Packages input data (BytesIO) into the format that is required for pypowsybl
    :param list_of_zip: list of data instances
    """
    all_models = []
    boundaries = []
    list_of_igms = []
    for zip_file in list_of_zip:
        try:
            # if zipfile.is_zipfile(zip_file):
            if not isinstance(zip_file, BytesIO):
                zip_file_bytes = BytesIO(zip_file)
            else:
                zip_file_bytes = zip_file
            zip_file_bytes.seek(0)
            zip_file_zip = ZipFile(zip_file_bytes)
            content = {name: zip_file_zip.read(name) for name in zip_file_zip.namelist()}
            for file_name_instance in content:
                file_content = BytesIO(content[file_name_instance])
                file_content.name = file_name_instance
                list_of_igms.append(file_content)
        except UnicodeDecodeError:
            print("Error received")
    models_data, boundary_data = group_files_by_origin(list_of_files=list_of_igms)
    if models_data:
        for tso in models_data:
            all_models.append(get_one_set_of_igms_from_local_storage(file_data=models_data[tso],
                                                                     tso_name=tso))
    if boundary_data:
        for tso in boundary_data:
            boundaries.append(get_one_set_of_boundaries_from_local_storage(file_names=boundary_data[tso]))
            if len(boundaries) > 0:
                boundaries = boundaries[0]
    return all_models, boundaries


def get_latest_boundary_from_minio(minio_client: minio.ObjectStorage = None,
                                   minio_bucket: str = 'opdm-data',
                                   minio_folder: str = 'CGMES/ENTSOE',
                                   check_later_versions: bool = True):
    """
    A temporary hack to check if minio has later version of boundary data available compared to elastic
    :param minio_client: instance of minio
    :param minio_bucket: bucket in minio where models are stored
    :param minio_folder: prefix (path) in bucket to models
    :param check_later_versions: if true then check if there are newer models in minio that are not in ELK
    :return boundary data
    """
    latest_elk_boundary = get_latest_boundary()
    if not check_later_versions:
        return latest_elk_boundary
    minio_client = minio_client or minio.ObjectStorage()
    list_of_files = minio_client.list_objects(bucket_name=minio_bucket,
                                              prefix=minio_folder,
                                              recursive=True)
    file_names = [file_name.object_name for file_name in list_of_files]
    file_name_frame = pandas.DataFrame([metadata_from_filename(os.path.basename(file_name)) | {'file_name': file_name}
                                        for file_name in file_names])
    file_name_frame['pmd:validFrom'] = pandas.to_datetime(file_name_frame['pmd:validFrom'])
    if file_name_frame['pmd:validFrom'].max() <= parse_datetime(latest_elk_boundary['pmd:validFrom']):
        return latest_elk_boundary
    last_set = file_name_frame[file_name_frame['pmd:validFrom'] == file_name_frame['pmd:validFrom'].max()]
    last_set_models = []
    for index, row in last_set.iterrows():
        profile = BytesIO(minio_client.download_object(bucket_name=minio_bucket, object_name=row['file_name']))
        profile.name = row['file_name']
        last_set_models.append(profile)
    models_data, boundary_data = group_files_by_origin(list_of_files=last_set_models)
    boundaries = [get_one_set_of_boundaries_from_local_storage(file_names=boundary_data[tso]) for tso in boundary_data]
    if len(boundaries) > 0:
        return boundaries[0]
    return latest_elk_boundary


def get_opc_data(index_name: str = 'filtered_opc*',
                 opc_start_datetime: datetime.datetime | str = datetime.datetime.today(),
                 opc_end_datetime: datetime.datetime | str = None,
                 start_datetime_field_name: str = 'start_date',
                 end_date_time_field_name: str = 'end_date',
                 max_query_size: int = 10000):
    """
    Checks and gets the version number from elastic
    Note that it works only if logger.info(f"Publishing {instance_file.name} to OPDM")
    is used when publishing files to OPDM
    :param index_name: index from where to search
    :param opc_start_datetime: datetime instance from where to look, if not set then takes current day
    :param opc_end_datetime: for period search
    :param max_query_size: maximum size of query, hopefully 10000 is enough for one timestamp
    :param start_datetime_field_name: field name that contains start dates
    :param end_date_time_field_name: field name that contains end dates
    :return version number as a string
    """
    query_components = []
    # Slice in time: take all outages start before or on the opc_start_datetime and end after or on
    # opc_end_datetime (if given)/opc_start_datetime
    if isinstance(opc_start_datetime, datetime.datetime):
        opc_start_datetime = opc_start_datetime.strftime("%Y-%m-%dT%H:%M:%S")
    query_components.append({"range": {start_datetime_field_name: {"lte": opc_start_datetime}}})
    if opc_end_datetime:
        if isinstance(opc_end_datetime, datetime.datetime):
            opc_end_datetime = opc_end_datetime.strftime("%Y-%m-%dT%H:%M:%S")
        query_components.append({"range": {end_date_time_field_name: {"gte": opc_end_datetime}}})
    else:
        query_components.append({"range": {end_date_time_field_name: {"gte": opc_start_datetime}}})
    opc_query = {"bool": {"must": query_components}}
    elastic_client = Elastic()
    results = elastic_client.get_docs_by_query(index=index_name, query=opc_query, size=max_query_size)
    return results


def flatten_list(x):
    """
    Flattens nested list to single level list recursively
    :param x: input list
    """
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        return [a for i in x for a in flatten_list(i)]
    return [x]


def filter_opc_data_by_tso(row_data, field_name, tso_eic_codes: dict | list):
    """
    Filters dataframe rows by tso eic codes
    :param row_data: row from pandas.DataFrame
    :param field_name: field where tso eic code is
    :param tso_eic_codes: list of tso eic code prefixes
    """
    if isinstance(tso_eic_codes, dict):
        tso_eic_codes = [*tso_eic_codes.values()]
    if isinstance(tso_eic_codes, list):
        tso_eic_codes = flatten_list(tso_eic_codes)
    else:
        tso_eic_codes = [tso_eic_codes]
    eic_code = row_data[field_name].str.upper()
    return eic_code[:2] in tso_eic_codes


def get_eic_to_mrid_mapping(elastic_index: str = 'config-network-elements*',
                            field_names: list = None):
    """
    Queries eic to mrid map from elastic
    :param elastic_index: name of the index
    :param field_names: names of the columns for eic and mrid
    :return pandas.Dataframe
    """
    if not field_names:
        field_names = ['eic', 'mrid']
    query_part = {"match_all": {}}
    try:
        elastic_client = Elastic()
        results = elastic_client.get_docs_by_query(index=elastic_index,
                                                   query=query_part,
                                                   size=1000,
                                                   return_df=True)
        if results.empty:
            raise NoContentFromElasticException
        return results[field_names]
    except (NoContentFromElasticException, KeyError, ValueError):
        logger.info(f"no data found, or missing fields")
    except Exception as ex:
        logger.warning(f"Got elastic error: {ex}")
    return pandas.DataFrame


def generate_sample_task(task_scenario_date,
                         task_included_models,
                         task_excluded_models: list = None,
                         task_models_from_minio: list = None,
                         task_time_horizon: str = "WK",
                         task_merging_area: str = 'BA',
                         task_merging_entity: str = 'BALTICRSC',
                         task_version_number: str = "101",
                         task_mas_name: str = "http://www.baltic-rsc.eu/OperationalPlanning/CGM",
                         task_replace: bool = False,
                         task_scale: bool = False,
                         task_to_opdm: bool = False,
                         task_to_minio: bool = False,
                         task_send_report: bool = False):
    created_at = datetime.datetime.utcnow()
    task_creation_time = created_at.isoformat()
    task_duration = 'PT1H'
    task_gate = 'PT15M'
    job_period_start = convert_to_utc(created_at)
    job_gate_start = convert_to_utc(created_at)
    job_gate_end = convert_to_utc(job_gate_start + parse_duration(task_gate))
    job_period_end = convert_to_utc(job_period_start + parse_duration(task_duration))

    if not task_excluded_models:
        task_excluded_models = []
    if not task_models_from_minio:
        task_models_from_minio = []
    return {
        "@context": "https://example.com/task_context.jsonld",
        "@type": "Task",
        "@id": f"urn:uuid:{str(uuid4())}",
        "process_id": "https://example.com/processes/CGM_CREATION",
        "run_id": "https://example.com/runs/IntraDayCGM/1",
        "job_id": f"urn:uuid:{str(uuid4())}",
        "task_type": "manual",
        "task_initiator": os.getlogin(),
        "task_priority": "normal",
        "task_creation_time": task_creation_time,
        "task_update_time": task_creation_time,
        "task_status": "created",
        "task_status_trace": [
            {"status": "created", "timestamp": task_creation_time}
        ],
        "task_dependencies": [],
        "task_tags": [],
        "task_retry_count": 0,
        "task_timeout": task_duration,
        "task_gate_open": job_gate_start.isoformat(),
        "task_gate_close": job_gate_end.isoformat(),
        "job_period_start": job_period_start.isoformat(),
        "job_period_end": job_period_end.isoformat(),
        "task_properties": {
            "timestamp_utc": task_scenario_date,
            "merge_type": task_merging_area,
            "merging_entity": task_merging_entity,
            "included": task_included_models,
            "excluded": task_excluded_models,
            "local_import": task_models_from_minio,
            "time_horizon": task_time_horizon,
            "version": task_version_number,
            "mas": task_mas_name,
            "replacement": str(task_replace).lower(),
            "scaling": str(task_scale).lower(),
            "upload_to_opdm": str(task_to_opdm).lower(),
            "upload_to_minio": str(task_to_minio).lower(),
            "send_merge_report": str(task_send_report).lower()
        }
    }


if __name__ == "__main__":

    import sys

    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    tso_codes = {'ELERING': ['EE', '38'],
                 'AST': ['LV', '43'],
                 'LITGRID': ['LT', '41'],
                 'PSE': ['PL', '19']}

    merge_prefix = 'RMM'
    minio_merged_folder = f"EMFOS/{merge_prefix}"
    test_version = '123'
    test_merging_entity = 'BALTICRSC'
    test_merging_area = 'BA'
    test_mas = 'http://www.baltic-rsc.eu/OperationalPlanning/CGM'

    minio_service = minio.ObjectStorage()
    start_scenario_date = "2024-10-10T08:30:00+00:00"
    existing_time_horizons = ['1D', '2D']
    virtual_time_horizons = ['3D', '4D', '5D', '6D', '7D', '8D', '9D']
    included_models = [
        'PSE'
    ]
    included_models_from_minio = [
        'LITGRID',
        'ELERING',
        'AST'
    ]

    # # opc-report index
    opc_index_name = 'opc-report*'
    opc_name_field = 'TimeSeries.RegisteredResource.name'
    opc_start_date_field = 'TimeSeries.outage_Period.timeInterval.start'
    opc_end_date_field = 'TimeSeries.outage_Period.timeInterval.end'
    # # Change this, in opc-report index everybody is using different field for storing mRID (EIC code)
    opc_eic_column_name = 'TimeSeries.RegisteredResource.mRID.#text'

    # # opc-filtered index
    # opc_index_name = 'filtered_opc*'
    # opc_name_field = 'name'
    # opc_start_date_field = 'start_date'
    # opc_end_date_field = 'end_date'
    # opc_eic_column_name = 'EIC'

    elastic_index_for_mapping = 'config-network-elements*'
    eic_mrid_map_eic = 'eic'
    eic_mrid_map_mrid = 'mrid'
    eic_mrid_map = get_eic_to_mrid_mapping(elastic_index_for_mapping, field_names=[eic_mrid_map_eic, eic_mrid_map_mrid])
    retrieved_WK_files = get_week_ahead_models(bucket_name='opde-confidential-models',
                                               prefix='IGM',
                                               current_scenario_date=start_scenario_date)

    # The 'retrieved_files' list will contain all matching future files with the latest version
    print("\nRetrieved future files with latest versions list:", retrieved_WK_files)

    all_horizons = [*existing_time_horizons, *virtual_time_horizons]
    some_results = calculate_scenario_datetime_and_time_horizon_by_shifts(given_datetime=start_scenario_date,
                                                                          time_horizons=all_horizons,
                                                                          shifts=shift_mapping,
                                                                          # Change this back, during daytime no next day
                                                                          # models
                                                                          offset='P0D'
                                                                          )
    # existing_folder = './weekly_models'
    existing_folder = r'E:\margus.ratsep\CGM_dump'
    current_run = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    existing_folder = os.path.join(existing_folder, current_run)
    # existing_folder = './existing'

    save_to_local = False
    save_to_minio = False
    save_cgm_to_minio = True
    save_cgm_to_local = True
    send_merge_report_to_elk = True
    print(pandas.DataFrame(some_results.values()).to_string())
    special_case_time_horizon = 'WK'
    for result in some_results:
        existing_tsos = []
        old_time_horizon = some_results[result].get('existing_time_horizon')
        old_scenario_date = some_results[result].get('existing_time_scenario')
        new_time_horizon = some_results[result].get('future_time_horizon')
        new_scenario_date = some_results[result].get('future_time_scenario')
        sample_task = generate_sample_task(task_scenario_date=new_scenario_date,
                                           task_included_models=[*included_models_from_minio, *included_models],
                                           task_models_from_minio=included_models_from_minio,
                                           task_time_horizon=new_time_horizon,
                                           task_merging_area=test_merging_area,
                                           task_merging_entity=test_merging_entity,
                                           task_version_number=test_version,
                                           task_mas_name=test_mas,
                                           task_to_minio=save_cgm_to_minio,
                                           task_send_report=send_merge_report_to_elk)
        merge_log = {"uploaded_to_opde": 'false',
                     "uploaded_to_minio": 'false',
                     "scaled": 'false',
                     "exclusion_reason": [],
                     "replacement": 'false',
                     "replaced_entity": [],
                     "replacement_reason": []}

        logger.info(f"Processing {new_scenario_date}")
        # 1. Take all existing models from minio opde-confidential-models/IGM
        wk_file_name = parse_datetime(new_scenario_date).strftime("%Y%m%dT%H%MZ")
        # take all
        # matching_file_names = [file_name for file_name in retrieved_WK_files if f"{wk_file_name}-WK" in file_name]
        # or filter by tso
        matching_file_names = [file_name for file_name in retrieved_WK_files
                               if (f"{wk_file_name}-WK" in file_name and
                                   get_tso_name_from_string(file_name)
                                   in [*included_models, *included_models_from_minio])]
        weekly_models = []
        filtered_models = []
        # 1.1 Download the models
        for matching_file_name in matching_file_names:
            weekly_models.append(minio_service.download_object(bucket_name='opde-confidential-models',
                                                               object_name=matching_file_name))
        # 1.2 mark all tsos who's igms were received
        existing_tsos.extend([get_tso_name_from_string(file_name) for file_name in matching_file_names])
        logger.info(f"WK from MINIO IGM: {', '.join(matching_file_names)}")
        # 2. Get designated models from opde (exclude those that were already received from minio)
        included_models = [tso_name for tso_name in included_models if tso_name not in existing_tsos]
        if included_models:
            if len(included_models) == 1:
                filtered_models = get_latest_models_and_download(time_horizon=old_time_horizon,
                                                                 scenario_date=old_scenario_date,
                                                                 tso=included_models[0])
            else:
                prefiltered_models = get_latest_models_and_download(time_horizon=old_time_horizon,
                                                                    scenario_date=old_scenario_date)
                filtered_models = filter_models(prefiltered_models, included_models, filter_on='pmd:TSO')
            if filtered_models:
                opdm_tsos = [model.get('pmd:TSO') for model in filtered_models]
                existing_tsos.extend(opdm_tsos)
                logger.info(f"OPDM: {old_scenario_date}-{old_time_horizon} , GOT {', '.join(opdm_tsos)} ")
        # 3. Get models from opde-confidential-models/IGM, exclude those that were got from OPDE and already are WK
        minio_igm_model_tsos = [tso for tso in included_models_from_minio if tso not in existing_tsos]
        if minio_igm_model_tsos:
            minio_models = minio_service.get_latest_models_and_download(time_horizon=old_time_horizon,
                                                                        scenario_datetime=old_scenario_date,
                                                                        model_entity=minio_igm_model_tsos,
                                                                        bucket_name='opde-confidential-models',
                                                                        prefix='IGM')
            # Add tsos whose models where in 'opde-confidential-models to existing ones
            if minio_models:
                minio_tsos = [minio_model.get('pmd:TSO') for minio_model in minio_models]
                existing_tsos.extend(minio_tsos)
                logger.info(f"MINIO IGM: {old_scenario_date}-{old_time_horizon} , GOT {', '.join(minio_tsos)} ")
                filtered_models.extend(minio_models)
        # 4. Get models from opde-confidential-models/RCC_WK_IGMS, exclude those found in #1 and #2 and #3
        weekly_tsos = [tso_name for tso_name in included_models_from_minio if tso_name not in existing_tsos]
        if weekly_tsos:
            week_igms = minio_service.get_latest_models_and_download(time_horizon=new_time_horizon,
                                                                     scenario_datetime=new_scenario_date,
                                                                     model_entity=weekly_tsos,
                                                                     bucket_name='opde-confidential-models',
                                                                     prefix='RCC_WK_IGMS')
            if week_igms:
                week_tsos = [minio_model.get('pmd:TSO') for minio_model in week_igms]
                existing_tsos.extend(week_tsos)
                logger.info(f"MINIO RCC WK IGM: {new_scenario_date}-{new_time_horizon} , GOT {', '.join(week_tsos)}")
                # Need to decide in which format they come, used here the same approach as the ones in IGM folder
                # filtered_models.extend(week_igms)
                weekly_models.extend(week_igms)
        # logger.info(f"{result}: found {', '.join(existing_tsos)}")
        if filtered_models:
            # updated_models = change_change_scenario_date_time_horizon(existing_models=filtered_models,
            #                                                           time_horizon_to_set=new_time_horizon,
            #                                                           scenario_time_to_set=new_scenario_date,
            #                                                           existing_outages=old_filtered_outages,
            #                                                           outages_to_set=new_filtered_outages)
            updated_models = change_change_scenario_date_time_horizon_new(existing_models=filtered_models,
                                                                          time_horizon_to_set=new_time_horizon,
                                                                          scenario_time_to_set=new_scenario_date)
            old_filtered_outages = get_opc_data(index_name=opc_index_name,
                                                opc_start_datetime=old_scenario_date,
                                                start_datetime_field_name=opc_start_date_field,
                                                end_date_time_field_name=opc_end_date_field)
            new_filtered_outages = get_opc_data(index_name=opc_index_name,
                                                opc_start_datetime=new_scenario_date,
                                                start_datetime_field_name=opc_start_date_field,
                                                end_date_time_field_name=opc_end_date_field)
            # fix eic codes:
            if 'filtered_opc' in opc_index_name:
                old_filtered_outages[opc_eic_column_name] = (old_filtered_outages['eic']
                                                             .combine_first(old_filtered_outages['element_id_eic']))
                new_filtered_outages[opc_eic_column_name] = (new_filtered_outages['eic']
                                                             .combine_first(new_filtered_outages['element_id_eic']))
            old_filtered_outages = old_filtered_outages.drop_duplicates(subset=[opc_name_field],
                                                                        keep='first')
            new_filtered_outages = new_filtered_outages.drop_duplicates(subset=[opc_name_field],
                                                                        keep='first')
            outage_changes = old_filtered_outages[[opc_name_field]].merge(new_filtered_outages[[opc_name_field]],
                                                                          on=opc_name_field,
                                                                          how='outer',
                                                                          indicator=True)
            outage_changes = outage_changes[
                (outage_changes['_merge'] == 'left_only') | (outage_changes['_merge'] == 'right_only')]
            if not outage_changes.empty:
                logger.info(f"Following outages must be updated:")
                print(outage_changes.to_string())
                updated_models = update_outages(model_data=updated_models,
                                                existing_outages=old_filtered_outages,
                                                outages_to_set=new_filtered_outages,
                                                eic_to_mrid_map=eic_mrid_map,
                                                outage_eic_codes_column=opc_eic_column_name,
                                                mrid_mrid_codes_column=eic_mrid_map_mrid,
                                                mrid_eic_codes_column=eic_mrid_map_eic,
                                                outage_name_field=opc_name_field)
            updated_serialized_data = export_to_cgmes_zip([updated_models])
            updated_model_data, _ = group_files_by_origin(list_of_files=updated_serialized_data)
            for updated_tso in updated_model_data:
                updated_tso_model = get_one_set_of_igms_from_local_storage(file_data=updated_model_data[updated_tso],
                                                                           tso_name=updated_tso)
                wk_model_zip = package_wk_model_to_zip(updated_tso_model)
                if save_to_local:
                    save_merged_model_to_local_storage(wk_model_zip, local_storage_location=existing_folder)
                if save_to_minio:
                    save_model_to_minio(data=wk_model_zip, data_file_name=wk_model_zip.name)
                weekly_models.append(wk_model_zip)
                merge_log.get('replaced_entity').append({'tso': updated_tso,
                                                         'replacement_time_horizon': old_time_horizon,
                                                         'replacement_scenario_date': old_scenario_date})
            merge_log.update({'replacement': 'true'})
        print(len(weekly_models))
        missing_tsos = [tso for tso in [*included_models, *included_models_from_minio] if tso not in existing_tsos]
        merge_log.get('exclusion_reason').extend(
            [{'tso': tso, 'reason': 'Model missing from Minio'} for tso in missing_tsos])
        weekly_models, boundary = package_models_for_pypowsybl(list_of_zip=weekly_models)
        if not boundary:
            # boundary = get_latest_boundary()
            boundary = get_latest_boundary_from_minio()
        merge_start = datetime.datetime.utcnow()
        test_model, test_network = merge_models(list_of_models=weekly_models,
                                                latest_boundary=boundary,
                                                time_horizon=special_case_time_horizon,
                                                scenario_datetime=new_scenario_date,
                                                merging_area=test_merging_area,
                                                merging_entity=test_merging_entity,
                                                merge_prefix=merge_prefix,
                                                mas=test_mas,
                                                version=test_version)
        merge_end = datetime.datetime.utcnow()
        if save_cgm_to_local:
            check_and_create_the_folder_path(existing_folder)
            full_name = existing_folder.removesuffix('/') + '/' + test_model.name.removeprefix('/')
            with open(full_name, 'wb') as write_file:
                write_file.write(test_model.getbuffer())
        # Do we need to update it to opde also
        # merge_log.update({'uploaded_to_opde': 'True'})
        if save_cgm_to_minio:
            save_model_to_minio(data=test_model,
                                subfolder_name=minio_merged_folder,
                                data_file_name=test_model.name)
            merge_log.update({'uploaded_to_minio': 'true'})
        file_name_bare = test_model.name.removesuffix('.zip')
        file_name_full = f"{minio_merged_folder}/{file_name_bare}.zip"

        merge_log.update({'task': sample_task,
                          'small_island_size': SMALL_ISLAND_SIZE,
                          'loadflow_settings': MERGE_LOAD_FLOW_SETTINGS,
                          'merge_duration': f'{(merge_end - merge_start).total_seconds()}',
                          'content_reference': file_name_full,
                          'cgm_name': file_name_bare})

        try:
            merge_report = merge_functions.generate_merge_report(test_network, weekly_models, merge_log)
            try:
                if send_merge_report_to_elk:
                    response = elastic.Elastic.send_to_elastic(index=PY_REPORT_ELK_INDEX,
                                                               json_message=merge_report)
                    if response.status_code != requests.codes.OK and response.status_code != requests.codes.created:
                        logger.error(f"Unable to upload report to elastic: {response.text}")
                else:
                    json_file_name = (existing_folder.removesuffix('/') + '/' +
                                      test_model.name.removeprefix('/').remove.suffix('.zip') + '.json')
                    json.dump(merge_report, open(json_file_name, 'w'))
            except Exception as error:
                logger.error(f"Merge report sending to Elastic failed: {error}")
        except Exception as error:
            logger.error(f"Failed to create merge report: {error}")

        logger.info(f"{new_scenario_date} DONE")
    logger.info("ALL DONE")
