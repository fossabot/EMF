import logging
import os
import re
from datetime import datetime
from io import BytesIO
from zipfile import ZipFile

import pandas

from emf.common.integrations.object_storage.file_system import group_files_by_origin, \
    get_one_set_of_igms_from_local_storage, get_one_set_of_boundaries_from_local_storage, \
    save_merged_model_to_local_storage
from emf.common.integrations.object_storage.file_system_general import check_the_folder_path
from emf.loadflow_tool.model_merger.merge_from_local import merge_models
from emf.task_generator.time_helper import parse_duration
import pytz
import triplets
import config
import datetime
from emf.loadflow_tool.helper import load_opdm_data, metadata_from_filename
from emf.task_generator.time_helper import parse_datetime
from emf.common.config_parser import parse_app_properties

try:
    from emf.common.integrations import minio
except ImportError:
    from emf.common.integrations import minio_api as minio
from emf.common.integrations.object_storage.models import get_latest_models_and_download, get_latest_boundary
from emf.loadflow_tool.model_merger.merge_functions import filter_models, export_to_cgmes_zip

# from rcc_common_tools import minio_api

logger = logging.getLogger(__name__)
parse_app_properties(caller_globals=globals(), path=config.paths.cgm_worker.merger)

SEPARATOR_SYMBOL = '/'
WINDOWS_SEPARATOR = '\\'

# Shift mapping logic for future scenario dates
shift_mapping = {
    # Change these pack
    '1D': 'P0D',  # Shift for '1D' is +2 days
    '2D': 'P1D',  # Shift for '2D' is +3 days
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


def get_date_from_time_horizon(time_horizon: str):
    """
    Parses number of dates from time horizon
    :param time_horizon: time horizon as string
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
    :param imaginary_time_horizons: time horizons to be substitute
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
    except Exception:
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
    :param shifts: time horizons to be substitute
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


def change_change_scenario_date_time_horizon(existing_models: list,
                                             new_scenario_time: str,
                                             new_time_horizon: str):
    """
    Changes scenario dates and time horizons in the opde models
    :param existing_models: list of opde models
    :param new_scenario_time: new scenario time to use
    :param new_time_horizon: new time horizon to use
    """
    parsed_scenario_time = parse_datetime(new_scenario_time)

    # Change to 'WK' if new_time_horizon is between 1D and 9D
    if new_time_horizon in ['1D', '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D']:
        new_time_horizon = 'WK'

    for model in existing_models:
        content_reference = None
        for component in model.get('opde:Component'):
            opdm_profile = component.get('opdm:Profile')
            if not opdm_profile:
                raise ValueError(f"Model in unknown format")
            profile = load_opdm_data(opdm_objects=[model],
                                     profile=opdm_profile.get('pmd:cgmesProfile'))
            # Change scenario time in headers
            new_scenario_string = f"{parsed_scenario_time:%Y-%m-%dT%H:%M:00Z}"
            profile.loc[profile.query('KEY == "Model.scenarioTime"').index, 'VALUE'] = new_scenario_string
            profile.loc[profile.query('KEY == "Model.processType"').index, 'VALUE'] = new_time_horizon

            # Update file names:
            file_names = profile[profile['KEY'] == 'label']
            for index, file_name_row in file_names.iterrows():
                file_name = file_name_row['VALUE']
                file_name_meta = metadata_from_filename(file_name)
                file_name_meta['pmd:validFrom'] = f"{parsed_scenario_time:%Y%m%dT%H%MZ}"
                file_name_meta['pmd:timeHorizon'] = new_time_horizon  # Apply the 'WK' here
                updated_file_name = reduced_filename_from_metadata(file_name_meta)
                profile.loc[index, 'VALUE'] = updated_file_name

            profile = triplets.cgmes_tools.update_FullModel_from_filename(profile)
            serialized_data = export_to_cgmes_zip([profile])
            if len(serialized_data) == 1:
                serialized = serialized_data[0]
                serialized.seek(0)
                opdm_profile['DATA'] = serialized.read()
                opdm_profile['pmd:scenarioDate'] = new_scenario_string
                opdm_profile['pmd:timeHorizon'] = new_time_horizon
                opdm_profile['pmd:fileName'] = serialized.name
                meta_data = metadata_from_filename(serialized.name)
                opdm_profile['pmd:content-reference'] = (f"CGMES/"
                                                         f"{new_time_horizon}/"
                                                         f"{meta_data['pmd:modelPartReference']}/"
                                                         f"{parsed_scenario_time:%Y%m%d}/"
                                                         f"{parsed_scenario_time:%H%M%S}/"
                                                         f"{meta_data['pmd:cgmesProfile']}/"
                                                         f"{serialized.name}")
                if not content_reference:
                    content_reference = opdm_profile['pmd:content-reference']
        if content_reference:
            model['pmd:content-reference'] = content_reference
    return existing_models


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
    :param minio_bucket: name of minio bucket
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
                date_to_str = model.object_name.split('-')[3]
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
                    key = (scenario_date_to, model.object_name.split('-')[5])
                    # If the key doesn't exist or the current model has a higher version, update the latest model
                    if key not in latest_models or latest_models[key]['version'] < version_number:
                        latest_models[key] = {'version': version_number, 'object_name': model.object_name}
            except Exception:
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
    """
    try:
        components = os.path.basename(str_value).split('-')
        if len(components) == 7:
            return components[5]
        return None
    except Exception:
        return None


def package_models_for_pypowsybl(list_of_zip: list):
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


if __name__ == "__main__":

    import sys

    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    merge_prefix = 'RMM'
    minio_merged_folder = f"EMFOS/{merge_prefix}"
    test_version = '123'
    test_merging_entity = 'BALTICRSC'
    test_merging_area = 'BA'
    test_mas = 'http://www.baltic-rsc.eu/OperationalPlanning/CGM'

    minio_service = minio.ObjectStorage()
    current_scenario_date = "2024-09-18T10:30:00+02:00"
    existing_time_horizons = ['1D', '2D']
    virtual_time_horizons = ['3D', '4D', '5D', '6D', '7D', '8D', '9D']
    included_models = ['PSE']
    included_models_from_minio = ['LITGRID', 'ELERING', 'AST']

    retrieved_WK_files = get_week_ahead_models(bucket_name='opde-confidential-models',
                                               prefix='IGM',
                                               current_scenario_date=current_scenario_date)

    # The 'retrieved_files' list will contain all matching future files with the latest version
    print("\nRetrieved future files with latest versions list:", retrieved_WK_files)

    all_horizons = [*existing_time_horizons, *virtual_time_horizons]
    some_results = calculate_scenario_datetime_and_time_horizon_by_shifts(given_datetime=current_scenario_date,
                                                                          time_horizons=all_horizons,
                                                                          shifts=shift_mapping,
                                                                          # Change this back, during daytime no next day
                                                                          # models
                                                                          offset='P0D')
    existing_folder = './existing'
    updated_folder = './updated'
    save_to_local_storage = False
    save_to_minio = False
    save_cgm_to_minio = True
    save_cgm_to_local = False
    print(pandas.DataFrame(some_results.values()).to_string())
    special_case_time_horizon = 'WK'
    for result in some_results:
        existing_tsos = []
        old_time_horizon = some_results[result].get('existing_time_horizon')
        old_scenario_date = some_results[result].get('existing_time_scenario')
        new_time_horizon = some_results[result].get('future_time_horizon')
        new_scenario_date = some_results[result].get('future_time_scenario')
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
                models = get_latest_models_and_download(time_horizon=old_time_horizon,
                                                        scenario_date=old_scenario_date)
                filtered_models = filter_models(models, included_models, filter_on='pmd:TSO')
            if filtered_models:
                opdm_tsos = [model.get('pmd:TSO') for model in filtered_models]
                existing_tsos.extend(opdm_tsos)
                logger.info(f"OPDM: {old_scenario_date}-{old_time_horizon} , GOT {', '.join(opdm_tsos)} ")
        # 3. Get models from opde-confidential-models/IGM, exclude those that were got from OPDE and already are WK
        minio_model_tsos = [tso for tso in included_models_from_minio if tso not in existing_tsos]
        if minio_model_tsos:
            minio_models = minio_service.get_latest_models_and_download(time_horizon=old_time_horizon,
                                                                        scenario_datetime=old_scenario_date,
                                                                        model_entity=minio_model_tsos,
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
            updated_models = change_change_scenario_date_time_horizon(existing_models=filtered_models,
                                                                      new_time_horizon=new_time_horizon,
                                                                      new_scenario_time=new_scenario_date)

            for updated_model in updated_models:
                wk_model_zip = package_wk_model_to_zip(updated_model)
                if save_to_local_storage:
                    save_merged_model_to_local_storage(wk_model_zip, local_storage_location=existing_folder)
                if save_to_minio:
                    save_model_to_minio(data=wk_model_zip, data_file_name=wk_model_zip.name)
                weekly_models.append(wk_model_zip)
        print(len(weekly_models))
        weekly_models, boundary = package_models_for_pypowsybl(list_of_zip=weekly_models)
        if not boundary:
            boundary = get_latest_boundary()

        test_model = merge_models(list_of_models=weekly_models,
                                  latest_boundary=boundary,
                                  time_horizon=special_case_time_horizon,
                                  scenario_datetime=new_scenario_date,
                                  merging_area=test_merging_area,
                                  merging_entity=test_merging_entity,
                                  merge_prefix=merge_prefix,
                                  mas=test_mas,
                                  version=test_version)
        if save_cgm_to_local:
            check_and_create_the_folder_path(existing_folder)
            full_name = existing_folder.removesuffix('/') + '/' + test_model.name.removeprefix('/')
            with open(full_name, 'wb') as write_file:
                write_file.write(test_model.getbuffer())
        if save_cgm_to_minio:
            save_model_to_minio(data=test_model,
                                subfolder_name=minio_merged_folder,
                                data_file_name=test_model.name)
        logger.info(f"{new_scenario_date} DONE")
