import logging
import os
import re
from datetime import datetime
from io import BytesIO

from emf.task_generator.time_helper import parse_duration
from datetime import timedelta
import pytz
import triplets
import config
import datetime
from emf.loadflow_tool.helper import load_opdm_data, metadata_from_filename
from emf.task_generator.time_helper import parse_datetime
from emf.common.config_parser import parse_app_properties
from emf.common.integrations import minio_api as minio
from emf.common.integrations.object_storage.models import get_latest_models_and_download
from emf.loadflow_tool.model_merger.merge_functions import filter_models, export_to_cgmes_zip
# from rcc_common_tools import minio_api

logger = logging.getLogger(__name__)
parse_app_properties(caller_globals=globals(), path=config.paths.cgm_worker.merger)

SEPARATOR_SYMBOL = '/'
WINDOWS_SEPARATOR = '\\'

# Shift mapping logic for future scenario dates
shift_mapping = {
    '1D': 'P2D',  # Shift for '1D' is +2 days
    '2D': 'P3D',  # Shift for '2D' is +3 days
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
    imaginary_offset_duration = parse_duration(imaginary_offset)

    # Map 'WK' to 7 days for the default time horizon
    if default_time_horizon == 'WK':
        default_time_horizon = '7D'

    # Apply the new shift logic for actual time horizons
    for time_horizon in actual_time_horizons:
        time_horizon_str = get_date_from_time_horizon(time_horizon)
        shift_duration = shift_mapping.get(time_horizon_str, actual_offset_duration)

        time_delta = parse_duration(shift_duration)
        future_scenario_date = starting_timestamp + time_delta
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
        future_scenario_date = starting_timestamp + time_delta + actual_offset_duration
        existing_scenario_date = future_scenario_date + actual_offset_duration + imaginary_offset_duration
        set_time_horizon = existing_timestamps.get(existing_scenario_date.isoformat(), default_time_horizon)
        time_horizon_value = {'future_time_scenario': future_scenario_date.isoformat(),
                              'future_time_horizon': time_horizon,
                              'existing_time_scenario': existing_scenario_date.isoformat(),
                              'existing_time_horizon': set_time_horizon}
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
    return existing_models


def check_the_folder_path(folder_path: str):
    """
    Checks folder path for special characters
    :param folder_path: input given
    :return checked folder path
    """
    if not folder_path.endswith(SEPARATOR_SYMBOL):
        folder_path = folder_path + SEPARATOR_SYMBOL
    double_separator = SEPARATOR_SYMBOL + SEPARATOR_SYMBOL
    folder_path = folder_path.replace(double_separator, SEPARATOR_SYMBOL)
    folder_path = folder_path.replace(WINDOWS_SEPARATOR, SEPARATOR_SYMBOL)
    return folder_path


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


def save_models_to_local_storage(models_to_save: list, root_folder: str = 'RCC_WK_IGMS'):
    """
    Saves models to local storage, uses pmd:content-reference as file path.
    All files will be saved in the 'RCC_WK_IGMS' folder.

    :param models_to_save: list of opdm object to save
    :param root_folder: location where to save (default: 'RCC_WK_IGMS')
    """
    for model in models_to_save:
        for component in model.get('opde:Component'):
            data = component.get('opdm:Profile', {}).get('DATA')
            file_name = component.get('opdm:Profile', {}).get('pmd:fileName')
            file_path = component.get('opdm:Profile', {}).get('pmd:content-reference')

            # Ensure all files go into the 'RCC_WK_IGMS' folder
            file_root_folder = root_folder
            check_and_create_the_folder_path(file_root_folder)

            file_name = file_root_folder.removesuffix('/') + '/' + file_name.removeprefix('/')
            with open(file_name, 'wb') as save_file_name:
                save_file_name.write(data)


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
            file_path = component.get('opdm:Profile', {}).get('pmd:content-reference')
            file_loc = file_name  # Store based on the file name only

            # Check if the file already exists in the retrieved WK files
            if any(file_name in wk_file for wk_file in retrieved_wk_files):
                logger.info(f"Skipping file {file_name} as it already exists in WK files.")
                continue

            # Store all files in the same folder 'RCC_WK_IGMS'
            if subfolder_name:
                file_loc = subfolder_name.removesuffix('/') + '/' + file_loc.removeprefix('/')
            logger.info(f"Uploading file to MINIO: {minio_bucket}/{file_loc}")
            data.name = file_loc
            try:
                save_minio_service.upload_object(data, bucket_name=minio_bucket)
            except Exception as e:
                logger.error(f"Error uploading {file_name} to Minio: {e}", exc_info=True)


def filter_minio_object_for_wk(objects, included_models_from_minio, start_date, end_date):
    result_list = []
    for obj in objects:
        file_name = obj.object_name

        # Extract scenario date from the file name (first 15 chars)
        scenario_date_str = file_name[:15]
        time_horizon = file_name.split('-')[1]
        model_name = file_name.split('-')[2]

        scenario_date = datetime.strptime(scenario_date_str, "%Y%m%dT%H%MZ")

        # Check if the scenario date is within the next 9 days and the time horizon is 'WK'
        if start_date <= scenario_date <= end_date and time_horizon == "WK" and model_name in included_models_from_minio:
            result_list.append(obj)

    return result_list


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
                key = (scenario_date_to, model.object_name.split('-')[4])  # e.g., (20240907T0830Z, 'ELERING')

                # If the key doesn't exist or the current model has a higher version, update the latest model
                if key not in latest_models or latest_models[key]['version'] < version_number:
                    latest_models[key] = {'version': version_number, 'object_name': model.object_name}

    # Collect the latest models into the retrieved_models list
    for key, data in latest_models.items():
        retrieved_models.append(data['object_name'])

    # Return the list of all retrieved models (latest versions only)
    if retrieved_models:
        return retrieved_models
    else:
        print("No models found within the future date range and with valid TSO names.")
        return []


if __name__ == "__main__":

    import sys

    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    minio_service = minio.ObjectStorage()
    current_scenario_date = "2024-09-18T10:30:00+02:00"
    existing_time_horizons = ['1D', '2D']
    virtual_time_horizons = ['3D', '4D', '5D', '6D', '7D', '8D', '9D']
    included_models = ['PSE']
    included_models_from_minio = ['LITGRID', 'ELERING', 'AST']

    retrieved_WK_files = get_week_ahead_models(bucket_name='opde-confidential-models', prefix='IGM',
                                               current_scenario_date=current_scenario_date)

    # The 'retrieved_files' list will contain all matching future files with the latest version
    print("\nRetrieved future files with latest versions list:", retrieved_WK_files)

    # The 'retrieved_files' list will contain all matching files
    print(retrieved_WK_files)

    some_results = calculate_scenario_datetime_and_time_horizon(given_datetime=current_scenario_date,
                                                                actual_time_horizons=existing_time_horizons,
                                                                imaginary_time_horizons=virtual_time_horizons)
    existing_folder = './existing'
    updated_folder = './updated'
    print(some_results)
    for result in some_results:
        old_time_horizon = some_results[result].get('existing_time_horizon')
        old_scenario_date = some_results[result].get('existing_time_scenario')
        models = get_latest_models_and_download(time_horizon=old_time_horizon,
                                                scenario_date=old_scenario_date)
        filtered_models = filter_models(models, included_models, filter_on='pmd:TSO')
        minio_service = minio.ObjectStorage()
        minio_models = minio_service.get_latest_models_and_download(time_horizon=old_time_horizon,
                                                                    scenario_datetime=old_scenario_date,
                                                                    model_names=included_models_from_minio,
                                                                    bucket_name='opde-confidential-models',
                                                                    prefix='IGM')
        filtered_models.extend(minio_models)
        updated_models = change_change_scenario_date_time_horizon(existing_models=filtered_models,
                                                                  new_time_horizon=some_results[result].get(
                                                                      'future_time_horizon'),
                                                                  new_scenario_time=some_results[result].get(
                                                                      'future_time_scenario'))

        save_models_to_minio(models_to_save=updated_models, retrieved_wk_files=retrieved_WK_files)
