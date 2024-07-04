import logging
import os
import re

from datetime import datetime
from io import BytesIO

from emf.task_generator.time_helper import parse_duration

import pytz
import triplets
import config
import datetime
from emf.loadflow_tool.helper import load_opdm_data, metadata_from_filename
from emf.task_generator.time_helper import parse_datetime
from emf.common.config_parser import parse_app_properties
from emf.common.integrations import minio
from emf.common.integrations.object_storage.models import get_latest_models_and_download
from emf.loadflow_tool.model_merger.merge_functions import filter_models, export_to_cgmes_zip

logger = logging.getLogger(__name__)
parse_app_properties(caller_globals=globals(), path=config.paths.cgm_worker.merger)

SEPARATOR_SYMBOL = '/'
WINDOWS_SEPARATOR = '\\'


def get_date_from_time_horizon(time_horizon: str):
    """
    Parses number of dates from time horizon
    :param time_horizon: time horizon as string
    """
    if 'D' in time_horizon:
        possible_days = [int(s) for s in re.findall(r'\d+', time_horizon)]
        if len(possible_days) == 1:
            return possible_days[0]
    raise ValueError(f"Cannot parse day from {time_horizon}")


def calculate_scenario_datetime_and_time_horizon(given_datetime: str | datetime.datetime = None,
                                                 default_time_horizon: str = '1D',
                                                 actual_offset: str = '-P1D',
                                                 imaginary_offset: str = '-P6D',
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
    default_time_horizon = f"{get_date_from_time_horizon(default_time_horizon)}D"
    for time_horizon in actual_time_horizons:
        time_horizon = f"{get_date_from_time_horizon(time_horizon)}D"
        time_delta = parse_duration(f"P{get_date_from_time_horizon(time_horizon)}D")
        future_scenario_date = starting_timestamp + time_delta + actual_offset_duration
        existing_scenario_date = future_scenario_date
        time_horizon_value = {'future_time_scenario': future_scenario_date.isoformat(),
                              'future_time_horizon': time_horizon,
                              'existing_time_scenario': existing_scenario_date.isoformat(),
                              'existing_time_horizon': time_horizon}
        existing_timestamps[existing_scenario_date.isoformat()] = time_horizon
        timestamps[time_horizon] = time_horizon_value
    for time_horizon in imaginary_time_horizons:
        time_delta = parse_duration(f"P{get_date_from_time_horizon(time_horizon)}D")
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
    for model in existing_models:
        # Take one profile at a time. It is easier to substitute updated file to profile afterwards
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
                file_name_meta['pmd:timeHorizon'] = new_time_horizon
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
    # Escape '//'
    folder_path = folder_path.replace(double_separator, SEPARATOR_SYMBOL)
    # Escape '\'
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


def save_models_to_local_storage(models_to_save: list, root_folder: str):
    """
    Saves models to local storage, uses pmd:content-reference as file path
    :param models_to_save: list of opdm object to save
    :param root_folder: location where to save
    """
    for model in models_to_save:
        for component in model.get('opde:Component'):
            data = component.get('opdm:Profile', {}).get('DATA')
            file_name = component.get('opdm:Profile', {}).get('pmd:fileName')
            file_path = component.get('opdm:Profile', {}).get('pmd:content-reference')
            file_root_folder = root_folder
            if file_path:
                file_folder = str(os.path.dirname(file_path))
                file_root_folder = root_folder.removesuffix('/') + '/' + file_folder.removeprefix('/')
            check_and_create_the_folder_path(file_root_folder)
            file_name = file_root_folder.removesuffix('/') + '/' + file_name.removeprefix('/')
            with open(file_name, 'wb') as save_file_name:
                save_file_name.write(data)


def save_models_to_minio(models_to_save: list,
                         minio_bucket: str = OUTPUT_MINIO_BUCKET,
                         subfolder_name: str = 'EMF_test_merge_find_better_place/testing_week_ahead',
                         save_minio_service: minio.ObjectStorage = minio.ObjectStorage()):
    """
    Saves models to minio, uses pmd:content-reference as file path
    :param models_to_save: list of models to save
    :param minio_bucket: name of minio bucket
    :param subfolder_name: prefix for the files if needed
    :param save_minio_service: minio instance if given, created otherwise
    """
    for model in models_to_save:
        for component in model.get('opde:Component'):
            data = component.get('opdm:Profile', {}).get('DATA')
            if isinstance(data, bytes):
                data = BytesIO(data)
            file_name = component.get('opdm:Profile', {}).get('pmd:fileName')
            file_path = component.get('opdm:Profile', {}).get('pmd:content-reference')
            file_loc = file_path or file_name
            if subfolder_name:
                file_loc = subfolder_name.removesuffix('/') + '/' + file_loc.removeprefix('/')
            logger.info(f"Uploading RMM to MINO {minio_bucket}/{file_loc}")
            data.name = file_loc
            try:
                save_minio_service.upload_object(data, bucket_name=minio_bucket)
            except:
                logging.error(f"""Unexpected error on uploading to Object Storage:""", exc_info=True)


if __name__ == "__main__":

    import sys
    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    current_scenario_date = "2024-07-04T10:30:00+02:00"
    existing_time_horizons = ['1D', '2D']
    virtual_time_horizons = ['3D', '4D', '5D', '6D', '7D', '8D', '9D']
    included_models = ['ELERING', 'AST', 'PSE']
    included_models_from_minio = ['LITGIRD']
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
        # Filter out models that are not to be used in merge
        filtered_models = filter_models(models, included_models, filter_on='pmd:TSO')
        minio_service = minio.ObjectStorage()
        # Get additional models directly from Minio
        minio_models = minio_service.get_latest_models_and_download(time_horizon=old_time_horizon,
                                                                    scenario_datetime=old_scenario_date,
                                                                    model_names=included_models_from_minio,
                                                                    bucket_name=INPUT_MINIO_BUCKET,
                                                                    prefix=INPUT_MINIO_FOLDER)
        filtered_models.extend(minio_models)
        # save_models_to_local_storage(models_to_save=filtered_models, root_folder=existing_folder)
        updated_models = change_change_scenario_date_time_horizon(existing_models=filtered_models,
                                                                  new_time_horizon=some_results[result].get(
                                                                      'future_time_horizon'),
                                                                  new_scenario_time=some_results[result].get(
                                                                      'future_time_scenario'))
        # save_models_to_local_storage(models_to_save=updated_models, root_folder=updated_folder)
        save_models_to_minio(models_to_save=updated_models)
