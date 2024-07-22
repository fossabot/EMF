import logging
from io import BytesIO
from uuid import uuid4
from zipfile import ZipFile

from aniso8601 import parse_datetime

import config
from emf.common.config_parser import parse_app_properties
from emf.common.integrations import minio
from emf.common.integrations.object_storage import file_system, models
from emf.common.integrations.object_storage.file_system import save_merged_model_to_local_storage
from emf.common.logging.pypowsybl_logger import PyPowsyblLogGatheringHandler, PyPowsyblLogReportingPolicy
from emf.loadflow_tool import loadflow_settings
from emf.loadflow_tool.helper import load_model
# from emf.loadflow_tool.model_merger.handlers.rmm_handler import set_brell_lines_to_zero_in_models
from emf.loadflow_tool.model_merger.merge_functions import run_lf, create_sv_and_updated_ssh, fix_sv_shunts, \
    fix_sv_tapsteps, export_to_cgmes_zip, filter_models
from emf.loadflow_tool.model_validator.validator import validate_model

logger = logging.getLogger(__name__)
parse_app_properties(caller_globals=globals(), path=config.paths.cgm_worker.merger)


def merge_models(list_of_models: list,
                 latest_boundary: dict,
                 time_horizon: str,
                 scenario_datetime: str,
                 merging_area: str,
                 merging_entity: str,
                 mas: str,
                 version: str = '001'):
    # Load all selected models
    input_models = list_of_models + [latest_boundary]
    # SET BRELL LINE VALUES
    # input_models = set_brell_lines_to_zero_in_models(input_models)
    # END OF MODIFICATION
    # FIX DANGLING REFERENCES ISSUE
    parameters = {"iidm.import.cgmes.import-node-breaker-as-bus-breaker": 'true'}
    merged_model = load_model(input_models, parameters=parameters)
    # END OF FIX
    # merged_model = load_model(input_models)
    # TODO - run other LF if default fails
    solved_model = run_lf(merged_model, loadflow_settings=loadflow_settings.CGM_DEFAULT)

    # TODO - get version dynamically form ELK
    sv_data, ssh_data = create_sv_and_updated_ssh(solved_model, input_models, scenario_datetime, time_horizon, version,
                                                  merging_area, merging_entity, mas)

    # Fix SV
    sv_data = fix_sv_shunts(sv_data, input_models)
    sv_data = fix_sv_tapsteps(sv_data, ssh_data)

    # Package both input models and exported CGM profiles to in memory zip files
    serialized_data = export_to_cgmes_zip([ssh_data, sv_data])

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
    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout),
                  # Set up the log gatherer:
                  # topic name: currently used as a start of a file name
                  # send_it_to_elastic: send the triggered log entry to elastic
                  # upload_to_minio: upload log file to minio (parameters are defined in custom_logger.properties)
                  # report_on_command: trigger reporting explicitly
                  # logging policy: choose according to the need. Currently:
                  #   ALL_ENTRIES: gathers all log entries no matter of what
                  #   ENTRIES_IF_LEVEL_REACHED: outputs all the log when at least one entry was over level
                  #   ENTRIES_ON_LEVEL: gathers all entries that were at least on the level specified
                  # print_to_console: propagate log to parent
                  # reporting_level: level that triggers policy
                  PyPowsyblLogGatheringHandler(topic_name='cgm_merge',
                                               send_to_elastic=False,
                                               upload_to_minio=False,
                                               logging_policy=PyPowsyblLogReportingPolicy.ALL_ENTRIES,
                                               print_to_console=False,
                                               report_level=logging.ERROR)]
    )

    # folder_to_study = 'OPDM_IOP_JULY'

    use_local = False
    validation_needed = False
    # path and folder for importing from local storage
    folder_to_study = 'cgmiopjulyv2'
    examples_path = '../../../loadflow_tool/example_models/'
    # path where to save merged model
    local_storage = './cgm_merge'
    # Model data
    test_time_horizon = '1D'
    # included_models = []
    included_models = ['REE', 'REN']
    excluded_models = []
    # local_import_models = ['LITGRID']
    local_import_models = []
    test_scenario_datetime_values = ['2024-07-12T15:30:00+00:00']
    test_version = '002'
    test_merging_entity = 'BALTICRSC'
    test_merging_area = 'EU'
    test_mas = 'http://www.baltic-rsc.eu/OperationalPlanning/CGM'
    # test_merging_area = 'BA'
    # test_mas = 'http://www.baltic-rsc.eu/OperationalPlanning/RMM'
    for test_scenario_datetime in test_scenario_datetime_values:
        logger.info(f"Executing {test_scenario_datetime}")
        if use_local:
            loaded_boundary = file_system.get_latest_boundary(path_to_directory=folder_to_study,
                                                              local_folder_for_examples=examples_path)
            valid_models = file_system.get_latest_models_and_download(path_to_directory=folder_to_study,
                                                                      local_folder_for_examples=examples_path)
        else:
            loaded_boundary = models.get_latest_boundary()
            valid_models = models.get_latest_models_and_download(test_time_horizon, test_scenario_datetime, valid=True)
        # Filter out models that are not to be used in merge
        filtered_models = filter_models(valid_models, included_models, excluded_models, filter_on='pmd:TSO')

        # Get additional models directly from Minio
        if local_import_models and not use_local:
            minio_service = minio.ObjectStorage()
            additional_models = minio_service.get_latest_models_and_download(time_horizon=test_time_horizon,
                                                                             scenario_datetime=test_scenario_datetime,
                                                                             model_names=local_import_models,
                                                                             bucket_name=INPUT_MINIO_BUCKET,
                                                                             prefix=INPUT_MINIO_FOLDER)
            test_input_models = filtered_models + additional_models
        else:
            test_input_models = filtered_models

        # SET BRELL LINE VALUES
        # test_input_models = set_brell_lines_to_zero_in_models(input_models)
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

        test_model = merge_models(list_of_models=available_models,
                                  latest_boundary=loaded_boundary,
                                  time_horizon=test_time_horizon,
                                  scenario_datetime=test_scenario_datetime,
                                  merging_area=test_merging_area,
                                  merging_entity=test_merging_entity,
                                  mas=test_mas,
                                  version=test_version)
        save_merged_model_to_local_storage(cgm_files=test_model, local_storage_location=local_storage)
        # check_and_create_the_folder_path(local_storage)
        # full_name = local_storage.removesuffix('/') + '/' + test_model.name.removeprefix('/')
        # with open(full_name, 'wb') as write_file:
        #     write_file.write(test_model.getbuffer())
        # test_model.name = 'EMF_test_merge_find_better_place/CGM/' + test_model.name
        # save_minio_service = minio.ObjectStorage()
        # try:
        #     save_minio_service.upload_object(test_model, bucket_name=OUTPUT_MINIO_BUCKET)
        # except:
        #     logging.error(f"""Unexpected error on uploading to Object Storage:""", exc_info=True)

