import logging
import config
import json
from uuid import uuid4
import datetime
from emf.task_generator.time_helper import parse_datetime
from io import BytesIO
from zipfile import ZipFile
from emf.common.config_parser import parse_app_properties
from emf.common.integrations import opdm, minio_api, elastic
from emf.common.integrations.object_storage.models import get_latest_boundary, get_latest_models_and_download
from emf.loadflow_tool import loadflow_settings
from emf.loadflow_tool.helper import opdmprofile_to_bytes, create_opdm_objects
from emf.loadflow_tool.model_merger import merge_functions
from emf.task_generator.task_generator import update_task_status
from emf.common.logging.custom_logger import get_elk_logging_handler
from emf.loadflow_tool.replacement import run_replacement, get_available_tsos
import triplets

logger = logging.getLogger(__name__)
parse_app_properties(caller_globals=globals(), path=config.paths.cgm_worker.merger)

# TODO - move this async solution to some common module
from concurrent.futures import ThreadPoolExecutor
from lxml import etree

executor = ThreadPoolExecutor(max_workers=20)
def async_call(function, callback=None, *args, **kwargs):

    future = executor.submit(function, *args, **kwargs)
    if callback:
        future.add_done_callback(lambda f: callback(f.result()))
    return future
def log_opdm_response(response):
    logger.debug(etree.tostring(response, pretty_print=True).decode())


class HandlerCreateCGM:

    def __init__(self):
        self.opdm_service = opdm.OPDM()
        self.minio_service = minio_api.ObjectStorage()
        self.elk_logging_handler = get_elk_logging_handler()

    def handle(self, task_object: dict, **kwargs):

        start_time = datetime.datetime.utcnow()
        merge_log = {"uploaded_to_opde": 'False',
                     "uploaded_to_minio": 'False',
                     "scaled": 'False',
                     "exclusion_reason": [],
                     "replacement": 'False',
                     "replaced_entity": [],
                     "replacement_reason": []}

        # Parse relevant data from Task
        task = task_object

        if not isinstance(task, dict):
            task = json.loads(task_object)

        # TODO - make it to a wrapper once it is settled/standardized how this info is exchanged
        # Initialize trace
        self.elk_logging_handler.start_trace(task)
        logger.debug(task)

        # Set task to started
        update_task_status(task, "started")

        task_creation_time = task.get('task_creation_time')
        task_properties = task.get('task_properties', {})
        included_models = task_properties.get('included', [])
        excluded_models = task_properties.get('excluded', [])
        task['task_properties']['local_import'] = []

        time_horizon = task_properties["time_horizon"]
        scenario_datetime = task_properties["timestamp_utc"]
        merging_area = task_properties["merge_type"]
        merging_entity = task_properties["merging_entity"]
        mas = task_properties["mas"]
        model_replacement = task_properties["replacement"]
        version = task_properties["version"]

        # Collect valid models from ObjectStorage
        downloaded_models = get_latest_models_and_download(time_horizon, scenario_datetime, valid=False)
        latest_boundary = get_latest_boundary()

        # Filter out models that are not to be used in merge
        filtered_models = merge_functions.filter_models(downloaded_models, included_models, excluded_models, filter_on='pmd:TSO')

        # Check model validity and availability
        valid_models = [model for model in filtered_models if model['valid'] == 'True' or model['valid'] == True]
        invalid_models = [model['pmd:TSO'] for model in filtered_models if model not in valid_models]
        if invalid_models:
            merge_log.get('exclusion_reason').extend([{'tso': tso, 'reason': 'Model is not valid'} for tso in invalid_models])

        if included_models:
            missing_models = [model for model in included_models if model not in [model['pmd:TSO'] for model in valid_models]]
            if missing_models:
                merge_log.get('exclusion_reason').extend([{'tso': tso, 'reason': 'Model missing from OPDM'} for tso in missing_models])
        else:
            if model_replacement == 'True':
                available_tsos = get_available_tsos()
                missing_models = [model for model in available_tsos if model not in [model['pmd:TSO'] for model in valid_models] + excluded_models]
            else:
                missing_models = []

        # Run replacement on missing/invalid models
        if model_replacement == 'True' and missing_models:
            try:
                logger.info(f"Running replacement for missing models")
                replacement_models = run_replacement(missing_models, time_horizon, scenario_datetime)
                if replacement_models:
                    logger.info(f"Replacement model(s) found: {[model['pmd:fileName'] for model in replacement_models]}")
                    merge_log.get('replaced_entity').extend([{'tso': model['pmd:TSO'],
                                                              'replacement_time_horizon': model['pmd:timeHorizon'],
                                                              'replacement_scenario_date': model['pmd:scenarioDate']} for model in replacement_models])
                    valid_models = valid_models + replacement_models
                    merge_log.update({'replacement': 'True'})
                    # TODO put exclusion_reason logging under replacement
            except Exception as error:
                logger.error(f"Failed to run replacement: {error}")

        #Run Process only if you find some models to merge, otherwise return None
        if not valid_models:
            logger.warning("Found no valid models to merge, Returning NONE")
            return None
        else:
            # Load all selected models
            input_models = valid_models + [latest_boundary]
            if len(input_models) < 2:
                logger.warning("Found no Models To Merge, Returning NONE")
                return None
            #merged_model = merge_functions.load_model(input_models)
            assembeled_data = merge_functions.load_opdm_data(input_models)
            assembeled_data = triplets.cgmes_tools.update_FullModel_from_filename(assembeled_data)
            # assembeled_data = merge_functions.configure_paired_boundarypoint_injections(assembeled_data)
            assembeled_data = merge_functions.configure_paired_boundarypoint_injections_by_nodes(assembeled_data)
            escape_upper_xml = assembeled_data[assembeled_data['VALUE'].astype(str).str.contains('.XML')]
            if not escape_upper_xml.empty:
                escape_upper_xml['VALUE'] = escape_upper_xml['VALUE'].str.replace('.XML', '.xml')
                assembeled_data = triplets.rdf_parser.update_triplet_from_triplet(assembeled_data, escape_upper_xml,
                                                                                  update=True,
                                                                                  add=False)

            input_models = create_opdm_objects([merge_functions.export_to_cgmes_zip([assembeled_data])])
            del assembeled_data

            merge_start = datetime.datetime.utcnow()
            merged_model = merge_functions.load_model(input_models)

            # TODO - run other LF if default fails
            solved_model = merge_functions.run_lf(merged_model, loadflow_settings=getattr(loadflow_settings, MERGE_LOAD_FLOW_SETTINGS))#getattr(loadflow_settings, MERGE_LOAD_FLOW_SETTINGS))
            merge_end = datetime.datetime.utcnow()
            logger.info(f"Loadflow status of main island - {solved_model['LOADFLOW_RESULTS'][0]['status_text']}")

            # Update time_horizon in case of generic ID process type
            if time_horizon.upper() == "ID":
                _task_creation_time = parse_datetime(task_creation_time, keep_timezone=False)
                _scenario_datetime = parse_datetime(scenario_datetime, keep_timezone=False)

                time_horizon = f"{int((_scenario_datetime - _task_creation_time).seconds/3600):02d}"
                logger.info(f"Setting ID TimeHorizon to {time_horizon}")

            # TODO - get version dynamically form ELK
            sv_data, ssh_data = merge_functions.create_sv_and_updated_ssh(solved_model, input_models, scenario_datetime, time_horizon, version, merging_area, merging_entity, mas)

            # Fix SV
            sv_data = merge_functions.fix_sv_shunts(sv_data, input_models)
            sv_data = merge_functions.fix_sv_tapsteps(sv_data, ssh_data)
            sv_data = merge_functions.remove_small_islands(sv_data, int(SMALL_ISLAND_SIZE))
            models_as_triplets = merge_functions.load_opdm_data(input_models)
            sv_data = merge_functions.remove_duplicate_sv_voltages(cgm_sv_data=sv_data,
                                                                original_data=models_as_triplets)
            sv_data = merge_functions.check_and_fix_dependencies(cgm_sv_data=sv_data,
                                                                cgm_ssh_data=ssh_data,
                                                                original_data=models_as_triplets)
            sv_data, ssh_data = merge_functions.disconnect_equipment_if_flow_sum_not_zero(cgm_sv_data=sv_data,
                                                                                        cgm_ssh_data=ssh_data,
                                                                                        original_data=models_as_triplets)
            # Package both input models and exported CGM profiles to in memory zip files
            serialized_data = merge_functions.export_to_cgmes_zip([ssh_data, sv_data])


            ### Upload to OPDM ###

            try:
                for item in serialized_data:
                    logger.info(f"Uploading to OPDM -> {item.name}")
                    async_call(function=self.opdm_service.publication_request, callback=log_opdm_response, file_path_or_file_object=item)
                    merge_log.update({'uploaded_to_opde': 'True'})
            except:
                logging.error(f"""Unexpected error on uploading to OPDM:""", exc_info=True)

            ### Upload to Object Storage ###

            # Set CGM name
            cgm_name = f"CGM_{time_horizon}_{version}_{parse_datetime(scenario_datetime):%Y%m%dT%H%MZ}_{merging_area}_{uuid4()}"

            cgm_data = BytesIO()
            with ZipFile(cgm_data, "w") as cgm_zip:

                # Include CGM model files
                for item in serialized_data:
                    cgm_zip.writestr(item.name, item.getvalue())

                # Include original IGM files
                for object in input_models:
                    for instance in object['opde:Component']:
                            file_object = opdmprofile_to_bytes(instance)
                            logging.info(f"Adding file: {file_object.name}")
                            cgm_zip.writestr(file_object.name, file_object.getvalue())

            # Upload to Object Storage
            cgm_object = cgm_data
            cgm_object.name = f"{OUTPUT_MINIO_FOLDER}/{cgm_name}.zip"
            logger.info(f"Uploading CGM to MINO {OUTPUT_MINIO_BUCKET}/{cgm_object.name}")

            try:
                response = self.minio_service.upload_object(cgm_object, bucket_name=OUTPUT_MINIO_BUCKET)
                if response:
                    merge_log.update({'uploaded_to_minio': 'True'})
            except:
                logging.error(f"""Unexpected error on uploading to Object Storage:""", exc_info=True)

            logger.info(f"CGM creation done for {cgm_name}")
            end_time = datetime.datetime.utcnow()
            task_duration = end_time - start_time
            logger.info(f"Task ended at {end_time}, total run time {task_duration}",
                        extra={"task_duration": task_duration.total_seconds(),
                            "task_start_time": start_time.isoformat(),
                            "task_end_time": end_time.isoformat()})

            # Set task to started
            update_task_status(task, "finished")

            logger.debug(task)

            # Stop Trace
            self.elk_logging_handler.stop_trace()

            merge_log.update({'task': task,
                               'small_island_size': SMALL_ISLAND_SIZE,
                               'loadflow_settings': MERGE_LOAD_FLOW_SETTINGS,
                               'merge_duration': f'{(merge_end - merge_start).total_seconds()}',
                               'content_reference': cgm_object.name,
                               'cgm_name': cgm_name})

            # Send merge report to Elastic
            try:
                merge_report = merge_functions.generate_merge_report(solved_model, valid_models, merge_log)
                try:
                    response = elastic.Elastic.send_to_elastic(index=MERGE_REPORT_ELK_INDEX, json_message=merge_report)
                except Exception as error:
                    logger.error(f"Merge report sending to Elastic failed: {error}")
            except Exception as error:
                logger.error(f"Failed to create merge report: {error}")

            return task


if __name__ == "__main__":

    import sys
    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    sample_task = {
        "@context": "https://example.com/task_context.jsonld",
        "@type": "Task",
        "@id": "urn:uuid:ee3c57bf-fa4e-402c-82ac-7352c0d8e118",
        "process_id": "https://example.com/processes/CGM_CREATION",
        "run_id": "https://example.com/runs/IntraDayCGM/1",
        "job_id": "urn:uuid:d9343f48-23cd-4d8a-ae69-1940a0ab1837",
        "task_type": "automatic",
        "task_initiator": "cgm_handler_unit_test",
        "task_priority": "normal",
        "task_creation_time": "2024-05-28T20:39:42.448064",
        "task_update_time": "",
        "task_status": "created",
        "task_status_trace": [
            {
                "status": "created",
                "timestamp": "2024-05-28T20:39:42.448064"
            }
        ],
        "task_dependencies": [],
        "task_tags": [],
        "task_retry_count": 0,
        "task_timeout": "PT1H",
        "task_gate_open": "2024-05-24T21:00:00+00:00",
        "task_gate_close": "2024-05-24T21:15:00+00:00",
        "job_period_start": "2024-05-24T22:00:00+00:00",
        "job_period_end": "2024-05-25T06:00:00+00:00",
        "task_properties": {
            "timestamp_utc": "2024-09-13T13:30:00+00:00",
            "merge_type": "EU",
            "merging_entity": "BALTICRSC",
            "included": [],
            "excluded": ["APG", "FI"],
            "time_horizon": "ID",
            "version": "123",

            "mas": "http://www.baltic-rsc.eu/OperationalPlanning",
            "replacement": "True",
        }
    }

    worker = HandlerCreateCGM()
    finished_task = worker.handle(sample_task)
