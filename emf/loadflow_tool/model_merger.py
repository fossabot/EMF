import copy
from datetime import timedelta
from enum import Enum
from io import BytesIO
from zipfile import ZipFile

import pypowsybl
import pytz
import zeep.exceptions

import config
from emf.common.config_parser import parse_app_properties
from emf.common.integrations import minio, opdm, elastic
from emf.common.integrations.elastic import Elastic
from emf.common.integrations.object_storage import models
from emf.loadflow_tool.helper import (load_model, load_opdm_data, filename_from_metadata,
                                      get_metadata_from_filename, attr_to_dict, export_model_escaped)
from emf.common.integrations.object_storage.file_system import get_one_set_of_igms_from_local_storage
from emf.common.integrations.object_storage.file_system_general import OPDE_COMPONENT_KEYWORD, \
    OPDM_PROFILE_KEYWORD, DATA_KEYWORD, PMD_FILENAME_KEYWORD, PMD_CGMES_PROFILE_KEYWORD, \
    PMD_MODEL_PART_REFERENCE_KEYWORD, PMD_MERGING_ENTITY_KEYWORD, \
    PMD_MERGING_AREA_KEYWORD, PMD_SCENARIO_DATE_KEYWORD, OPDE_OBJECT_TYPE_KEYWORD, PMD_TSO_KEYWORD, \
    PMD_VERSION_NUMBER_KEYWORD, PMD_TIME_HORIZON_KEYWORD, PMD_CREATION_DATE_KEYWORD, PMD_MODEL_ID_KEYWORD, \
    PMD_MODELING_AUTHORITY_SET_KEYWORD, BOUNDARY_OBJECT_TYPE, IGM_OBJECT_TYPE, CGM_OBJECT_TYPE, \
    MODEL_MESSAGE_TYPE_KEYWORD, MODEL_MODELING_ENTITY_KEYWORD, MODEL_MERGING_ENTITY_KEYWORD, MODEL_DOMAIN_KEYWORD, \
    MODEL_SCENARIO_TIME_KEYWORD, MODEL_PROCESS_TYPE_KEYWORD, MODEL_VERSION_KEYWORD, get_meta_from_filename, \
    VALIDATION_STATUS_KEYWORD, VALID_KEYWORD, NETWORK_KEYWORD, NETWORK_META_KEYWORD, SEPARATOR_SYMBOL, \
    check_the_folder_path
import logging
import json
from emf.loadflow_tool import loadflow_settings
import sys
from emf.common.integrations.opdm import OPDM
from aniso8601 import parse_datetime
import os
import triplets
import pandas
import datetime
from uuid import uuid4

from emf.loadflow_tool.validator import validate_model

# Update SSH

logger = logging.getLogger(__name__)

parse_app_properties(caller_globals=globals(), path=config.paths.cgm_worker.model_merge)

logging.basicConfig(
    format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)

PY_EMF_OS_MINIO_LOCAL_IGM_BUCKET = EMF_OS_MINIO_LOCAL_IGM_BUCKET
PY_EMF_OS_MINIO_LOCAL_IGM_FOLDER = EMF_OS_MINIO_LOCAL_IGM_FOLDER
PY_EMF_OS_MINIO_OPDE_MODELS_BUCKET = EMF_OS_MINIO_OPDE_MODELS_BUCKET
PY_EMF_OS_MINIO_OPDE_MODELS_FOLDER = EMF_OS_MINIO_OPDE_MODELS_FOLDER
PY_MERGING_ENTITY = MERGING_ENTITY

UPDATE_MAP = [
    {
        "from_class": "SvPowerFlow",
        "from_ID": "Terminal.ConductingEquipment",
        "from_attribute": "SvPowerFlow.p",
        "to_attribute": "EnergyConsumer.p",
    },
    {
        "from_class": "SvPowerFlow",
        "from_ID": "Terminal.ConductingEquipment",
        "from_attribute": "SvPowerFlow.q",
        "to_attribute": "EnergyConsumer.q",
    },
    {
        "from_class": "SvPowerFlow",
        "from_ID": "Terminal.ConductingEquipment",
        "from_attribute": "SvPowerFlow.p",
        "to_attribute": "RotatingMachine.p",
    },
    {
        "from_class": "SvPowerFlow",
        "from_ID": "Terminal.ConductingEquipment",
        "from_attribute": "SvPowerFlow.q",
        "to_attribute": "RotatingMachine.q",
    },
    {
        "from_class": "SvTapStep",
        "from_ID": "SvTapStep.TapChanger",
        "from_attribute": "SvTapStep.position",
        "to_attribute": "TapChanger.step",
    },
    {
        "from_class": "SvShuntCompensatorSections",
        "from_ID": "SvShuntCompensatorSections.ShuntCompensator",
        "from_attribute": "SvShuntCompensatorSections.sections",
        "to_attribute": "ShuntCompensator.sections",
    }
]

FILENAME_MASK = ("{scenarioTime:%Y%m%dT%H%MZ}_{processType}_"
                 "{mergingEntity}-{domain}-{forEntity}_{messageType}_{version:03d}")

NAMESPACE_MAP = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "cim": "http://iec.ch/TC57/2013/CIM-schema-cim16#",
    "md": "http://iec.ch/TC57/61970-552/ModelDescription/1#",
    "entsoe": "http://entsoe.eu/CIM/SchemaExtension/3/1#",
    # "cgmbp": "http://entsoe.eu/CIM/Extensions/CGM-BP/2020#"
}
RDF_MAP_JSON = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'entsoe_v2.4.15_2014-08-07.json')
PATTERN_WITHOUT_TIMEZONE = '%Y-%m-%dT%H:%M:%S'

INTRA_DAY_TIME_HORIZON = 'ID'

# Variables used for local testing
TIME_HORIZON = '1D'
SCENARIO_DATE = '2024-03-14T09:30'
DEFAULT_AREA = 'EU'
VERSION = "104"
PUBLISH_TO_OPDM = False
USE_LOCAL_FILES = True
LOCAL_FOLDER = 'TC3_T1_Conform'

PROCESS_ID_KEYWORD = "process_id"
RUN_ID_KEYWORD = 'run_id'
JOB_ID_KEYWORD = 'job_id'

FULL_PATH_KEYWORD = 'full_path'

LOCAL_STORAGE_LOCATION = './merged_examples/'
IGM_TYPE = 'IGM'
CGM_TYPE = 'CGM'
BOUNDARY_TYPE = 'BOUNDARY'
OPDM_FORMAT = 'OPDM_FORMAT'
IGM_FILES = ['EQ', 'TP']
CGM_FILES = ['SSH', 'SV']
BOUNDARY_FILES = ['EQ_BD', 'TP_BD']

MODELS_KEYWORD = 'Models'
TASK_KEYWORD = 'Task'

TASK_PROPERTIES_KEYWORD = 'task_properties'
TIMESTAMP_KEYWORD = 'timestamp_utc'
MERGE_TYPE_KEYWORD = 'merge_type'
TIME_HORIZON_KEYWORD = 'time_horizon'
AREA_KEYWORD = 'area'
INCLUDED_TSO_KEYWORD = 'included'
EXCLUDED_TSO_KEYWORD = 'excluded'
IMPORT_TSO_LOCALLY_KEYWORD = 'local_import'
VERSION_NUMBER_KEYWORD = 'version_no'
MERGING_ENTITY_KEYWORD = 'merging_entity'


class DownloadModels(Enum):
    """
    For determining from where to download files
    """
    OPDM = 1
    MINIO = 2
    OPDM_AND_MINIO = 3


class CgmModelType(Enum):
    BOUNDARY = 1
    IGM = 2


class CgmExportType(Enum):
    """
    Define the set of files for export
    Note that Pypowsybl requires a full set ('EQ', 'TP', 'SSH', 'SV') and boundary ('EQBD' mad 'TPBD') files. At the
    same time there should not be present two same process files per tso (meaning that in merged model SSH files from
    tso's must be overwritten by merged SSH files)
    So, to have a custom format:
    a) define which files from igms are needed (EQ, TP, SSH, SV)
    b) define which files from the cgm are needed (SSH, SV)
    c) define which files from boundary are needed (EQ_BD, TP_BD)
    d) choose either dictionary representation (OPDM_FORMAT= True, is used to import to pypowsybl), or not (get files
    as BytesIO objects)
    """
    # Get everything
    FULL = {IGM_OBJECT_TYPE: IGM_FILES,
            CGM_OBJECT_TYPE: CGM_FILES,
            BOUNDARY_OBJECT_TYPE: BOUNDARY_FILES,
            OPDM_FORMAT: True}
    FULL_FILES_ONLY = {IGM_OBJECT_TYPE: IGM_FILES,
                       CGM_OBJECT_TYPE: CGM_FILES,
                       BOUNDARY_OBJECT_TYPE: BOUNDARY_FILES,
                       OPDM_FORMAT: False}
    ALL = {}
    ALL_FILES_ONLY = {OPDM_FORMAT: False}
    # Get cgm as is
    BARE = {}

    @classmethod
    def _missing_(cls, value: object):
        return cls.BARE


class IntraDayPastScenarioDateException(Exception):
    pass


class PyPowsyblError(Exception):
    pass


class NotEnoughInputDataError(Exception):
    pass


class NoContentFromElasticException(Exception):
    pass


def load_rdf_map(file_name: str = RDF_MAP_JSON):
    """
    Loads rdf map file
    :param file_name: from where to load
    :return: rdf map
    """
    with open(file_name, 'r') as file_object:
        rdf_map = json.load(file_object)
    return rdf_map


def check_dataframe(first_input=None, second_input=None):
    """
    Escapes first input if not given
    :param first_input: first element to be checked
    :param second_input: second element to be checked
    :return: first_input ro second_input (for dataframes)
    """
    if first_input is not None and isinstance(first_input, pandas.DataFrame):
        return first_input
    return second_input


def get_latest_boundary_data(opdm_client: OPDM = None):
    """
    Tries to get the boundary data from OPDM, if not successful, fallback to Minio and take the latest
    Alternative would be to check depends on
    :param opdm_client: OPDM client
    :return boundary data
    """
    try:
        opdm_client = opdm_client or OPDM()
        boundary_data = opdm_client.get_latest_boundary()
    except zeep.exceptions.Fault as fault:
        logger.error(f"OPDM connection error when getting boundaries: {fault}")
        boundary_data = models.get_latest_boundary()
    except Exception as ex:
        logger.error(f"Undefined exception when getting boundary data: {ex}")
        boundary_data = models.get_latest_boundary()
    return boundary_data


def validate_models(igm_models: list = None, boundary_data: dict = None):
    """
    Validates the raw output from the opdm
    :param igm_models: list of igm models
    :param boundary_data: dictionary containing the boundary data
    :return list of validated models
    """
    valid_models = []
    invalid_models = []
    # Validate models
    if not igm_models or not boundary_data:
        logger.error(f"Missing input data")
        return valid_models
    for igm_model in igm_models:

        try:
            validation_response = validate_model([igm_model, boundary_data])
            igm_model[VALIDATION_STATUS_KEYWORD] = validation_response
            if validation_response[VALID_KEYWORD]:
                valid_models.append(igm_model)
            else:
                invalid_models.append(igm_model)
        except Exception as ex:
            invalid_models.append(igm_model)
            tso_name = igm_model.get(PMD_TSO_KEYWORD, "")
            logger.error(f"Unable to validate {tso_name}: {ex}")
    return valid_models


def handle_opdm_models(opdm_models=None,
                       boundary_data=None,
                       validate: bool = True):
    """
    Decides what to do with raw models received from opdm: either run them through model retriever pipeline, validate
    them or do nothing
    :param opdm_models: list of igm models got from opdm
    :param boundary_data: dictionary of (latest) boundary data
    :param validate: true-> validate igm models, take only those which are valid
    :return updated (valid) opdm_models
    """
    if opdm_models and boundary_data:
        if validate:
            opdm_models = validate_models(igm_models=opdm_models, boundary_data=boundary_data)
    return opdm_models


def get_models(time_horizon: str = TIME_HORIZON,
               scenario_date: str = SCENARIO_DATE,
               included_tsos: list | str = None,
               excluded_tsos: list | str = None,
               locally_imported_tsos: list | str = None,
               download_policy: DownloadModels = DownloadModels.OPDM_AND_MINIO,
               opdm_client: OPDM = None):
    """
    Gets models from opdm and/or minio
    NB! Priority is given to Minio!
    NB! Extend this according to the need and means available
    Workflow:
    1) Get models from local storage (only present in minio, no trace in elastic)
    2) Get models from opdm if selected
    3) Get models from minio if selected or opdm failed
    4) If requested from both, take data from minio and extend it from opdm
    5) Merge 4, 3, 2 with 1, give priority to models from local storage
    6) By default get boundary from opdm
    7) Fallback: get boundary from minio
    :param time_horizon: time horizon of the igms
    :param scenario_date: the date of the scenario for which the igm was created
    :param included_tsos: list or string of tso names, that should be included
    :param excluded_tsos: list or string of tso names, that should be excluded
    :param locally_imported_tsos: list or string of tso names, that should be loaded locally
    :param download_policy: from where to download models
    :param opdm_client: client for the opdm
    """
    igm_models = []
    opdm_models = None
    minio_models = None
    # 1 Get boundary data
    boundary_data = get_latest_boundary_data(opdm_client=opdm_client)
    # 2 if there are some tsos specified to be locally imported, find them first (priority)
    if locally_imported_tsos:
        igm_models.extend(get_igm_models_from_minio_by_metadata(tsos=locally_imported_tsos,
                                                                scenario_date=scenario_date,
                                                                time_horizon=time_horizon))
    # 3 if opdm is selected, try to download from there
    if download_policy == DownloadModels.OPDM or download_policy == DownloadModels.OPDM_AND_MINIO:
        opdm_models = get_models_from_opdm(time_horizon=time_horizon,
                                           scenario_date=scenario_date,
                                           included_tsos=included_tsos,
                                           excluded_tsos=excluded_tsos,
                                           opdm_client=opdm_client)
    # 4 if minio is selected or opdm failed, download data from there
    if download_policy == DownloadModels.MINIO or download_policy == DownloadModels.OPDM_AND_MINIO or not opdm_models:
        minio_models = models.get_latest_models_and_download(time_horizon=time_horizon,
                                                             scenario_date=scenario_date)
        minio_models = filter_models_by_tsos(igm_models=minio_models,
                                             included_tsos=included_tsos,
                                             excluded_tsos=excluded_tsos)
    # 5 Merge minio and opdm, giving minio priority and validating opdm results
    if download_policy == DownloadModels.OPDM_AND_MINIO:
        found_models = minio_models or []
        if minio_models:
            minio_tsos = [model.get(PMD_TSO_KEYWORD) for model in minio_models]
            if opdm_models and minio_tsos:
                opdm_models = [model for model in opdm_models if model.get(PMD_TSO_KEYWORD) not in minio_tsos]
        opdm_models = handle_opdm_models(opdm_models=opdm_models,
                                         boundary_data=boundary_data)
        if opdm_models:
            found_models.extend(opdm_models)
    else:
        if opdm_models:
            opdm_models = handle_opdm_models(opdm_models=opdm_models,
                                             boundary_data=boundary_data)
        found_models = minio_models or opdm_models
    # 6 Merge local models with minio-opdm combination giving priority to first
    tsos_local = [model.get(PMD_TSO_KEYWORD) for model in igm_models]
    igm_models.extend([model for model in found_models if not model.get(PMD_TSO_KEYWORD) in tsos_local])
    return igm_models, boundary_data


def filter_models_by_tsos(igm_models: list, included_tsos: list | str = None, excluded_tsos: list | str = None):
    """
    Filters the list of models to include or to exclude specific tsos if they are given
    :param igm_models: list of igm models
    :param included_tsos: list or string of tso names, if given, only matching models are returned
    :param excluded_tsos: list or string of tso names, if given, matching models will be discarded
    :return updated list of igms
    """
    if included_tsos:
        included_tsos = [included_tsos] if isinstance(included_tsos, str) else included_tsos
        igm_models = [model for model in igm_models if model.get(PMD_TSO_KEYWORD) in included_tsos]
    if excluded_tsos:
        excluded_tsos = [excluded_tsos] if isinstance(excluded_tsos, str) else excluded_tsos
        igm_models = [model for model in igm_models if not model.get(PMD_TSO_KEYWORD) in excluded_tsos]
    return igm_models


def get_models_from_opdm(time_horizon: str,
                         scenario_date: str,
                         included_tsos: list | str = None,
                         excluded_tsos: list | str = None,
                         opdm_client: OPDM = None):
    """
    Gets models from opdm
    :param time_horizon: time horizon of the igms
    :param scenario_date: the date of the scenario for which the igm was created
    :param included_tsos: list or string of tso names, if given, only matching models are returned
    :param excluded_tsos: list or string of tso names, if given, matching models will be discarded
    :param opdm_client: client for the opdm
    :return list of models if found, None otherwise
    """
    available_models = None
    try:
        opdm_client = opdm_client or OPDM()
        scenario_date_iso = datetime.datetime.fromisoformat(scenario_date)
        converted_scenario_date = scenario_date_iso.strftime(PATTERN_WITHOUT_TIMEZONE)
        received_models = opdm_client.get_latest_models_and_download(time_horizon=time_horizon,
                                                                     scenario_date=converted_scenario_date)
        available_models = filter_models_by_tsos(igm_models=received_models,
                                                 included_tsos=included_tsos,
                                                 excluded_tsos=excluded_tsos)
    except zeep.exceptions.Fault as fault:
        logger.error(f"Could not connect to OPDM: {fault}")
    except Exception as ex:
        logger.error(f"Unknown exception when getting data from opdm: {ex}")
    finally:
        return available_models


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
                file_metadata[FULL_PATH_KEYWORD] = file_name.object_name
                file_name_list.append(file_metadata)
        except ValueError:
            continue
        except Exception as ex:
            logger.warning(f"Exception when parsing the filename: {ex}")
            continue
    exploded_results = pandas.DataFrame(file_name_list)
    return exploded_results


def get_igm_models_from_minio_by_metadata(tsos: list | str,
                                          time_horizon: str,
                                          scenario_date: datetime.datetime | str,
                                          minio_client: minio.ObjectStorage = None,
                                          minio_bucket: str = PY_EMF_OS_MINIO_LOCAL_IGM_BUCKET,
                                          minio_folder: str = PY_EMF_OS_MINIO_LOCAL_IGM_FOLDER,
                                          minio_metadata_field: str = EMF_OS_MINIO_LOCAL_IGM_METADATA_FIELD):
    """
    Gets models from minio by filtering the files in given bucket by minio_metadata_field.
    Code borrowed from m-karo
    Note that this is one implementation. Actual possibilities may vary
    :param minio_client: instance of Minio
    :param minio_bucket: bucket where to search data
    :param minio_folder: prefix in minio storage
    :param minio_metadata_field: parameter by which to search
    :param tsos: list of tsos
    :param time_horizon: time horizon
    :param scenario_date: scenario date
    :return list of models
    """
    if isinstance(tsos, str):
        tsos = [tsos]
    minio_client = minio_client or minio.ObjectStorage()
    if isinstance(scenario_date, str):
        scenario_date = parse_datetime(scenario_date)
    if scenario_date.tzinfo:
        scenario_date = scenario_date.astimezone(pytz.utc)
    else:
        logger.warning(f"Time zone is not defined for scenario_time variable, localizing as UTC time zone")
        scenario_date = scenario_date.tz_localize(pytz.utc)

    # Define model search pattern and query/download
    if time_horizon == INTRA_DAY_TIME_HORIZON:
        # takes any integer between 0-32 which can be in network model name
        model_name_pattern = f"{scenario_date:%Y%m%dT%H%M}Z-({'0[0-9]|1[0-9]|2[0-9]|3[0-6]'})-({'|'.join(tsos)})"
    else:
        model_name_pattern = f"{scenario_date:%Y%m%dT%H%M}Z-{time_horizon}-({'|'.join(tsos)})"
    external_model_metadata = {minio_metadata_field: model_name_pattern}
    external_models = minio_client.query_objects(bucket_name=minio_bucket,
                                                 prefix=minio_folder,
                                                 metadata=external_model_metadata,
                                                 use_regex=True)

    igm_models = []
    if external_models:
        logger.info(f"Number of external models received: {len(external_models)}")
        for model in external_models:
            logger.info(f"Retrieving file from Minio storage: {model.object_name}")
            model_datum = BytesIO(minio_client.download_object(bucket_name=minio_bucket, object_name=model.object_name))
            model_datum.name = model.object_name
            # Convert received model to OPDM format, may need some fields
            model_datum.seek(0)
            model_datum = ZipFile(model_datum)
            model_data = [BytesIO(model_datum.read(file_name)) for file_name in model_datum.namelist()]
            model_data = get_one_set_of_igms_from_local_storage(file_data=model_data)
            igm_models.append(model_data)
    else:
        logger.warning(f"No external models returned from Minio with metadata: {external_model_metadata}")
    return igm_models


def get_version_number_from_elastic(index_name: str = ELASTIC_LOGS_INDEX,
                                    start_looking: datetime.datetime | str = datetime.datetime.today(),
                                    scenario_date: str = None,
                                    time_horizon: str = None,
                                    modeling_entity: str = None):
    """
    Checks and gets the version number from elastic
    Note that it works only if logger.info(f"Publishing {instance_file.name} to OPDM")
    is used when publishing files to OPDM
    :param index_name: index from where to search
    :param start_looking: datetime instance from where to look, if not set then takes current day
    :param scenario_date: filter the file names by scenario date
    :param time_horizon: filter file names by time horizon
    :param modeling_entity: filter file names by modeling entity
    :return version number as a string
    """
    must_elements = []
    query_part = {"query_string": {"default_field": "message", "query": "*Publishing* AND *to OPDM"}}
    must_elements.append(query_part)
    new_version_number = 1
    if start_looking:
        if isinstance(start_looking, datetime.datetime):
            start_looking = start_looking.strftime("%Y-%m-%dT%H:%M:%S")
        range_part = {"range": {"log_timestamp": {"gte": start_looking}}}
        must_elements.append(range_part)
    previous_cgm_query = {"bool": {"must": must_elements}}
    try:
        elastic_client = Elastic()
        results = elastic_client.get_data(index=index_name,
                                          query=previous_cgm_query,
                                          fields=['message'])
        if results.empty:
            raise NoContentFromElasticException
        # Get the filenames and explode them
        exploded_results = (results["message"].
                            str.removesuffix(' to OPDM').
                            str.removeprefix('Publishing ').
                            map(get_metadata_from_filename).
                            apply(pandas.Series))
        # Filter the results if needed
        new_version_number = get_largest_version_from_filename_dataframe(exploded_data=exploded_results,
                                                                         scenario_date=scenario_date,
                                                                         time_horizon=time_horizon,
                                                                         modeling_entity=modeling_entity)
    except (NoContentFromElasticException, KeyError, ValueError):
        logger.info(f"No previous entries found, starting with version number {new_version_number:03}")
    except Exception as ex:
        logger.warning(f"Got elastic error: {ex}, starting with version number {new_version_number:03}")
    finally:
        return f"{new_version_number:03}"


def get_version_number_from_minio(minio_bucket: str = PY_EMF_OS_MINIO_OPDE_MODELS_BUCKET,
                                  sub_folder: str = PY_EMF_OS_MINIO_OPDE_MODELS_FOLDER,
                                  minio_client: minio.ObjectStorage = None,
                                  scenario_date: str | datetime.datetime = None,
                                  modeling_entity: str = PY_MERGING_ENTITY,
                                  time_horizon: str = None):
    """
    Gets file list from minio, explodes it and retrieves the biggest matched version number
    :param minio_client: if given
    :param minio_bucket: the name of the bucket
    :param sub_folder: prefix
    :param scenario_date: date of the merge
    :param modeling_entity: name of the merging entity
    :param time_horizon: the time horizon
    """
    new_version_number = 1
    try:
        exploded_results = get_filename_dataframe_from_minio(minio_bucket=minio_bucket,
                                                             minio_client=minio_client,
                                                             sub_folder=sub_folder)
        new_version_number = get_largest_version_from_filename_dataframe(exploded_data=exploded_results,
                                                                         scenario_date=scenario_date,
                                                                         time_horizon=time_horizon,
                                                                         modeling_entity=modeling_entity)
    except (ValueError, KeyError):
        logger.info(f"No previous entries found, starting with version number {new_version_number:03}")
    except Exception as ex:
        logger.warning(f"Got minio error: {ex}, starting with version number {new_version_number:03}")
    return f"{new_version_number:03}"


def get_largest_version_from_filename_dataframe(exploded_data: pandas.DataFrame,
                                                scenario_date: str = None,
                                                time_horizon: str = None,
                                                modeling_entity: str = None):
    """
    Searches largest version number from a dict. Optionally the dict can be filtered beforehand
    :param exploded_data: the dictionary containing exploded filenames (used get_metadata_from_filename)
    :param scenario_date: optionally filter filenames by scenario date
    :param time_horizon: optionally filter filenames by time horizon
    :param modeling_entity: optionally filter filenames by checking if modelling entity is in the field
    :return: largest found file number or 1 if key error or not found
    """
    try:
        if modeling_entity is not None:
            exploded_data = exploded_data[exploded_data[MODEL_MODELING_ENTITY_KEYWORD].str.contains(modeling_entity)]
        if scenario_date is not None:
            scenario_date = f"{parse_datetime(scenario_date):%Y%m%dT%H%MZ}"
            exploded_data = exploded_data[exploded_data[MODEL_SCENARIO_TIME_KEYWORD].str.contains(scenario_date)]
        if time_horizon is not None:
            exploded_data = exploded_data[exploded_data[MODEL_PROCESS_TYPE_KEYWORD].str.contains(time_horizon)]
        # Get the largest version number and increment it by 1
        new_version_number = max(pandas.to_numeric(exploded_data[MODEL_VERSION_KEYWORD])) + 1
        logger.info(f"Continuing with version number {new_version_number:03}")
    except KeyError as key_error:
        logger.info(f"{key_error}")
        new_version_number = 1
    return new_version_number


def get_version_number(scenario_date: str,
                       time_horizon: str,
                       modeling_entity: str,
                       start_looking: str | datetime.date = None,
                       use_elastic: bool = True,
                       use_minio: bool = True,
                       default_version_value='104'):
    """
    Gets a version number from elastic and or minio.
    :param scenario_date: the date by which to look the version number
    :param time_horizon: the time horizon
    :param modeling_entity: the author of the previous versions
    :param start_looking: can be used to cut down the elastic logs
    :param use_elastic: search version number from elastic
    :param use_minio: search version number from minio
    :param default_version_value: return value if not found
    :return largest version number from minio, elastic or default one
    """
    version_number = default_version_value
    version_number_minio = None
    version_number_elastic = None
    if use_minio:
        version_number_minio = get_version_number_from_minio(time_horizon=time_horizon,
                                                             scenario_date=scenario_date,
                                                             modeling_entity=modeling_entity)
    if use_elastic:
        if start_looking:
            version_number_elastic = get_version_number_from_elastic(start_looking=start_looking,
                                                                     modeling_entity=modeling_entity,
                                                                     time_horizon=time_horizon,
                                                                     scenario_date=scenario_date)
        else:
            version_number_elastic = get_version_number_from_elastic(modeling_entity=modeling_entity,
                                                                     time_horizon=time_horizon,
                                                                     scenario_date=scenario_date)
    if version_number_minio and version_number_elastic:
        version_number = version_number_minio if int(version_number_minio) > int(version_number_elastic) \
            else version_number_elastic
    else:
        version_number = version_number_minio or version_number_elastic or version_number
    return version_number


def get_time_horizon_for_intra_day(time_horizon: str, scenario_date: str, skip_past_scenario_dates: bool = True):
    """
    Taken as is from previous code
    :param time_horizon: time_horizon of the merged model
    :param scenario_date: scenario date of the merged model
    :param skip_past_scenario_dates: either to skip past intraday scenarios
    :return updated time horizon value
    """
    if time_horizon == "ID":
        utc_now = datetime.datetime.now(datetime.timezone.utc)
        parsed_date = parse_datetime(scenario_date)
        time_delta = parsed_date.replace(tzinfo=None) - utc_now.replace(tzinfo=None)
        parsed_date_utc = parsed_date.replace(tzinfo=datetime.timezone.utc)
        if (skip_past_scenario_dates and
                (parsed_date_utc.hour + parsed_date_utc.minute / 60) < (utc_now.hour + utc_now.minute / 60)):
            raise IntraDayPastScenarioDateException(f"Skipping merge, past timestamp for intra day")
        time_horizon = f"{int(time_delta.seconds / 3600) + 1 :02d}"
    return time_horizon


def get_profiles_by_cgmes_type(profiles: [], cgmes_types: []):
    """
    Filters profiles by given types
    :param profiles: list of dictionaries containing pmd:cgmesProfile dictionaries
    :param cgmes_types: list of CGMES types (['SSH', 'SV'] for example)
    :return updated profiles list
    """
    if isinstance(cgmes_types, list):
        return [profile for profile in profiles
                if profile.get(OPDM_PROFILE_KEYWORD, {}).get(PMD_CGMES_PROFILE_KEYWORD) in cgmes_types]
    return profiles


def get_files_from_opdm_objects(opdm_objects: list):
    """
    From the input gets data field and filename as list of BytesIO objects
    :param opdm_objects: list of opdm objects
    """
    file_data = []
    for component in opdm_objects:
        for profile in component.get(OPDE_COMPONENT_KEYWORD, []):
            profile = profile.get(OPDM_PROFILE_KEYWORD, {})
            if DATA_KEYWORD in profile:
                file_datum = BytesIO(profile.get(DATA_KEYWORD))
                file_datum.name = profile.get(PMD_FILENAME_KEYWORD)
                file_data.append(file_datum)
    return file_data


class CgmModelComposer:
    """
    Class for gathering the data and running the merge function (copy from merge.py)
    """

    def __init__(self,
                 igm_models=None,
                 boundary_data=None,
                 version: str = VERSION,
                 time_horizon: str = TIME_HORIZON,
                 area: str = DEFAULT_AREA,
                 scenario_date: str = SCENARIO_DATE,
                 merging_entity: str = PY_MERGING_ENTITY,
                 namespace_map=None,
                 rdf_map_loc: str = RDF_MAP_JSON,
                 task_data: dict = None,
                 debugging: bool = False):
        """
        Constructor, note that data gathering and filtering must be done beforehand
        This just stores and merges
        A refactored version of merge.py
        :param igm_models: the individual grid models of the tso's
        :param boundary_data: the boundary data of the region
        :param version: the version number to use for the merged model
        :param time_horizon: the time horizon for the merge
        :param area: the area of the merge
        :param scenario_date: the date of the scenario
        :param merging_entity: the author of the merged model
        :param namespace_map:
        :param rdf_map_loc:
        :param task_data:
        """
        if igm_models is None:
            igm_models = []
        self.igm_models = igm_models
        self.boundary_data = boundary_data
        if namespace_map is None:
            namespace_map = NAMESPACE_MAP
        self.sv_data = None
        self.ssh_data = None

        self.time_horizon = time_horizon
        self.area = area
        self.scenario_date = scenario_date

        self.version = version
        self.merging_entity = merging_entity
        self._merged_model = None
        self.merge_report = {}
        self._opdm_data = None
        self._opdm_object_meta = None
        self.namespace_map = namespace_map
        self._cgm = None
        self.rdf_map = load_rdf_map(rdf_map_loc)
        self.task_data = task_data
        self.debugging = debugging

    def get_tso_list(self):
        return ', '.join([model.get(PMD_TSO_KEYWORD, '') for model in self.igm_models])

    def get_log_message(self):
        return (f"Merge at {self.scenario_date}, "
                f"time horizon {self.time_horizon}, "
                f"area {self.area}, "
                f"tsos: {self.get_tso_list()}")

    def get_task_data(self):
        """
        Packages necessary fields to rabbit task for passing on
        """
        task_content = self.task_data.get(TASK_PROPERTIES_KEYWORD, {})
        task_content[TIMESTAMP_KEYWORD] = self.scenario_date
        task_content[TIME_HORIZON_KEYWORD] = self.time_horizon
        task_content[MERGE_TYPE_KEYWORD] = self.area
        task_content[VERSION_NUMBER_KEYWORD] = self.version
        task_content[MERGING_ENTITY_KEYWORD] = self.merging_entity
        self.task_data[TASK_PROPERTIES_KEYWORD] = task_content
        return self.task_data

    def set_task_data(self, task_data: dict = None):
        if task_data:
            task_content = task_data.get(TASK_PROPERTIES_KEYWORD, {})
            self.scenario_date = task_content.get(TIMESTAMP_KEYWORD, self.scenario_date)
            self.time_horizon = task_content.get(TIME_HORIZON, self.time_horizon)
            self.area = task_content.get(MERGE_TYPE_KEYWORD, self.area)
            self.version = task_content.get(VERSION_NUMBER_KEYWORD, self.version)
            self.merging_entity = task_content.get(MERGING_ENTITY_KEYWORD, self.merging_entity)
            self.task_data = task_data

    def get_content(self):
        """
        Gets data in specified format, currently, dict or json
        :return content of the CgmComposer
        """
        return {TASK_KEYWORD: self.get_task_data(),
                MODELS_KEYWORD: self.get_cgm_igms_boundary_as_opde_object(export_type=CgmExportType.ALL)}

    def set_models(self, models_data: dict | list = None):
        """
        Divides the input to igms, cgm and boundary
        :param models_data:
        """
        # If it is dictionary take the values
        if not models_data:
            return False
        if isinstance(models_data, dict):
            if IGM_OBJECT_TYPE in models_data:
                self.igm_models = models_data.get(IGM_OBJECT_TYPE, self.igm_models)
            if BOUNDARY_OBJECT_TYPE in models_data:
                self.boundary_data = models_data.get(BOUNDARY_OBJECT_TYPE, self.boundary_data)
            if CGM_OBJECT_TYPE in models_data:
                self._cgm = models_data.get(CGM_OBJECT_TYPE, self._cgm)
        elif isinstance(models_data, list):
            self.igm_models = [model for model in models_data
                               if model.get(OPDE_OBJECT_TYPE_KEYWORD) == IGM_OBJECT_TYPE]
            self.boundary_data = next((model for model in models_data
                                       if model.get(OPDE_OBJECT_TYPE_KEYWORD) == BOUNDARY_OBJECT_TYPE), None)
            self._cgm = next((model for model in models_data
                              if model.get(OPDE_OBJECT_TYPE_KEYWORD) == CGM_OBJECT_TYPE), None)
            if self._cgm:
                cgm_data = [self._cgm] if not isinstance(self._cgm, list) else self._cgm
                file_data = get_files_from_opdm_objects(cgm_data)
                self._cgm = file_data
        return (self.igm_models is not None) or (self.boundary_data is not None) or (self._cgm is not None)

    @property
    def merged_model(self):
        """
        Gets merged model
        """
        if self._merged_model is None and self.igm_models and self.boundary_data:
            # Dependencies, timeframes needed from somewhere
            # Extract dependencies from the metadata from the igms
            # IGM created, processType some additional field
            self._merged_model = load_model(self.igm_models + [self.boundary_data])
            # Run LF
            self.merge_report = {}
            loadflow_report = pypowsybl.report.Reporter()
            try:
                loadflow_result = pypowsybl.loadflow.run_ac(network=self._merged_model[NETWORK_KEYWORD],
                                                            parameters=loadflow_settings.CGM_DEFAULT,
                                                            reporter=loadflow_report)
                loadflow_result_dict = [attr_to_dict(island) for island in loadflow_result]
                self.merge_report["LOADFLOW_REPORT"] = json.loads(loadflow_report.to_json())
                self.merge_report["LOADFLOW_RESULTS"] = loadflow_result_dict
            except pypowsybl._pypowsybl.PyPowsyblError as p_error:
                logger.error(f"Error at calculating loadflow: {p_error}")
                raise PyPowsyblError(p_error)
        return self._merged_model

    @property
    def opdm_data(self):
        """
        Gets opdm data (igm models and boundary data combined)
        """
        if isinstance(self._opdm_data, pandas.DataFrame):
            return self._opdm_data
        if self.igm_models and self.boundary_data:
            self._opdm_data = load_opdm_data(self.igm_models + [self.boundary_data])
        return self._opdm_data

    @property
    def opdm_object_meta(self):
        """
        Gets base for opdm object meta
        """
        if self._opdm_object_meta is None:
            sv_id = self.merged_model[NETWORK_META_KEYWORD]['id'].split("uuid:")[-1]
            self.time_horizon = get_time_horizon_for_intra_day(self.time_horizon, self.scenario_date)
            self._opdm_object_meta = {'pmd:fullModel_ID': sv_id,
                                      'pmd:creationDate': f"{datetime.datetime.utcnow():%Y-%m-%dT%H:%M:%S.%fZ}",
                                      'pmd:timeHorizon': self.time_horizon,
                                      'pmd:cgmesProfile': 'SV',
                                      'pmd:contentType': 'CGMES',
                                      'pmd:modelPartReference': '',
                                      'pmd:mergingEntity': f"{self.merging_entity}",
                                      'pmd:mergingArea': self.area,
                                      'pmd:validFrom': f"{parse_datetime(self.scenario_date):%Y%m%dT%H%MZ}",
                                      'pmd:modelingAuthoritySet': 'http://www.baltic-rsc.eu/OperationalPlanning',
                                      'pmd:scenarioDate': f"{parse_datetime(self.scenario_date):%Y-%m-%dT%H:%M:00Z}",
                                      'pmd:modelid': sv_id,
                                      'pmd:description': f"""<MDE>
                                    <BP>{self.time_horizon}</BP>
                                    <TOOL>pypowsybl_{pypowsybl.__version__}</TOOL>
                                    <RSC>{self.merging_entity}</RSC>
                                    </MDE>""",
                                      'pmd:versionNumber': self.version,
                                      'file_type': "xml"}
        return self._opdm_object_meta

    def get_cgm_meta_for_qas(self, default_value: str = ''):
        meta_data = {'creationDate': self.opdm_object_meta.get(PMD_CREATION_DATE_KEYWORD, default_value),
                     'modelid': self.opdm_object_meta.get(PMD_MODEL_ID_KEYWORD, default_value),
                     'scenarioDate': self.opdm_object_meta.get(PMD_SCENARIO_DATE_KEYWORD, default_value),
                     'versionNumber': self.opdm_object_meta.get(PMD_VERSION_NUMBER_KEYWORD, default_value),
                     'timeHorizon': self.opdm_object_meta.get(PMD_TIME_HORIZON_KEYWORD, default_value),
                     'mergingEntity': self.opdm_object_meta.get(PMD_MERGING_ENTITY_KEYWORD, default_value),
                     'mergingArea': self.opdm_object_meta.get(PMD_MERGING_AREA_KEYWORD, default_value)
                     }
        return {'MergeInformation': {'MetaData': meta_data}}

    def get_igm_metas_for_qas(self, default_value: str = ''):
        igm_metas = []
        for igm in self.igm_models:
            meta_data = {'creationDate': igm.get(PMD_CREATION_DATE_KEYWORD, default_value),
                         'timeHorizon': igm.get(PMD_TIME_HORIZON_KEYWORD, default_value),
                         'scenarioDate': igm.get(PMD_SCENARIO_DATE_KEYWORD, default_value),
                         'modelingAuthoritySet': igm.get(PMD_MODELING_AUTHORITY_SET_KEYWORD, default_value),
                         'modelPartReference': igm.get(PMD_MODEL_PART_REFERENCE_KEYWORD, default_value),
                         'versionNumber': igm.get(PMD_VERSION_NUMBER_KEYWORD, default_value),
                         'valid': igm.get('valid', default_value)}
            components = [{'modelid': profile.get(OPDM_PROFILE_KEYWORD, {}).get(PMD_MODEL_ID_KEYWORD, default_value)}
                          for profile in igm.get(OPDE_COMPONENT_KEYWORD, [])]
            meta_data['Component'] = components
            igm_metas.append({'MetaData': meta_data})
        return {'ModelInformation': igm_metas}

    def get_loadflow_results_for_qas(self):
        load_flow_results = {}
        if self.merge_report:
            load_flow_results = {'LoadflowResults': self.merge_report.get('LOADFLOW_RESULTS', [])}
            for island in load_flow_results.get('LoadflowResults'):
                # Convert ComponentStatus.CONVERGED to CONVERGED
                try:
                    island['status'] = str(island['status'].name)
                except Exception:
                    pass
        return load_flow_results

    def get_data_for_qas(self):
        """
        Gets data for qas report
        Note that the content is currently preliminary and might change according to the need of the report
        """
        # Get cgm meta and igm metas
        # return self.get_cgm_meta_for_qas() | self.get_igm_metas_for_qas()
        # Get cgm meta, igm metas and loadflow results
        return self.get_cgm_meta_for_qas() | self.get_igm_metas_for_qas() | self.get_loadflow_results_for_qas()

    def get_cgm_as_opde_object(self):
        """
        Returns cgm as opde object
        # TODO what is object type for the CGM, is it 'CGM' or area abbreviation ('BA', 'EU')
        """
        cgm_value = {OPDE_OBJECT_TYPE_KEYWORD: CGM_OBJECT_TYPE, OPDE_COMPONENT_KEYWORD: []}
        if not self._cgm:
            # self.compose_cgm()
            return None
        for file_instance in self._cgm:
            file_name = file_instance.name
            meta_for_data = get_meta_from_filename(file_name)
            opdm_profile_content = meta_for_data
            opdm_profile_content[DATA_KEYWORD] = file_instance.getvalue()
            cgm_value[OPDE_COMPONENT_KEYWORD].append({OPDM_PROFILE_KEYWORD: opdm_profile_content})
        return cgm_value

    def get_cgm_igms_boundary_as_opde_object(self, export_type: CgmExportType = CgmExportType.ALL):
        """
        Packages igms, boundary and cgm to be ready to be imported to pypowsybl
        :param export_type: specify which files from where are exported. Note that export works 'exists' mode
        :return list of requested instances either dicts or file objects
        """
        type_values = export_type.value
        cgm_file_list = type_values.get(CGM_OBJECT_TYPE)
        igm_file_list = type_values.get(IGM_OBJECT_TYPE)
        boundary_file_list = type_values.get(BOUNDARY_OBJECT_TYPE)
        opdm_format = type_values.get(OPDM_FORMAT, True)
        cgm_set = self.get_cgm_as_opde_object()
        opdm_objects = []
        if cgm_set:
            if cgm_file_list:
                cgm_set[OPDE_COMPONENT_KEYWORD] = get_profiles_by_cgmes_type(
                    profiles=cgm_set[OPDE_COMPONENT_KEYWORD],
                    cgmes_types=cgm_file_list)
            opdm_objects = [cgm_set]
        if igm_file_list:
            for old_model in self.igm_models:
                model = copy.deepcopy(old_model)
                model[OPDE_COMPONENT_KEYWORD] = get_profiles_by_cgmes_type(profiles=model[OPDE_COMPONENT_KEYWORD],
                                                                           cgmes_types=igm_file_list)
                opdm_objects.append(model)
        else:
            opdm_objects.extend(self.igm_models)
        if boundary_file_list:
            boundary = copy.deepcopy(self.boundary_data)
            boundary[OPDE_COMPONENT_KEYWORD] = get_profiles_by_cgmes_type(profiles=boundary[OPDE_COMPONENT_KEYWORD],
                                                                          cgmes_types=boundary_file_list)
            opdm_objects.append(boundary)
        else:
            opdm_objects.append(self.boundary_data)
        if not opdm_format:
            return get_files_from_opdm_objects(opdm_objects=opdm_objects)
        return opdm_objects

    def get_cgm(self, export_type: CgmExportType = CgmExportType.BARE):
        """
        Gets cgm model
        :param export_type: specify export type (enum consisting dicts with file types)
        :return requested cgm model
        """
        if export_type == CgmExportType.BARE:
            return self._cgm
        return self.get_cgm_igms_boundary_as_opde_object(export_type=export_type)

    def set_sv_file(self,
                    merged_model=None,
                    opdm_object_meta=None):
        merged_model = merged_model or self.merged_model
        opdm_object_meta = opdm_object_meta or self.opdm_object_meta
        # export_report = pypowsybl.report.Reporter()
        exported_model = export_model_escaped(network=merged_model[NETWORK_KEYWORD],
                                              opdm_object_meta=opdm_object_meta,
                                              profiles=["SV"],
                                              debugging=self.debugging)
        if self.debugging and not exported_model:
            raise Exception(f"Failed to export model")

        logger.info(f"Exporting merged model to {exported_model.name}")
        # Load SV data
        sv_data = pandas.read_RDF([exported_model])
        # Update SV filename
        sv_data.set_VALUE_at_KEY(key='label', value=filename_from_metadata(opdm_object_meta))
        # Update SV description
        sv_data.set_VALUE_at_KEY(key='Model.description', value=opdm_object_meta['pmd:description'])
        # Update SV created time
        sv_data.set_VALUE_at_KEY(key='Model.created', value=opdm_object_meta['pmd:creationDate'])
        # Update SSH Model.scenarioTime
        sv_data.set_VALUE_at_KEY('Model.scenarioTime', opdm_object_meta['pmd:scenarioDate'])
        # Update SV metadata
        sv_data = triplets.cgmes_tools.update_FullModel_from_filename(sv_data)
        self.sv_data = sv_data
        return sv_data, opdm_object_meta

    def set_ssh_files(self,
                      valid_models=None,
                      latest_boundary=None,
                      sv_data=None,
                      opdm_object_meta=None,
                      update_map=None):

        valid_models = valid_models or self.igm_models
        latest_boundary = latest_boundary or self.boundary_data
        sv_data = check_dataframe(sv_data, self.sv_data)
        opdm_object_meta = opdm_object_meta or self.opdm_object_meta
        update_map = update_map or UPDATE_MAP

        ssh_data = load_opdm_data(valid_models, "SSH")
        ssh_data = triplets.cgmes_tools.update_FullModel_from_filename(ssh_data)

        # Update SSH Model.scenarioTime
        ssh_data.set_VALUE_at_KEY('Model.scenarioTime', opdm_object_meta['pmd:scenarioDate'])

        # Load full original data to fix issues
        data = load_opdm_data(valid_models + [latest_boundary])
        terminals = data.type_tableview("Terminal")

        # Update SSH data from SV
        updated_ssh_data = ssh_data.copy()
        for update in update_map:
            source_data = sv_data.type_tableview(update['from_class']).reset_index(drop=True)

            # Merge with terminal, if needed
            if terminal_reference := \
                    [column_name if ".Terminal" in column_name else None for column_name in source_data.columns][0]:
                source_data = source_data.merge(terminals, left_on=terminal_reference, right_on='ID')
                logger.debug(f"Added Terminals to {update['from_class']}")

            updated_ssh_data = updated_ssh_data.update_triplet_from_triplet(
                source_data.rename(columns={update['from_ID']: 'ID', update['from_attribute']: update['to_attribute']})[
                    ['ID', update['to_attribute']]].set_index('ID').tableview_to_triplet(), add=False)

        # Generate new UUID for updated SSH
        updated_ssh_id_map = {}
        for old_id in updated_ssh_data.query("KEY == 'Type' and VALUE == 'FullModel'").ID.unique():
            new_id = str(uuid4())
            updated_ssh_id_map[old_id] = new_id
            logger.info(f"Assigned new UUID for updated SSH: {old_id} -> {new_id}")

        # Update SSH ID-s
        updated_ssh_data = updated_ssh_data.replace(updated_ssh_id_map)

        # Update in SV SSH references
        sv_data = sv_data.replace(updated_ssh_id_map)

        # Add SSH supersedes reference to old SSH
        ssh_supersedes_data = pandas.DataFrame(
            [{"ID": item[1], "KEY": "Model.Supersedes", "VALUE": item[0]} for item in updated_ssh_id_map.items()])
        ssh_supersedes_data['INSTANCE_ID'] = updated_ssh_data.query("KEY == 'Type'").merge(ssh_supersedes_data.ID)[
            'INSTANCE_ID']
        updated_ssh_data = updated_ssh_data.update_triplet_from_triplet(ssh_supersedes_data)

        # Update SSH metadata
        updated_ssh_data = triplets.cgmes_tools.update_FullModel_from_dict(updated_ssh_data, {
            "Model.version": opdm_object_meta['pmd:versionNumber'],
            "Model.created": opdm_object_meta['pmd:creationDate'],
            "Model.mergingEntity": opdm_object_meta['pmd:mergingEntity'],
            "Model.domain": opdm_object_meta['pmd:mergingArea']
        })
        self.ssh_data = updated_ssh_data
        self.sv_data = sv_data
        return updated_ssh_data, sv_data

    def set_cgm(self, updated_ssh_data=None,
                sv_data=None,
                valid_models=None,
                latest_boundary=None,
                opdm_object_meta=None,
                filename_mask: str = FILENAME_MASK,
                namespace_map=None):
        # Update SSH filenames
        updated_ssh_data = check_dataframe(updated_ssh_data, self.ssh_data)
        sv_data = check_dataframe(sv_data, self.sv_data)
        valid_models = valid_models or self.igm_models
        latest_boundary = latest_boundary or self.boundary_data
        opdm_object_meta = opdm_object_meta or self.opdm_object_meta
        namespace_map = namespace_map or NAMESPACE_MAP
        data = load_opdm_data(valid_models + [latest_boundary])
        updated_ssh_data = triplets.cgmes_tools.update_filename_from_FullModel(updated_ssh_data,
                                                                               filename_mask=filename_mask)

        # Update SV metadata
        sv_metadata = {"Model.version": opdm_object_meta['pmd:versionNumber'],
                       "Model.created": opdm_object_meta['pmd:creationDate']}
        sv_data = triplets.cgmes_tools.update_FullModel_from_dict(sv_data, sv_metadata)

        # Fix SV - Remove Shunt Sections for EQV Shunts
        equiv_shunt = data.query("KEY == 'Type' and VALUE == 'EquivalentShunt'")
        if len(equiv_shunt) > 0:
            shunts_to_remove = sv_data.merge(sv_data.query("KEY == 'SvShuntCompensatorSections.ShuntCompensator'").
                                             merge(equiv_shunt.ID,
                                                   left_on='VALUE',
                                                   right_on="ID", how='inner',
                                                   suffixes=('', '_EQVShunt')).ID)
            if len(shunts_to_remove) > 0:
                logger.warning(f'Removing invalid SvShuntCompensatorSections for EquivalentShunt')
                sv_data = triplets.rdf_parser.remove_triplet_from_triplet(sv_data, shunts_to_remove)

        # Fix SV - add missing SV Tap Steps
        ssh_tap_steps = updated_ssh_data.query("KEY == 'TapChanger.step'")
        sv_tap_steps = sv_data.query("KEY == 'SvTapStep.TapChanger'")
        missing_sv_tap_steps = ssh_tap_steps.merge(sv_tap_steps[['VALUE']], left_on='ID', right_on="VALUE", how='left',
                                                   indicator=True, suffixes=('', '_SV')).query("_merge == 'left_only'")

        tap_steps_to_be_added = []
        sv_instance_id = sv_data.INSTANCE_ID.iloc[0]
        for tap_changer in missing_sv_tap_steps.itertuples():
            id_value = str(uuid4())
            logger.warning(f'Missing SvTapStep for {tap_changer.ID}, adding SvTapStep {id_value} '
                           f'and taking tap value {tap_changer.VALUE} from SSH')
            tap_steps_to_be_added.extend([
                (id_value, 'Type', 'SvTapStep', sv_instance_id),
                (id_value, 'SvTapStep.TapChanger', tap_changer.ID, sv_instance_id),
                (id_value, 'SvTapStep.position', tap_changer.VALUE, sv_instance_id),
            ])

        sv_data = pandas.concat(
            [sv_data, pandas.DataFrame(tap_steps_to_be_added, columns=['ID', 'KEY', 'VALUE', 'INSTANCE_ID'])],
            ignore_index=True)

        export = (pandas.concat([updated_ssh_data, sv_data], ignore_index=True).
                  export_to_cimxml(rdf_map=self.rdf_map,
                                   namespace_map=namespace_map,
                                   export_undefined=False,
                                   export_type="xml_per_instance_zip_per_xml",
                                   debug=False,
                                   export_to_memory=True))
        self._cgm = export
        return export

    def compose_cgm(self):
        """
        Composes the cgm
        """
        if not self.igm_models or not self.boundary_data:
            raise NotEnoughInputDataError(f"Missing {'igm data' if not self.igm_models else ''} "
                                          f"{'and' if (not self.igm_models and not self.boundary_data) else ''} "
                                          f"{'boundary data' if not self.boundary_data else ''}")
        logger.info(f"Merging at {self.scenario_date}, "
                    f"time horizon: {self.time_horizon}, "
                    f"version: {self.version}, "
                    f"area: {self.area}, "
                    f"tsos: {', '.join([model.get(PMD_TSO_KEYWORD) for model in self.igm_models])}")
        self.set_sv_file()
        self.set_ssh_files()
        self.set_cgm()
        return self.get_cgm()

    def get_consolidated_metadata(self, rabbit_data: dict = None, additional_fields: dict = None):
        """
        Combines existing metadata with rabbit data for reporting
        NOTE! Change this
        """
        if not rabbit_data:
            rabbit_data = self.task_data
        consolidated_data = self.opdm_object_meta
        consolidated_data[PROCESS_ID_KEYWORD] = rabbit_data.get(PROCESS_ID_KEYWORD)
        consolidated_data[RUN_ID_KEYWORD] = rabbit_data.get(RUN_ID_KEYWORD)
        consolidated_data[JOB_ID_KEYWORD] = rabbit_data.get(JOB_ID_KEYWORD)
        if additional_fields:
            consolidated_data.update(additional_fields)
        return consolidated_data

    def get_folder_name(self):
        """
        Modify this to generate folder name/path where to save the models in local storage
        NB! This is meant for testing purposes only
        """
        model_date = f"{parse_datetime(self.scenario_date):%Y%m%dT%H%MZ}"
        operator_name = '-'.join([self.merging_entity, self.area])
        folder_name = '_'.join([model_date, self.time_horizon, operator_name, self.version])
        return folder_name


def save_merged_model_to_local_storage(cgm_files,
                                       cgm_folder_name: str = None,
                                       local_storage_location: str = LOCAL_STORAGE_LOCATION):
    """
    Saves merged cgm to local storage. This is meant for testing purposes only
    :param cgm_files: list of cgm_files
    :param cgm_folder_name: sub folder name where to gather files
    :param local_storage_location: path to store
    :return: None
    """
    if not local_storage_location:
        return
    if cgm_folder_name is not None:
        local_storage_location = local_storage_location + '/' + cgm_folder_name
        local_storage_location = check_the_folder_path(local_storage_location)
    if not os.path.exists(local_storage_location):
        os.makedirs(local_storage_location)
    for cgm_file in cgm_files:
        full_file_name = local_storage_location + cgm_file.name
        with open(full_file_name, 'wb') as f:
            f.write(cgm_file.getbuffer())


def publish_merged_model_to_opdm(opdm_client: opdm.OPDM = None,
                                 cgm_files: list = None):
    """
    Sends files to opdm
    :param opdm_client: client instance
    :param cgm_files: list of files to be sent
    :return tuple of results
    """
    # Post files if given
    result = ()
    # Send files out if given
    if cgm_files and len(cgm_files) > 0:
        opdm_publication_responses = []
        for instance_file in cgm_files:
            try:
                opdm_client = opdm_client or opdm.OPDM()
                logger.info(f"Publishing {instance_file.name} to OPDM")
                file_response = opdm_client.publication_request(instance_file, "CGMES")
                opdm_publication_responses.append({"name": instance_file.name, "response": file_response})
                if "OperationFailure" in file_response.tag:
                    logger.error(f"Failed to publish {instance_file.name} to OPDM, OPDM OperationFailure")
            except zeep.exceptions.Fault as fault:
                logger.error(f"Failed to publish {instance_file.name} to OPDM, connection failure: {fault}")
            except Exception as ex_message:
                logger.error(f"Failed to publish {instance_file.name} to OPDM, unknown error: {ex_message}")
        logger.info(f"Publishing results: {opdm_publication_responses}")
        result = result + (opdm_publication_responses,)
    return result


def save_merged_model_to_minio(minio_bucket: str = PY_EMF_OS_MINIO_OPDE_MODELS_BUCKET,
                               folder_in_bucket: str = PY_EMF_OS_MINIO_OPDE_MODELS_FOLDER,
                               minio_client: minio.ObjectStorage = None,
                               time_horizon: str = None,
                               scenario_datetime: str = None,
                               merging_entity: str = None,
                               area: str = None,
                               version: str = None,
                               file_type: str = None,
                               cgm_files: [] = None):
    """
    Posts cgm files to minio
    implementation of path <minio_bucket>/<folder_in_bucket>/<cgm_folder_name>/cgm_files
    :param minio_bucket: bucket in minio
    :param minio_client: the instance of object storage client
    :param time_horizon: time_horizon for the path tree
    :param scenario_datetime: scenario_date for the path tree
    :param merging_entity: the entity responsible for the merge
    :param version: the version number
    :param area: the area where the merge was done
    :param cgm_files: list of individual cgm files
    :param file_type: type of the file if specified
    :param folder_in_bucket: general folder (prefix) in bucket where
    :return: file name and link to file, the link to the file
    """
    links_to_file = {}
    minio_client = minio_client or minio.ObjectStorage()
    if cgm_files is not None:
        # check if the given bucket exists
        if not minio_client.client.bucket_exists(bucket_name=minio_bucket):
            logger.warning(f"{minio_bucket} does not exist")
            return links_to_file
        for cgm_file in cgm_files:
            file_name = cgm_file.name
            file_name_exploded = get_metadata_from_filename(file_name)
            file_time_horizon = time_horizon or file_name_exploded.get(MODEL_PROCESS_TYPE_KEYWORD, '')
            file_scenario_datetime = scenario_datetime or file_name_exploded.get(MODEL_SCENARIO_TIME_KEYWORD, None)
            if file_scenario_datetime:
                file_scenario_datetime = parse_datetime(file_scenario_datetime)
            file_merging_entity = merging_entity or file_name_exploded.get(MODEL_MERGING_ENTITY_KEYWORD)
            file_merging_entity = file_merging_entity or file_name_exploded.get(MODEL_MODELING_ENTITY_KEYWORD, '')
            file_area = area or file_name_exploded.get(MODEL_DOMAIN_KEYWORD, '')
            file_version = version or file_name_exploded.get(MODEL_VERSION_KEYWORD)
            scenario_date = ''
            scenario_time = ''
            if file_scenario_datetime:
                scenario_date = f"{file_scenario_datetime:%Y%m%d}"
                scenario_time = f"{file_scenario_datetime:%H%M00}"
            file_file_type = file_type or file_name_exploded.get(MODEL_MESSAGE_TYPE_KEYWORD)
            file_path_elements = [folder_in_bucket, file_time_horizon, file_merging_entity, file_area,
                                  scenario_date, scenario_time, file_version, file_file_type, cgm_file.name]
            full_file_name = SEPARATOR_SYMBOL.join(file_path_elements)
            full_file_name = full_file_name.replace('//', '/')
            cgm_file.name = full_file_name
            minio_client.upload_object(file_path_or_file_object=cgm_file, bucket_name=minio_bucket)
            time_to_expire = timedelta(days=7)
            link_to_file = minio_client.client.get_presigned_url(method="GET",
                                                                 bucket_name=minio_bucket,
                                                                 object_name=cgm_file.name,
                                                                 expires=time_to_expire)
            cgm_file.name = file_name
            links_to_file[file_name] = link_to_file
        return links_to_file


def publish_metadata_to_elastic(cgm_index: str, metadata: dict, elastic_server: str = elastic.ELK_SERVER):
    """
    Publishes metadata to elastic
    :param cgm_index: table name
    :param metadata: metadata information
    :param elastic_server: address to elastic server
    :return response
    """
    if metadata:
        response = elastic.Elastic.send_to_elastic(index=cgm_index,
                                                   json_message=metadata,
                                                   server=elastic_server)
        return response


if __name__ == '__main__':
    # Run the entire pipeline in functions
    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        # Some files have logging at the beginning in the file which is read in, so overwrite the conf
        # force=True,
        handlers=[logging.StreamHandler(sys.stdout)]
        # handlers=[logging.StreamHandler(sys.stdout), ElkLoggingHandler(), PyPowsyblLogGatheringHandler()]
    )

    take_data_from_local = True
    testing_merging_entity = PY_MERGING_ENTITY

    testing_area = 'EU'
    testing_time_horizon = '1D'
    testing_scenario_date = "2024-05-06T21:30:00+00:00"
    # testing_scenario_date = '2024-04-17T09:30:00+00:00'
    wanted_tsos = []
    unwanted_tsos = []
    test_version_number = None

    if take_data_from_local:
        # folder_to_study = 'apg_case'
        # test_version_number = '002'
        from emf.common.integrations.object_storage import file_system
        igm_model_data = file_system.get_latest_models_and_download()
        latest_boundary_data = file_system.get_latest_boundary()
        igm_model_data = filter_models_by_tsos(igm_models=igm_model_data,
                                               included_tsos=wanted_tsos,
                                               excluded_tsos=unwanted_tsos)
    else:
        igm_model_data, latest_boundary_data = get_models(time_horizon=testing_time_horizon,
                                                          scenario_date=testing_scenario_date,
                                                          included_tsos=wanted_tsos,
                                                          excluded_tsos=unwanted_tsos,
                                                          download_policy=DownloadModels.MINIO)
    version_number_from_storage = get_version_number(scenario_date=testing_scenario_date,
                                                     time_horizon=testing_time_horizon,
                                                     modeling_entity=f"{testing_merging_entity}-"
                                                                     f"{testing_area}")
    test_version_number = test_version_number or version_number_from_storage
    if not igm_model_data or not latest_boundary_data:
        logger.error(f"Terminating")
        sys.exit()
    cgm_input = CgmModelComposer(igm_models=igm_model_data,
                                 boundary_data=latest_boundary_data,
                                 time_horizon=testing_time_horizon,
                                 scenario_date=testing_scenario_date,
                                 area=testing_area,
                                 merging_entity=testing_merging_entity,
                                 version=test_version_number,
                                 debugging=True)

    # Save to opdm
    bare_cgm = cgm_input.compose_cgm()
    # publish_merged_model_to_opdm(cgm_files=bare_cgm)

    # Save to local storage
    test_folder_name = cgm_input.get_folder_name()
    all_files = cgm_input.get_cgm(export_type=CgmExportType.ALL_FILES_ONLY)
    save_merged_model_to_local_storage(cgm_files=all_files, cgm_folder_name=test_folder_name)

    # Save to minio
    full_files = cgm_input.get_cgm(export_type=CgmExportType.FULL_FILES_ONLY)
    # save_merged_model_to_minio(cgm_files=full_files,
    #                            minio_bucket=PY_EMF_OS_MINIO_OPDE_MODELS_BUCKET,
    #                            folder_in_bucket=PY_EMF_OS_MINIO_OPDE_MODELS_FOLDER,
    #                            # comment next lines if following the path by file name is needed
    #                            # if uncommented, gathers files to <t.horizon>/<m.entity>/<area>/<date>/<ver>
    #                            merging_entity=cgm_input.merging_entity,
    #                            time_horizon=cgm_input.time_horizon,
    #                            scenario_datetime=cgm_input.scenario_date,
    #                            area=cgm_input.area,
    #                            version=cgm_input.version,
    #                            file_type='CGM'
    #                            )

    logger.info("The end...")
