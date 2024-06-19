import ast
import json
import logging
import sys
from enum import Enum
from http import HTTPMethod, HTTPStatus

import ndjson
import requests

import config
from emf.common.config_parser import parse_app_properties

# from common import save_content_to_local_storage, load_content_from_local_storage


logger = logging.getLogger(__name__)

parse_app_properties(globals(), config.paths.integrations.kibana_api)


def send_prepared_request(method_name: HTTPMethod,
                          url_path: str,
                          headers=None,
                          auth=None,
                          data=None,
                          params=None,
                          files=None,
                          json_content=None,
                          verify=False):
    """
    Prepares and sends the request and returns response
    :param method_name: http method
    :param url_path: url of request
    :param headers: headers of the request if specified
    :param auth: authentication if provided
    :param data: payload if given
    :param params: parameters if given
    :param files: files if given
    :param json_content: pass dict as json
    :param verify: verify SSL connection
    :return: response of the request
    """
    session = requests.Session()
    request = requests.Request(method=method_name,
                               url=url_path,
                               headers=headers,
                               auth=auth,
                               data=data,
                               params=params,
                               json=json_content,
                               files=files)
    response = session.send(request.prepare(), verify=verify)
    return response


class SavedObjectType(Enum):
    """
    Extend this
    """
    DASHBOARD = 'dashboard'
    INDEX_PATTERN = 'index-pattern'
    LENS = 'lens'
    SEARCH = 'search'
    TAG = 'tag'
    VISUALIZATION = 'visualization'
    ALL_OBJECTS = '*'

    def __str__(self):
        return self.value

    @classmethod
    def value_of(cls, value):
        for v in cls.__members__.values():
            if v.__str__() == value:
                return v
        else:
            raise ValueError(f"{value} is not recognised {cls.__name__} type")


PY_FROM_KIBANA_URL = FROM_KIBANA_URL
PY_TO_KIBANA_URL = TO_KIBANA_URL
PY_EXPORT_OBJECT_ENDPOINT = EXPORT_OBJECT_ENDPOINT
PY_IMPORT_OBJECT_ENDPOINT = IMPORT_OBJECT_ENDPOINT
PY_SAVED_OBJECT_TYPE = SavedObjectType.value_of(SAVED_OBJECT_TYPE)
PY_SAVED_OBJECT_NAME = SAVED_OBJECT_NAME
PY_INCLUDE_REFERENCES_DEEP = json.loads(INCLUDE_REFERENCES_DEEP.lower())
PY_EXCLUDE_EXPORT_DETAILS = json.loads(EXCLUDE_EXPORT_DETAILS.lower())
PY_SPACE_ID = ast.literal_eval(SPACE_ID)
PY_CREATE_NEW_COPY = json.loads(CREATE_NEW_COPY.lower())
PY_OVERWRITE = json.loads(OVERWRITE.lower())
PY_COMPATIBILITY_MODE = json.loads(COMPATIBILITY_MODE.lower())


def get_saved_object_api_address(kibana_url: str = PY_FROM_KIBANA_URL,
                                 endpoint: str = PY_IMPORT_OBJECT_ENDPOINT,
                                 space_id: str = PY_SPACE_ID):
    """
    Composes kibana url for saved objects api
    :param kibana_url: base url for kibana
    :param endpoint: specific api endpoint
    :param space_id: if space id is used
    """
    if space_id:
        return (kibana_url.removesuffix('/') + '/s/'
                + space_id.removesuffix('/').removeprefix('/') + '/'
                + endpoint.removeprefix('/'))
    return kibana_url.removesuffix('/') + '/' + endpoint.removeprefix('/')


def compose_export_request_body(saved_object_type: str | list = None,
                                saved_objects: list = None,
                                include_references_deep: bool = PY_INCLUDE_REFERENCES_DEEP,
                                exclude_export_details: bool = PY_EXCLUDE_EXPORT_DETAILS):
    """
    Function to generate request body for export save_object endpoint
    :param saved_object_type: saved object types to include in export, use "*" for all types
    :param saved_objects: a list of objects to export
    :param include_references_deep: Include all the referenced objects in the exported object
    :param exclude_export_details: Do not add export details entry at the end of the stream
    :return request body as dict
    """
    content = {}
    if saved_object_type:
        content["type"] = saved_object_type
    if saved_objects:
        content["objects"] = saved_objects
    if include_references_deep:
        content["includeReferencesDeep"] = include_references_deep
    if exclude_export_details:
        content["excludeExportDetails"] = exclude_export_details
    return json.dumps(content)


def compose_import_request_body(data):
    """
    Function to generate request body for import save_object endpoint
    :param data: bytes or string following ndjson format
    :return: dict
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    multipart_data = {'file': ('request.ndjson', data)}
    return multipart_data


def export_saved_objects(export_content: str | dict,
                         kibana_url: str = PY_FROM_KIBANA_URL,
                         space_id: str = PY_SPACE_ID,
                         export_endpoint: str = PY_EXPORT_OBJECT_ENDPOINT):
    """
    Generic function against export endpoint of kibana (note that export-> get uses POST here)
    :param export_content: payload
    :param kibana_url: url of kibana
    :param space_id: space where and if the object belongs to
    :param export_endpoint: api endpoint
    :return content if response was OK, None otherwise
    """
    export_url = get_saved_object_api_address(kibana_url=kibana_url,
                                              space_id=space_id,
                                              endpoint=export_endpoint)
    export_headers = {'Content-Type': 'application/json', 'kbn-xsrf': 'true'}
    export_response = send_prepared_request(method_name=HTTPMethod.POST,
                                            headers=export_headers,
                                            json_content=export_content,
                                            url_path=export_url)
    if export_response.status_code == HTTPStatus.OK:
        return export_response.content
    return None


def ndjson_to_dict(content: str = None):
    """
    Convert input string from Kibana to dict
    :param content: input as string
    """
    if content:
        return ndjson.loads(content)
    return None


def dict_to_ndjson(content: dict = None):
    """
    Convert input dict to ndjson formatted string
    :param content: input as dict
    """
    if content:
        return ndjson.dumps(content)
    return None


def import_saved_objects(data,
                         kibana_url: str = PY_TO_KIBANA_URL,
                         space_id: str = PY_SPACE_ID,
                         import_endpoint: str = PY_IMPORT_OBJECT_ENDPOINT,
                         create_new_copies: bool = PY_CREATE_NEW_COPY,
                         overwrite: bool = PY_OVERWRITE,
                         compatibility_mode: bool = PY_COMPATIBILITY_MODE):
    """
    Imports saved objects to kibana
    :data: payload (in ndjson format)
    :kibana_url: url of location
    :import_endpoint: api endpoint for import
    :space_id: id of space where to import (if specified)
    :create_new_copies: whether to create a new copy (overwrites 'overwrite' and 'compatibility_mode')
    :overwrite: overwrites the existing one (create_new_copies must be set to False)
    :compatibility_mode: switch version friendly mode on
    :return: response code
    """
    import_headers = {'kbn-xsrf': 'true'}
    import_url = get_saved_object_api_address(kibana_url=kibana_url,
                                              space_id=space_id,
                                              endpoint=import_endpoint)
    # Fill the parameters
    parameters = {}
    if create_new_copies:
        parameters['createNewCopies'] = 'true'
    else:
        if overwrite:
            parameters['overwrite'] = 'true'
        if compatibility_mode:
            parameters['compatibilityMode'] = 'true'
    # Escape parameters to None if empty
    parameters = parameters if parameters else None
    multipart_data = compose_import_request_body(data=data)
    import_response = send_prepared_request(method_name=HTTPMethod.POST,
                                            headers=import_headers,
                                            url_path=import_url,
                                            params=parameters,
                                            files=multipart_data)
    return import_response


def get_list_of_saved_objects_by_type(kibana_url: str = PY_FROM_KIBANA_URL,
                                      space_id: str = PY_SPACE_ID,
                                      export_endpoint: str = PY_EXPORT_OBJECT_ENDPOINT,
                                      object_type: SavedObjectType | list[SavedObjectType] = PY_SAVED_OBJECT_TYPE,
                                      include_references_deep: bool = PY_INCLUDE_REFERENCES_DEEP,
                                      exclude_export_details: bool = PY_EXCLUDE_EXPORT_DETAILS):
    """
    Gets objects by type from kibana
    :param kibana_url: url of kibana
    :param space_id: space id if used
    :param export_endpoint: api endpoint
    :param object_type: saved object type specified
    :param include_references_deep: include all references
    :param exclude_export_details: exclude export report
    """
    if isinstance(object_type, SavedObjectType):
        object_type = [object_type]
    types = [element.value for element in object_type]
    request_body = compose_export_request_body(saved_object_type=types,
                                               include_references_deep=include_references_deep,
                                               exclude_export_details=exclude_export_details)
    return export_saved_objects(export_content=request_body,
                                kibana_url=kibana_url,
                                space_id=space_id,
                                export_endpoint=export_endpoint)


def get_saved_object_by_id(saved_object_id: str,
                           object_type: SavedObjectType = SavedObjectType.DASHBOARD,
                           kibana_url: str = PY_FROM_KIBANA_URL,
                           space_id: str = PY_SPACE_ID,
                           export_endpoint: str = PY_EXPORT_OBJECT_ENDPOINT,
                           include_references_deep: bool = PY_INCLUDE_REFERENCES_DEEP,
                           exclude_export_details: bool = PY_EXCLUDE_EXPORT_DETAILS):
    """
    Exports saved object by id
    Note: object_type must be passed along
    :param saved_object_id: id of the saved object
    :param kibana_url: url of kibana
    :param space_id: space id if used
    :param export_endpoint: api endpoint
    :param object_type: saved object type specified
    :param include_references_deep: include all references
    :param exclude_export_details: exclude export report
    :return: ndjson as string
    """
    by_id = {"id": saved_object_id, "type": object_type.value}
    request_body = compose_export_request_body(saved_objects=[by_id],
                                               include_references_deep=include_references_deep,
                                               exclude_export_details=exclude_export_details)
    return export_saved_objects(export_content=request_body,
                                kibana_url=kibana_url,
                                space_id=space_id,
                                export_endpoint=export_endpoint)


def get_saved_object_by_name(saved_object_title: str,
                             object_type: SavedObjectType = SavedObjectType.DASHBOARD,
                             kibana_url: str = PY_FROM_KIBANA_URL,
                             space_id: str = PY_SPACE_ID,
                             export_endpoint: str = PY_EXPORT_OBJECT_ENDPOINT,
                             include_references_deep: bool = PY_INCLUDE_REFERENCES_DEEP,
                             exclude_export_details: bool = PY_EXCLUDE_EXPORT_DETAILS):
    """
    Exports saved object by its name (title)
    Note: object_type must be passed along
    :param saved_object_title: title of saved object
    :param kibana_url: url of kibana
    :param space_id: space id if used
    :param export_endpoint: api endpoint
    :param object_type: saved object type specified
    :param include_references_deep: include all references
    :param exclude_export_details: exclude export report
    :return: ndjson as string
    """
    all_elements = get_list_of_saved_objects_by_type(kibana_url=kibana_url,
                                                     space_id=space_id,
                                                     export_endpoint=export_endpoint,
                                                     object_type=object_type,
                                                     include_references_deep=False,
                                                     exclude_export_details=False)
    all_elements = ndjson_to_dict(all_elements)
    match = next((element for element in all_elements
                  if element.get('attributes', {}).get('title') == saved_object_title), None)
    if match:
        saved_object = get_saved_object_by_id(saved_object_id=match.get('id'),
                                              object_type=object_type,
                                              kibana_url=kibana_url,
                                              space_id=space_id,
                                              export_endpoint=export_endpoint,
                                              include_references_deep=include_references_deep,
                                              exclude_export_details=exclude_export_details)
        return saved_object
    return None


def get_all_elements_by_tag(tag_name: str):
    """
    Example to get all saved_objects which refer to the tag:
    1) Get tag id by name
    2) Get all elements, filter them which have the id of the tag in the reference
    3) Get ids and types of filtered elements
    4) Query those elements with all dependencies included
    :param tag_name: name of the tag (string)
    :return: saved objects or none
    """
    if tags := ndjson_to_dict(get_list_of_saved_objects_by_type(object_type=SavedObjectType.TAG,
                                                                include_references_deep=False)):
        if tag := next((tag for tag in tags if tag.get('attributes', {}).get('name') == tag_name), None):
            tag_id = tag.get('id')
            all_elements = ndjson_to_dict(get_list_of_saved_objects_by_type(object_type=SavedObjectType.ALL_OBJECTS,
                                                                            include_references_deep=False))
            if all_elements:
                objects = []
                for element in all_elements:
                    references = element.get('references')
                    if next((reference for reference in references if reference.get('id') == tag_id), None):
                        objects.append({"type": element.get("type"), "id": element.get("id")})
                response_body = compose_export_request_body(saved_objects=objects,
                                                            include_references_deep=True,
                                                            exclude_export_details=True)
                final_response = export_saved_objects(export_content=response_body)
                return final_response
    return None


if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    migrate_object_name = PY_SAVED_OBJECT_NAME
    from_kibana = PY_FROM_KIBANA_URL
    to_kibana = PY_TO_KIBANA_URL
    # file_name = './migrate_object.ndjson'
    # 1. Get everything
    # all_stuff = get_list_of_saved_objects_by_type(kibana_url=from_kibana,
    #                                               object_type=SavedObjectType.ALL_OBJECTS)
    # all_stuff = ndjson_to_dict(all_stuff)
    # object_types = set(saved_object.get('type') for saved_object in all_stuff)
    # logger.info(f"{from_kibana} contains {len(all_stuff)} saved_objects, these are: {','.join(object_types)}")
    # 2. Get some types
    some_things = get_list_of_saved_objects_by_type(kibana_url=from_kibana,
                                                    object_type=[SavedObjectType.TAG],
                                                    include_references_deep=True)
    some_things = ndjson_to_dict(some_things)

    # only_boards = [element for element in some_things if element.get("type") == SavedObjectType.DASHBOARD.value]
    # only_indices = [element for element in some_things if element.get("type") == SavedObjectType.INDEX_PATTERN.value]
    # logger.info(f"Got {len(only_boards)} {SavedObjectType.DASHBOARD.value}s, "
    #             f"{len(only_indices)} {SavedObjectType.INDEX_PATTERN.value}s, "
    #             f"in total {len(some_things)} objects")
    # 3. Get all dashboards
    # dashboards_bare = get_list_of_saved_objects_by_type(kibana_url=from_kibana,
    #                                                     object_type=SavedObjectType.DASHBOARD,
    #                                                     include_references_deep=False)
    # dashboards_full = get_list_of_saved_objects_by_type(kibana_url=from_kibana,
    #                                                     object_type=SavedObjectType.DASHBOARD)
    # dashboards_bare = ndjson_to_dict(dashboards_bare)
    # dashboards_full = ndjson_to_dict(dashboards_full)
    # logger.info(f"{from_kibana} contains {len(dashboards_bare)} dashboards, "
    #             f"along with {len(dashboards_full) - len(dashboards_bare)} dependencies")
    # 4. Get dashboard by its id
    # random_dashboard_id = 'f27bb190-233b-11ef-b23f-295bc02b62e3'
    # random_dashboard = get_saved_object_by_id(kibana_url=from_kibana,
    #                                           saved_object_id=random_dashboard_id,
    #                                           object_type=SavedObjectType.DASHBOARD)
    # random_dashboard = ndjson_to_dict(random_dashboard)
    # print(random_dashboard)
    # 5. Get dashboard by its name
    dashboard_name = 'EMFOS TASKS'
    tasks_dashboard = get_saved_object_by_name(kibana_url=from_kibana,
                                               object_type=SavedObjectType.DASHBOARD,
                                               saved_object_title=dashboard_name)
    tasks_dashboard = ndjson_to_dict(tasks_dashboard)
    print(tasks_dashboard)
    emfos_saved_objects = get_all_elements_by_tag("EMFOS")
    emfos_saved_objects = ndjson_to_dict(emfos_saved_objects)
    print(emfos_saved_objects)
    # 6. import dashboard to kibana
    # tasks_dashboard = dict_to_ndjson(tasks_dashboard)
    # response = import_saved_objects(data=tasks_dashboard, kibana_url=from_kibana)
    # print(response.status_code)
