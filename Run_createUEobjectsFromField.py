import unreal as ue
import UE5lib as UE5lib
import importlib
import requests
from pathlib import Path
my_module = importlib.import_module('UE5lib')
importlib.reload(my_module)

search_string = "URUK32"
ue.log(search_string)

config = UE5lib.loadconfigs('E:/UEpythonutils/URUKVR_config.json')

actors = UE5lib.get_static_mesh_actors_by_name(search_string)

top_center_positions = UE5lib.get_top_center_of_0to1m_mesh(actors)
ue.log(top_center_positions)

# Log the names of the actors found
for actor in actors:
    ue.log(f"Found StaticMeshActor: {actor.get_actor_label()}")
api = UE5lib.couchDB_APIs(config)
#targetDOCs= UE5lib.getAllDocs(api)


matching_docs = UE5lib.getDocsIfContainStringInIdentifier(api, search_string)
sedimentcore_docs = UE5lib.filter_docs_by_category(matching_docs, "Sedimentcore")
related_docs = UE5lib.get_related_docs_by_liesWithin(api, sedimentcore_docs)
for doc in related_docs:
    y_coordinates = UE5lib.extract_polygon_geometries(doc)
    #next iteration if y_coordinates is empty
    if not y_coordinates:
        continue
    ue.log(y_coordinates)
    min_y, max_y = UE5lib.calculate_min_max_y(y_coordinates)
    ue.log(f"min_y: {min_y}, max_y: {max_y}")
    translated_min_y, translated_max_y = UE5lib.height_translate_UTM2UrukVR(min_y, max_y)
    ue.log(f"trans_min_y: {translated_min_y}, trans_max_y: {translated_max_y}")
    # Assuming top_center_positions is defined and represents the top-center position
    UE5lib.create_cylinder_at_position(top_center_positions, translated_min_y, translated_max_y)

