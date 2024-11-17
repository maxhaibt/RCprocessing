import unreal as ue
import UE5lib as UE5lib
import importlib
import requests
from pathlib import Path
my_module = importlib.import_module('UE5lib')
importlib.reload(my_module)

search_string = "URUK26"
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
# get related docs from related_docs and append them
subrelated_docs = UE5lib.get_related_docs_by_liesWithin(api, related_docs)
related_docs += subrelated_docs
ue.log(related_docs)
for doc in related_docs:
    resource_identifier = doc['resource'].get('identifier') # Get the resource identifier
    y_coordinates, needs_vertical_extent = UE5lib.extract_polygon_geometries(doc)
    
    if y_coordinates:
        ue.log(y_coordinates)
        min_y, max_y = UE5lib.calculate_min_max_y(y_coordinates)
        top_value, bottom_value = UE5lib.height_translate_UTM2UrukVR(min_y, max_y)
        ue.log(f"top_value: {top_value}, bottom_value: {bottom_value}")

        # Create the cylinder with polygon coordinates
        resource3DGeometry =UE5lib.create_dynamic_cylinder_and_save(top_center_positions, top_value, bottom_value, resource_identifier)
        UE5lib.spawn_resource_actor(resource3DGeometry, doc)
    elif needs_vertical_extent:
        # Extract vertical extent since Y-coordinates were not found
        top_value,bottom_value = UE5lib.extract_verticalextent(doc)
        actual_top_value, actual_bottom_value = UE5lib.calculate_cylinder_position(top_center_positions,top_value, bottom_value)
        
        if actual_top_value is None or actual_bottom_value is None:
            ue.log_warning("Vertical extent values are None, skipping cylinder creation.")
            continue  # Skip if vertical extent values are not valid
        

        # Create the cylinder with vertical extent values
        resource3DGeometry = UE5lib.create_dynamic_cylinder_and_save(top_center_positions, actual_top_value, actual_bottom_value, resource_identifier)
        #UE5lib.transform_and_save_static_mesh(resource3DGeometry, resource_identifier=resource_identifier)
        UE5lib.spawn_resource_actor(resource3DGeometry, doc)