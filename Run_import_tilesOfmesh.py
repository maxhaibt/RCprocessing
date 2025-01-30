
import unreal as ue
import sys
import os

import importlib
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))
import UE5lib as UE5lib
my_module = importlib.import_module('UE5lib')
importlib.reload(my_module)


ue.log('How are You?')

config = UE5lib.loadconfigs('C:/Users/tronc/Documents/GitHub/RCprocessing/config_boattiles.json')
meshdf = UE5lib.parse_mesh_directory(config['meshfolder'])
#print number of parts
print(len(meshdf))
for part in meshdf: 
    
    texturetask = UE5lib.build_import_texture_tasks(part['textures'], config['UEdestination'])
    UE5lib.import_asset(texturetask)
        #print(texturetask[0].imported_object_paths)
    if not ue.EditorAssetLibrary.does_asset_exist(config['UEdestination'] + '/' + part['partname']):
        materialinstance_part = UE5lib.create_constant_material_instance(config['UEbasematerialpath'], config['UEdestination'], part['partname'], texturetask[0].imported_object_paths[0])

    #importedtextureref=
    texture_params = list(materialinstance_part.texture_parameter_values)
    for param in texture_params:
        print(param.parameter_info.name, 'is already in the list.')
    tilesfolder = config['UEdestination'] + part['partname'] + '_tiles'  
    if not ue.EditorAssetLibrary.does_directory_exist(tilesfolder):
        ue.EditorAssetLibrary.make_directory(tilesfolder)
    for tile in part['OBJtilesAndMtls']:
        print(tile)
        mtldf = UE5lib.parse_mtl(tile["Mtl"])
        print('Goeshere')
        print('this is',mtldf)
        mtldf_modified = UE5lib.modify_mtl(mtldf)
        UE5lib.write_mtl(mtldf_modified, tile["Mtl"])
        obj = tile["OBJtile"]
        print('the meshtile path to obj',obj)
        options = UE5lib.build_meshtile_import_options()

        meshtask = UE5lib.build_meshtile_import_tasks(obj, tilesfolder, options)
        imported_tiles = UE5lib.import_asset(meshtask)
        imported_mesh_path = meshtask[0].imported_object_paths[0]
        imported_mesh = ue.EditorAssetLibrary.load_asset(imported_mesh_path)

        if not imported_mesh:
            ue.log_error(f"Failed to import mesh: {imported_mesh_path}")

        ue.log(f"Imported mesh: {imported_mesh_path}")

        # Assign the material to the first material slot of the mesh
        imported_mesh.set_material(0, materialinstance_part)

        # Save the changes
        ue.EditorAssetLibrary.save_asset(imported_mesh_path)

        ue.log(f"Material assigned successfully to: {imported_mesh_path}")

    


