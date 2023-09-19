import unreal
import UE5lib as UE5lib
import importlib
from pathlib import Path
my_module = importlib.import_module('UE5lib')
importlib.reload(my_module)

unreal.log('How are You?')

config = UE5lib.loadconfigs('C:/Users/tronc/Documents/GitHub/RCprocessing/config_UrukModel.json')
meshdf = UE5lib.provide_meshdf(config['meshfolder'])
print(meshdf)
for index, mesh in meshdf.iterrows():
    #mtldf = UE5lib.parse_mtl(mesh['MTL File'])
    #mtldf_modified = UE5lib.modify_mtl(mtldf)
    #UE5lib.write_mtl(mtldf_modified, mesh['MTL File'])
    #print(mtldf)
    texturestemname = UE5lib.get_texturestemname(mesh['Texture Files'])
    print(texturestemname)
    options = UE5lib.build_import_options(texturestemname)
    #print(options.get_editor_property('static_mesh_import_data'))
    
    tasks = UE5lib.build_import_tasks(mesh['OBJ File'], mesh['Texture Files'], config['UEdestination'], options)
    UE5lib.import_static_mesh(tasks)
    
    UE5lib.post_import_process(Path(mesh['OBJ File']).stem, texturestemname, config['UEdestination'])

