
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
meshdf = UE5lib.parse_mesh_andsubtiles_directory(config['meshfolder'])
#meshdf = UE5lib.parse_mesh_directory(config['meshfolder'])
#print number of parts
#print(len(meshdf))
#print(type(meshdf))
#print list of dicts prettyP
#print(meshdf[0])
#print(meshdf)
#stop script
#sys.exit()
for part in meshdf[0:1]:  

    tilesfolder = config['UEdestination'] + part['partname'] + '_editsubtiles/' 
    #sublevelpath = config['UEdestination'] + part['partname'] + '_editsublevel'
    UE5lib.create_dynamic_mesh_actors_from_part(config['parentlevel'],  tilesfolder)
#dyn_actor = UE5lib.convert_testcube_to_dynamic_mesh()



    


