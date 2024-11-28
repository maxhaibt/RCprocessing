import unreal as ue
import UE5lib as UE5lib
import importlib
import requests
from pathlib import Path
my_module = importlib.import_module('UE5lib')
importlib.reload(my_module)

search_string = "URUK28"
ue.log(search_string)

#config = UE5lib.loadconfigs('E:/UEpythonutils/URUKVR_config.json')

#UE5lib.shift_sediment_cores_to_player(search_string)
UE5lib.create_thick_component(
    blueprint_path="/Game/CoordinateDisplay/ThickBP",
    cylinder_diameter=0.3,
    cylinder_length=2.0,
    label_distance=2.0
)                                                                                                                                                             