from pxr import Usd

stage = Usd.Stage.Open("E:/Your/Path/YourFile.usd")
for prim in stage.Traverse():
    print(prim.GetPath())