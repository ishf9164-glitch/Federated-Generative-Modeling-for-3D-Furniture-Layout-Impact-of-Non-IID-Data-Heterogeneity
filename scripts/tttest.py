import argparse
import os
import sys
from tqdm import tqdm
import numpy as np
class A():
     def __init__(self):
        self.config = "LIN"
        print("1")
def main(argv):
    parser = argparse.ArgumentParser(
        description="Prepare the 3D-FRONT scenes to train our model"
    )   
        
    parser.add_argument(
        "--background",
        type=lambda x: map(float, x.split(",")),
        default="0,0,0,1",
        help="Set the background of the scene"
    )
    args = parser.parse_args(argv)
    print(args.background)
if __name__ == '__main__':
    # main(sys.argv[1:])
    # q=[1,2,3]
    # w=2
    # e=3
    # print(q)
    # aa = {"1":"222"}
    # aa["1"] = A()
    # print(aa["1"].config)
    # path_to_scene_layouts = [
    #     os.path.join("../3dfront/3dfront", f)
    #     for f in sorted(os.listdir('../3dfront/3dfront'))
    #     if f.endswith(".json")
    # ]
    # print(len(path_to_scene_layouts))
    
    # for char in enumerate(tqdm(path_to_scene_layouts)):
    #     print(char[1])
    # furniture_in_scene={}
    # furniture_in_scene["LIN"]= "qqq"
    # print(furniture_in_scene)
    # a = 1
    a = {i: [] for i in range(20)}
    b = [1,2,3]
    a[b].append(222)