import json
import os
import pickle
import _pickle as cPickle
from collections import defaultdict
from itertools import chain

import numpy as np
from tqdm import tqdm
import queue

from synthesis.datasets.Common import ModelInfo
from synthesis.datasets.FUTURE import FutureModel, FutureExtra, Door
from synthesis.datasets.Room import Room

def parse_threed_front_scenes(dataset_directory,
                              path_to_model_info,
                              path_to_models,
                              path_to_room_masks_dir=None,
                              debug=False):
    limit = 0
    if os.path.exists(f"../dump/threed_front.pkl"):
        print("loading scene cache...")
        scenes = cPickle.load(open(f"../dump/threed_front.pkl", "rb"))  #存在threed_front.pkl则直接加载，不在在则第一次创建
    else:
        # Parse model info at first time
        model_info = ModelInfo.load_file(path_to_model_info)  # 1. model_info为路径(path_to_model_info)下的ModelInfo类
        model_info = model_info.model_info    # 2. 此时的model_info为model_info ()函数传回来的    key为id，value为一个初始化的Furniture类？  #TODO：待验证

        path_to_scene_layouts = [        #6813个layout
            os.path.join(dataset_directory, f)
            for f in sorted(os.listdir(dataset_directory))
            if f.endswith(".json")
        ]

        scenes = []
        unique_room_ids = set()

        for i, m in enumerate(tqdm(path_to_scene_layouts)):
            with open(m) as f:
                data = json.load(f)
                # Parse the furniture of the scene
            furniture_in_scene = dict()
            
            for furniture in data["furniture"]:         
                if "valid" in furniture and furniture["valid"]:     #furniture_in_scene  是一个字典   {furniture["uid"]]  :   }
                    furniture_in_scene[furniture["uid"]] = dict(
                        model_uid=furniture["uid"],
                        model_jid=furniture["jid"],
                        model_info=model_info[furniture["jid"]]
                    )

            # Parse the extra meshes of the scene e.g walls, doors,
            # windows etc.
            meshes_in_scene = defaultdict()    # defaultdict()取不不在的值时不会报keyerror，返回默认值
            for mesh in data["mesh"]:          # TODO：meshes_in_scene代表的家具是墙？
                meshes_in_scene[mesh["uid"]] = dict(
                    mesh_uid=mesh["uid"],
                    mesh_jid=mesh["jid"],
                    mesh_xyz=np.asarray(mesh["xyz"]).reshape(-1, 3),
                    mesh_faces=np.asarray(mesh["faces"]).reshape(-1, 3),
                    mesh_type=mesh["type"]
                )

            # Parse the rooms of the scene
            scene = data["scene"]     #scene代表一整个场景，即一个完整的房子，有很多个room在里面
            # Keep track of the parsed rooms
            rooms = []

            for r in scene['room']:     #每个房间
                # Keep track of the furniture in the room
                furniture_in_room = []
                # Keep track of the extra meshes in the room
                extra_meshes_in_room = []
                # Keep track of the doors in the room
                # Flag to keep track of invalid scenes
                is_valid_scene = True

                for children in r["children"]:  #房间里的每个家具
                    if children["ref"] in furniture_in_scene:
                        furniture = furniture_in_scene[children["ref"]]   #furniture为字典，内容如下
                        # If scale is very small/big ignore this scene    # dict(model_uid=furniture["uid"],model_jid=furniture["jid"], model_info=model_info[furniture["jid"]])
                        if any(si < 1e-5 for si in children["scale"]):
                            is_valid_scene = False
                            break
                        if any(si > 5 for si in children["scale"]):
                            is_valid_scene = False
                            break
                        furniture_in_room.append(FutureModel(
                            furniture["model_uid"],
                            furniture["model_jid"],
                            children["instanceid"],
                            furniture["model_info"],
                            children["pos"],
                            children["rot"],
                            children["scale"],
                            path_to_models
                        ))
                    elif children["ref"] in meshes_in_scene:          #
                        mf = meshes_in_scene[children["ref"]]
                        extra_meshes_in_room.append(FutureExtra(    #TODO：FutureExtra类的作用
                            mf["mesh_uid"],
                            mf["mesh_jid"],
                            children["instanceid"],
                            mf["mesh_xyz"],
                            mf["mesh_faces"],
                            mf["mesh_type"],
                            children["pos"],
                            children["rot"],
                            children["scale"]
                        ))
                    else:
                        continue
                if len(furniture_in_room) > 1 and is_valid_scene:        #房间的家具数>1，且所有家具都是有效的
                    # Check whether a room with the same instanceid has
                    # already been added to the list of rooms
                    if r["instanceid"] not in unique_room_ids:
                        unique_room_ids.add(r["instanceid"])
                        # Add to the list
                        rooms.append(Room(                                #TODO：Room类的
                            r["instanceid"],  # scene_id
                            r["type"].lower(),  # scene_type
                            furniture_in_room,  # bounding boxes  房间里的所有家具
                            extra_meshes_in_room,  # extras e.g. walls,
                            m.split("/")[-1].split(".")[0],  # json_path
                            path_to_room_masks_dir
                        ))
            scenes.extend(rooms)    #所有场景的所有房间。注意extend与append的区别

        if not os.path.exists("../dump"):
            os.mkdir("../dump")
        print("Saving threed_front.pkl ...")
        cPickle.dump(scenes, open(f"../dump/threed_front.pkl", "wb"), True)
        print("Saved threed_front.pkl ...")
    return scenes     # #scenes是一个包含所有场景的所有房间的列表，不区别场景，即是一个由所有有效房间对应的Room类组成的列表


def parse_threed_future_models(
        dataset_directory, path_to_models, path_to_model_info
):
    if os.path.exists("../dump/threed_future_model.pkl"):
        furniture = cPickle.load(
            open(os.getenv("../dump/threed_future_model.pkl"), "rb")
        )
    else:
        # Parse the model info
        mf = ModelInfo.load_file(path_to_model_info)
        model_info = mf.model_info

        path_to_scene_layouts = [
            os.path.join(dataset_directory, f)
            for f in sorted(os.listdir(dataset_directory))
            if f.endswith(".json")
        ]
        # List to keep track of all available furniture in the dataset
        furniture = []
        unique_furniture_ids = set()

        # Start parsing the dataset
        print("Loading dataset ", end="")
        for i, m in enumerate(path_to_scene_layouts):
            with open(m) as f:
                data = json.load(f)
                # Parse the furniture of the scene
                furniture_in_scene = defaultdict()
                for ff in data["furniture"]:
                    if "valid" in ff and ff["valid"]:
                        furniture_in_scene[ff["uid"]] = dict(
                            model_uid=ff["uid"],
                            model_jid=ff["jid"],
                            model_info=model_info[ff["jid"]]
                        )
                # Parse the rooms of the scene
                scene = data["scene"]
                for rr in scene["room"]:
                    # Flag to keep track of invalid scenes
                    is_valid_scene = True
                    for cc in rr["children"]:
                        if cc["ref"] in furniture_in_scene:
                            tf = furniture_in_scene[cc["ref"]]
                            # If scale is very small/big ignore this scene
                            if any(si < 1e-5 for si in cc["scale"]):
                                is_valid_scene = False
                                break
                            if any(si > 5 for si in cc["scale"]):
                                is_valid_scene = False
                                break
                            if tf["model_uid"] not in unique_furniture_ids:
                                unique_furniture_ids.add(tf["model_uid"])
                                furniture.append(FutureModel(
                                    tf["model_uid"],
                                    tf["model_jid"],
                                    tf["model_info"],
                                    cc["pos"],
                                    cc["rot"],
                                    cc["scale"],
                                    path_to_models
                                ))
                        else:
                            continue
            s = "{:5d} / {:5d}".format(i, len(path_to_scene_layouts))
            print(s, flush=True, end="\b" * len(s))
        print()
        if not os.path.exists("../dump"):
            os.mkdir("./dump")

        cPickle.dump(furniture, open("../dump/threed_future_model.pkl", "wb"), True)

