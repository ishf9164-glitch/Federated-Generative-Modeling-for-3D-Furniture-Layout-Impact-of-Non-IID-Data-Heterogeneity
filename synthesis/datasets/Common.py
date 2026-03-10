import json
import os
import sys
from collections import Counter
from typing import TypeVar, Any, Optional, List, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset

# 修复核心报错：在 PyTorch 新版本中，T_co 不再直接从 dataset 暴露或更名为 _T_co
# 使用 typing 自定义兼容
T_co = TypeVar('T_co', covariant=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthesis.datasets import THREED_FRONT_FURNITURE, CSVSplitsBuilder
from synthesis.datasets.FUTURE import Furniture


class InfiniteDataset(IterableDataset):
    """
    Decorate any Dataset instance to provide an infinite IterableDataset version of it.
    """
    def __init__(self, dataset: Dataset, shuffle: bool = True):
        super().__init__()
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self):
        N = len(self.dataset)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = 0
            end = N
        else:
            num_workers = worker_info.num_workers
            per_worker = (N + num_workers - 1) // num_workers
            start = worker_info.id * per_worker
            end = min(start + per_worker, N)

        indices = np.arange(start, end)
        while True:
            if self.shuffle:
                np.random.shuffle(indices)
            for i in indices:
                yield self.dataset[i]

    # Python 3.12 建议显式定义类型，即使 pass 也要符合接口
    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]


class ModelInfo(object):
    def __init__(self, model_info_data):
        self.model_info_data = model_info_data
        self._model_info = None
        self._styles = []
        self._themes = []
        self._categories = []
        self._super_categories = []
        self._materials = []

    @property
    def model_info(self):    
        if self._model_info is None:
            self._model_info = {}
            for m in self.model_info_data:
                # 收集属性
                for key, attr_list in [
                    ("style", self._styles),
                    ("theme", self._themes),
                    ("super-category", self._super_categories),
                    ("category", self._categories),
                    ("material", self._materials)
                ]:
                    val = m.get(key)
                    if val is not None and val not in attr_list:
                        attr_list.append(val)

                super_cat = "unknown_super-category"
                if m.get("super-category") is not None:
                    super_cat = m["super-category"].lower().replace(" / ", "/")

                cat = "unknown"
                if m.get("category") is not None:
                    cat = m["category"].lower().replace(" / ", "/")
                
                # 改名逻辑映射
                cat = THREED_FRONT_FURNITURE.get(cat, cat) 

                self._model_info[m["model_id"]] = Furniture(
                    super_cat,
                    cat,
                    m.get("style"),
                    m.get("theme"),
                    m.get("material")
                )
        return self._model_info

    @property
    def styles(self): return self._styles

    @property
    def themes(self): return self._themes

    @property
    def materials(self): return self._materials

    @property
    def categories(self):
        return set([s.lower().replace(" / ", "/") for s in self._categories])

    @property
    def super_categories(self):
        return set([s.lower().replace(" / ", "/") for s in self._super_categories])

    @classmethod
    def load_file(cls, path_to_model_info):
        with open(path_to_model_info, "r", encoding='utf-8') as f: # Python 3.12 推荐显式编码
            model_info = json.load(f)
        return cls(model_info)


class BaseDataset(Dataset):
    def __init__(self, scenes):
        assert len(scenes) > 0
        self.scenes = scenes
        self._class_labels = {}

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        return self.scenes[idx]

    @property
    def num_class(self):
        return len(self._class_labels)

    @staticmethod
    def with_room(scene_type):
        # 修正逻辑：判断 scene_type 是否包含在场景类型字符串中
        def inner(scene):
            return scene if scene_type.lower() in scene.scene_type.lower() else False
        return inner

    @staticmethod
    def with_scene_ids(scene_ids):
        def inner(scene):
            return scene if scene.scene_id in scene_ids else False
        return inner

    @staticmethod
    def with_object_types(objects):
        def inner(scene):
            return scene if all(b.label in objects for b in scene.bboxes) else False
        return inner

    @staticmethod
    def contains_object_types(objects):
        def inner(scene):
            return scene if any(b.label in objects for b in scene.bboxes) else False
        return inner

    @staticmethod
    def with_generic_classes(box_types_map):
        def inner(scene):
            for box in scene.bboxes:
                box.label = box_types_map.get(box.label, box.label)
            return scene
        return inner

    @staticmethod
    def without_object_types(objects):
        def inner(scene):
            return False if any(b.label in objects for b in scene.bboxes) else scene
        return inner

    @staticmethod
    def without_box_types(box_types):
        def inner(scene):
            # 倒序删除以避免索引偏移问题
            for i in range(len(scene.bboxes) - 1, -1, -1):
                if scene.bboxes[i].label in box_types:
                    scene.bboxes.pop(i)
            return scene
        return inner

    @staticmethod
    def room_smaller_than_along_axis(max_size, axis=1):
        def inner(scene):
            return scene if scene.bbox[1][axis] <= max_size else False
        return inner

    @staticmethod
    def room_larger_than_along_axis(min_size, axis=1):
        def inner(scene):
            return scene if scene.bbox[0][axis] >= min_size else False
        return inner

    @staticmethod
    def at_least_boxes(n):
        def inner(scene):
            return scene if len(scene.bboxes) >= n else False
        return inner

    @staticmethod
    def at_most_boxes(n):
        def inner(scene):
            return scene if len(scene.bboxes) <= n else False
        return inner

    @staticmethod
    def with_valid_scene_ids(invalid_scene_ids):
        def inner(scene):
            return scene if scene.scene_id not in invalid_scene_ids else False
        return inner

    @staticmethod
    def with_valid_bbox_jids(invalid_bbox_jds):
        def inner(scene):
            return False if any(b.model_jid in invalid_bbox_jds for b in scene.bboxes) else scene
        return inner

    @staticmethod
    def with_valid_boxes(box_types):
        def inner(scene):
            for i in range(len(scene.bboxes) - 1, -1, -1):
                if scene.bboxes[i].label not in box_types:
                    scene.bboxes.pop(i)
            return scene
        return inner

    @staticmethod
    def floor_plan_with_limits(limit_x, limit_y, axis=None):
        if axis is None: axis = [0, 2]
        def inner(scene):
            min_bbox, max_bbox = scene.floor_plan_bbox
            t_x = max_bbox[axis[0]] - min_bbox[axis[0]]
            t_y = max_bbox[axis[1]] - min_bbox[axis[1]]
            return scene if t_x <= limit_x and t_y <= limit_y else False
        return inner

    @staticmethod
    def filter_compose(*filters):
        def inner(scene):
            s = scene
            for f in filters:
                if not s: break
                s = f(s)
            return s
        return inner


def filter_function(config, split=None, without_lamps=False):
    if split is None:
        split = ["train", "val"]
    
    print(f"Applying {config['room_type_filter']} filtering")
    
    if config["room_type_filter"] == "no_filtering":
        return lambda s: s

    # 统一使用 UTF-8 读取
    with open(config["path_to_invalid_scene_ids"], "r", encoding='utf-8') as f:
        invalid_scene_ids = set(l.strip() for l in f)

    with open(config["path_to_invalid_bbox_jids"], "r", encoding='utf-8') as f:
        invalid_bbox_jids = set(l.strip() for l in f)

    splits_builder = CSVSplitsBuilder(config["annotation_file"])
    split_scene_ids = splits_builder.get_splits(split)

    # 常用过滤器组合
    common_filters = [
        BaseDataset.at_least_boxes(3),
        BaseDataset.with_object_types(list(THREED_FRONT_FURNITURE.values())),
        BaseDataset.with_valid_scene_ids(invalid_scene_ids),
        BaseDataset.with_valid_bbox_jids(invalid_bbox_jids),
        BaseDataset.room_larger_than_along_axis(-0.005, axis=1),
        BaseDataset.with_scene_ids(split_scene_ids)
    ]

    room_filter = config["room_type_filter"]
    
    if "bedroom" in room_filter:
        return BaseDataset.filter_compose(
            BaseDataset.with_room("bed"),
            BaseDataset.at_most_boxes(13),
            BaseDataset.contains_object_types(["bed"]),
            BaseDataset.room_smaller_than_along_axis(6.0, axis=1),
            BaseDataset.floor_plan_with_limits(8, 8),
            BaseDataset.without_box_types(["lamp"] if without_lamps else [""]),
            *common_filters
        )
    elif "livingroom" in room_filter:
        return BaseDataset.filter_compose(
            BaseDataset.with_room("living"),
            BaseDataset.at_most_boxes(21),
            BaseDataset.room_smaller_than_along_axis(4.0, axis=1),
            BaseDataset.floor_plan_with_limits(12, 12),
            BaseDataset.without_box_types(["lamp", "sofa"] if without_lamps else [""]),
            BaseDataset.without_box_types(["bed"]),
            *common_filters
        )
    elif "diningroom" in room_filter:
        return BaseDataset.filter_compose(
            BaseDataset.with_room("dining"),
            BaseDataset.at_most_boxes(21),
            BaseDataset.room_smaller_than_along_axis(4.0, axis=1),
            BaseDataset.floor_plan_with_limits(12, 12),
            BaseDataset.contains_object_types(["dining_chair", "dining_table"]),
            BaseDataset.without_box_types(["lamp", "dining_table"] if without_lamps else [""]),
            BaseDataset.without_box_types(["bed"]),
            *common_filters
        )
    elif "library" in room_filter:
        return BaseDataset.filter_compose(
            BaseDataset.with_room("library"),
            BaseDataset.floor_plan_with_limits(6, 6),
            BaseDataset.contains_object_types(["shelf"]),
            BaseDataset.without_box_types(["lamp"] if without_lamps else [""]),
            BaseDataset.without_box_types(["bed"]),
            *common_filters
        )
    else:
        return lambda s: s if len(s.bboxes) > 0 else False