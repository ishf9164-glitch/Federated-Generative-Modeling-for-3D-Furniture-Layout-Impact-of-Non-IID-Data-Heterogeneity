# synthesis/datasets/FutureDataset.py
import os
import sys
import pickle
from typing import List, Optional, Sequence

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthesis.datasets.Utils import parse_threed_future_models


class FutureDataset(object):
    """
    Robust FutureDataset:
    - 预过滤缺失模型文件（bbox_vertices.npy / raw_model.obj）
    - label 下无候选 -> 回退到全体可用对象，避免 IndexError
    - 支持从 pickle 反序列化后自动补齐 cache 字段，避免 AttributeError
    """

    def __init__(self, objects):
        assert len(objects) > 0
        self.objects = objects
        self._ensure_caches()

    # --------- pickle safety ----------
    def __setstate__(self, state):
        # pickle load 时会调用：把旧对象 state 装回来，然后补齐新字段
        self.__dict__.update(state)
        self._ensure_caches()

    def _ensure_caches(self):
        # 兼容：旧版 pickle 对象没有这些字段
        if not hasattr(self, "_valid_objects_cache") or self._valid_objects_cache is None:
            self._valid_objects_cache = {}
        if not hasattr(self, "_valid_objects_cache_by_label") or self._valid_objects_cache_by_label is None:
            self._valid_objects_cache_by_label = {}

    # ---------------------------------
    def __len__(self):
        return len(self.objects)

    def __str__(self):
        return "Dataset contains {} objects".format(len(self))

    def __getitem__(self, idx):
        return self.objects[idx]

    @property
    def labels(self):
        return list(set([oi.label for oi in self.objects]))

    def _filter_objects_by_label(self, label):
        return [oi for oi in self.objects if oi.label == label]

    # ----------------------------
    # Validity filtering / caching
    # ----------------------------
    def _exists_any_geom(self, oi) -> bool:
        p_bbox = getattr(oi, "path_to_bbox_vertices", None)
        p_raw = getattr(oi, "raw_model_path", None)
        ok_bbox = bool(p_bbox) and os.path.exists(p_bbox)
        ok_raw = bool(p_raw) and os.path.exists(p_raw)
        return ok_bbox or ok_raw

    def _get_valid_objects(self, invalid_list: Sequence[str]) -> List[object]:
        self._ensure_caches()

        key = tuple(sorted(invalid_list)) if invalid_list is not None else tuple()
        if key in self._valid_objects_cache:
            return self._valid_objects_cache[key]

        valid = []
        invalid_set = set(invalid_list) if invalid_list is not None else set()

        for oi in self.objects:
            jid = getattr(oi, "model_jid", None)
            if jid in invalid_set:
                continue
            if not self._exists_any_geom(oi):
                continue
            valid.append(oi)

        self._valid_objects_cache[key] = valid
        # 每个 invalid_list key 对应一份 label cache
        self._valid_objects_cache_by_label[key] = {}
        return valid

    def _get_valid_objects_by_label(self, label: str, invalid_list: Sequence[str]) -> List[object]:
        self._ensure_caches()

        key = tuple(sorted(invalid_list)) if invalid_list is not None else tuple()
        if key not in self._valid_objects_cache:
            self._get_valid_objects(invalid_list)

        by_label = self._valid_objects_cache_by_label.get(key)
        if by_label is None:
            by_label = {}
            self._valid_objects_cache_by_label[key] = by_label

        if label in by_label:
            return by_label[label]

        candidates = [oi for oi in self._valid_objects_cache[key] if oi.label == label]
        by_label[label] = candidates
        return candidates

    # ----------------------------
    # Nearest furniture selection
    # ----------------------------
    def _mse3(self, oi, query_size_xyz: np.ndarray) -> Optional[float]:
        try:
            scale = oi.size * oi.scale * 2
            size = np.array([scale[2], scale[0], scale[1]], dtype=np.float32)
            return float(np.sum((size - query_size_xyz) ** 2, axis=-1))
        except (FileNotFoundError, ValueError, OSError, Exception):
            return None

    def get_closest_furniture_to_box(self, query_label, query_size, invalid_list):
        self._ensure_caches()

        query_size = np.asarray(query_size, dtype=np.float32)

        candidates = self._get_valid_objects_by_label(query_label, invalid_list)
        if not candidates:
            candidates = self._get_valid_objects(invalid_list)

        if not candidates:
            raise RuntimeError(
                "No valid 3D-FUTURE objects found. Check --threed_future_dataset_directory / "
                "--furniture_path / model file structure."
            )

        best_oi = None
        best_mse = None
        for oi in candidates:
            mse = self._mse3(oi, query_size)
            if mse is None:
                continue
            if best_mse is None or mse < best_mse:
                best_mse = mse
                best_oi = oi

        if best_oi is None:
            return candidates[0]
        return best_oi

    def get_closest_furniture_to_2dbox(self, query_label, query_size):
        query_size = np.asarray(query_size, dtype=np.float32)

        candidates = [oi for oi in self.objects if oi.label == query_label and self._exists_any_geom(oi)]
        if not candidates:
            candidates = [oi for oi in self.objects if self._exists_any_geom(oi)]
        if not candidates:
            raise RuntimeError("No valid 3D-FUTURE objects found (2D).")

        best_oi = None
        best_mse = None
        for oi in candidates:
            try:
                mse = float((oi.size[0] - query_size[0]) ** 2 + (oi.size[2] - query_size[1]) ** 2)
            except Exception:
                continue
            if best_mse is None or mse < best_mse:
                best_mse = mse
                best_oi = oi

        if best_oi is None:
            return candidates[0]
        return best_oi

    # ----------------------------
    # Builders
    # ----------------------------
    @classmethod
    def from_dataset_directory(cls, dataset_directory, path_to_model_info, path_to_models):
        objects = parse_threed_future_models(dataset_directory, path_to_models, path_to_model_info)
        return cls(objects)

    @classmethod
    def from_pickled_dataset(cls, path_to_pickled_dataset):
        with open(path_to_pickled_dataset, "rb") as f:
            dataset = pickle.load(f)

        # 关键：pickle 出来的可能是 FutureDataset 实例，但没有新字段；补齐
        if isinstance(dataset, cls):
            dataset._ensure_caches()
            return dataset

        # 兜底：如果 pickle 里存的是 objects 列表
        if hasattr(dataset, "__len__") and not hasattr(dataset, "get_closest_furniture_to_box"):
            return cls(dataset)

        return dataset
