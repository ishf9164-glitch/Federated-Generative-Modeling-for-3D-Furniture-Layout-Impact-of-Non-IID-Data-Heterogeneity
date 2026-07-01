import blenderproc as bproc

import os
import json
import argparse
import numpy as np
import imageio.v2 as imageio
import bpy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a single bedroom from 3D-FRONT by importing only that bedroom's furniture."
    )
    parser.add_argument("json_path", help="Path to one 3D-FRONT apartment json file")
    parser.add_argument("future_model_path", help="Path to 3D-FUTURE-model directory")
    parser.add_argument("output_dir", help="Output directory")

    parser.add_argument("--bedroom_index", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=1600)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--camera_mode", type=str, default="paper_view", choices=["paper_view", "topdown"])
    parser.add_argument("--margin", type=float, default=0.8)
    parser.add_argument("--wall_height", type=float, default=2.7)
    parser.add_argument("--wall_thickness", type=float, default=0.06)
    return parser.parse_args()


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_room_type(room_type):
    if room_type is None:
        return ""
    return str(room_type).strip().lower()


def find_bedrooms(front_json):
    bedrooms = []
    scenes = front_json.get("scene", [])
    if isinstance(scenes, dict):
        scenes = [scenes]

    for scene in scenes:
        rooms = scene.get("room", [])
        for room in rooms:
            rtype = normalize_room_type(room.get("type", ""))
            if rtype in ["bedroom", "masterbedroom", "secondbedroom"]:
                bedrooms.append(room)
    return bedrooms


def build_furniture_meta(front_json):
    meta = {}
    for item in front_json.get("furniture", []):
        uid = item.get("uid")
        if uid is not None:
            meta[uid] = item
    return meta


def quaternion_to_yaw_z_deg(q):
    if q is None or len(q) != 4:
        return 0.0
    x, y, z, w = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))
    return float(yaw)


def find_future_model_files(future_model_dir, model_jid):
    if model_jid is None:
        return None, None

    folder = os.path.join(future_model_dir, model_jid)
    if not os.path.isdir(folder):
        return None, None

    obj_candidates = [
        os.path.join(folder, "raw_model.obj"),
        os.path.join(folder, "normalized_model.obj"),
        os.path.join(folder, "model.obj"),
    ]
    obj_path = None
    for p in obj_candidates:
        if os.path.exists(p):
            obj_path = p
            break

    if obj_path is None:
        for fn in os.listdir(folder):
            if fn.lower().endswith(".obj"):
                obj_path = os.path.join(folder, fn)
                break

    tex_candidates = [
        os.path.join(folder, "texture.png"),
        os.path.join(folder, "texture.jpg"),
        os.path.join(folder, "albedo.png"),
        os.path.join(folder, "albedo.jpg"),
    ]
    tex_path = None
    for p in tex_candidates:
        if os.path.exists(p):
            tex_path = p
            break

    return obj_path, tex_path


def import_single_furniture(obj_path):
    existing = set(bpy.data.objects.keys())

    try:
        bpy.ops.wm.obj_import(filepath=obj_path)
    except Exception as e1:
        try:
            bpy.ops.import_scene.obj(filepath=obj_path)
        except Exception as e2:
            raise RuntimeError(
                f"OBJ 导入失败: {obj_path}\n"
                f"wm.obj_import error: {e1}\n"
                f"import_scene.obj error: {e2}"
            )

    new_objs = [obj for obj in bpy.data.objects if obj.name not in existing]
    mesh_objs = [obj for obj in new_objs if obj.type == "MESH"]
    return mesh_objs


def join_mesh_objects(mesh_objs, new_name):
    if len(mesh_objs) == 0:
        return None
    if len(mesh_objs) == 1:
        mesh_objs[0].name = new_name
        return mesh_objs[0]

    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = new_name
    return joined


def world_bbox_coords(obj):
    if obj.type != "MESH" or len(obj.data.vertices) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    coords = np.array([obj.matrix_world @ v.co for v in obj.data.vertices], dtype=np.float32)
    return coords


def ground_object_to_floor(obj, floor_z=0.0, offset=0.01):
    coords = world_bbox_coords(obj)
    if coords.shape[0] == 0:
        return
    min_z = float(coords[:, 2].min())
    obj.location.z += (floor_z - min_z + offset)
    bpy.context.view_layer.update()


def apply_texture_material(obj, texture_path, material_name_prefix="Mat"):
    """
    强制给 OBJ 重新绑定一个简单可靠的彩色贴图材质。
    """
    if texture_path is None or not os.path.exists(texture_path):
        return

    mat = bpy.data.materials.new(name=f"{material_name_prefix}_{obj.name}")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    tex_node = nodes.new(type="ShaderNodeTexImage")
    tex_node.location = (-400, 0)
    tex_node.image = bpy.data.images.load(texture_path, check_existing=True)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (-50, 0)
    bsdf.inputs["Roughness"].default_value = 0.75
    bsdf.inputs["Specular IOR Level"].default_value = 0.2

    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (250, 0)

    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def load_only_bedroom_furniture(front_json, room, future_model_dir):
    furniture_meta = build_furniture_meta(front_json)
    children = room.get("children", [])

    loaded = []
    failed = []

    for idx, child in enumerate(children):
        ref = child.get("ref")
        pos = child.get("pos", [0, 0, 0])
        rot = child.get("rot", [0, 0, 0, 1])
        scale = child.get("scale", [1, 1, 1])

        meta = furniture_meta.get(ref)
        if meta is None:
            continue

        model_jid = meta.get("jid")
        obj_path, tex_path = find_future_model_files(future_model_dir, model_jid)
        if obj_path is None:
            failed.append((ref, model_jid, "obj_not_found"))
            continue

        try:
            imported_meshes = import_single_furniture(obj_path)
            joined = join_mesh_objects(imported_meshes, f"{ref}_{model_jid}_{idx}")
            if joined is None:
                failed.append((ref, model_jid, "join_failed"))
                continue

            # JSON x,z -> Blender x,y ; JSON y -> Blender z
            joined.location = (float(pos[0]), float(pos[2]), float(pos[1]))
            joined.scale = (float(scale[0]), float(scale[2]), float(scale[1]))

            yaw_deg = quaternion_to_yaw_z_deg(rot)
            joined.rotation_euler[2] = np.radians(yaw_deg)

            bpy.context.view_layer.update()

            ground_object_to_floor(joined, floor_z=0.0, offset=0.01)
            apply_texture_material(joined, tex_path, material_name_prefix="FurnitureTex")

            loaded.append(joined)

        except Exception as e:
            failed.append((ref, model_jid, str(e)))

    print(f"Loaded bedroom furniture count: {len(loaded)}")
    if failed:
        print(f"Failed furniture count: {len(failed)}")
        print("Examples of failed furniture:", failed[:10])

    return loaded


def compute_loaded_scene_bbox(objs, margin=0.5):
    xs_min, xs_max = [], []
    ys_min, ys_max = [], []

    for obj in objs:
        coords = world_bbox_coords(obj)
        if coords.shape[0] == 0:
            continue
        xs_min.append(float(coords[:, 0].min()))
        xs_max.append(float(coords[:, 0].max()))
        ys_min.append(float(coords[:, 1].min()))
        ys_max.append(float(coords[:, 1].max()))

    if not xs_min:
        raise RuntimeError("没有任何成功加载的家具几何，无法计算场景范围。")

    min_x = min(xs_min) - margin
    max_x = max(xs_max) + margin
    min_y = min(ys_min) - margin
    max_y = max(ys_max) + margin

    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "center_x": (min_x + max_x) / 2.0,
        "center_y": (min_y + max_y) / 2.0,
        "span_x": max_x - min_x,
        "span_y": max_y - min_y,
    }


def create_material(name, base_color=(0.85, 0.85, 0.85, 1.0), roughness=0.9):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = base_color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Specular IOR Level"].default_value = 0.1
    return mat


def create_simple_floor(scene_bbox):
    cx = scene_bbox["center_x"]
    cy = scene_bbox["center_y"]
    sx = scene_bbox["span_x"]
    sy = scene_bbox["span_y"]

    bpy.ops.mesh.primitive_plane_add(location=(cx, cy, 0.0))
    floor = bpy.context.active_object
    floor.scale = (sx / 2.0, sy / 2.0, 1.0)
    floor.name = "BedroomFloor"
    floor.data.materials.append(
        create_material("FloorMat", base_color=(0.83, 0.83, 0.83, 1.0), roughness=0.95)
    )
    return floor


def create_simple_walls(scene_bbox, wall_height=2.7, wall_thickness=0.06):
    min_x = scene_bbox["min_x"]
    max_x = scene_bbox["max_x"]
    min_y = scene_bbox["min_y"]
    max_y = scene_bbox["max_y"]
    cx = scene_bbox["center_x"]
    cy = scene_bbox["center_y"]
    sx = scene_bbox["span_x"]
    sy = scene_bbox["span_y"]

    wall_mat = create_material("WallMat", base_color=(0.92, 0.92, 0.92, 1.0), roughness=0.98)

    walls = []

    # north wall
    bpy.ops.mesh.primitive_cube_add(location=(cx, max_y + wall_thickness / 2.0, wall_height / 2.0))
    w1 = bpy.context.active_object
    w1.scale = (sx / 2.0, wall_thickness / 2.0, wall_height / 2.0)
    w1.name = "Wall_North"
    w1.data.materials.append(wall_mat)
    walls.append(w1)

    # south wall
    bpy.ops.mesh.primitive_cube_add(location=(cx, min_y - wall_thickness / 2.0, wall_height / 2.0))
    w2 = bpy.context.active_object
    w2.scale = (sx / 2.0, wall_thickness / 2.0, wall_height / 2.0)
    w2.name = "Wall_South"
    w2.data.materials.append(wall_mat)
    walls.append(w2)

    # east wall
    bpy.ops.mesh.primitive_cube_add(location=(max_x + wall_thickness / 2.0, cy, wall_height / 2.0))
    w3 = bpy.context.active_object
    w3.scale = (wall_thickness / 2.0, sy / 2.0, wall_height / 2.0)
    w3.name = "Wall_East"
    w3.data.materials.append(wall_mat)
    walls.append(w3)

    # west wall
    bpy.ops.mesh.primitive_cube_add(location=(min_x - wall_thickness / 2.0, cy, wall_height / 2.0))
    w4 = bpy.context.active_object
    w4.scale = (wall_thickness / 2.0, sy / 2.0, wall_height / 2.0)
    w4.name = "Wall_West"
    w4.data.materials.append(wall_mat)
    walls.append(w4)

    return walls


def find_anchor_object(objs):
    bed_candidates = []
    all_candidates = []

    for obj in objs:
        if obj.type != "MESH":
            continue
        coords = world_bbox_coords(obj)
        if coords.shape[0] == 0:
            continue

        extent = coords.max(axis=0) - coords.min(axis=0)
        footprint = float(extent[0] * extent[1])
        all_candidates.append((footprint, obj, coords))

        if "bed" in obj.name.lower():
            bed_candidates.append((footprint, obj, coords))

    if bed_candidates:
        bed_candidates.sort(key=lambda x: x[0], reverse=True)
        return bed_candidates[0][1], bed_candidates[0][2]

    if all_candidates:
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        return all_candidates[0][1], all_candidates[0][2]

    return None, None


def setup_world_lighting():
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs[1].default_value = 0.6

    # 加一个 area light，让颜色更正常
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 3.8))
    light = bpy.context.active_object
    light.data.energy = 800
    light.data.shape = 'RECTANGLE'
    light.data.size = 4.5
    light.data.size_y = 4.5


def setup_topdown_camera(scene_bbox, resolution):
    cam_location = np.array([scene_bbox["center_x"], scene_bbox["center_y"], 10.0], dtype=np.float32)
    look_at = np.array([scene_bbox["center_x"], scene_bbox["center_y"], 0.0], dtype=np.float32)

    forward_vec = look_at - cam_location
    rotation_matrix = bproc.camera.rotation_from_forward_vec(forward_vec)
    cam2world_matrix = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    cam_obj = scene.camera
    cam_data = cam_obj.data
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = max(scene_bbox["span_x"], scene_bbox["span_y"]) * 1.05
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1000.0

    print("===== Topdown Camera Setup =====")
    print("camera location:", cam_location)
    print("ortho_scale:", cam_data.ortho_scale)


def setup_paper_view_camera(scene_bbox, loaded_objs, resolution):
    center_x = scene_bbox["center_x"]
    center_y = scene_bbox["center_y"]
    span_x = scene_bbox["span_x"]
    span_y = scene_bbox["span_y"]

    anchor_obj, anchor_bbox = find_anchor_object(loaded_objs)
    if anchor_bbox is not None:
        anchor_center = anchor_bbox.mean(axis=0)
        target_floor = 0.7 * anchor_center[:2] + 0.3 * np.array([center_x, center_y], dtype=np.float32)
        print("Anchor object:", anchor_obj.name)
        print("Anchor center:", anchor_center)
    else:
        target_floor = np.array([center_x, center_y], dtype=np.float32)
        print("[Warning] No anchor object found, fallback to scene center.")

    px = max(span_x * 0.22, 0.8)
    py = max(span_y * 0.22, 0.8)

    candidate_xy = [
        np.array([scene_bbox["min_x"] + px, scene_bbox["min_y"] + py], dtype=np.float32),
        np.array([scene_bbox["min_x"] + px, scene_bbox["max_y"] - py], dtype=np.float32),
        np.array([scene_bbox["max_x"] - px, scene_bbox["min_y"] + py], dtype=np.float32),
        np.array([scene_bbox["max_x"] - px, scene_bbox["max_y"] - py], dtype=np.float32),
    ]

    eye_height = 1.65
    target_dist = max(span_x, span_y) * 0.42

    best_xy = None
    best_score = 1e9
    for xy in candidate_xy:
        dist = np.linalg.norm(xy - target_floor)
        score = abs(dist - target_dist)
        if score < best_score:
            best_score = score
            best_xy = xy

    cam_location = np.array([best_xy[0], best_xy[1], eye_height], dtype=np.float32)
    look_at = np.array([target_floor[0], target_floor[1], 0.95], dtype=np.float32)

    forward_vec = look_at - cam_location
    rotation_matrix = bproc.camera.rotation_from_forward_vec(forward_vec)
    cam2world_matrix = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    cam_obj = scene.camera
    cam_data = cam_obj.data
    cam_data.type = "PERSP"
    cam_data.lens = 32
    cam_data.clip_start = 0.05
    cam_data.clip_end = 1000.0

    print("===== Paper-view Camera Setup =====")
    print("camera location:", cam_location)
    print("look_at:", look_at)
    print("lens:", cam_data.lens)


def setup_renderer(samples):
    try:
        bproc.renderer.set_max_amount_of_samples(samples)
        print(f"Renderer samples set by set_max_amount_of_samples({samples})")
        return
    except Exception:
        pass

    try:
        bproc.renderer.set_samples(samples)
        print(f"Renderer samples set by set_samples({samples})")
        return
    except Exception:
        pass

    try:
        bpy.context.scene.cycles.samples = samples
        print(f"Renderer samples set by bpy.context.scene.cycles.samples = {samples}")
        return
    except Exception:
        pass

    print("[Warning] Failed to set renderer samples, using default settings.")


def save_outputs(output_dir, data, camera_mode):
    os.makedirs(output_dir, exist_ok=True)
    bproc.writer.write_hdf5(output_dir, data)

    img = data["colors"][0]
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    filename = "bedroom_topdown.png" if camera_mode == "topdown" else "bedroom_paper_view.png"
    path = os.path.join(output_dir, filename)
    imageio.imwrite(path, img)
    print(f"Saved PNG to: {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    bproc.init()

    front_json = load_json(args.json_path)
    bedrooms = find_bedrooms(front_json)

    if len(bedrooms) == 0:
        raise RuntimeError("No bedroom found in this apartment JSON.")
    if args.bedroom_index >= len(bedrooms):
        raise RuntimeError(f"bedroom_index={args.bedroom_index} out of range, only found {len(bedrooms)} bedrooms.")

    target_room = bedrooms[args.bedroom_index]

    print("===== Selected Bedroom =====")
    print("room type:", target_room.get("type"))
    print("room instanceid:", target_room.get("instanceid"))
    print("camera_mode:", args.camera_mode)

    loaded_objs = load_only_bedroom_furniture(front_json, target_room, args.future_model_path)
    if len(loaded_objs) == 0:
        raise RuntimeError("没有成功加载任何 bedroom 家具，无法渲染。")

    scene_bbox = compute_loaded_scene_bbox(loaded_objs, margin=args.margin)
    print("scene_bbox_from_loaded_objects:", scene_bbox)

    create_simple_floor(scene_bbox)
    create_simple_walls(scene_bbox, wall_height=args.wall_height, wall_thickness=args.wall_thickness)

    setup_world_lighting()

    if args.camera_mode == "topdown":
        setup_topdown_camera(scene_bbox, args.resolution)
    else:
        setup_paper_view_camera(scene_bbox, loaded_objs, args.resolution)

    setup_renderer(args.samples)

    data = bproc.renderer.render()
    save_outputs(args.output_dir, data, args.camera_mode)

    print("Render finished.")


if __name__ == "__main__":
    main()