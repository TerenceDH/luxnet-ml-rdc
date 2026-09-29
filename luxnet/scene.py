"""Scene encoder and per-luminaire inference extracted from the annual workflow."""
import math
import numpy as np
from .model import load_model
DEFAULT_G_CKPT_PATH = None
DEFAULT_TRAIN_GLOBAL_MM2PX = 0.020747
DEFAULT_AUTO_COMPUTE_SCALE_FOR_SINGLE_ROOM = False
DEFAULT_IMG_SIZE = 128
DEFAULT_UNIFORM_SCALE_MARGIN = 0.02
DEFAULT_WALL_BAND_PX = 4
DEFAULT_LAMP_POINT_RADIUS = 2
def build_generator(cond_dim, g_ckpt_path):
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, mean, std = load_model(g_ckpt_path, device)
    return model, mean, std, device
def parse_pair(line):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 2:
        raise ValueError("不是合法的 x,y 行: {}".format(line))
    return float(parts[0]), float(parts[1])


def parse_triplet(line):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        raise ValueError("不是合法的 x,y,z 行: {}".format(line))
    return float(parts[0]), float(parts[1]), float(parts[2])


def extract_rho_from_material_name(name):
    try:
        return float(name.rsplit("_", 1)[1])
    except Exception:
        raise ValueError("材质名无法解析反射率: {}".format(name))


def is_float_str(s):
    try:
        float(s)
        return True
    except Exception:
        return False


def is_triplet_line(line):
    parts = [p.strip() for p in line.split(",")]
    return len(parts) == 3 and all(is_float_str(p) for p in parts)


def to_numpy_2d(arr, shape_last=None, name="array"):
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim < 2:
        raise ValueError(f"{name} 至少应为 2 维，实际 shape={out.shape}")
    if shape_last is not None and out.shape[-1] != shape_last:
        raise ValueError(f"{name} 最后一维应为 {shape_last}，实际为 {out.shape}")
    return out


def geom_lib():
    from .encoding import (
        get_room_size_from_vertices,
        get_scale_and_offset_for_size,
        world_to_pixel_xy_up_uniform,
        up_to_down_y,
        polygon_mask_from_vertices_up,
        make_wall_mask_and_segidx_from_poly,
        build_window_mask_on_left_wall_from_rects,
        draw_lamps_hard_dots,
    )
    return (
        get_room_size_from_vertices,
        get_scale_and_offset_for_size,
        world_to_pixel_xy_up_uniform,
        up_to_down_y,
        polygon_mask_from_vertices_up,
        make_wall_mask_and_segidx_from_poly,
        build_window_mask_on_left_wall_from_rects,
        draw_lamps_hard_dots,
    )


def parse_combine_txt(combine_path):
    get_room_size_from_vertices, _, _, _, _, _, _, _ = geom_lib()

    with open(combine_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 16:
        raise ValueError("combine.txt 行数过少: {}".format(combine_path))

    sample_name = lines[0]

    room_vertices = np.array([parse_pair(lines[i]) for i in range(1, 5)], dtype=np.float32)
    H = float(lines[5])

    materials = lines[6:13]
    if len(materials) != 7:
        raise ValueError("combine.txt 材质行数量不对: {}".format(combine_path))

    # 顺序固定：floor, left, bottom, right, top, ceiling, window
    floor_rho = extract_rho_from_material_name(materials[0])
    left_rho = extract_rho_from_material_name(materials[1])
    bottom_rho = extract_rho_from_material_name(materials[2])
    right_rho = extract_rho_from_material_name(materials[3])
    top_rho = extract_rho_from_material_name(materials[4])
    ceiling_rho = extract_rho_from_material_name(materials[5])
    window_rho = extract_rho_from_material_name(materials[6])

    # 与训练输入保持一致：bottom, right, top, left
    wall_rhos = np.array([bottom_rho, right_rho, top_rho, left_rho], dtype=np.float32)
    surface_rhos = np.array([floor_rho, ceiling_rho], dtype=np.float32)

    n_win = int(float(lines[13]))

    idx = 14
    anchor_lines = []
    while idx < len(lines) and is_triplet_line(lines[idx]):
        anchor_lines.append(parse_triplet(lines[idx]))
        idx += 1
        if idx < len(lines) and is_float_str(lines[idx]):
            break

    if idx >= len(lines):
        raise ValueError("combine.txt 缺少 win_w / win_h: {}".format(combine_path))

    win_w = float(lines[idx])
    idx += 1

    if idx >= len(lines):
        raise ValueError("combine.txt 缺少 win_h: {}".format(combine_path))

    win_h = float(lines[idx])
    idx += 1

    lamp_positions = []
    while idx < len(lines):
        lamp_positions.append(parse_triplet(lines[idx]))
        idx += 1

    if len(lamp_positions) == 0:
        raise ValueError("combine.txt 未解析到任何灯具坐标: {}".format(combine_path))

    lamp_positions = np.array(lamp_positions, dtype=np.float32)

    # anchor: x0,y0,z0；这里窗位于左墙，所以有效的是 y0,z0
    window_rects = []
    valid_count = 0
    for _x0, y0, z0 in anchor_lines:
        if y0 >= 0 and z0 >= 0 and valid_count < n_win:
            window_rects.append([y0, z0, win_w, win_h])
            valid_count += 1
        else:
            window_rects.append([-1.0, -1.0, -1.0, -1.0])

    if valid_count < n_win:
        raise ValueError("combine.txt 有效窗锚点少于 n_win: {}".format(combine_path))

    window_rects = np.asarray(window_rects, dtype=np.float32)

    L, W = get_room_size_from_vertices(room_vertices)
    room_dims = np.array([L, W, H], dtype=np.float32)

    valid_rects = [r for r in window_rects if r[2] > 0 and r[3] > 0]
    if len(valid_rects) == 0:
        raise ValueError("combine.txt 未解析到有效窗: {}".format(combine_path))

    sill_h = float(valid_rects[0][1])

    return {
        "sample_name": sample_name,
        "room_vertices_mm": room_vertices,
        "room_dims_mm": room_dims,
        "lamp_positions_mm": lamp_positions,
        "wall_rhos": wall_rhos,
        "surface_rhos": surface_rhos,
        "window_rects_mm": window_rects,
        "win_h_mm": float(win_h),
        "sill_h_mm": float(sill_h),
        "window_rho": float(window_rho),
        "win_w_mm": float(win_w),
        "n_win": int(n_win),
    }


def compute_single_room_mm2px_from_vertices(room_vertices, img_size, margin):
    get_room_size_from_vertices, _, _, _, _, _, _, _ = geom_lib()
    L, W = get_room_size_from_vertices(room_vertices)
    s_nominal = min(
        (img_size - 1) / max(L, 1e-9),
        (img_size - 1) / max(W, 1e-9),
    )
    return s_nominal * (1.0 - float(margin))


def build_A_float(
    verts_xy,
    lamp_xyz_list,
    window_rects_i,
    img_size,
    s_global_base,
    wall_band_px,
    lamp_point_radius,
):
    (
        get_room_size_from_vertices,
        get_scale_and_offset_for_size,
        world_to_pixel_xy_up_uniform,
        up_to_down_y,
        polygon_mask_from_vertices_up,
        make_wall_mask_and_segidx_from_poly,
        build_window_mask_on_left_wall_from_rects,
        draw_lamps_hard_dots,
    ) = geom_lib()

    from PIL import Image

    verts_xy = np.asarray(verts_xy, dtype=float)
    lamp_xyz_list = np.asarray(lamp_xyz_list, dtype=float).reshape(-1, 3)

    L, W = get_room_size_from_vertices(verts_xy)
    s_px, off_x, off_y = get_scale_and_offset_for_size(L, W, img_size, s_global_base)

    verts_up = world_to_pixel_xy_up_uniform(verts_xy, s_px, off_x, off_y)
    poly_mask_up = polygon_mask_from_vertices_up(img_size, verts_up)

    _, seg_idx_map = make_wall_mask_and_segidx_from_poly(
        verts_up,
        poly_mask_up,
        band_px=wall_band_px,
    )

    window_mask_up = build_window_mask_on_left_wall_from_rects(
        seg_idx_map,
        s_px,
        off_y,
        window_rects_i,
    )

    A_room = np.flipud(poly_mask_up.astype(np.float32))
    A_window = np.flipud(window_mask_up.astype(np.float32))

    lamp_xy = lamp_xyz_list[:, :2]
    lamps_up = world_to_pixel_xy_up_uniform(lamp_xy, s_px, off_x, off_y)
    lamps_down = np.stack(
        [lamps_up[:, 0], up_to_down_y(lamps_up[:, 1], img_size)],
        axis=1,
    )

    img_lamp = Image.new("L", (img_size, img_size), color=0)
    draw_lamps_hard_dots(img_lamp, lamps_down, radius=lamp_point_radius)
    A_lamp = np.asarray(img_lamp, dtype=np.float32) / 255.0

    return np.stack([A_lamp, A_room, A_window], axis=0).astype(np.float32)


def build_cond_vec_for_one_lamp(
    verts_xy,
    room_dims_3,
    lamp_xyz,
    wall_rhos_4,
    surface_rhos_2,
    win_h_mm,
    sill_h_mm,
    window_rho,
):
    get_room_size_from_vertices, _, _, _, _, _, _, _ = geom_lib()

    verts_xy = np.asarray(verts_xy, dtype=float)
    room_dims_3 = np.asarray(room_dims_3, dtype=float).ravel()
    lamp_xyz = np.asarray(lamp_xyz, dtype=float).ravel()
    wall_rhos_4 = np.asarray(wall_rhos_4, dtype=float).ravel()
    surface_rhos_2 = np.asarray(surface_rhos_2, dtype=float).ravel()

    L_mm_v, W_mm_v = get_room_size_from_vertices(verts_xy)
    L_mm, W_mm, H_mm = [float(x) for x in room_dims_3]

    if abs(L_mm - L_mm_v) > 1e-4 or abs(W_mm - W_mm_v) > 1e-4:
        raise ValueError(
            f"room_dims 与 room_vertices 不一致："
            f"room_dims=({L_mm},{W_mm}), verts=({L_mm_v},{W_mm_v})"
        )

    lamp_z_m = float(lamp_xyz[2]) / 1000.0
    rho_bottom, rho_right, rho_top, rho_left = [float(x) for x in wall_rhos_4.tolist()]
    floor_rho = float(surface_rhos_2[0])
    ceil_rho = float(surface_rhos_2[1])

    room_h_m = float(H_mm / 1000.0)
    sill_h_m = float(sill_h_mm / 1000.0)
    win_h_m = float(win_h_mm / 1000.0)

    return np.array([
        room_h_m,
        lamp_z_m,
        sill_h_m,
        win_h_m,
        rho_bottom,
        rho_right,
        rho_top,
        rho_left,
        floor_rho,
        ceil_rho,
        float(window_rho),
    ], dtype=np.float32)


def forward_once(G, A_np, cond_np, device):
    import torch

    A_t = torch.from_numpy(np.clip(A_np, 0.0, 1.0)).unsqueeze(0).to(device)
    cond_t = torch.from_numpy(cond_np.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        fake_t = G(A_t, cond_t)

    fake_lux = fake_t[0, 0].detach().cpu().numpy()
    return np.clip(fake_lux, 0.0, None).astype(np.float32)


def map_world_pts_to_pixel_indices(
    room_vertices_mm,
    pts_xyz_mm,
    img_size,
    s_global_base,
):
    (
        _get_room_size_from_vertices,
        get_scale_and_offset_for_size,
        world_to_pixel_xy_up_uniform,
        _up_to_down_y,
        polygon_mask_from_vertices_up,
        _make_wall_mask_and_segidx_from_poly,
        _build_window_mask_on_left_wall_from_rects,
        _draw_lamps_hard_dots,
    ) = geom_lib()

    room_vertices_mm = np.asarray(room_vertices_mm, dtype=np.float32)
    pts_xyz_mm = np.asarray(pts_xyz_mm, dtype=np.float32)

    L, W = _get_room_size_from_vertices(room_vertices_mm)
    s_px, off_x, off_y = get_scale_and_offset_for_size(L, W, img_size, s_global_base)

    xy_mm = pts_xyz_mm[:, :2]
    pts_up = world_to_pixel_xy_up_uniform(xy_mm, s_px, off_x, off_y)

    x_px = np.rint(pts_up[:, 0]).astype(np.int32)
    y_up = np.rint(pts_up[:, 1]).astype(np.int32)
    y_px = (img_size - 1 - y_up).astype(np.int32)

    verts_up = world_to_pixel_xy_up_uniform(room_vertices_mm, s_px, off_x, off_y)
    poly_mask_up = polygon_mask_from_vertices_up(img_size, verts_up)
    room_mask_down = np.flipud(poly_mask_up.astype(bool))

    valid_mask = (
        (x_px >= 0) & (x_px < img_size) &
        (y_px >= 0) & (y_px < img_size)
    )

    valid_ids = np.where(valid_mask)[0]
    valid_mask2 = np.zeros_like(valid_mask, dtype=bool)

    if valid_ids.size > 0:
        valid_mask2[valid_ids] = room_mask_down[y_px[valid_ids], x_px[valid_ids]]

    valid_mask = valid_mask & valid_mask2
    return x_px, y_px, valid_mask


def sample_lux_map_at_points(lux_map_hw, x_px, y_px):
    return lux_map_hw[y_px, x_px].astype(np.float32)


def infer_lamp_arrays_at_points(
    room_vertices_mm,
    room_dims_mm,
    lamp_positions_mm,
    wall_rhos,
    surface_rhos,
    window_rects_mm,
    win_h_mm,
    sill_h_mm,
    window_rho,
    pts_xyz_mm,
    g_ckpt_path=DEFAULT_G_CKPT_PATH,
    train_global_mm2px=DEFAULT_TRAIN_GLOBAL_MM2PX,
    auto_compute_scale_for_single_room=DEFAULT_AUTO_COMPUTE_SCALE_FOR_SINGLE_ROOM,
    img_size=DEFAULT_IMG_SIZE,
    uniform_scale_margin=DEFAULT_UNIFORM_SCALE_MARGIN,
    wall_band_px=DEFAULT_WALL_BAND_PX,
    lamp_point_radius=DEFAULT_LAMP_POINT_RADIUS,
):
    room_vertices = to_numpy_2d(room_vertices_mm, shape_last=2, name="room_vertices_mm")
    room_dims = np.asarray(room_dims_mm, dtype=np.float32).ravel()
    lamp_positions = to_numpy_2d(lamp_positions_mm, shape_last=3, name="lamp_positions_mm")
    wall_rhos = np.asarray(wall_rhos, dtype=np.float32).ravel()
    surface_rhos = np.asarray(surface_rhos, dtype=np.float32).ravel()
    window_rects = to_numpy_2d(window_rects_mm, shape_last=4, name="window_rects_mm")

    if auto_compute_scale_for_single_room:
        s_global_mm2px = compute_single_room_mm2px_from_vertices(
            room_vertices,
            img_size=img_size,
            margin=uniform_scale_margin,
        )
    else:
        s_global_mm2px = float(train_global_mm2px)
        if not math.isfinite(s_global_mm2px) or s_global_mm2px <= 0:
            raise ValueError("train_global_mm2px 必须是正数")

    example_cond = build_cond_vec_for_one_lamp(
        room_vertices,
        room_dims,
        lamp_positions[0],
        wall_rhos,
        surface_rhos,
        win_h_mm,
        sill_h_mm,
        window_rho,
    )
    cond_dim = int(example_cond.shape[0])

    G, cond_mean, cond_std, device = build_generator(
        cond_dim=cond_dim,
        g_ckpt_path=g_ckpt_path,
    )

    x_px, y_px, valid_mask = map_world_pts_to_pixel_indices(
        room_vertices_mm=room_vertices,
        pts_xyz_mm=pts_xyz_mm,
        img_size=img_size,
        s_global_base=s_global_mm2px,
    )

    if np.sum(valid_mask) == 0:
        raise RuntimeError("所有 pts 都未落入房间有效像素范围")

    x_valid = x_px[valid_mask]
    y_valid = y_px[valid_mask]

    lamp_arrays = []
    lamp_maps = []

    for lamp_xyz in lamp_positions:
        A_one = build_A_float(
            room_vertices,
            lamp_xyz.reshape(1, 3),
            window_rects,
            img_size=img_size,
            s_global_base=s_global_mm2px,
            wall_band_px=wall_band_px,
            lamp_point_radius=lamp_point_radius,
        )

        cond_vec = build_cond_vec_for_one_lamp(
            room_vertices,
            room_dims,
            lamp_xyz,
            wall_rhos,
            surface_rhos,
            win_h_mm,
            sill_h_mm,
            window_rho,
        )
        cond_norm = ((cond_vec - cond_mean) / cond_std).astype(np.float32)

        pred_one = forward_once(G, A_one, cond_norm, device)
        lamp_arr = sample_lux_map_at_points(pred_one, x_valid, y_valid)

        lamp_maps.append(pred_one.astype(np.float32))
        lamp_arrays.append(lamp_arr)

    lamp_arrays = np.stack(lamp_arrays, axis=0).astype(np.float32)
    lamp_maps = np.stack(lamp_maps, axis=0).astype(np.float32)

    return lamp_arrays, lamp_maps, valid_mask, x_valid, y_valid, s_global_mm2px
