# -*- coding: utf-8 -*-
"""
矩形房间数据 -> 双头输入监督数据（新采样版）
--------------------------------------------------
A_float: shape = [3, 128, 128]
通道顺序：
    [lamp, room_mask, window_layout]

其中：
- lamp           : 灯具平面位置
- room_mask      : 室内区域
- window_layout  : 左墙窗布置在平面图上的投影（按左墙墙带上的窗口跨度来画）

额外导出：
    cond_vec/   每个样本一条数值条件向量
    meta_json/  每个样本一份完整元数据
    B_float/    目标照度图
    A_png/B_png/AB  预览图

当前输入文件要求：
1) room_vertices.npy   [n, 4, 2]
2) room_dims.npy       [n, 3] = [L, W, H]
3) lamp_positions.npy  [n, 3] = [x, y, z]
4) wall_rhos.npy       [n, 4] = [bottom, right, top, left]
5) surface_rhos.npy    [n, 2] = [floor, ceiling]
6) window_params.npy   [n, 9] =
   [wwr_target, wwr_actual, n_win, win_aspect_target,
    win_w_mm, win_h_mm, sill_h_mm, gap_mm, window_rho]
7) window_rects.npy    [n, 4, 4]
   每扇窗 [y0, z0, width, height]，无效窗位全 -1
8) calc_points_list.npy
9) all_total_lux.npy

新的 cond_vec 顺序：
[
  room_h_m,
  lamp_z_m,
  sill_h_m,
  win_h_m,
  rho_bottom, rho_right, rho_top, rho_left,
  floor_rho, ceiling_rho,
  window_rho
]
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
from matplotlib.path import Path
from collections import deque
import scipy.ndimage as ndimage

# ===================== 基本配置 =====================

DATA_DIR = "data/raw"

# 全局 px/mm : 0.020749
# 自动计算全局最大照度值 LUX_MAX = 893.6190 lux

OUT_DIR = os.path.join(DATA_DIR, "ab_out_dualhead_128")

IMG_SIZE = 128
UNIFORM_SCALE_MARGIN = 0.02

WALL_BAND_PX = 4
INNER_BLACK_BAND_PX = 0
LAMP_POINT_RADIUS = 2

LUX_NORM_MAX = None
RANDOM_SEED = 11

B_WALL_FILL_ZERO = True

# ===================================================


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_dataset(data_dir: str):
    room_vertices = np.load(os.path.join(data_dir, "room_vertices.npy"), allow_pickle=True)   # [n,4,2]
    room_dims = np.load(os.path.join(data_dir, "room_dims.npy"), allow_pickle=True)            # [n,3]
    lamp_positions = np.load(os.path.join(data_dir, "lamp_positions.npy"), allow_pickle=True)  # [n,3]
    wall_rhos = np.load(os.path.join(data_dir, "wall_rhos.npy"), allow_pickle=True)            # [n,4]
    surface_rhos = np.load(os.path.join(data_dir, "surface_rhos.npy"), allow_pickle=True)      # [n,2]
    window_params = np.load(os.path.join(data_dir, "window_params.npy"), allow_pickle=True)    # [n,9]
    window_rects = np.load(os.path.join(data_dir, "window_rects.npy"), allow_pickle=True)      # [n,4,4]

    calc_pts_list = np.load(os.path.join(data_dir, "calc_points_list.npy"), allow_pickle=True)
    total_lux_list = np.load(os.path.join(data_dir, "all_total_lux.npy"), allow_pickle=True)

    n = len(room_vertices)
    assert all(len(x) == n for x in [
        room_dims, lamp_positions, wall_rhos, surface_rhos,
        window_params, window_rects, calc_pts_list, total_lux_list
    ]), "各输入 .npy 长度不一致，请检查。"

    return {
        "room_vertices": room_vertices,
        "room_dims": room_dims,
        "lamp_positions": lamp_positions,
        "wall_rhos": wall_rhos,
        "surface_rhos": surface_rhos,
        "window_params": window_params,
        "window_rects": window_rects,
        "calc_pts_list": calc_pts_list,
        "total_lux_list": total_lux_list,
        "n": n,
    }


def get_global_lux_max(total_lux_list):
    all_vals = []
    for v in total_lux_list:
        arr = np.asarray(v, dtype=np.float64).ravel()
        all_vals.append(arr)
    all_vals = np.concatenate(all_vals, axis=0)
    all_vals = all_vals[all_vals >= 0.0]
    return float(all_vals.max())


# ===================== 坐标缩放与变换 =====================

def compute_global_mm2px_from_vertices(room_vertices, img_size=IMG_SIZE, margin=UNIFORM_SCALE_MARGIN):
    L_max = 0.0
    W_max = 0.0
    for verts in room_vertices:
        verts = np.asarray(verts, dtype=float)
        xs = verts[:, 0]
        ys = verts[:, 1]
        L = float(xs.max() - xs.min())
        W = float(ys.max() - ys.min())
        L_max = max(L_max, L)
        W_max = max(W_max, W)

    s_nominal = min(
        (img_size - 1) / max(L_max, 1e-9),
        (img_size - 1) / max(W_max, 1e-9)
    )
    return s_nominal * (1.0 - float(margin))


def get_room_size_from_vertices(verts_xy):
    verts_xy = np.asarray(verts_xy, dtype=float)
    xs = verts_xy[:, 0]
    ys = verts_xy[:, 1]
    L = float(xs.max() - xs.min())
    W = float(ys.max() - ys.min())
    return L, W


def get_scale_and_offset_for_size(L, W, target_size, s_global_base, img_size_base=IMG_SIZE):
    s_px = s_global_base * (target_size / float(img_size_base))
    off_x = (target_size - L * s_px) * 0.5
    off_y = (target_size - W * s_px) * 0.5
    return s_px, off_x, off_y


def world_to_pixel_xy_up_uniform(xy, s_px, off_x, off_y):
    xy = np.asarray(xy, dtype=float)
    px = off_x + xy[:, 0] * s_px
    py = off_y + xy[:, 1] * s_px
    return np.stack([px, py], axis=1)


def up_to_down_y(py_up, img_size):
    return (img_size - 1) - py_up


def polygon_mask_from_vertices_up(img_size, verts_up):
    gx, gy = np.meshgrid(np.arange(img_size), np.arange(img_size))
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    path = Path(verts_up)
    inside = path.contains_points(pts)
    return inside.reshape((img_size, img_size))


# ===================== 墙带与窗几何辅助 =====================

def compute_inner_band_from_wall(poly_mask_up, wall_mask_down, band_px):
    wall_mask_up = np.flipud(wall_mask_down.astype(bool))
    poly_mask_up = np.asarray(poly_mask_up, dtype=bool)

    region = poly_mask_up & (~wall_mask_up)
    h, w = region.shape
    if band_px <= 0 or not region.any():
        return np.zeros_like(region, dtype=bool)

    up = np.zeros_like(region);    up[:-1, :] = wall_mask_up[1:, :]
    down = np.zeros_like(region);  down[1:, :] = wall_mask_up[:-1, :]
    left = np.zeros_like(region);  left[:, :-1] = wall_mask_up[:, 1:]
    right = np.zeros_like(region); right[:, 1:] = wall_mask_up[:, :-1]

    neighbor_wall = up | down | left | right
    inner_edge = region & neighbor_wall

    dist = np.full((h, w), -1, dtype=np.int32)
    q = deque()
    ys, xs = np.nonzero(inner_edge)
    for y, x in zip(ys, xs):
        dist[y, x] = 0
        q.append((y, x))

    while q:
        y, x = q.popleft()
        if dist[y, x] >= band_px - 1:
            continue
        d_next = dist[y, x] + 1

        if y > 0 and region[y - 1, x] and dist[y - 1, x] == -1:
            dist[y - 1, x] = d_next; q.append((y - 1, x))
        if y < h - 1 and region[y + 1, x] and dist[y + 1, x] == -1:
            dist[y + 1, x] = d_next; q.append((y + 1, x))
        if x > 0 and region[y, x - 1] and dist[y, x - 1] == -1:
            dist[y, x - 1] = d_next; q.append((y, x - 1))
        if x < w - 1 and region[y, x + 1] and dist[y, x + 1] == -1:
            dist[y, x + 1] = d_next; q.append((y, x + 1))

    return (dist >= 0) & (dist < band_px)


def build_outer_wall_band_from_poly_mask(poly_mask_up, band_px):
    poly = np.asarray(poly_mask_up, dtype=bool)
    if band_px <= 0 or not poly.any():
        return np.zeros_like(poly, dtype=bool)

    h, w = poly.shape
    outside = ~poly

    up = np.zeros_like(poly);   up[:-1, :] = poly[1:, :]
    down = np.zeros_like(poly); down[1:, :] = poly[:-1, :]
    left = np.zeros_like(poly); left[:, :-1] = poly[:, 1:]
    right = np.zeros_like(poly); right[:, 1:] = poly[:, :-1]

    neighbor_inside = up | down | left | right
    start = outside & neighbor_inside

    dist = np.full((h, w), -1, dtype=np.int32)
    q = deque()
    ys, xs = np.nonzero(start)
    for y, x in zip(ys, xs):
        dist[y, x] = 0
        q.append((y, x))

    while q:
        y, x = q.popleft()
        if dist[y, x] >= band_px - 1:
            continue
        d_next = dist[y, x] + 1
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w:
                if outside[ny, nx] and dist[ny, nx] == -1:
                    dist[ny, nx] = d_next
                    q.append((ny, nx))

    return (dist >= 0) & (dist < band_px)


def make_wall_mask_and_segidx_from_poly(verts_up, poly_mask_up, band_px=WALL_BAND_PX):
    """
    段索引顺序与 room_vertices 顶点顺序一致：
      seg 0: bottom  (0,0) -> (L,0)
      seg 1: right   (L,0) -> (L,W)
      seg 2: top     (L,W) -> (0,W)
      seg 3: left    (0,W) -> (0,0)
    """
    poly_mask_up = np.asarray(poly_mask_up, dtype=bool)
    verts_up = np.asarray(verts_up, dtype=np.float32)
    n_verts = len(verts_up)

    wall_mask_up = build_outer_wall_band_from_poly_mask(poly_mask_up, band_px)

    h, w = poly_mask_up.shape
    seg_idx_map = np.full((h, w), -1, dtype=np.int32)
    if not wall_mask_up.any() or n_verts == 0:
        return wall_mask_up, seg_idx_map

    ys, xs = np.nonzero(wall_mask_up)
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)

    M = pts.shape[0]
    dist2_all = np.empty((n_verts, M), dtype=np.float32)

    for k in range(n_verts):
        p0 = verts_up[k]
        p1 = verts_up[(k + 1) % n_verts]
        v = p1 - p0
        vv = float(v[0] * v[0] + v[1] * v[1])

        if vv == 0.0:
            diff = pts - p0
            dist2_all[k] = np.sum(diff * diff, axis=1)
            continue

        wv = pts - p0
        t = (wv @ v) / vv
        t = np.clip(t, 0.0, 1.0)
        proj = p0 + t[:, None] * v
        diff = pts - proj
        dist2_all[k] = np.sum(diff * diff, axis=1)

    seg_idx = np.argmin(dist2_all, axis=0)
    seg_idx_map[ys, xs] = seg_idx
    return wall_mask_up, seg_idx_map


def rect_is_valid(rect):
    rect = np.asarray(rect, dtype=float).ravel()
    if rect.shape[0] != 4:
        return False
    y0, z0, w, h = rect
    return (w > 0.0) and (h > 0.0) and (y0 >= 0.0) and (z0 >= 0.0)


def build_window_mask_on_left_wall_from_rects(seg_idx_map, s_px, off_y, window_rects_i):
    """
    在左墙（seg_idx == 3）的墙带上，根据 window_rects 的 [y0, z0, width, height]
    画出窗在平面图上的投影。
    注意：平面图只表达沿左墙长度方向的跨度，因此只使用 [y0, width]。
    """
    left_mask = (seg_idx_map == 3)
    out = np.zeros_like(seg_idx_map, dtype=bool)
    if not left_mask.any():
        return out

    rects = np.asarray(window_rects_i, dtype=float)
    valid_rects = [np.asarray(r, dtype=float).ravel() for r in rects if rect_is_valid(r)]
    if len(valid_rects) == 0:
        return out

    ys, xs = np.nonzero(left_mask)
    y_world = (ys.astype(np.float32) - float(off_y)) / max(float(s_px), 1e-9)

    in_any_window = np.zeros_like(y_world, dtype=bool)
    for rect in valid_rects:
        y0, _z0, w, _h = rect
        y1 = y0 + w
        in_any_window |= ((y_world >= y0) & (y_world <= y1))

    out[ys[in_any_window], xs[in_any_window]] = True
    return out


# ===================== 规则网格构造（B 图） =====================

def build_regular_grid_from_calc(calc_pts_xyz, lux_vals, tol_round_dec=3):
    P = np.asarray(calc_pts_xyz, dtype=float)
    x = P[:, 0]
    y = P[:, 1]
    v = np.asarray(lux_vals, dtype=float).ravel()

    x_round = np.round(x, tol_round_dec)
    y_round = np.round(y, tol_round_dec)

    xu = np.unique(x_round); xu.sort()
    yu = np.unique(y_round); yu.sort()

    Nx = len(xu)
    Ny = len(yu)

    xmin, ymin = float(xu.min()), float(yu.min())
    dx = float(np.median(np.diff(xu))) if Nx > 1 else 1.0
    dy = float(np.median(np.diff(yu))) if Ny > 1 else 1.0

    ix = np.rint((x_round - xmin) / max(dx, 1e-9)).astype(int)
    iy = np.rint((y_round - ymin) / max(dy, 1e-9)).astype(int)

    sumZ = np.zeros((Ny, Nx), dtype=float)
    cntZ = np.zeros((Ny, Nx), dtype=int)

    for k in range(len(v)):
        j, i = iy[k], ix[k]
        if 0 <= j < Ny and 0 <= i < Nx:
            sumZ[j, i] += v[k]
            cntZ[j, i] += 1

    Z = np.full((Ny, Nx), np.nan, dtype=float)
    mask = cntZ > 0
    Z[mask] = sumZ[mask] / cntZ[mask]

    invalid_mask = np.isnan(Z)
    if invalid_mask.any() and not invalid_mask.all():
        ind = ndimage.distance_transform_edt(
            invalid_mask,
            return_distances=False,
            return_indices=True
        )
        Z = Z[tuple(ind)]

    return xmin, ymin, dx, dy, xu, yu, Z


# ===================== A 图 =====================

def draw_lamps_hard_dots(imgL, lamp_px_down, radius=LAMP_POINT_RADIUS):
    if len(lamp_px_down) == 0:
        return
    d = ImageDraw.Draw(imgL)
    for x, y in lamp_px_down:
        x0, y0 = float(x - radius), float(y - radius)
        x1, y1 = float(x + radius), float(y + radius)
        d.ellipse([x0, y0, x1, y1], fill=255, outline=255)


def make_A_data_and_preview(verts_xy, lamp_xyz, window_rects_i, img_size, s_global_base):
    """
    返回：
      A_lamp, A_room, A_window, imgA_png
    """
    verts_xy = np.asarray(verts_xy, dtype=float)

    L, W = get_room_size_from_vertices(verts_xy)
    s_px, off_x, off_y = get_scale_and_offset_for_size(L, W, img_size, s_global_base)

    verts_up = world_to_pixel_xy_up_uniform(verts_xy, s_px, off_x, off_y)
    poly_mask_up = polygon_mask_from_vertices_up(img_size, verts_up)

    wall_mask_up, seg_idx_map = make_wall_mask_and_segidx_from_poly(
        verts_up, poly_mask_up, band_px=WALL_BAND_PX
    )

    window_mask_up = build_window_mask_on_left_wall_from_rects(
        seg_idx_map, s_px, off_y, window_rects_i
    )

    A_room = np.flipud(poly_mask_up.astype(np.float32))
    A_window = np.flipud(window_mask_up.astype(np.float32))

    lamp_xy = np.asarray([[lamp_xyz[0], lamp_xyz[1]]], dtype=float)
    lamps_up = world_to_pixel_xy_up_uniform(lamp_xy, s_px, off_x, off_y)
    lamps_down = np.stack(
        [lamps_up[:, 0], up_to_down_y(lamps_up[:, 1], img_size)],
        axis=1
    )

    imgLamp = Image.new("L", (img_size, img_size), color=0)
    draw_lamps_hard_dots(imgLamp, lamps_down, radius=LAMP_POINT_RADIUS)
    A_lamp = (np.asarray(imgLamp, dtype=np.float32) / 255.0)

    rgb = np.zeros((img_size, img_size, 3), dtype=np.float32)
    rgb[..., 0] = A_lamp
    rgb[..., 1] = A_room * 0.8
    rgb[..., 2] = A_window

    lamp_mask = A_lamp > 0.5
    rgb[lamp_mask] = 1.0

    imgA_png = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    return A_lamp, A_room, A_window, imgA_png


# ===================== 条件向量 =====================

def build_cond_and_meta(verts_xy, room_dims_3, lamp_xyz, wall_rhos_4, surface_rhos_2, window_params_9, window_rects_i):
    verts_xy = np.asarray(verts_xy, dtype=float)
    room_dims_3 = np.asarray(room_dims_3, dtype=float).ravel()
    lamp_xyz = np.asarray(lamp_xyz, dtype=float).ravel()
    wall_rhos_4 = np.asarray(wall_rhos_4, dtype=float).ravel()
    surface_rhos_2 = np.asarray(surface_rhos_2, dtype=float).ravel()
    window_params_9 = np.asarray(window_params_9, dtype=float).ravel()
    window_rects_i = np.asarray(window_rects_i, dtype=float)

    if room_dims_3.shape[0] != 3:
        raise ValueError(f"room_dims 应为 3 维 [L,W,H]，实际 {room_dims_3.shape}")
    if wall_rhos_4.shape[0] != 4:
        raise ValueError(f"wall_rhos 应为 4 维 [bottom,right,top,left]，实际 {wall_rhos_4.shape}")
    if surface_rhos_2.shape[0] != 2:
        raise ValueError(f"surface_rhos 应为 2 维 [floor,ceiling]，实际 {surface_rhos_2.shape}")
    if window_params_9.shape[0] != 9:
        raise ValueError(f"window_params 应为 9 维，实际 {window_params_9.shape}")

    L_mm_v, W_mm_v = get_room_size_from_vertices(verts_xy)
    L_mm, W_mm, H_mm = [float(x) for x in room_dims_3]

    # 做一个弱一致性检查（只检查 L/W）
    if abs(L_mm - L_mm_v) > 1e-4 or abs(W_mm - W_mm_v) > 1e-4:
        raise ValueError(
            f"room_dims 与 room_vertices 不一致："
            f"room_dims=({L_mm},{W_mm}), verts=({L_mm_v},{W_mm_v})"
        )

    lamp_z_m = float(lamp_xyz[2]) / 1000.0

    rho_bottom, rho_right, rho_top, rho_left = [float(x) for x in wall_rhos_4.tolist()]
    floor_rho = float(surface_rhos_2[0])
    ceil_rho = float(surface_rhos_2[1])

    # window_params 顺序：
    # [wwr_target, wwr_actual, n_win, win_aspect_target,
    #  win_w_mm, win_h_mm, sill_h_mm, gap_mm, window_rho]
    n_win = int(round(float(window_params_9[2])))
    win_w_mm = float(window_params_9[4])
    win_h_mm = float(window_params_9[5])
    sill_h_mm = float(window_params_9[6])
    gap_mm = float(window_params_9[7])
    window_rho = float(window_params_9[8])

    room_h_m = H_mm / 1000.0
    sill_h_m = sill_h_mm / 1000.0
    win_h_m = win_h_mm / 1000.0

    cond_items = [
        float(room_h_m),
        float(lamp_z_m),
        float(sill_h_m),
        float(win_h_m),
        float(rho_bottom), float(rho_right), float(rho_top), float(rho_left),
        float(floor_rho), float(ceil_rho),
        float(window_rho),
    ]
    cond_vec = np.array(cond_items, dtype=np.float32)

    valid_rects = [np.asarray(r, dtype=float).ravel().tolist() for r in window_rects_i if rect_is_valid(r)]

    meta = {
        "room_size_mm": [float(L_mm), float(W_mm), float(H_mm)],
        "room_size_m": [float(L_mm / 1000.0), float(W_mm / 1000.0), float(H_mm / 1000.0)],
        "lamp_xyz_mm": [float(lamp_xyz[0]), float(lamp_xyz[1]), float(lamp_xyz[2])],
        "lamp_xyz_m": [float(lamp_xyz[0] / 1000.0), float(lamp_xyz[1] / 1000.0), float(lamp_xyz[2] / 1000.0)],
        "wall_rhos_4": [float(x) for x in wall_rhos_4.tolist()],
        "floor_rho": float(floor_rho),
        "ceiling_rho": float(ceil_rho),
        "window_rho": float(window_rho),
        "n_win": int(n_win),
        "win_w_mm": float(win_w_mm),
        "win_h_mm": float(win_h_mm),
        "sill_h_mm": float(sill_h_mm),
        "gap_mm": float(gap_mm),
        "window_rects_valid": valid_rects,
        "cond_dim": int(cond_vec.shape[0]),
        "cond_order": [
            "room_h_m",
            "lamp_z_m",
            "sill_h_m",
            "win_h_m",
            "rho_bottom",
            "rho_right",
            "rho_top",
            "rho_left",
            "floor_rho",
            "ceiling_rho",
            "window_rho",
        ],
    }

    return cond_vec, meta


# ===================== B 图 =====================

def build_wall_value_map(seg_idx_map, rhos_by_wall_4):
    out = np.zeros_like(seg_idx_map, dtype=np.float32)
    for k in range(4):
        out[seg_idx_map == k] = float(np.clip(rhos_by_wall_4[k], 0.0, 1.0))
    return out


def _bilinear_sample_uniform_from_reggrid(verts_xy, calc_pts_xyz, lux_vals,
                                          target_size, s_global_base):
    xmin, ymin, dx, dy, xs, ys, Z = build_regular_grid_from_calc(calc_pts_xyz, lux_vals)
    Ny, Nx = Z.shape

    L, W = get_room_size_from_vertices(verts_xy)
    s_px, off_x, off_y = get_scale_and_offset_for_size(L, W, target_size, s_global_base)

    gx, gy = np.meshgrid(np.arange(target_size), np.arange(target_size))
    Xw = (gx - off_x) / max(s_px, 1e-9)
    Yw = (gy - off_y) / max(s_px, 1e-9)

    verts_up = world_to_pixel_xy_up_uniform(np.asarray(verts_xy), s_px, off_x, off_y)
    poly_mask_up = polygon_mask_from_vertices_up(target_size, verts_up)

    u = (Xw - xmin) / max(dx, 1e-9)
    v = (Yw - ymin) / max(dy, 1e-9)

    i0 = np.floor(u).astype(int)
    j0 = np.floor(v).astype(int)
    i1 = np.clip(i0 + 1, 0, Nx - 1)
    j1 = np.clip(j0 + 1, 0, Ny - 1)
    i0 = np.clip(i0, 0, Nx - 1)
    j0 = np.clip(j0, 0, Ny - 1)

    du = np.clip(u - i0, 0.0, 1.0)
    dv = np.clip(v - j0, 0.0, 1.0)

    Z00 = Z[j0, i0]
    Z10 = Z[j0, i1]
    Z01 = Z[j1, i0]
    Z11 = Z[j1, i1]

    valid00 = ~np.isnan(Z00)
    valid10 = ~np.isnan(Z10)
    valid01 = ~np.isnan(Z01)
    valid11 = ~np.isnan(Z11)

    w00 = (1.0 - du) * (1.0 - dv) * valid00
    w10 = du * (1.0 - dv) * valid10
    w01 = (1.0 - du) * dv * valid01
    w11 = du * dv * valid11
    w_sum = w00 + w10 + w01 + w11

    Z00_safe = np.where(valid00, Z00, 0.0)
    Z10_safe = np.where(valid10, Z10, 0.0)
    Z01_safe = np.where(valid01, Z01, 0.0)
    Z11_safe = np.where(valid11, Z11, 0.0)

    img_float = (Z00_safe * w00 + Z10_safe * w10 + Z01_safe * w01 + Z11_safe * w11)

    nonzero_mask = w_sum > 0
    img_float[nonzero_mask] = img_float[nonzero_mask] / w_sum[nonzero_mask]
    img_float[~nonzero_mask] = 0.0

    img_float[~poly_mask_up] = 0.0
    return img_float, poly_mask_up, verts_up


def make_B_image_grid_uniform(verts_xy, calc_pts_xyz, lux_vals,
                              img_size, s_global_base,
                              wall_rhos_4, lux_max):
    img_lux_up, poly_mask_up, verts_up = _bilinear_sample_uniform_from_reggrid(
        verts_xy, calc_pts_xyz, lux_vals,
        target_size=img_size, s_global_base=s_global_base
    )
    lux_up = np.clip(img_lux_up, 0.0, None).astype(np.float32)

    wall_mask_up, seg_idx_map = make_wall_mask_and_segidx_from_poly(
        verts_up, poly_mask_up, band_px=WALL_BAND_PX
    )
    wall_val_up = build_wall_value_map(seg_idx_map, wall_rhos_4)

    wall_mask_down = np.flipud(wall_mask_up)
    wall_float_down = np.flipud(wall_val_up)

    band_up = compute_inner_band_from_wall(poly_mask_up, wall_mask_down, band_px=INNER_BLACK_BAND_PX)
    lux_up[band_up] = 0.0
    lux_up[~poly_mask_up] = 0.0

    lux_down = np.flipud(lux_up)
    B_float = lux_down.copy()

    if B_WALL_FILL_ZERO:
        B_float[wall_mask_down] = 0.0
    else:
        B_float[wall_mask_down] = wall_float_down[wall_mask_down]

    interior_mask_down = np.flipud(poly_mask_up) & (~wall_mask_down)
    vis = np.zeros_like(B_float, dtype=np.float32)

    lux_max_vis = float(lux_max) if (lux_max is not None and lux_max > 0) else max(float(np.max(lux_down)), 1.0)
    vis[interior_mask_down] = np.clip(lux_down[interior_mask_down] / lux_max_vis, 0.0, 1.0)

    if B_WALL_FILL_ZERO:
        vis[wall_mask_down] = 0.0
    else:
        vis[wall_mask_down] = wall_float_down[wall_mask_down]

    imgB = Image.fromarray((vis * 255.0).astype(np.uint8), mode="L")
    return imgB, B_float


# ===================== 主流程 =====================

def save_AB_images(dataset, out_dir, img_size=IMG_SIZE):
    ensure_dir(out_dir)

    dir_A_png = os.path.join(out_dir, "A_png")
    dir_B_png = os.path.join(out_dir, "B_png")
    dir_AB = os.path.join(out_dir, "AB")
    dir_A_float = os.path.join(out_dir, "A_float")
    dir_B_float = os.path.join(out_dir, "B_float")
    dir_cond_vec = os.path.join(out_dir, "cond_vec")
    dir_meta_json = os.path.join(out_dir, "meta_json")

    for d in [dir_A_png, dir_B_png, dir_AB, dir_A_float, dir_B_float, dir_cond_vec, dir_meta_json]:
        ensure_dir(d)

    n = dataset["n"]
    room_vertices = dataset["room_vertices"]
    room_dims = dataset["room_dims"]
    lamp_positions = dataset["lamp_positions"]
    wall_rhos = dataset["wall_rhos"]
    surface_rhos = dataset["surface_rhos"]
    window_params = dataset["window_params"]
    window_rects = dataset["window_rects"]
    calc_pts_list = dataset["calc_pts_list"]
    total_lux_list = dataset["total_lux_list"]

    S_GLOBAL_MM2PX = compute_global_mm2px_from_vertices(
        room_vertices, img_size=img_size, margin=UNIFORM_SCALE_MARGIN
    )
    print(f"  全局 px/mm : {S_GLOBAL_MM2PX:.6f}")

    if (LUX_NORM_MAX is not None) and (LUX_NORM_MAX > 0):
        lux_max = float(LUX_NORM_MAX)
        print(f"使用外部指定的可视化归一化上限 LUX_MAX = {lux_max:.4f} lux")
    else:
        lux_max = get_global_lux_max(total_lux_list)
        print(f"自动计算全局最大照度值 LUX_MAX = {lux_max:.4f} lux")

    for i in range(n):
        verts_xy = np.asarray(room_vertices[i], dtype=float)
        room_dims_i = np.asarray(room_dims[i], dtype=float).ravel()
        lamp_xyz = np.asarray(lamp_positions[i], dtype=float).ravel()
        wall_rhos_i = np.asarray(wall_rhos[i], dtype=float).ravel()
        surface_rhos_i = np.asarray(surface_rhos[i], dtype=float).ravel()
        window_params_i = np.asarray(window_params[i], dtype=float).ravel()
        window_rects_i = np.asarray(window_rects[i], dtype=float)

        calc_pts = np.asarray(calc_pts_list[i], dtype=float)
        lux_vals = np.asarray(total_lux_list[i], dtype=float).ravel()

        A_lamp, A_room, A_window, imgA = make_A_data_and_preview(
            verts_xy, lamp_xyz, window_rects_i,
            img_size, S_GLOBAL_MM2PX
        )
        A_float = np.stack([A_lamp, A_room, A_window], axis=0).astype(np.float32)

        cond_vec, meta = build_cond_and_meta(
            verts_xy, room_dims_i, lamp_xyz, wall_rhos_i, surface_rhos_i, window_params_i, window_rects_i
        )

        imgB, B_float = make_B_image_grid_uniform(
            verts_xy, calc_pts, lux_vals,
            img_size=img_size, s_global_base=S_GLOBAL_MM2PX,
            wall_rhos_4=wall_rhos_i, lux_max=lux_max
        )

        w, h = imgA.size
        ab_img = Image.new("RGB", (w * 2, h), color=(0, 0, 0))
        ab_img.paste(imgA, (0, 0))
        ab_img.paste(imgB.convert("RGB"), (w, 0))

        stem = f"sample_{i:05d}"
        imgA.save(os.path.join(dir_A_png, f"{stem}.png"))
        imgB.save(os.path.join(dir_B_png, f"{stem}.png"))
        ab_img.save(os.path.join(dir_AB, f"{stem}.png"))

        np.save(os.path.join(dir_A_float, f"{stem}.npy"), A_float)
        np.save(os.path.join(dir_B_float, f"{stem}.npy"), B_float.astype(np.float32))
        np.save(os.path.join(dir_cond_vec, f"{stem}.npy"), cond_vec.astype(np.float32))

        with open(os.path.join(dir_meta_json, f"{stem}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"[{i+1}/{n}] 生成完成")

    print("全部数据已生成：")
    print("  A_float 通道顺序 = [lamp, room_mask, window_layout]")
    print("  cond_vec 顺序 = [room_h_m, lamp_z_m, sill_h_m, win_h_m, "
          "rho_bottom, rho_right, rho_top, rho_left, floor_rho, ceiling_rho, window_rho]")
    print("  cond_dim   =", 11)
    print("  A_png      :", dir_A_png)
    print("  B_png      :", dir_B_png)
    print("  AB         :", dir_AB)
    print("  A_float    :", dir_A_float, " (shape = [3, H, W])")
    print("  B_float    :", dir_B_float, " (shape = [H, W])")
    print("  cond_vec   :", dir_cond_vec)
    print("  meta_json  :", dir_meta_json)
    print(f"  全局 px/mm : {S_GLOBAL_MM2PX:.6f}")


