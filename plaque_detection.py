import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import shutil
import glob

INPUT_ROOT = r"D:/Internship/Teeth Segment Result"
OUTPUT_ROOT = r"D:/Internship/Plaque Result Curves"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

TOOTH_FILES = [
    "tooth11.png",
    "tooth12.png",
    "tooth13.png",
    "tooth21.png",
    "tooth22.png",
    "tooth23.png",
]

TOOTH_NAME_TH = {
    11: "ฟันตัดกลางบนขวา",
    12: "ฟันตัดข้างบนขวา",
    13: "ฟันเขี้ยวบนขวา",
    21: "ฟันตัดกลางบนซ้าย",
    22: "ฟันตัดข้างบนซ้าย",
    23: "ฟันเขี้ยวบนซ้าย",
}

# ลำดับการวางภาพ summary
SUMMARY_ORDER = [11, 12, 13, 21, 22, 23]

# ปรับตรงนี้ได้
QHPI_SCORE2_BAND_MM = 0.5   # เส้นเหลืองลึกจากขอบเหงือกประมาณ 1–2 mm
SUMMARY_CELL_W = 540        # ความกว้างช่องในภาพ summary
SUMMARY_CELL_H = 430        # ความสูงช่องในภาพ summary
SUMMARY_IMAGE_H = 270       # พื้นที่รูปด้านบนของแต่ละช่อง


# =========================
# Utility
# =========================
def get_tooth_mask_from_rgba(image_rgba):
    alpha = image_rgba[:, :, 3]
    return alpha > 0


def draw_dashed_polyline(img, pts, color, thickness=2, dash_length=10, gap_length=6):
    pts = np.asarray(pts, dtype=np.int32)

    if len(pts) < 2:
        return img

    for i in range(len(pts) - 1):
        p1 = pts[i].astype(np.float32)
        p2 = pts[i + 1].astype(np.float32)

        vec = p2 - p1
        dist = np.linalg.norm(vec)

        if dist == 0:
            continue

        direction = vec / dist
        current = 0.0

        while current < dist:
            start = p1 + direction * current
            end = p1 + direction * min(current + dash_length, dist)

            cv2.line(
                img,
                tuple(start.astype(np.int32)),
                tuple(end.astype(np.int32)),
                color,
                thickness,
                cv2.LINE_AA
            )

            current += dash_length + gap_length

    return img


# =========================
# Image Enhancement
# =========================
def enhance_tooth_image(image_rgb, scale=2):
    h, w = image_rgb.shape[:2]

    up = cv2.resize(
        image_rgb,
        (w * scale, h * scale),
        interpolation=cv2.INTER_CUBIC
    )

    lab = cv2.cvtColor(up, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    return out


# =========================
# Plaque Detection
# =========================
def detect_plaque_mask(image_rgb, tooth_mask):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    A = lab[:, :, 1]
    B = lab[:, :, 2]

    mask_pink = cv2.inRange(
        hsv,
        np.array([140, 20, 40], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8)
    )

    mask_purple = cv2.inRange(
        hsv,
        np.array([110, 20, 40], dtype=np.uint8),
        np.array([145, 255, 255], dtype=np.uint8)
    )

    mask_bluepurple = cv2.inRange(
        hsv,
        np.array([95, 25, 35], dtype=np.uint8),
        np.array([125, 255, 255], dtype=np.uint8)
    )

    color_mask = (mask_pink > 0) | (mask_purple > 0) | (mask_bluepurple > 0)
    sat_mask = S > 25
    val_mask = V > 35
    lab_mask = (A > 120) & (B < 150)

    mask = color_mask & sat_mask & val_mask & lab_mask & tooth_mask
    mask = (mask.astype(np.uint8) * 255)

    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    min_area = 20
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned


# =========================
# Visualization
# =========================
def visualize_plaque(image_rgb, plaque_mask, alpha=0.45):
    out = image_rgb.copy().astype(np.float32)

    red = np.zeros_like(out)
    red[:, :, 0] = 255

    mask_bool = plaque_mask > 0
    out[mask_bool] = out[mask_bool] * (1 - alpha) + red[mask_bool] * alpha

    return out.astype(np.uint8)


# =========================
# Shape-based zone split PHP
# =========================
def build_shape_profile(tooth_mask):
    h, w = tooth_mask.shape
    ys, xs = np.where(tooth_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    y_min, y_max = ys.min(), ys.max()

    left_x = np.full(h, -1, dtype=np.int32)
    right_x = np.full(h, -1, dtype=np.int32)

    for y in range(y_min, y_max + 1):
        row_x = np.where(tooth_mask[y])[0]
        if len(row_x) > 0:
            left_x[y] = row_x.min()
            right_x[y] = row_x.max()

    return {
        "y_min": int(y_min),
        "y_max": int(y_max),
        "left_x": left_x,
        "right_x": right_x,
    }


def get_x_on_row(left_x, right_x, y, frac):
    xl = left_x[y]
    xr = right_x[y]

    if xl < 0 or xr < 0 or xr <= xl:
        return None

    return int(round(xl + frac * (xr - xl)))


def make_shape_based_zone_masks(tooth_mask, tooth_id=None, top_label="I"):
    profile = build_shape_profile(tooth_mask)

    if profile is None:
        return None, None

    h, w = tooth_mask.shape

    y_min = profile["y_min"]
    y_max = profile["y_max"]
    left_x = profile["left_x"]
    right_x = profile["right_x"]

    height = y_max - y_min + 1
    y1 = y_min + height // 3
    y2 = y_min + (2 * height) // 3

    active_zones = ["M", top_label, "C", "G", "D"]

    if tooth_id == 13:
        active_zones = [top_label, "C", "G", "D"]
    elif tooth_id == 23:
        active_zones = ["M", top_label, "C", "G"]

    zone_masks = {}

    for z in active_zones:
        zone_masks[z] = np.zeros((h, w), dtype=np.uint8)

    for y in range(y_min, y_max + 1):
        xl = left_x[y]
        xr = right_x[y]

        if xl < 0 or xr < 0 or xr <= xl:
            continue

        x_l_mid = get_x_on_row(left_x, right_x, y, 1 / 3)
        x_r_mid = get_x_on_row(left_x, right_x, y, 2 / 3)

        if x_l_mid is None or x_r_mid is None:
            continue

        if tooth_id == 13:
            bound_left = xl
            bound_right = x_r_mid
            zone_masks["D"][y, bound_right:xr + 1] = 1

        elif tooth_id == 23:
            bound_left = x_l_mid
            bound_right = xr
            zone_masks["M"][y, xl:bound_left] = 1

        else:
            bound_left = x_l_mid
            bound_right = x_r_mid
            zone_masks["M"][y, xl:bound_left] = 1
            zone_masks["D"][y, bound_right:xr + 1] = 1

        if y < y1:
            if top_label in active_zones:
                zone_masks[top_label][y, bound_left:bound_right] = 1

        elif y < y2:
            if "C" in active_zones:
                zone_masks["C"][y, bound_left:bound_right] = 1

        else:
            if "G" in active_zones:
                zone_masks["G"][y, bound_left:bound_right] = 1

    for k in zone_masks:
        zone_masks[k] = (zone_masks[k] > 0) & tooth_mask

    guide = {
        "profile": profile,
        "y1": int(y1),
        "y2": int(y2),
        "top_label": top_label,
        "tooth_id": tooth_id
    }

    return zone_masks, guide


def score_php_from_zone_masks(plaque_mask, zone_masks, min_plaque_pixels=1):
    score = 0
    detail = {}

    for zone_name, zone_mask in zone_masks.items():
        zone_pixels = int(np.sum(zone_mask))
        plaque_pixels = int(np.sum((plaque_mask > 0) & zone_mask))
        has_plaque = plaque_pixels >= min_plaque_pixels

        detail[zone_name] = {
            "score": 1 if has_plaque else 0,
            "plaque_pixels": plaque_pixels,
            "zone_pixels": zone_pixels,
        }

        score += detail[zone_name]["score"]

    max_score = len(zone_masks)

    return score, max_score, detail


def draw_shape_based_php_zones(image_rgb, tooth_mask, guide, detail):
    out = image_rgb.copy()

    profile = guide["profile"]
    y_min = profile["y_min"]
    y_max = profile["y_max"]
    left_x = profile["left_x"]
    right_x = profile["right_x"]

    y1 = guide["y1"]
    y2 = guide["y2"]
    top_label = guide["top_label"]
    tooth_id = guide.get("tooth_id", None)

    line_color = (0, 255, 0)
    text_color = (255, 255, 0)

    contours, _ = cv2.findContours(
        (tooth_mask.astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    cv2.drawContours(out, contours, -1, line_color, 2)

    pts_left_mid = []
    pts_right_mid = []

    for y in range(y_min, y_max + 1):
        xl = left_x[y]
        xr = right_x[y]

        if xl < 0 or xr < 0 or xr <= xl:
            continue

        x1 = get_x_on_row(left_x, right_x, y, 1 / 3)
        x2 = get_x_on_row(left_x, right_x, y, 2 / 3)

        if x1 is not None:
            pts_left_mid.append([x1, y])

        if x2 is not None:
            pts_right_mid.append([x2, y])

    if tooth_id == 13:
        if len(pts_right_mid) > 1:
            cv2.polylines(out, [np.array(pts_right_mid, dtype=np.int32)], False, line_color, 2)

    elif tooth_id == 23:
        if len(pts_left_mid) > 1:
            cv2.polylines(out, [np.array(pts_left_mid, dtype=np.int32)], False, line_color, 2)

    else:
        if len(pts_left_mid) > 1:
            cv2.polylines(out, [np.array(pts_left_mid, dtype=np.int32)], False, line_color, 2)

        if len(pts_right_mid) > 1:
            cv2.polylines(out, [np.array(pts_right_mid, dtype=np.int32)], False, line_color, 2)

    row1 = []
    row2 = []

    for y in [y1, y2]:
        xl = left_x[y]
        xr = right_x[y]

        if xl >= 0 and xr >= 0 and xr > xl:
            x1 = get_x_on_row(left_x, right_x, y, 1 / 3)
            x2 = get_x_on_row(left_x, right_x, y, 2 / 3)

            if x1 is not None and x2 is not None:
                if tooth_id == 13:
                    b_left, b_right = xl, x2
                elif tooth_id == 23:
                    b_left, b_right = x1, xr
                else:
                    b_left, b_right = x1, x2

                if y == y1:
                    row1 = [[x, y] for x in range(b_left, b_right + 1)]
                else:
                    row2 = [[x, y] for x in range(b_left, b_right + 1)]

    if len(row1) > 1:
        cv2.polylines(out, [np.array(row1, dtype=np.int32)], False, line_color, 2)

    if len(row2) > 1:
        cv2.polylines(out, [np.array(row2, dtype=np.int32)], False, line_color, 2)

    zone_masks, _ = make_shape_based_zone_masks(
        tooth_mask,
        tooth_id=tooth_id,
        top_label=top_label
    )

    for zone_name, zone_mask in zone_masks.items():
        ys, xs = np.where(zone_mask)

        if len(xs) == 0:
            continue

        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

        label = f"{zone_name}:{detail[zone_name]['score']}"

        cv2.putText(
            out,
            label,
            (cx - 20, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            2,
            cv2.LINE_AA
        )

    return out


# =========================
# QHPI helpers
# =========================
def estimate_pixels_per_mm(tooth_mask):
    ys, xs = np.where(tooth_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    y_min = ys.min()
    y_max = ys.max()

    tooth_height_px = y_max - y_min + 1

    approx_crown_height_mm = 8.0
    px_per_mm = tooth_height_px / approx_crown_height_mm

    return max(px_per_mm, 1.0)


def get_row_segments(binary_row):
    xs = np.where(binary_row)[0]

    if len(xs) == 0:
        return []

    segments = []

    start = xs[0]
    prev = xs[0]

    for x in xs[1:]:
        if x == prev + 1:
            prev = x
        else:
            segments.append((start, prev))
            start = x
            prev = x

    segments.append((start, prev))

    return segments


def measure_gingival_band_properties(plaque_mask, tooth_mask, px_per_mm):
    ys, xs = np.where(tooth_mask)

    if len(xs) == 0 or len(ys) == 0:
        return {
            "max_vertical_depth_px": 0,
            "max_band_width_ratio": 0.0,
            "fleck_count": 0
        }

    y_min = ys.min()
    y_max = ys.max()

    height = y_max - y_min + 1

    gingival_depth_px = int(round(2.0 * px_per_mm))
    gingival_depth_px = max(1, min(gingival_depth_px, height))

    max_vertical_depth_px = 0
    max_band_width_ratio = 0.0
    fleck_count = 0

    profile = build_shape_profile(tooth_mask)
    left_x = profile["left_x"]
    right_x = profile["right_x"]

    for y in range(y_min, y_min + gingival_depth_px):
        if y >= tooth_mask.shape[0]:
            break

        xl = left_x[y]
        xr = right_x[y]

        if xl < 0 or xr < 0 or xr <= xl:
            continue

        tooth_row = tooth_mask[y, xl:xr + 1]
        plaque_row = (plaque_mask[y, xl:xr + 1] > 0) & tooth_row

        row_width = int(np.sum(tooth_row))

        if row_width == 0:
            continue

        segments = get_row_segments(plaque_row)

        small_flecks = sum(
            1
            for s, e in segments
            if (e - s + 1) <= max(2, int(round(0.15 * row_width)))
        )

        fleck_count += small_flecks

        if segments:
            longest = max((e - s + 1) for s, e in segments)
            max_band_width_ratio = max(max_band_width_ratio, longest / row_width)

    xs_all = np.where(tooth_mask.any(axis=0))[0]

    for x in xs_all:
        col_tooth = tooth_mask[y_min:y_min + gingival_depth_px, x]
        col_plaque = (plaque_mask[y_min:y_min + gingival_depth_px, x] > 0) & col_tooth

        if not np.any(col_plaque):
            continue

        ys_local = np.where(col_plaque)[0]
        max_vertical_depth_px = max(max_vertical_depth_px, int(ys_local.max() + 1))

    return {
        "max_vertical_depth_px": max_vertical_depth_px,
        "max_band_width_ratio": max_band_width_ratio,
        "fleck_count": fleck_count,
    }


def compute_qhpi_from_mask(plaque_mask, tooth_mask, pixels_per_mm=None):
    h, w = tooth_mask.shape

    ys, xs = np.where(tooth_mask)

    if len(xs) == 0 or len(ys) == 0:
        return 0, {
            "ratio": 0.0,
            "pixels_per_mm": None,
            "vertical_depth_mm": 0.0,
            "band_width_ratio": 0.0,
            "top_edge": None,
        }

    y_min, y_max = ys.min(), ys.max()
    H = y_max - y_min + 1

    if pixels_per_mm is None:
        pixels_per_mm = estimate_pixels_per_mm(tooth_mask)

    top_edge = np.full(w, -1, dtype=np.int32)

    for x in range(w):
        ys_col = np.where(tooth_mask[:, x])[0]

        if len(ys_col) > 0:
            top_edge[x] = ys_col.min()

    shift_1 = H / 3.0
    shift_2 = 2.0 * H / 3.0

    # เส้นเหลืองสำหรับช่วยแยก QHPI 1 / 2
    # ใช้หน่วย mm จริงโดยประมาณ ไม่ใช้ครึ่งหนึ่งของ gingival third แล้ว
    shift_score2_band = QHPI_SCORE2_BAND_MM * pixels_per_mm

    # กันไม่ให้เส้นเหลืองลึกเกิน gingival third
    shift_score2_band = min(shift_score2_band, shift_1 * 0.45)

    zone1_mask = np.zeros_like(tooth_mask)
    zone2_mask = np.zeros_like(tooth_mask)

    gingival_mask = np.zeros_like(tooth_mask)
    middle_mask = np.zeros_like(tooth_mask)
    incisal_mask = np.zeros_like(tooth_mask)

    for x in range(w):
        if top_edge[x] == -1:
            continue

        y_t = top_edge[x]

        y_score2 = int(round(y_t + shift_score2_band))
        y_c1 = int(round(y_t + shift_1))
        y_c2 = int(round(y_t + shift_2))

        ys_col = np.where(tooth_mask[:, x])[0]

        for y in ys_col:
            if y < y_score2:
                zone1_mask[y, x] = 1
                gingival_mask[y, x] = 1

            elif y < y_c1:
                zone2_mask[y, x] = 1
                gingival_mask[y, x] = 1

            elif y < y_c2:
                middle_mask[y, x] = 1

            else:
                incisal_mask[y, x] = 1

    gingival_props = measure_gingival_band_properties(
        plaque_mask,
        tooth_mask,
        pixels_per_mm
    )

    vertical_depth_mm = (
        gingival_props["max_vertical_depth_px"] / pixels_per_mm
        if pixels_per_mm
        else 0.0
    )

    band_width_ratio = gingival_props["max_band_width_ratio"]

    plaque_area = int(np.sum((plaque_mask > 0) & tooth_mask))
    tooth_area = int(np.sum(tooth_mask))

    ratio = plaque_area / tooth_area if tooth_area > 0 else 0.0

    has_zone1 = np.sum((plaque_mask > 0) & zone1_mask) > 0
    has_zone2 = np.sum((plaque_mask > 0) & zone2_mask) > 0
    has_gingival = np.sum((plaque_mask > 0) & gingival_mask) > 0
    has_middle = np.sum((plaque_mask > 0) & middle_mask) > 0
    has_incisal = np.sum((plaque_mask > 0) & incisal_mask) > 0

    if plaque_area == 0:
        score = 0

    elif has_incisal:
        score = 5

    elif has_middle:
        score = 4

    elif has_gingival:
        if vertical_depth_mm > 1.0:
            score = 3
        elif has_zone2 and band_width_ratio >= 0.50:
            score = 2
        elif has_zone1 or has_zone2:
            score = 1
        else:
            score = 0

    else:
        score = 0

    qh_detail = {
        "ratio": ratio,
        "pixels_per_mm": pixels_per_mm,
        "vertical_depth_mm": float(vertical_depth_mm),
        "band_width_ratio": float(band_width_ratio),
        "top_edge": top_edge,
        "shift_score2_band": shift_score2_band,
        "shift_1": shift_1,
        "shift_2": shift_2,
        "score2_band_mm": QHPI_SCORE2_BAND_MM,
        "has_zone1": bool(has_zone1),
        "has_zone2": bool(has_zone2),
        "has_gingival": bool(has_gingival),
        "has_middle": bool(has_middle),
        "has_incisal": bool(has_incisal),
    }

    return score, qh_detail


def draw_qhpi_visual(image_rgb, tooth_mask, plaque_mask, qh_score, qh_detail):
    out = image_rgb.copy()

    colors = {
        0: (0, 255, 0),
        1: (120, 255, 0),
        2: (255, 255, 0),
        3: (255, 170, 0),
        4: (255, 100, 0),
        5: (255, 0, 0)
    }

    overlay_color = np.array(
        colors.get(qh_score, (255, 255, 255)),
        dtype=np.float32
    )

    out_f = out.astype(np.float32)

    mask_bool = plaque_mask > 0
    alpha = 0.55

    out_f[mask_bool] = (
        out_f[mask_bool] * (1 - alpha)
        + overlay_color * alpha
    )

    out = out_f.astype(np.uint8)

    contours, _ = cv2.findContours(
        (tooth_mask.astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)

    top_edge = qh_detail.get("top_edge")

    if top_edge is not None:
        shift_score2 = qh_detail["shift_score2_band"]
        shift_1 = qh_detail["shift_1"]
        shift_2 = qh_detail["shift_2"]

        pts_score2 = []
        pts1 = []
        pts2 = []

        for x in range(len(top_edge)):
            if top_edge[x] == -1:
                continue

            ys_col = np.where(tooth_mask[:, x])[0]

            if len(ys_col) == 0:
                continue

            y_b = ys_col.max()

            y_score2 = int(round(top_edge[x] + shift_score2))
            y1 = int(round(top_edge[x] + shift_1))
            y2 = int(round(top_edge[x] + shift_2))

            if y_score2 <= y_b:
                pts_score2.append([x, y_score2])

            if y1 <= y_b:
                pts1.append([x, y1])

            if y2 <= y_b:
                pts2.append([x, y2])

        # เส้นเหลือง = ระยะประมาณ 1–2 mm จากขอบเหงือก
        if len(pts_score2) > 1:
            cv2.polylines(
                out,
                [np.array(pts_score2, dtype=np.int32)],
                False,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )

        # เส้นเขียวหลัก = เส้นประ
        if len(pts1) > 1:
            draw_dashed_polyline(
                out,
                pts1,
                color=(0, 255, 0),
                thickness=2,
                dash_length=10,
                gap_length=6
            )

        if len(pts2) > 1:
            draw_dashed_polyline(
                out,
                pts2,
                color=(0, 255, 0),
                thickness=2,
                dash_length=10,
                gap_length=6
            )

    ratio_pct = qh_detail["ratio"] * 100
    depth_mm = qh_detail["vertical_depth_mm"]
    width_pct = qh_detail["band_width_ratio"] * 100
    score2_mm = qh_detail["score2_band_mm"]

    cv2.putText(out, f"QHPI: {qh_score}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(out, f"Coverage: {ratio_pct:.1f}%", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(out, f"Depth: {depth_mm:.2f} mm  Band: {width_pct:.1f}%", (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(out, f"Yellow line: {score2_mm:.1f} mm", (10, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

    return out



# =========================
# Case Summary Image
# =========================
def load_summary_font(size=22):
    """โหลดฟอนต์ที่รองรับไทยให้มากที่สุด ถ้าไม่มีจะ fallback เป็น default."""
    candidates = [
        r"C:/Windows/Fonts/tahoma.ttf",
        r"C:/Windows/Fonts/arial.ttf",
        r"C:/Windows/Fonts/THSarabunNew.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    ]

    for path in candidates:
        try:
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
        except Exception:
            pass

    return ImageFont.load_default()


def paste_image_center(canvas, img, box):
    """ย่อรูปให้พอดีกับ box แล้ววางตรงกลาง"""
    x1, y1, x2, y2 = box
    max_w = x2 - x1
    max_h = y2 - y1

    img = img.convert("RGB")
    img_fit = ImageOps.contain(img, (max_w, max_h))

    px = x1 + (max_w - img_fit.width) // 2
    py = y1 + (max_h - img_fit.height) // 2

    canvas.paste(img_fit, (px, py))


def draw_text_center(draw, box, text, font, fill=(20, 20, 20)):
    """เขียนข้อความตรงกลางกล่อง ใช้กับ label PHP/QHPI เหนือรูป"""
    x1, y1, x2, y2 = box

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)

    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 - th) // 2
    draw.text((tx, ty), text, fill=fill, font=font)


def create_case_summary_image(case_name, case_results, save_dir):
    """
    สร้างภาพรวม 6 ซี่ต่อเคส
    แต่ละช่องมี 2 รูป:
      - รูป PHP zone visualization
      - รูป QHPI visualization
    พร้อมข้อความ PHP / QHPI / Plaque ratio ใต้ภาพ
    """
    if not case_results:
        return None

    result_map = {
        r.get("tooth_id"): r
        for r in case_results
        if r.get("tooth_id") is not None
    }

    cols = 3
    rows = 2
    header_h = 70
    margin = 24

    cell_w = SUMMARY_CELL_W
    cell_h = SUMMARY_CELL_H

    canvas_w = cols * cell_w + margin * 2
    canvas_h = rows * cell_h + header_h + margin * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    font_title = load_summary_font(28)
    font_head = load_summary_font(22)
    font_body = load_summary_font(19)
    font_small = load_summary_font(16)
    font_label = load_summary_font(18)

    title = f"Case Summary PHP + QHPI: {case_name}"
    draw.text((margin, 22), title, fill=(20, 20, 20), font=font_title)

    for idx, tooth_id in enumerate(SUMMARY_ORDER):
        row = idx // cols
        col = idx % cols

        x = margin + col * cell_w
        y = header_h + margin + row * cell_h

        # card background
        draw.rounded_rectangle(
            [x + 8, y + 8, x + cell_w - 8, y + cell_h - 8],
            radius=18,
            fill=(255, 255, 255),
            outline=(210, 210, 210),
            width=2
        )

        r = result_map.get(tooth_id)

        if r is None:
            draw.text((x + 28, y + 28), f"Tooth {tooth_id}", fill=(40, 40, 40), font=font_head)
            draw.text((x + 28, y + 65), "Not found", fill=(160, 0, 0), font=font_body)
            continue

        php_vis_path = r.get("php_vis_path")
        qhpi_vis_path = r.get("qhpi_vis_path")
        tooth_name = TOOTH_NAME_TH.get(tooth_id, "")

        # image area: 2 รูปต่อ 1 ช่อง
        label_y1 = y + 22
        label_y2 = y + 48
        img_y1 = y + 52
        img_y2 = y + SUMMARY_IMAGE_H

        gap = 14
        inner_x1 = x + 22
        inner_x2 = x + cell_w - 22
        half_w = (inner_x2 - inner_x1 - gap) // 2

        php_box = (
            inner_x1,
            img_y1,
            inner_x1 + half_w,
            img_y2,
        )

        qhpi_box = (
            inner_x1 + half_w + gap,
            img_y1,
            inner_x2,
            img_y2,
        )

        draw_text_center(
            draw,
            (php_box[0], label_y1, php_box[2], label_y2),
            "PHP",
            font_label,
            fill=(20, 90, 20)
        )

        draw_text_center(
            draw,
            (qhpi_box[0], label_y1, qhpi_box[2], label_y2),
            "QHPI",
            font_label,
            fill=(120, 40, 20)
        )

        # กรอบรูปเล็ก ๆ
        draw.rounded_rectangle(php_box, radius=10, outline=(225, 225, 225), width=1)
        draw.rounded_rectangle(qhpi_box, radius=10, outline=(225, 225, 225), width=1)

        if php_vis_path and os.path.exists(php_vis_path):
            try:
                img = Image.open(php_vis_path)
                paste_image_center(canvas, img, php_box)
            except Exception:
                draw.text((php_box[0] + 10, php_box[1] + 60), "PHP image error", fill=(160, 0, 0), font=font_small)
        else:
            draw.text((php_box[0] + 10, php_box[1] + 60), "No PHP image", fill=(160, 0, 0), font=font_small)

        if qhpi_vis_path and os.path.exists(qhpi_vis_path):
            try:
                img = Image.open(qhpi_vis_path)
                paste_image_center(canvas, img, qhpi_box)
            except Exception:
                draw.text((qhpi_box[0] + 10, qhpi_box[1] + 60), "QHPI image error", fill=(160, 0, 0), font=font_small)
        else:
            draw.text((qhpi_box[0] + 10, qhpi_box[1] + 60), "No QHPI image", fill=(160, 0, 0), font=font_small)

        # text summary
        text_y = y + SUMMARY_IMAGE_H + 16
        ratio_pct = r.get("ratio", 0.0) * 100
        php_score = r.get("php_score", 0)
        php_max = r.get("php_max", 0)
        qhpi_score = r.get("qhpi_score", 0)

        draw.text(
            (x + 26, text_y),
            f"{tooth_id} - {tooth_name}",
            fill=(20, 20, 20),
            font=font_head
        )

        draw.text(
            (x + 26, text_y + 36),
            f"PHP: {php_score}/{php_max}    QHPI: {qhpi_score}/5",
            fill=(30, 30, 30),
            font=font_body
        )

        draw.text(
            (x + 26, text_y + 68),
            f"Plaque coverage: {ratio_pct:.2f}%",
            fill=(70, 70, 70),
            font=font_small
        )

    save_path = os.path.join(save_dir, "case_summary_php_qhpi.png")
    canvas.save(save_path)
    return save_path

# =========================
# Process Each Tooth
# =========================
def process_tooth(tooth_path, save_dir, top_label="I", min_plaque_pixels=1, scale=2):
    print("READING:", tooth_path)

    rgba = np.array(Image.open(tooth_path).convert("RGBA"))
    rgb = rgba[:, :, :3].copy()

    tooth_mask = get_tooth_mask_from_rgba(rgba)

    rgb_masked = np.zeros_like(rgb)
    rgb_masked[tooth_mask] = rgb[tooth_mask]

    enhanced_rgb = enhance_tooth_image(rgb_masked, scale=scale)

    tooth_mask_up = cv2.resize(
        tooth_mask.astype(np.uint8),
        (enhanced_rgb.shape[1], enhanced_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST
    ) > 0

    enhanced_rgb_masked = np.zeros_like(enhanced_rgb)
    enhanced_rgb_masked[tooth_mask_up] = enhanced_rgb[tooth_mask_up]

    base = os.path.splitext(os.path.basename(tooth_path))[0]

    tooth_id = None

    try:
        tooth_id = int("".join(filter(str.isdigit, base)))
    except:
        pass

    Image.fromarray(rgba).save(os.path.join(save_dir, f"{base}_input_rgba.png"))
    Image.fromarray(rgb_masked).save(os.path.join(save_dir, f"{base}_input_rgb.png"))
    Image.fromarray(enhanced_rgb_masked).save(
        os.path.join(save_dir, f"{base}_input_rgb_enhanced.png")
    )

    plaque_mask = detect_plaque_mask(
        enhanced_rgb_masked,
        tooth_mask=tooth_mask_up
    )

    plaque_vis = visualize_plaque(enhanced_rgb_masked, plaque_mask)

    plaque_area = int(np.sum(plaque_mask > 0))
    tooth_area = int(np.sum(tooth_mask_up > 0))

    ratio = plaque_area / tooth_area if tooth_area > 0 else 0.0

    zone_masks, guide = make_shape_based_zone_masks(
        tooth_mask_up,
        tooth_id=tooth_id,
        top_label=top_label
    )

    php_score, php_max, detail = score_php_from_zone_masks(
        plaque_mask,
        zone_masks,
        min_plaque_pixels=min_plaque_pixels
    )

    php_vis = draw_shape_based_php_zones(
        plaque_vis,
        tooth_mask_up,
        guide,
        detail
    )

    qh_score, qh_detail = compute_qhpi_from_mask(
        plaque_mask,
        tooth_mask_up,
        pixels_per_mm=None
    )

    qh_vis = draw_qhpi_visual(
        enhanced_rgb_masked,
        tooth_mask_up,
        plaque_mask,
        qh_score,
        qh_detail
    )

    plaque_mask_path = os.path.join(save_dir, f"{base}_plaque_mask.png")
    plaque_vis_path = os.path.join(save_dir, f"{base}_plaque_vis.png")
    php_vis_path = os.path.join(save_dir, f"{base}_php_shape_vis.png")
    qhpi_vis_path = os.path.join(save_dir, f"{base}_qhpi_vis.png")

    Image.fromarray(plaque_mask).save(plaque_mask_path)
    Image.fromarray(plaque_vis).save(plaque_vis_path)
    Image.fromarray(php_vis).save(php_vis_path)
    Image.fromarray(qh_vis).save(qhpi_vis_path)

    return {
        "tooth_id": tooth_id,
        "base": base,
        "tooth_name_th": TOOTH_NAME_TH.get(tooth_id, ""),
        "ratio": ratio,
        "plaque_area": plaque_area,
        "tooth_area": tooth_area,
        "php_score": php_score,
        "php_max": php_max,
        "qhpi_score": qh_score,
        "qh_detail": qh_detail,
        "detail": detail,
        "plaque_mask_path": plaque_mask_path,
        "plaque_vis_path": plaque_vis_path,
        "php_vis_path": php_vis_path,
        "qhpi_vis_path": qhpi_vis_path,
    }


# =========================
# Main
# =========================
def main():
    case_folders = [
        d
        for d in os.listdir(INPUT_ROOT)
        if os.path.isdir(os.path.join(INPUT_ROOT, d))
    ]

    tooth_results = []

    print("INPUT_ROOT =", INPUT_ROOT)
    print("FOUND CASES =", len(case_folders))

    for case_name in case_folders:
        case_dir = os.path.join(INPUT_ROOT, case_name)
        save_dir = os.path.join(OUTPUT_ROOT, case_name)

        os.makedirs(save_dir, exist_ok=True)

        print("\nCASE:", case_name)

        original_files = glob.glob(os.path.join(case_dir, "*_original.*"))
        segmented_files = glob.glob(os.path.join(case_dir, "*_segmented_teeth_only.*"))

        for f in original_files + segmented_files:
            try:
                shutil.copy2(f, save_dir)
                print(f"Copied: {os.path.basename(f)}")
            except Exception as e:
                print(f"Error copying {f}: {e}")

        report_lines = [f"Case: {case_name}\n\n"]

        total_php = 0
        total_php_max = 0
        total_qhpi = 0
        counted_teeth = 0
        case_results = []

        for tooth_file in TOOTH_FILES:
            tooth_path = os.path.join(case_dir, tooth_file)

            if not os.path.exists(tooth_path):
                report_lines.append(f"{tooth_file}: not found\n\n")
                continue

            result = process_tooth(
                tooth_path,
                save_dir,
                top_label="I",
                min_plaque_pixels=10,
                scale=2
            )

            tooth_results.append(result)
            case_results.append(result)

            total_php += result["php_score"]
            total_php_max += result["php_max"]
            total_qhpi += result["qhpi_score"]
            counted_teeth += 1

            detail = result["detail"]
            qh_detail = result["qh_detail"]

            report_lines.append(f"{tooth_file}\n")
            report_lines.append(
                f"  Plaque ratio : {result['ratio']:.4f} "
                f"({result['ratio'] * 100:.2f}%)\n"
            )
            report_lines.append(f"  PHP score    : {result['php_score']}/{result['php_max']}\n")
            report_lines.append(f"  QHPI score   : {result['qhpi_score']}/5\n")
            report_lines.append(f"  QH detail    : coverage={qh_detail['ratio'] * 100:.2f}%\n")
            report_lines.append(
                f"                 depth={qh_detail['vertical_depth_mm']:.2f} mm, "
                f"band={qh_detail['band_width_ratio'] * 100:.1f}%\n"
            )
            report_lines.append(
                f"                 yellow_line={qh_detail['score2_band_mm']:.1f} mm, "
                f"zone1={qh_detail['has_zone1']}, "
                f"zone2={qh_detail['has_zone2']}, "
                f"middle={qh_detail['has_middle']}, "
                f"incisal={qh_detail['has_incisal']}\n"
            )
            report_lines.append(
                "  PHP Zone detail: "
                + ", ".join([f"{z}={detail[z]['score']}" for z in detail.keys()])
                + "\n\n"
            )

        avg_php_normalized = (
            total_php / total_php_max * 5.0
            if total_php_max > 0
            else 0.0
        )

        avg_qhpi = (
            total_qhpi / counted_teeth
            if counted_teeth > 0
            else 0.0
        )

        report_lines.append(f"Total PHP (Raw)    : {total_php}/{total_php_max}\n")
        report_lines.append(f"Average PHP (0-5)  : {avg_php_normalized:.2f}\n")
        report_lines.append(f"Average QHPI (0-5) : {avg_qhpi:.2f}\n")

        summary_path = create_case_summary_image(
            case_name,
            case_results,
            save_dir
        )

        if summary_path is not None:
            report_lines.append(f"Summary image      : {os.path.basename(summary_path)}\n")
            print(f"Summary saved: {summary_path}")

        with open(
            os.path.join(save_dir, "plaque_php_qhpi_report.txt"),
            "w",
            encoding="utf-8"
        ) as f:
            f.writelines(report_lines)

        print(
            f"[DONE] {case_name} "
            f"AVG PHP = {avg_php_normalized:.2f} "
            f"AVG QHPI = {avg_qhpi:.2f}"
        )


if __name__ == "__main__":
    main()