import os
import cv2
import numpy as np
from PIL import Image

INPUT_ROOT = r"D:/Internship/Teeth Segment Result"
OUTPUT_ROOT = r"D:/Internship/Plaque Result"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

TOOTH_FILES = [
    "tooth11.png",
    "tooth12.png",
    "tooth13.png",
    "tooth21.png",
    "tooth22.png",
    "tooth23.png",
]

# =========================
# Utility
# =========================
def get_tooth_mask_from_rgba(image_rgba):
    alpha = image_rgba[:, :, 3]
    return alpha > 0

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
# Shape-based zone split
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

def make_shape_based_zone_masks(tooth_mask, top_label="I"):
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

    zone_masks = {
        "M": np.zeros((h, w), dtype=np.uint8),
        top_label: np.zeros((h, w), dtype=np.uint8),
        "C": np.zeros((h, w), dtype=np.uint8),
        "G": np.zeros((h, w), dtype=np.uint8),
        "D": np.zeros((h, w), dtype=np.uint8),
    }

    for y in range(y_min, y_max + 1):
        xl = left_x[y]
        xr = right_x[y]

        if xl < 0 or xr < 0 or xr <= xl:
            continue

        x_l_mid = get_x_on_row(left_x, right_x, y, 1/3)
        x_r_mid = get_x_on_row(left_x, right_x, y, 2/3)

        if x_l_mid is None or x_r_mid is None:
            continue

        zone_masks["M"][y, xl:x_l_mid] = 1
        zone_masks["D"][y, x_r_mid:xr+1] = 1

        if y < y1:
            zone_masks[top_label][y, x_l_mid:x_r_mid] = 1
        elif y < y2:
            zone_masks["C"][y, x_l_mid:x_r_mid] = 1
        else:
            zone_masks["G"][y, x_l_mid:x_r_mid] = 1

    for k in zone_masks:
        zone_masks[k] = (zone_masks[k] > 0) & tooth_mask

    guide = {
        "profile": profile,
        "y1": int(y1),
        "y2": int(y2),
        "top_label": top_label,
    }

    return zone_masks, guide

# =========================
# PHP Score from shape zones
# =========================
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

    return score, detail

# =========================
# Draw curved / tooth-shape guides
# =========================
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

        x1 = get_x_on_row(left_x, right_x, y, 1/3)
        x2 = get_x_on_row(left_x, right_x, y, 2/3)

        if x1 is not None:
            pts_left_mid.append([x1, y])
        if x2 is not None:
            pts_right_mid.append([x2, y])

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
            x1 = get_x_on_row(left_x, right_x, y, 1/3)
            x2 = get_x_on_row(left_x, right_x, y, 2/3)
            if x1 is not None and x2 is not None:
                if y == y1:
                    row1 = [[x, y] for x in range(x1, x2 + 1)]
                else:
                    row2 = [[x, y] for x in range(x1, x2 + 1)]

    if len(row1) > 1:
        cv2.polylines(out, [np.array(row1, dtype=np.int32)], False, line_color, 2)
    if len(row2) > 1:
        cv2.polylines(out, [np.array(row2, dtype=np.int32)], False, line_color, 2)

    zone_masks, _ = make_shape_based_zone_masks(tooth_mask, top_label=top_label)

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
# ด้านบนของภาพ = ด้านเหงือกของฟันบน
# =========================
def compute_qhpi_from_mask(plaque_mask, tooth_mask):
    """
    ใช้แนวคิด Quigley-Hein / Turesky แบบประมาณจาก mask:
    0 = ไม่มีคราบ
    1 = มีคราบเป็นจุดเล็กใกล้ขอบเหงือก
    2 = เป็นเส้นบางต่อเนื่องใกล้ขอบเหงือก
    3 = คราบครอบคลุม < 1/3 ของผิวฟัน
    4 = คราบครอบคลุม 1/3 ถึง < 2/3
    5 = คราบครอบคลุม >= 2/3
    """

    ys, xs = np.where(tooth_mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0, {
            "ratio": 0.0,
            "gingival_ratio": 0.0,
            "gingival_continuity": 0.0,
            "y_cut_1": 0,
            "y_cut_2": 0,
        }

    y_min = ys.min()
    y_max = ys.max()
    height = y_max - y_min + 1

    # ด้านบนของภาพ = gingival side
    gingival_h = max(1, int(round(height * 0.20)))
    y_g_end = min(y_max + 1, y_min + gingival_h)

    tooth_area = int(np.sum(tooth_mask))
    plaque_area = int(np.sum((plaque_mask > 0) & tooth_mask))
    ratio = plaque_area / tooth_area if tooth_area > 0 else 0.0

    gingival_band = np.zeros_like(tooth_mask, dtype=bool)
    gingival_band[y_min:y_g_end, :] = True
    gingival_band = gingival_band & tooth_mask

    gingival_area = int(np.sum(gingival_band))
    gingival_plaque = int(np.sum((plaque_mask > 0) & gingival_band))
    gingival_ratio = gingival_plaque / gingival_area if gingival_area > 0 else 0.0

    # ความต่อเนื่องตามแนวกว้างบริเวณเหงือก
    row_coverages = []
    for y in range(y_min, y_g_end):
        tooth_row = tooth_mask[y]
        plaque_row = (plaque_mask[y] > 0) & tooth_row

        tooth_w = int(np.sum(tooth_row))
        if tooth_w == 0:
            continue

        plaque_w = int(np.sum(plaque_row))
        row_coverages.append(plaque_w / tooth_w)

    gingival_continuity = max(row_coverages) if row_coverages else 0.0

    # ให้คะแนน
    if plaque_area == 0:
        score = 0
    elif ratio >= 2/3:
        score = 5
    elif ratio >= 1/3:
        score = 4
    elif ratio > 0:
        # น้อยกว่า 1/3 ให้แยกตามลักษณะใกล้ขอบเหงือก
        if gingival_plaque == 0:
            score = 3
        elif gingival_continuity >= 0.50:
            score = 2
        else:
            score = 1
    else:
        score = 0

    qh_detail = {
        "ratio": ratio,
        "gingival_ratio": gingival_ratio,
        "gingival_continuity": gingival_continuity,
        "y_cut_1": int(y_min + height / 3),
        "y_cut_2": int(y_min + 2 * height / 3),
        "y_min": int(y_min),
        "y_max": int(y_max),
        "y_g_end": int(y_g_end),
    }

    return score, qh_detail

def draw_qhpi_visual(image_rgb, tooth_mask, plaque_mask, qh_score, qh_detail):
    out = image_rgb.copy()

    colors = {
        0: (0, 255, 0),      # เขียว
        1: (120, 255, 0),
        2: (255, 255, 0),    # เหลือง
        3: (255, 170, 0),    # ส้ม
        4: (255, 100, 0),
        5: (255, 0, 0),      # แดง
    }

    overlay_color = np.array(colors[qh_score], dtype=np.float32)

    # overlay คราบตามสี score
    out_f = out.astype(np.float32)
    mask_bool = (plaque_mask > 0)
    alpha = 0.55
    out_f[mask_bool] = out_f[mask_bool] * (1 - alpha) + overlay_color * alpha
    out = out_f.astype(np.uint8)

    # วาดขอบฟัน
    contours, _ = cv2.findContours(
        (tooth_mask.astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)

    # เส้นแบ่งแนว coverage 1/3 และ 2/3
    y_cut_1 = qh_detail["y_cut_1"]
    y_cut_2 = qh_detail["y_cut_2"]
    y_min = qh_detail["y_min"]
    y_g_end = qh_detail["y_g_end"]

    for y_line, color in [(y_g_end, (255, 255, 255)), (y_cut_1, (0, 255, 0)), (y_cut_2, (0, 255, 0))]:
        xs = np.where(tooth_mask[y_line])[0] if 0 <= y_line < tooth_mask.shape[0] else []
        if len(xs) > 1:
            cv2.line(out, (int(xs.min()), y_line), (int(xs.max()), y_line), color, 2)

    # ข้อความ
    ratio_pct = qh_detail["ratio"] * 100
    ging_pct = qh_detail["gingival_ratio"] * 100
    cont_pct = qh_detail["gingival_continuity"] * 100

    cv2.putText(
        out,
        f"QHPI: {qh_score}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        out,
        f"Coverage: {ratio_pct:.1f}%",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        out,
        f"Gingival: {ging_pct:.1f}%  Cont: {cont_pct:.1f}%",
        (10, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return out

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

    Image.fromarray(rgba).save(os.path.join(save_dir, f"{base}_input_rgba.png"))
    Image.fromarray(rgb_masked).save(os.path.join(save_dir, f"{base}_input_rgb.png"))
    Image.fromarray(enhanced_rgb_masked).save(os.path.join(save_dir, f"{base}_input_rgb_enhanced.png"))

    plaque_mask = detect_plaque_mask(enhanced_rgb_masked, tooth_mask=tooth_mask_up)
    plaque_vis = visualize_plaque(enhanced_rgb_masked, plaque_mask)

    plaque_area = int(np.sum(plaque_mask > 0))
    tooth_area = int(np.sum(tooth_mask_up > 0))
    ratio = (plaque_area / tooth_area) if tooth_area > 0 else 0.0

    # PHP
    zone_masks, guide = make_shape_based_zone_masks(tooth_mask_up, top_label=top_label)
    php_score, detail = score_php_from_zone_masks(
        plaque_mask,
        zone_masks,
        min_plaque_pixels=min_plaque_pixels
    )
    php_vis = draw_shape_based_php_zones(plaque_vis, tooth_mask_up, guide, detail)

    # QHPI
    qh_score, qh_detail = compute_qhpi_from_mask(plaque_mask, tooth_mask_up)
    qh_vis = draw_qhpi_visual(enhanced_rgb_masked, tooth_mask_up, plaque_mask, qh_score, qh_detail)

    Image.fromarray(plaque_mask).save(os.path.join(save_dir, f"{base}_plaque_mask.png"))
    Image.fromarray(plaque_vis).save(os.path.join(save_dir, f"{base}_plaque_vis.png"))
    Image.fromarray(php_vis).save(os.path.join(save_dir, f"{base}_php_shape_vis.png"))
    Image.fromarray(qh_vis).save(os.path.join(save_dir, f"{base}_qhpi_vis.png"))

    return {
        "ratio": ratio,
        "plaque_area": plaque_area,
        "tooth_area": tooth_area,
        "php_score": php_score,
        "qhpi_score": qh_score,
        "qh_detail": qh_detail,
        "detail": detail,
    }

# =========================
# Main
# =========================
def main():
    case_folders = [
        d for d in os.listdir(INPUT_ROOT)
        if os.path.isdir(os.path.join(INPUT_ROOT, d))
    ]

    print("INPUT_ROOT =", INPUT_ROOT)
    print("FOUND CASES =", len(case_folders))

    for case_name in case_folders:
        case_dir = os.path.join(INPUT_ROOT, case_name)
        save_dir = os.path.join(OUTPUT_ROOT, case_name)
        os.makedirs(save_dir, exist_ok=True)

        print("\nCASE:", case_name)

        report_lines = [f"Case: {case_name}\n\n"]

        total_php = 0
        total_qhpi = 0
        counted_teeth = 0

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

            total_php += result["php_score"]
            total_qhpi += result["qhpi_score"]
            counted_teeth += 1

            detail = result["detail"]
            qh_detail = result["qh_detail"]

            report_lines.append(f"{tooth_file}\n")
            report_lines.append(f"  Plaque ratio : {result['ratio']:.4f} ({result['ratio'] * 100:.2f}%)\n")
            report_lines.append(f"  Plaque area  : {result['plaque_area']} pixels\n")
            report_lines.append(f"  Tooth area   : {result['tooth_area']} pixels\n")
            report_lines.append(f"  PHP score    : {result['php_score']}/5\n")
            report_lines.append(f"  QHPI score   : {result['qhpi_score']}/5\n")
            report_lines.append(
                f"  QH detail    : coverage={qh_detail['ratio']*100:.2f}%, "
                f"gingival={qh_detail['gingival_ratio']*100:.2f}%, "
                f"continuity={qh_detail['gingival_continuity']*100:.2f}%\n"
            )
            report_lines.append(
                "  Zone detail  : " +
                ", ".join([f"{z}={detail[z]['score']}" for z in detail.keys()]) + "\n"
            )
            report_lines.append(
                "  Zone pixels  : " +
                ", ".join([f"{z}({detail[z]['plaque_pixels']}/{detail[z]['zone_pixels']})" for z in detail.keys()]) + "\n\n"
            )

        avg_php = (total_php / counted_teeth) if counted_teeth > 0 else 0.0
        avg_qhpi = (total_qhpi / counted_teeth) if counted_teeth > 0 else 0.0

        report_lines.append(f"Average PHP score  : {avg_php:.2f}\n")
        report_lines.append(f"Average QHPI score : {avg_qhpi:.2f}\n")

        with open(os.path.join(save_dir, "plaque_php_qhpi_report.txt"), "w", encoding="utf-8") as f:
            f.writelines(report_lines)

        print("[DONE]", case_name, "AVG PHP =", f"{avg_php:.2f}", "AVG QHPI =", f"{avg_qhpi:.2f}")

if __name__ == "__main__":
    main()