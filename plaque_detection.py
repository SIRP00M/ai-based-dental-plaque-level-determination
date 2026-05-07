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
    """
    สำหรับทุกแถว y หา x ซ้ายสุด/ขวาสุดของตัวฟัน
    """
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
    """
    frac = 0.0 -> ขอบซ้าย
    frac = 1.0 -> ขอบขวา
    """
    xl = left_x[y]
    xr = right_x[y]
    if xl < 0 or xr < 0 or xr <= xl:
        return None
    return int(round(xl + frac * (xr - xl)))

def make_shape_based_zone_masks(tooth_mask, top_label="I"):
    """
    แบ่งโซนตามทรงฟันจริง:
    - M = ด้านซ้าย 1/3 ของความกว้างฟันในแต่ละแถว
    - D = ด้านขวา 1/3
    - ตรงกลางแบ่งแนวตั้งเป็น top / C / G
    """
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

        # Mesial
        zone_masks["M"][y, xl:x_l_mid] = 1

        # Distal
        zone_masks["D"][y, x_r_mid:xr+1] = 1

        # Middle column แบ่งเป็น 3 ส่วนแนวตั้ง
        if y < y1:
            zone_masks[top_label][y, x_l_mid:x_r_mid] = 1
        elif y < y2:
            zone_masks["C"][y, x_l_mid:x_r_mid] = 1
        else:
            zone_masks["G"][y, x_l_mid:x_r_mid] = 1

    # บังคับให้เหลือเฉพาะพื้นที่ฟันจริง
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

    # ขอบฟัน
    contours, _ = cv2.findContours(
        (tooth_mask.astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(out, contours, -1, line_color, 2)

    # เส้นแบ่งแนวตั้งตามทรงฟัน (1/3 และ 2/3 ของความกว้างในแต่ละแถว)
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

    # เส้นแบ่งแนวนอนในคอลัมน์กลาง
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

    # จุดกึ่งกลางแต่ละ zone สำหรับวาง text
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
# Process Each Tooth
# =========================
def process_tooth(tooth_path, save_dir, top_label="I", min_plaque_pixels=1):
    print("READING:", tooth_path)

    rgba = np.array(Image.open(tooth_path).convert("RGBA"))
    rgb = rgba[:, :, :3].copy()
    tooth_mask = get_tooth_mask_from_rgba(rgba)

    rgb_masked = np.zeros_like(rgb)
    rgb_masked[tooth_mask] = rgb[tooth_mask]

    base = os.path.splitext(os.path.basename(tooth_path))[0]

    Image.fromarray(rgba).save(os.path.join(save_dir, f"{base}_input_rgba.png"))
    Image.fromarray(rgb_masked).save(os.path.join(save_dir, f"{base}_input_rgb.png"))

    plaque_mask = detect_plaque_mask(rgb_masked, tooth_mask=tooth_mask)
    plaque_vis = visualize_plaque(rgb_masked, plaque_mask)

    plaque_area = int(np.sum(plaque_mask > 0))
    tooth_area = int(np.sum(tooth_mask > 0))
    ratio = (plaque_area / tooth_area) if tooth_area > 0 else 0.0

    zone_masks, guide = make_shape_based_zone_masks(tooth_mask, top_label=top_label)
    php_score, detail = score_php_from_zone_masks(
        plaque_mask,
        zone_masks,
        min_plaque_pixels=min_plaque_pixels
    )

    php_vis = draw_shape_based_php_zones(plaque_vis, tooth_mask, guide, detail)

    Image.fromarray(plaque_mask).save(os.path.join(save_dir, f"{base}_plaque_mask.png"))
    Image.fromarray(plaque_vis).save(os.path.join(save_dir, f"{base}_plaque_vis.png"))
    Image.fromarray(php_vis).save(os.path.join(save_dir, f"{base}_php_shape_vis.png"))

    return {
        "ratio": ratio,
        "plaque_area": plaque_area,
        "tooth_area": tooth_area,
        "php_score": php_score,
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
        counted_teeth = 0

        for tooth_file in TOOTH_FILES:
            tooth_path = os.path.join(case_dir, tooth_file)

            if not os.path.exists(tooth_path):
                report_lines.append(f"{tooth_file}: not found\n\n")
                continue

            # ฟันหน้ากำหนด top_label = I
            # ถ้าเป็นฟันกรามเปลี่ยนเป็น O
            result = process_tooth(
                tooth_path,
                save_dir,
                top_label="I",
                min_plaque_pixels=10
            )

            total_php += result["php_score"]
            counted_teeth += 1

            detail = result["detail"]

            report_lines.append(f"{tooth_file}\n")
            report_lines.append(f"  Plaque ratio : {result['ratio']:.4f} ({result['ratio'] * 100:.2f}%)\n")
            report_lines.append(f"  Plaque area  : {result['plaque_area']} pixels\n")
            report_lines.append(f"  Tooth area   : {result['tooth_area']} pixels\n")
            report_lines.append(f"  PHP score    : {result['php_score']}/5\n")
            report_lines.append(
                "  Zone detail  : " +
                ", ".join([f"{z}={detail[z]['score']}" for z in detail.keys()]) + "\n"
            )
            report_lines.append(
                "  Zone pixels  : " +
                ", ".join([f"{z}({detail[z]['plaque_pixels']}/{detail[z]['zone_pixels']})" for z in detail.keys()]) + "\n\n"
            )

        avg_php = (total_php / counted_teeth) if counted_teeth > 0 else 0.0
        report_lines.append(f"Average PHP score: {avg_php:.2f}\n")

        with open(os.path.join(save_dir, "plaque_php_shape_report.txt"), "w", encoding="utf-8") as f:
            f.writelines(report_lines)

        print("[DONE]", case_name, "AVG PHP =", f"{avg_php:.2f}")

if __name__ == "__main__":
    main()