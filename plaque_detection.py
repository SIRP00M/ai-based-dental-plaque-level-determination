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

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    A = lab[:, :, 1]
    B = lab[:, :, 2]

    # 1) ช่วงสีหลายโซน
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

    # 2) คัดความเข้มสี
    sat_mask = S > 25
    val_mask = V > 35

    # 3) LAB ช่วยคัด
    lab_mask = (A > 120) & (B < 150)

    # 4) รวมเงื่อนไข
    mask = color_mask & sat_mask & val_mask & lab_mask & tooth_mask
    mask = (mask.astype(np.uint8) * 255)

    # 5) Morphology
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # 6) ลบจุดเล็กเกิน
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
# PHP Zone Split
# =========================
def split_tooth_zones(tooth_mask, top_label="I"):
    """
    แบ่งฟันเป็น 5 zone ตามหลัก PHP:
    M = Mesial
    D = Distal
    top_label = I หรือ O
    C = Center
    G = Gingival
    """
    ys, xs = np.where(tooth_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # +1 กันเคส slice หายขอบสุดท้าย
    x_max += 1
    y_max += 1

    w = x_max - x_min
    h = y_max - y_min

    x1 = x_min + w // 3
    x2 = x_min + (2 * w) // 3

    y1 = y_min + h // 3
    y2 = y_min + (2 * h) // 3

    zones = {
        "M": (slice(y_min, y_max), slice(x_min, x1)),
        top_label: (slice(y_min, y1), slice(x1, x2)),
        "C": (slice(y1, y2), slice(x1, x2)),
        "G": (slice(y2, y_max), slice(x1, x2)),
        "D": (slice(y_min, y_max), slice(x2, x_max)),
    }

    return zones

# =========================
# PHP Score
# =========================
def score_php(plaque_mask, tooth_mask, zones, min_plaque_pixels=1):
    """
    ให้คะแนนแบบ PHP:
    - แต่ละ zone ถ้ามี plaque => 1
    - ไม่มี => 0
    """
    score = 0
    detail = {}

    for zone_name, (ys, xs) in zones.items():
        zone_plaque = plaque_mask[ys, xs]
        zone_tooth = tooth_mask[ys, xs]

        # เอาเฉพาะพื้นที่ที่อยู่ในตัวฟันจริง
        valid_pixels = zone_tooth > 0
        plaque_pixels = np.sum((zone_plaque > 0) & valid_pixels)

        has_plaque = plaque_pixels >= min_plaque_pixels

        detail[zone_name] = {
            "score": 1 if has_plaque else 0,
            "plaque_pixels": int(plaque_pixels),
            "zone_pixels": int(np.sum(valid_pixels)),
        }

        score += detail[zone_name]["score"]

    return score, detail

# =========================
# Draw PHP Zones
# =========================
def draw_php_zones(image_rgb, tooth_mask, zones, detail):
    out = image_rgb.copy()

    # วาดเส้นแบ่ง zone
    ys, xs = np.where(tooth_mask)
    if len(xs) == 0 or len(ys) == 0:
        return out

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x_max += 1
    y_max += 1

    w = x_max - x_min
    h = y_max - y_min

    x1 = x_min + w // 3
    x2 = x_min + (2 * w) // 3
    y1 = y_min + h // 3
    y2 = y_min + (2 * h) // 3

    line_color = (0, 255, 0)  # เขียว
    text_color = (255, 255, 0)  # ฟ้าอมเหลือง

    # เส้นแนวตั้ง
    cv2.line(out, (x1, y_min), (x1, y_max), line_color, 1)
    cv2.line(out, (x2, y_min), (x2, y_max), line_color, 1)

    # เส้นแนวนอนเฉพาะคอลัมน์กลาง
    cv2.line(out, (x1, y1), (x2, y1), line_color, 1)
    cv2.line(out, (x1, y2), (x2, y2), line_color, 1)

    # ใส่ข้อความคะแนนแต่ละ zone
    for zone_name, (ys_s, xs_s) in zones.items():
        cy = (ys_s.start + ys_s.stop) // 2
        cx = (xs_s.start + xs_s.stop) // 2
        zone_score = detail[zone_name]["score"]
        label = f"{zone_name}:{zone_score}"

        cv2.putText(
            out,
            label,
            (cx - 18, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            text_color,
            1,
            cv2.LINE_AA
        )

    return out

# =========================
# Process Each Tooth
# =========================
def process_tooth(tooth_path, save_dir, top_label="I"):
    print("READING:", tooth_path)

    rgba = np.array(Image.open(tooth_path).convert("RGBA"))
    print("SHAPE:", rgba.shape, "FILE:", tooth_path)

    rgb = rgba[:, :, :3].copy()
    tooth_mask = get_tooth_mask_from_rgba(rgba)

    # ลบนอกฟันออกจาก RGB
    rgb_masked = np.zeros_like(rgb)
    rgb_masked[tooth_mask] = rgb[tooth_mask]

    base = os.path.splitext(os.path.basename(tooth_path))[0]

    Image.fromarray(rgba).save(os.path.join(save_dir, f"{base}_input_rgba.png"))
    Image.fromarray(rgb_masked).save(os.path.join(save_dir, f"{base}_input_rgb.png"))

    plaque_mask = detect_plaque_mask(rgb_masked, tooth_mask=tooth_mask)
    vis = visualize_plaque(rgb_masked, plaque_mask)

    # ratio เดิม
    plaque_area = np.sum(plaque_mask > 0)
    tooth_area = np.sum(tooth_mask > 0)
    ratio = (plaque_area / tooth_area) if tooth_area > 0 else 0.0

    # PHP
    zones = split_tooth_zones(tooth_mask, top_label=top_label)
    php_score, detail = score_php(plaque_mask, tooth_mask, zones, min_plaque_pixels=1)
    zone_vis = draw_php_zones(vis, tooth_mask, zones, detail)

    Image.fromarray(plaque_mask).save(os.path.join(save_dir, f"{base}_plaque_mask.png"))
    Image.fromarray(vis).save(os.path.join(save_dir, f"{base}_plaque_vis.png"))
    Image.fromarray(zone_vis).save(os.path.join(save_dir, f"{base}_php_vis.png"))

    return {
        "ratio": ratio,
        "plaque_area": int(plaque_area),
        "tooth_area": int(tooth_area),
        "php_score": int(php_score),
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
        print("CASE_DIR:", case_dir)

        report_lines = [f"Case: {case_name}\n\n"]

        total_php = 0
        counted_teeth = 0

        for tooth_file in TOOTH_FILES:
            tooth_path = os.path.join(case_dir, tooth_file)

            if not os.path.exists(tooth_path):
                print("MISSING:", tooth_path)
                report_lines.append(f"{tooth_file}: not found\n\n")
                continue

            # ฟันชุดนี้เป็น anterior เลยใช้ I
            result = process_tooth(tooth_path, save_dir, top_label="I")

            total_php += result["php_score"]
            counted_teeth += 1

            detail = result["detail"]

            report_lines.append(f"{tooth_file}\n")
            report_lines.append(f"  Plaque ratio : {result['ratio']:.4f} ({result['ratio'] * 100:.2f}%)\n")
            report_lines.append(f"  Plaque area  : {result['plaque_area']} pixels\n")
            report_lines.append(f"  Tooth area   : {result['tooth_area']} pixels\n")
            report_lines.append(f"  PHP score    : {result['php_score']}/5\n")
            report_lines.append(
                "  Zone detail  : "
                + ", ".join(
                    [f"{z}={detail[z]['score']}" for z in detail.keys()]
                )
                + "\n"
            )
            report_lines.append(
                "  Zone pixels  : "
                + ", ".join(
                    [f"{z}({detail[z]['plaque_pixels']}/{detail[z]['zone_pixels']})" for z in detail.keys()]
                )
                + "\n\n"
            )

        if counted_teeth > 0:
            avg_php = total_php / counted_teeth
        else:
            avg_php = 0.0

        report_lines.append(f"Average PHP score: {avg_php:.2f}\n")

        with open(os.path.join(save_dir, "plaque_php_report.txt"), "w", encoding="utf-8") as f:
            f.writelines(report_lines)

        print("[DONE]", case_name, "AVG PHP =", f"{avg_php:.2f}")

if __name__ == "__main__":
    main()