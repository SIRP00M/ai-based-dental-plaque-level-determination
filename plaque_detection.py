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

def get_tooth_mask_from_rgba(image_rgba):
    alpha = image_rgba[:, :, 3]
    return alpha > 0

def detect_plaque_mask(image_rgb, tooth_mask):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    A = lab[:, :, 1]   # แดง-เขียว
    B = lab[:, :, 2]   # เหลือง-น้ำเงิน

    # ---------------------------
    # 1) หลายช่วงสีของคราบ
    # ---------------------------
    # ชมพู/ม่วงแดง
    mask_pink = cv2.inRange(hsv,
                            np.array([135, 25, 40], dtype=np.uint8),
                            np.array([179, 255, 255], dtype=np.uint8))

    # ม่วง
    mask_purple = cv2.inRange(hsv,
                              np.array([120, 20, 30], dtype=np.uint8),
                              np.array([155, 255, 255], dtype=np.uint8))

    # ม่วงอมฟ้า/น้ำเงิน
    mask_bluepurple = cv2.inRange(hsv,
                                  np.array([100, 20, 20], dtype=np.uint8),
                                  np.array([135, 255, 255], dtype=np.uint8))

    color_mask = cv2.bitwise_or(mask_pink, mask_purple)
    color_mask = cv2.bitwise_or(color_mask, mask_bluepurple)

    # ---------------------------
    # 2) ตัดขาวล้วน / มืดเกิน
    # ---------------------------
    valid_sv = (S > 18) & (V > 35)

    # ---------------------------
    # 3) ใช้ LAB ช่วย
    # A สูง = อมแดง/ม่วงมากขึ้น
    # B ต่ำ = อมน้ำเงินมากขึ้น
    # ---------------------------
    cond_lab = (A > 128) | (B < 135)

    # ---------------------------
    # 4) เฉพาะในฟัน
    # ---------------------------
    mask = (color_mask > 0) & valid_sv & cond_lab & tooth_mask

    mask = (mask.astype(np.uint8) * 255)

    # ---------------------------
    # 5) morphology
    # ---------------------------
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # ---------------------------
    # 6) ลบจุดเล็ก
    # ---------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    min_area = 20
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned

def visualize_plaque(image_rgb, plaque_mask, alpha=0.45):
    out = image_rgb.copy().astype(np.float32)
    red = np.zeros_like(out)
    red[:, :, 0] = 255

    mask_bool = plaque_mask > 0
    out[mask_bool] = out[mask_bool] * (1 - alpha) + red[mask_bool] * alpha

    return out.astype(np.uint8)

def process_tooth(tooth_path, save_dir):
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

    plaque_area = np.sum(plaque_mask > 0)
    tooth_area = np.sum(tooth_mask > 0)
    ratio = (plaque_area / tooth_area) if tooth_area > 0 else 0.0

    Image.fromarray(plaque_mask).save(os.path.join(save_dir, f"{base}_plaque_mask.png"))
    Image.fromarray(vis).save(os.path.join(save_dir, f"{base}_plaque_vis.png"))

    return ratio

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

        report_lines = [f"Case: {case_name}\n"]

        for tooth_file in TOOTH_FILES:
            tooth_path = os.path.join(case_dir, tooth_file)

            if not os.path.exists(tooth_path):
                print("MISSING:", tooth_path)
                report_lines.append(f"{tooth_file}: not found\n")
                continue

            ratio = process_tooth(tooth_path, save_dir)
            report_lines.append(f"{tooth_file}: {ratio:.4f} ({ratio*100:.2f}%)\n")

        with open(os.path.join(save_dir, "plaque_report.txt"), "w", encoding="utf-8") as f:
            f.writelines(report_lines)

        print("[DONE]", case_name)

if __name__ == "__main__":
    main()