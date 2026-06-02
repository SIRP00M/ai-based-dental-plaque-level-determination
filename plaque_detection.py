# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import cv2
import numpy as np
from PIL import Image
import shutil
import glob

# Thai tooth names
from plaque_summary import TOOTH_NAME_TH

# Calculation modules
from plaque_calculation import (
    get_tooth_mask_from_rgba,
    detect_plaque_mask,
    make_shape_based_zone_masks,
    score_php_from_zone_masks,
    compute_qhpi_from_mask
)

# Visualization modules
from plaque_visualization import (
    enhance_tooth_image,
    visualize_plaque,
    draw_shape_based_php_zones,
    draw_qhpi_visual
)


# ==============================================================================
# PATH SETTINGS
# ==============================================================================

INPUT_ROOT = r"D:/Internship/Teeth Segment Result"
OUTPUT_ROOT = r"D:/Internship/Plaque Result Curves"

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ==============================================================================
# TARGET TEETH
# ==============================================================================

TOOTH_FILES = [
    "tooth11.png",
    "tooth12.png",
    "tooth13.png",
    "tooth21.png",
    "tooth22.png",
    "tooth23.png",
]

# Note:
# Summary dashboard generation has been moved to plaque_summary.py


# ==============================================================================
# PROCESS SINGLE TOOTH
# ==============================================================================

def process_tooth(tooth_path, save_dir, top_label="I", min_plaque_pixels=1, scale=2):

    print("READING:", tooth_path)

    # Load RGBA tooth image
    rgba = np.array(Image.open(tooth_path).convert("RGBA"))
    rgb = rgba[:, :, :3].copy()

    # Extract tooth mask from alpha channel
    tooth_mask = get_tooth_mask_from_rgba(rgba)

    # Keep only tooth region
    rgb_masked = np.zeros_like(rgb)
    rgb_masked[tooth_mask] = rgb[tooth_mask]

    # Enhance image before plaque detection
    enhanced_rgb = enhance_tooth_image(rgb_masked, scale=scale)

    # Resize mask to match enhanced image
    tooth_mask_up = cv2.resize(
        tooth_mask.astype(np.uint8),
        (enhanced_rgb.shape[1], enhanced_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST
    ) > 0

    enhanced_rgb_masked = np.zeros_like(enhanced_rgb)
    enhanced_rgb_masked[tooth_mask_up] = enhanced_rgb[tooth_mask_up]

    base = os.path.splitext(os.path.basename(tooth_path))[0]

    # Extract tooth ID from filename
    tooth_id = None

    try:
        tooth_id = int("".join(filter(str.isdigit, base)))
    except:
        pass

    # Save preprocessing images
    Image.fromarray(rgba).save(os.path.join(save_dir, f"{base}_input_rgba.png"))
    Image.fromarray(rgb_masked).save(os.path.join(save_dir, f"{base}_input_rgb.png"))

    Image.fromarray(enhanced_rgb_masked).save(
        os.path.join(save_dir, f"{base}_input_rgb_enhanced.png")
    )

    # Detect plaque mask
    plaque_mask = detect_plaque_mask(
        enhanced_rgb_masked,
        tooth_mask=tooth_mask_up
    )

    # Create plaque overlay visualization
    plaque_vis = visualize_plaque(enhanced_rgb_masked, plaque_mask)

    # Plaque coverage ratio
    plaque_area = int(np.sum(plaque_mask > 0))
    tooth_area = int(np.sum(tooth_mask_up > 0))

    ratio = plaque_area / tooth_area if tooth_area > 0 else 0.0

    # Generate adaptive PHP zones
    zone_masks, guide = make_shape_based_zone_masks(
        tooth_mask_up,
        tooth_id=tooth_id,
        top_label=top_label
    )

    # Compute PHP score
    php_score, php_max, detail = score_php_from_zone_masks(
        plaque_mask,
        zone_masks,
        min_plaque_pixels=min_plaque_pixels
    )

    # Draw PHP visualization
    php_vis = draw_shape_based_php_zones(
        plaque_vis,
        tooth_mask_up,
        guide,
        detail
    )

    # Compute QHPI score
    qh_score, qh_detail = compute_qhpi_from_mask(
        plaque_mask,
        tooth_mask_up,
        pixels_per_mm=None
    )

    # Draw QHPI visualization
    qh_vis = draw_qhpi_visual(
        enhanced_rgb_masked,
        tooth_mask_up,
        plaque_mask,
        qh_score,
        qh_detail
    )

    # Output paths
    plaque_mask_path = os.path.join(save_dir, f"{base}_plaque_mask.png")
    plaque_vis_path = os.path.join(save_dir, f"{base}_plaque_vis.png")
    php_vis_path = os.path.join(save_dir, f"{base}_php_shape_vis.png")
    qhpi_vis_path = os.path.join(save_dir, f"{base}_qhpi_vis.png")

    # Save result images
    Image.fromarray(plaque_mask).save(plaque_mask_path)
    Image.fromarray(plaque_vis).save(plaque_vis_path)
    Image.fromarray(php_vis).save(php_vis_path)
    Image.fromarray(qh_vis).save(qhpi_vis_path)

    # Return structured result
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


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():

    # Find all case folders
    case_folders = [
        d
        for d in os.listdir(INPUT_ROOT)
        if os.path.isdir(os.path.join(INPUT_ROOT, d))
    ]

    tooth_results = []

    print("INPUT_ROOT =", INPUT_ROOT)
    print("FOUND CASES =", len(case_folders))

    # Process each case
    for case_name in case_folders:

        case_dir = os.path.join(INPUT_ROOT, case_name)
        save_dir = os.path.join(OUTPUT_ROOT, case_name)

        os.makedirs(save_dir, exist_ok=True)

        print("\nCASE:", case_name)

        # Copy original images for reference
        original_files = glob.glob(os.path.join(case_dir, "*_original.*"))

        segmented_files = glob.glob(
            os.path.join(case_dir, "*_segmented_teeth_only.*")
        )

        for f in original_files + segmented_files:

            try:
                shutil.copy2(f, save_dir)
                print(f"Copied: {os.path.basename(f)}")

            except Exception as e:
                print(f"Error copying {f}: {e}")

        report_lines = [f"Case: {case_name}\n\n"]

        # Case summary stats
        total_php = 0
        total_php_max = 0
        total_qhpi = 0
        counted_teeth = 0

        case_results = []

        # Process each tooth
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

            # Update summary scores
            total_php += result["php_score"]
            total_php_max += result["php_max"]
            total_qhpi += result["qhpi_score"]
            counted_teeth += 1

            detail = result["detail"]
            qh_detail = result["qh_detail"]

            # Report output
            report_lines.append(f"{tooth_file}\n")

            report_lines.append(
                f"  Plaque ratio : {result['ratio']:.4f} "
                f"({result['ratio'] * 100:.2f}%)\n"
            )

            report_lines.append(
                f"  PHP score    : {result['php_score']}/{result['php_max']}\n"
            )

            report_lines.append(
                f"  QHPI score   : {result['qhpi_score']}/5\n"
            )

            report_lines.append(
                f"  QH detail    : coverage={qh_detail['ratio'] * 100:.2f}%\n"
            )

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

        # Normalize PHP to 0-5 scale
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

        # Case summary lines
        report_lines.append(f"Total PHP (Raw)    : {total_php}/{total_php_max}\n")

        report_lines.append(
            f"Average PHP (0-5)  : {avg_php_normalized:.2f}\n"
        )

        report_lines.append(
            f"Average QHPI (0-5) : {avg_qhpi:.2f}\n"
        )

        # Save JSON for dashboard generation
        import json

        class NpEncoder(json.JSONEncoder):

            def default(self, obj):

                if isinstance(obj, np.ndarray):
                    return obj.tolist()

                return super(NpEncoder, self).default(obj)

        json_path = os.path.join(save_dir, "case_results.json")

        try:
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(
                    case_results,
                    jf,
                    cls=NpEncoder,
                    indent=4,
                    ensure_ascii=False
                )

            print(f"Saved case results JSON: {json_path}")

        except Exception as e:
            print(f"Error saving case results JSON: {e}")

        # Save TXT report
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


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()