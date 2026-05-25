"""
master_run_pipeline.py

Dental Plaque Pipeline แบบแยก 3 ขั้นตอนชัดเจน:
1) AI Teeth Segmentation / Mask R-CNN              -> Run_Model_1Class.py
2) Plaque Detection + PHP + QHPI + Report TXT     -> plaque_detection_curve.py
3) Summary Dashboard Image                         -> plaque_summary.py

แนวคิดสำคัญ:
- Step 2 ไม่ต้องตรวจ Summary image แล้ว เพราะแยกให้ Step 3 เป็นคนสร้างภาพ Summary โดยตรง
- Sync path ให้ไฟล์ย่อยทั้ง 3 ตัวก่อนรัน
- ใช้ Python executable เดียวกับ Spyder/Conda env ปัจจุบัน
- แสดง log สดระหว่างรัน subprocess
"""

import os
import re
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict


# ==============================================================================
# 0) Auto-detect Project Paths
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RUN_MODEL_SCRIPT = SCRIPT_DIR / "Run_Model_1Class.py"
PLAQUE_DETECTION_SCRIPT = SCRIPT_DIR / "plaque_detection_curve.py"
PLAQUE_SUMMARY_SCRIPT = SCRIPT_DIR / "plaque_summary.py"

# เลือก input image folder อัตโนมัติจาก candidate เหล่านี้
INPUT_IMAGE_CANDIDATES = [
    PROJECT_ROOT / "Data",
    PROJECT_ROOT / "Data Class 1" / "test",
    PROJECT_ROOT / "Data Class 1",
    SCRIPT_DIR / "Data",
]

# weight model
WEIGHT_FILE_CANDIDATES = [
    PROJECT_ROOT / "h5" / "mask_rcnn_tooth_single_class_0097.h5",
    SCRIPT_DIR / "h5" / "mask_rcnn_tooth_single_class_0097.h5",
    Path(r"D:/Internship/h5/mask_rcnn_tooth_single_class_0097.h5"),
    Path(r"D:/Internship Test/h5/mask_rcnn_tooth_single_class_0097.h5"),
]

TEETH_SEGMENT_RESULT_DIR = PROJECT_ROOT / "Teeth Segment Result"
PLAQUE_RESULT_DIR = PROJECT_ROOT / "Plaque Result Curves"

# โหมดข้ามการรัน AI ถ้ามีผลลัพธ์อยู่แล้ว
SKIP_AI_IF_EXISTS = True

SUMMARY_FILENAMES = {
    "case_summary_php_qhpi.png",
    "case_summary.png",
    "summary.png",
    "summary_dashboard.png",
}
REPORT_FILENAMES = {
    "plaque_php_qhpi_report.txt",
    "plaque_report.txt",
}


# ==============================================================================
# 1) Print / Helper
# ==============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_time(label: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {label}")


def safe_read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp874", "latin-1", "utf-16"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def safe_write_text(path: Path, text: str) -> None:
    # ใช้ utf-8 เพื่อไม่ให้ภาษาไทยพัง
    path.write_text(text, encoding="utf-8")


def backup_once(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists() and path.exists():
        shutil.copy2(path, bak)
        print(f"[BACKUP] {path.name} -> {bak.name}")


def normalize_path_for_py(path: Path) -> str:
    # ใช้ / เพื่อกันปัญหา escape บน Windows
    return str(path).replace("\\", "/")


def first_existing_file(candidates):
    for p in candidates:
        if Path(p).is_file():
            return Path(p)
    return None


def first_existing_dir_with_images(candidates):
    for p in candidates:
        p = Path(p)
        if p.is_dir() and count_images(p) > 0:
            return p
    return None


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png"}
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def count_case_folders(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_dir())


def find_case_folders(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_dir()])


def count_reports(folder: Path) -> int:
    count = 0
    for case_dir in find_case_folders(folder):
        if any((case_dir / name).exists() for name in REPORT_FILENAMES):
            count += 1
        else:
            count += len(list(case_dir.glob("*report*.txt")))
    return count


def count_summary_images(folder: Path) -> int:
    count = 0
    for case_dir in find_case_folders(folder):
        direct = any((case_dir / name).exists() for name in SUMMARY_FILENAMES)
        fuzzy = bool(list(case_dir.glob("*summary*.png"))) or bool(list(case_dir.glob("*dashboard*.png")))
        if direct or fuzzy:
            count += 1
    return count


def has_valid_ai_output(folder: Path) -> bool:
    """เช็คว่ามี case folder อย่างน้อย 1 โฟลเดอร์ที่มีไฟล์ tooth ครบทั้ง 6 ซี่หรือไม่"""
    if not folder.exists():
        return False
    required_teeth = {"tooth11.png", "tooth12.png", "tooth13.png", "tooth21.png", "tooth22.png", "tooth23.png"}
    for case_dir in find_case_folders(folder):
        existing_files = {p.name.lower() for p in case_dir.iterdir() if p.is_file()}
        if required_teeth.issubset(existing_files):
            return True
    return False


def has_valid_detection_output(folder: Path) -> bool:
    """เช็คว่ามี case folder อย่างน้อย 1 โฟลเดอร์ที่มีไฟล์ report หรือ case_results.json หรือไม่"""
    if not folder.exists():
        return False
    for case_dir in find_case_folders(folder):
        has_txt = any((case_dir / name).exists() for name in REPORT_FILENAMES)
        has_fuzzy_txt = bool(list(case_dir.glob("*report*.txt")))
        has_json = (case_dir / "case_results.json").exists()
        if has_txt or has_fuzzy_txt or has_json:
            return True
    return False


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"ไม่พบ {label}: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"ไม่พบ {label}: {path}")


# ==============================================================================
# 2) Patch Python Constants
# ==============================================================================

def replace_string_constant(content: str, var_name: str, value: Path) -> Tuple[str, bool]:
    """แทนค่า VAR = r"..." ถ้ามีในไฟล์"""
    value_str = normalize_path_for_py(value)
    pattern = rf'(^\s*{re.escape(var_name)}\s*=\s*)(?:r|R)?(["\']).*?\2'
    repl = rf'\1r"{value_str}"'
    new_content, n = re.subn(pattern, repl, content, flags=re.MULTILINE)
    return new_content, n > 0


def patch_file_constants(script_path: Path, mapping: Dict[str, Path], label: str) -> None:
    print_header(f"PATCH: {label}")
    require_file(script_path, label)

    content = safe_read_text(script_path)
    backup_once(script_path)

    changed_any = False
    for var_name, value in mapping.items():
        content, changed = replace_string_constant(content, var_name, value)
        if changed:
            changed_any = True
            print(f"[SYNC] {var_name} = {value}")
        else:
            print(f"[SKIP] ไม่พบตัวแปร {var_name} ใน {script_path.name}")

    if changed_any:
        safe_write_text(script_path, content)
        print(f"[OK] {script_path.name} sync แล้ว")
    else:
        print(f"[INFO] {script_path.name} ไม่มี constant ที่ต้องแก้ หรือใช้ path แบบอื่น")


# ==============================================================================
# 3) Run Subprocess with Live Log
# ==============================================================================

def run_script(script_path: Path, step_name: str) -> None:
    print_header(f"RUN: {step_name}")
    print_time(f"Start {script_path.name}")

    env = os.environ.copy()
    # เผื่อ summary/detection รุ่นใหม่อ่าน env var ได้
    env["DENTAL_INPUT_IMAGE_DIR"] = normalize_path_for_py(INPUT_IMAGE_DIR)
    env["DENTAL_TEETH_SEGMENT_RESULT_DIR"] = normalize_path_for_py(TEETH_SEGMENT_RESULT_DIR)
    env["DENTAL_PLAQUE_RESULT_DIR"] = normalize_path_for_py(PLAQUE_RESULT_DIR)

    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, str(script_path))

    print_time(f"Finished {step_name}")
    print(f"[OK] Return code = {process.returncode}")


# ==============================================================================
# 4) Validation
# ==============================================================================

def validate_before_run() -> None:
    print_header("VALIDATE BEFORE RUN")

    require_file(RUN_MODEL_SCRIPT, "Run_Model_1Class.py")
    require_file(PLAQUE_DETECTION_SCRIPT, "plaque_detection_curve.py")
    require_file(PLAQUE_SUMMARY_SCRIPT, "plaque_summary.py")

    require_dir(INPUT_IMAGE_DIR, "Input images folder")
    img_count = count_images(INPUT_IMAGE_DIR)
    if img_count == 0:
        raise RuntimeError(f"ไม่พบรูป .jpg/.jpeg/.png ใน {INPUT_IMAGE_DIR}")

    if WEIGHT_FILE is None:
        raise FileNotFoundError(
            "ไม่พบ weight file mask_rcnn_tooth_single_class_0097.h5\n"
            "ให้วางไว้ใน PROJECT_ROOT/h5 หรือแก้ WEIGHT_FILE_CANDIDATES ใน master_run_pipeline.py"
        )

    TEETH_SEGMENT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PLAQUE_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[OK] SCRIPT_DIR               = {SCRIPT_DIR}")
    print(f"[OK] PROJECT_ROOT             = {PROJECT_ROOT}")
    print(f"[OK] Input images             = {INPUT_IMAGE_DIR}")
    print(f"[OK] Found images             = {img_count}")
    print(f"[OK] Weight file              = {WEIGHT_FILE}")
    print(f"[OK] AI output                = {TEETH_SEGMENT_RESULT_DIR}")
    print(f"[OK] Detection output         = {PLAQUE_RESULT_DIR}")


def validate_after_ai() -> None:
    print_header("VALIDATE AFTER STEP 1 - AI")
    case_count = count_case_folders(TEETH_SEGMENT_RESULT_DIR)
    print(f"[INFO] AI output case folders = {case_count}")

    if case_count == 0:
        raise RuntimeError(
            "AI รันจบแล้ว แต่ไม่พบ case folder ใน Teeth Segment Result\n"
            f"Folder ที่เช็คคือ: {TEETH_SEGMENT_RESULT_DIR}\n"
            "ให้ตรวจว่า Run_Model_1Class.py detect ฟันได้จริง และ OUT_ROOT ตรงกับ pipeline"
        )


def validate_after_detection_only() -> None:
    print_header("VALIDATE AFTER STEP 2 - DETECTION ONLY")
    case_count = count_case_folders(PLAQUE_RESULT_DIR)
    report_count = count_reports(PLAQUE_RESULT_DIR)

    print(f"[INFO] Plaque result case folders = {case_count}")
    print(f"[INFO] Report files               = {report_count}")

    if case_count == 0:
        raise RuntimeError(
            "Detection รันจบแล้ว แต่ไม่พบ case folder ใน Plaque Result Curves\n"
            f"Folder ที่เช็คคือ: {PLAQUE_RESULT_DIR}"
        )

    if report_count == 0:
        raise RuntimeError(
            "Detection รันจบแล้ว แต่ไม่พบ report txt\n"
            "ให้ตรวจว่า plaque_detection_curve.py สร้าง plaque_php_qhpi_report.txt ได้จริงหรือไม่"
        )

    print("[OK] Step 2 ผ่านแล้ว: มีผล Detection/Report แล้ว ขั้นต่อไปให้ Step 3 สร้าง Summary image")


def validate_after_summary() -> None:
    print_header("VALIDATE AFTER STEP 3 - SUMMARY")
    case_count = count_case_folders(PLAQUE_RESULT_DIR)
    summary_count = count_summary_images(PLAQUE_RESULT_DIR)
    report_count = count_reports(PLAQUE_RESULT_DIR)

    print(f"[INFO] Plaque result case folders = {case_count}")
    print(f"[INFO] Summary images             = {summary_count}")
    print(f"[INFO] Report files               = {report_count}")

    if summary_count == 0:
        raise RuntimeError(
            "Summary step รันจบแล้ว แต่ยังไม่พบไฟล์ Summary image\n"
            "ไฟล์ที่ค้นหา: case_summary_php_qhpi.png หรือ *summary*.png ในแต่ละ case folder\n"
            "ถ้าการรัน plaque_summary.py เองสร้างชื่อไฟล์อื่น ให้เพิ่มชื่อไฟล์ใน SUMMARY_FILENAMES"
        )


def validate_for_detection_only() -> None:
    print_header("VALIDATE FOR DETECTION ONLY MODE")
    if not has_valid_ai_output(TEETH_SEGMENT_RESULT_DIR):
        raise RuntimeError(
            "ไม่สามารถรัน --detection-only ได้ เนื่องจากไม่พบผลลัพธ์ AI ที่สมบูรณ์\n"
            f"กรุณาตรวจสอบว่าในโฟลเดอร์ {TEETH_SEGMENT_RESULT_DIR} มีโฟลเดอร์เคสที่มีไฟล์รูป tooth11 ถึง tooth23 ครบถ้วน"
        )
    print("[OK] มีผลลัพธ์ AI พร้อมสำหรับการรัน Step 2")


def validate_for_summary_only() -> None:
    print_header("VALIDATE FOR SUMMARY ONLY MODE")
    if not has_valid_detection_output(PLAQUE_RESULT_DIR):
        raise RuntimeError(
            "ไม่สามารถรัน --summary-only ได้ เนื่องจากไม่พบผลลัพธ์ Detection\n"
            f"กรุณาตรวจสอบว่าในโฟลเดอร์ {PLAQUE_RESULT_DIR} มีไฟล์ report txt หรือ case_results.json"
        )
    print("[OK] มีผลลัพธ์ Detection พร้อมสำหรับการรัน Step 3")


# ==============================================================================
# 5) Main
# ==============================================================================

INPUT_IMAGE_DIR = first_existing_dir_with_images(INPUT_IMAGE_CANDIDATES)
WEIGHT_FILE = first_existing_file(WEIGHT_FILE_CANDIDATES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dental Plaque Pipeline")
    parser.add_argument("--force-ai", action="store_true", help="บังคับรัน AI ใหม่เสมอ")
    parser.add_argument("--detection-only", action="store_true", help="ข้าม Step 1 แล้วรันเฉพาะ Step 2 + Step 3")
    parser.add_argument("--summary-only", action="store_true", help="ข้าม Step 1 และ Step 2 แล้วรันเฉพาะ Step 3")
    args, unknown = parser.parse_known_args()
    
    force_ai = args.force_ai
    detection_only = args.detection_only
    summary_only = args.summary_only

    if summary_only:
        detection_only = False

    print_header("DENTAL PLAQUE PIPELINE: AI -> DETECTION -> SUMMARY")
    print(f"Python executable = {sys.executable}")

    try:
        validate_before_run()

        # Sync path ของไฟล์ย่อยก่อนรัน
        patch_file_constants(
            RUN_MODEL_SCRIPT,
            {
                "WEIGHTS": WEIGHT_FILE,
                "IMG_DIR": INPUT_IMAGE_DIR,
                "OUT_ROOT": TEETH_SEGMENT_RESULT_DIR,
            },
            "STEP 1 SCRIPT - Run_Model_1Class.py",
        )

        patch_file_constants(
            PLAQUE_DETECTION_SCRIPT,
            {
                "INPUT_ROOT": TEETH_SEGMENT_RESULT_DIR,
                "OUTPUT_ROOT": PLAQUE_RESULT_DIR,
            },
            "STEP 2 SCRIPT - plaque_detection_curve.py",
        )

        patch_file_constants(
            PLAQUE_SUMMARY_SCRIPT,
            {
                # รองรับหลายชื่อ เผื่อ summary script ใช้ชื่อไม่เหมือนกัน
                "INPUT_ROOT": PLAQUE_RESULT_DIR,
                "OUTPUT_ROOT": PLAQUE_RESULT_DIR,
                "PLAQUE_RESULT_DIR": PLAQUE_RESULT_DIR,
                "RESULT_ROOT": PLAQUE_RESULT_DIR,
                "ROOT_DIR": PLAQUE_RESULT_DIR,
                "output_root": PLAQUE_RESULT_DIR,
            },
            "STEP 3 SCRIPT - plaque_summary.py",
        )

        if summary_only:
            print_header("MODE: SUMMARY ONLY")
            validate_for_summary_only()
            
            # Step 3
            run_script(PLAQUE_SUMMARY_SCRIPT, "STEP 3/3 - Summary Dashboard Image")
            validate_after_summary()

        elif detection_only:
            print_header("MODE: DETECTION ONLY")
            validate_for_detection_only()
            
            # Step 2
            run_script(PLAQUE_DETECTION_SCRIPT, "STEP 2/3 - Plaque Detection + PHP + QHPI + Report")
            validate_after_detection_only()
            
            # Step 3
            run_script(PLAQUE_SUMMARY_SCRIPT, "STEP 3/3 - Summary Dashboard Image")
            validate_after_summary()

        else:
            # Step 1
            if SKIP_AI_IF_EXISTS and not force_ai and has_valid_ai_output(TEETH_SEGMENT_RESULT_DIR):
                print_header("STEP 1/3 - AI Teeth Segmentation / Mask R-CNN")
                print(f"[SKIP] พบโฟลเดอร์ผลลัพธ์ที่มีรูปฟันครบใน {TEETH_SEGMENT_RESULT_DIR.name} แล้ว ข้ามการรัน AI ใหม่")
                validate_after_ai()
            else:
                run_script(RUN_MODEL_SCRIPT, "STEP 1/3 - AI Teeth Segmentation / Mask R-CNN")
                validate_after_ai()
    
            # Step 2: Detection อย่างเดียว ไม่เช็ค summary แล้ว
            run_script(PLAQUE_DETECTION_SCRIPT, "STEP 2/3 - Plaque Detection + PHP + QHPI + Report")
            validate_after_detection_only()
    
            # Step 3: Summary แยกออกมาเป็นไฟล์ของตัวเอง
            run_script(PLAQUE_SUMMARY_SCRIPT, "STEP 3/3 - Summary Dashboard Image")
            validate_after_summary()

        print_header("PIPELINE COMPLETE")
        print("[SUCCESS] รันครบ 3 ขั้นตอนเรียบร้อย")
        print(f"AI output:                {TEETH_SEGMENT_RESULT_DIR}")
        print(f"Detection/Summary output: {PLAQUE_RESULT_DIR}")

    except Exception as e:
        print_header("PIPELINE FAILED")
        print(f"[ERROR] {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
