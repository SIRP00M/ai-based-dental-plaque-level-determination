"""
master_run_pipeline.py

ใช้สำหรับรัน Pipeline ทั้งหมดแบบต่อเนื่อง:
1) รัน Mask R-CNN Teeth Segmentation / Run Model
2) เมื่อข้อ 1 เสร็จสมบูรณ์แล้ว ค่อยรัน Plaque Detection + PHP + QHPI

หลักการ:
- ใช้ subprocess.run(..., check=True)
- ถ้า Run Model error โปรแกรมจะหยุดทันที และไม่รัน Plaque Detection ต่อ
- ใช้ Python ตัวเดียวกับที่เปิดไฟล์นี้อยู่ เพื่อให้ใช้ environment เดียวกันกับ Spyder / Anaconda
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime


# ==============================================================================
# 1) ตั้งค่า Path ของไฟล์ทั้ง 2 ตัว
# ==============================================================================

# ไฟล์ Run Model / Teeth Segmentation
RUN_MODEL_SCRIPT = Path(r"D:/Internship/Plaque Mask/Run_Model_1Class.py")

# ไฟล์ Plaque Detection + PHP + QHPI
PLAQUE_DETECTION_SCRIPT = Path(r"D:/Internship/Plaque Mask/plaque_detection_curve.py")


# ==============================================================================
# 2) ตั้งค่า Folder สำคัญที่ใช้ตรวจสอบผลลัพธ์
# ==============================================================================

# Folder รูป input ที่ Run Model ใช้อ่าน
INPUT_IMAGE_DIR = Path(r"D:/Internship/Data")

# Folder output จาก Run Model
TEETH_SEGMENT_RESULT_DIR = Path(r"D:/Internship/Teeth Segment Result")

# Folder output จาก Plaque Detection
PLAQUE_RESULT_DIR = Path(r"D:/Internship/Plaque Result Curves")


# ==============================================================================
# 3) Helper Functions
# ==============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_time(label: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {label}")


def check_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nไม่พบ {label}\n"
            f"Path ที่ตั้งไว้คือ:\n{path}\n\n"
            f"ให้แก้ path ใน master_run_pipeline.py ให้ตรงกับเครื่องก่อน"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"\n{label} ไม่ใช่ไฟล์ .py ที่ถูกต้อง:\n{path}"
        )


def check_folder_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nไม่พบ {label}\n"
            f"Path ที่ตั้งไว้คือ:\n{path}\n\n"
            f"ให้แก้ path ใน master_run_pipeline.py ให้ตรงกับเครื่องก่อน"
        )

    if not path.is_dir():
        raise FileNotFoundError(
            f"\n{label} ไม่ใช่โฟลเดอร์ที่ถูกต้อง:\n{path}"
        )


def count_input_images(folder: Path) -> int:
    exts = {".png", ".jpg", ".jpeg"}
    return sum(
        1
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def count_case_folders(folder: Path) -> int:
    if not folder.exists():
        return 0

    return sum(
        1
        for p in folder.iterdir()
        if p.is_dir()
    )


def run_python_script(script_path: Path, step_name: str) -> None:
    """
    รันไฟล์ Python อีกไฟล์หนึ่งแบบรอจนจบ
    ถ้าไฟล์นั้น error จะ raise CalledProcessError และหยุด pipeline
    """
    print_header(f"START: {step_name}")
    print_time(f"Running {script_path}")

    # ใช้ sys.executable เพื่อให้รันด้วย Python / Conda env เดียวกับไฟล์นี้
    command = [sys.executable, str(script_path)]

    result = subprocess.run(
        command,
        cwd=str(script_path.parent),
        check=True,
        text=True
    )

    print_time(f"Finished {step_name}")
    print(f"Return code: {result.returncode}")


def safe_read_file(path: Path) -> str:
    """อ่านไฟล์ด้วยการสุ่ม encoding เพื่อให้รองรับไฟล์ภาษาไทยบน Windows อย่างปลอดภัย"""
    for enc in ["utf-8", "cp874", "latin-1", "utf-16"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Cannot decode {path}")


def verify_path_synchronization() -> None:
    """
    ตรวจสอบว่าค่า Path ใน master_run_pipeline.py ตรงกับที่ตั้งไว้ในไฟล์ย่อยทั้ง 2 หรือไม่
    เพื่อป้องกันความลับสนเวลาผู้ใช้อยากเปลี่ยนโฟลเดอร์ทำงาน
    """
    print_header("VERIFYING PATH SYNCHRONIZATION")
    import re
    
    warnings_found = False

    # 1. อ่าน Run_Model_1Class.py
    try:
        content = safe_read_file(RUN_MODEL_SCRIPT)
        img_dir_match = re.search(r'IMG_DIR\s*=\s*r?["\'](.*?)["\']', content)
        out_root_match = re.search(r'OUT_ROOT\s*=\s*r?["\'](.*?)["\']', content)
        
        if img_dir_match and out_root_match:
            sub_img_dir = Path(img_dir_match.group(1).replace("\\", "/"))
            sub_out_root = Path(out_root_match.group(1).replace("\\", "/"))
            
            if Path(os.path.abspath(sub_img_dir)) != Path(os.path.abspath(INPUT_IMAGE_DIR)):
                print(f"[WARNING] ที่ตั้งรูปภาพไม่ตรงกัน (Input image directories mismatch):")
                print(f"  - ใน master_run_pipeline.py: {INPUT_IMAGE_DIR}")
                print(f"  - ใน Run_Model_1Class.py: {sub_img_dir}")
                warnings_found = True
            if Path(os.path.abspath(sub_out_root)) != Path(os.path.abspath(TEETH_SEGMENT_RESULT_DIR)):
                print(f"[WARNING] ที่บันทึกรูปฟันไม่ตรงกัน (Teeth Segment directories mismatch):")
                print(f"  - ใน master_run_pipeline.py: {TEETH_SEGMENT_RESULT_DIR}")
                print(f"  - ใน Run_Model_1Class.py: {sub_out_root}")
                warnings_found = True
    except Exception as e:
        print(f"[INFO] ไม่สามารถตรวจสอบความซิงค์ของ Run_Model_1Class.py ได้: {e}")

    # 2. อ่าน plaque_detection_curve.py
    try:
        content = safe_read_file(PLAQUE_DETECTION_SCRIPT)
        input_root_match = re.search(r'INPUT_ROOT\s*=\s*r?["\'](.*?)["\']', content)
        output_root_match = re.search(r'OUTPUT_ROOT\s*=\s*r?["\'](.*?)["\']', content)
        
        if input_root_match and output_root_match:
            sub_input_root = Path(input_root_match.group(1).replace("\\", "/"))
            sub_output_root = Path(output_root_match.group(1).replace("\\", "/"))
            
            if Path(os.path.abspath(sub_input_root)) != Path(os.path.abspath(TEETH_SEGMENT_RESULT_DIR)):
                print(f"[WARNING] ที่อ่านรูปฟันตรวจไม่ตรงกัน (Plaque detection inputs mismatch):")
                print(f"  - ใน master_run_pipeline.py: {TEETH_SEGMENT_RESULT_DIR}")
                print(f"  - ใน plaque_detection_curve.py: {sub_input_root}")
                warnings_found = True
            if Path(os.path.abspath(sub_output_root)) != Path(os.path.abspath(PLAQUE_RESULT_DIR)):
                print(f"[WARNING] ที่บันทึกผลลัพธ์คราบฟันไม่ตรงกัน (Plaque results mismatch):")
                print(f"  - ใน master_run_pipeline.py: {PLAQUE_RESULT_DIR}")
                print(f"  - ใน plaque_detection_curve.py: {sub_output_root}")
                warnings_found = True
    except Exception as e:
        print(f"[INFO] ไม่สามารถตรวจสอบความซิงค์ของ plaque_detection_curve.py ได้: {e}")

    if not warnings_found:
        print("[OK] ทุกโฟลเดอร์และสคริปต์ย่อยซิงค์ตรงกัน 100%!")


def validate_before_run() -> None:
    print_header("VALIDATING PATHS")

    check_file_exists(RUN_MODEL_SCRIPT, "ไฟล์ Run Model")
    print(f"[OK] Run Model script: {RUN_MODEL_SCRIPT}")

    check_file_exists(PLAQUE_DETECTION_SCRIPT, "ไฟล์ Plaque Detection")
    print(f"[OK] Plaque Detection script: {PLAQUE_DETECTION_SCRIPT}")

    # ดึงการตรวจสอบความสอดคล้องของ Path
    verify_path_synchronization()

    check_folder_exists(INPUT_IMAGE_DIR, "โฟลเดอร์รูป input")
    print(f"[OK] Input image folder: {INPUT_IMAGE_DIR}")

    input_count = count_input_images(INPUT_IMAGE_DIR)
    print(f"[INFO] Found input images: {input_count}")

    if input_count == 0:
        raise RuntimeError(
            f"\nไม่พบรูปภาพใน {INPUT_IMAGE_DIR}\n"
            f"ให้ใส่ไฟล์ .jpg / .jpeg / .png ก่อนรัน pipeline"
        )

    # สร้าง output folders ไว้ก่อน ถ้ายังไม่มี
    TEETH_SEGMENT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PLAQUE_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[OK] Teeth segment output folder: {TEETH_SEGMENT_RESULT_DIR}")
    print(f"[OK] Plaque result output folder: {PLAQUE_RESULT_DIR}")


def validate_after_run_model() -> None:
    """
    เช็คว่า Run Model สร้าง case folder ออกมาจริงไหม
    ถ้าไม่มี แปลว่าขั้นแรกอาจไม่เจอฟัน หรือ path output ไม่ตรงกับ Plaque Detection
    """
    print_header("CHECKING RUN MODEL OUTPUT")

    case_count = count_case_folders(TEETH_SEGMENT_RESULT_DIR)

    print(f"[INFO] Case folders in Teeth Segment Result: {case_count}")

    if case_count == 0:
        raise RuntimeError(
            "\nRun Model รันจบแล้ว แต่ไม่พบ case folder ใน Teeth Segment Result\n"
            f"Folder ที่เช็คคือ:\n{TEETH_SEGMENT_RESULT_DIR}\n\n"
            "สาเหตุที่เป็นไปได้:\n"
            "1) Run Model ไม่ detect ฟันเลย\n"
            "2) OUT_ROOT ใน Run Model ไม่ตรงกับ INPUT_ROOT ใน Plaque Detection\n"
            "3) รูป input อยู่ผิด folder\n"
            "4) ไฟล์ weight/model มีปัญหาแต่ไม่ได้แสดง error ชัดเจน\n"
        )


def validate_after_plaque_detection() -> None:
    print_header("CHECKING PLAQUE DETECTION OUTPUT")

    case_count = count_case_folders(PLAQUE_RESULT_DIR)

    print(f"[INFO] Case folders in Plaque Result: {case_count}")

    if case_count == 0:
        raise RuntimeError(
            "\nPlaque Detection รันจบแล้ว แต่ไม่พบผลลัพธ์ใน Plaque Result Curves\n"
            f"Folder ที่เช็คคือ:\n{PLAQUE_RESULT_DIR}\n\n"
            "ให้ตรวจสอบว่า INPUT_ROOT ในไฟล์ Plaque Detection ตรงกับ output ของ Run Model หรือไม่"
        )


# ==============================================================================
# 4) Main Pipeline
# ==============================================================================

def main() -> None:
    print_header("DENTAL PLAQUE FULL PIPELINE")
    print(f"Python executable: {sys.executable}")

    try:
        validate_before_run()

        # ----------------------------------------------------------------------
        # Step 1: Run Model / Teeth Segmentation
        # ----------------------------------------------------------------------
        run_python_script(
            RUN_MODEL_SCRIPT,
            step_name="Step 1 - Run Mask R-CNN Teeth Segmentation"
        )

        validate_after_run_model()

        # ----------------------------------------------------------------------
        # Step 2: Plaque Detection + PHP + QHPI
        # ----------------------------------------------------------------------
        run_python_script(
            PLAQUE_DETECTION_SCRIPT,
            step_name="Step 2 - Run Plaque Detection PHP QHPI"
        )

        validate_after_plaque_detection()

        print_header("PIPELINE COMPLETE")
        print("[SUCCESS] รันครบทั้ง 2 ขั้นตอนเรียบร้อย")
        print(f"Teeth Segment Result: {TEETH_SEGMENT_RESULT_DIR}")
        print(f"Plaque Result       : {PLAQUE_RESULT_DIR}")

    except subprocess.CalledProcessError as e:
        print_header("PIPELINE FAILED")
        print("[ERROR] มีไฟล์บางตัวรันแล้วเกิด error")
        print(f"Failed command: {e.cmd}")
        print(f"Return code   : {e.returncode}")
        print("\nระบบหยุดการทำงานแล้ว เพื่อไม่ให้รันขั้นตอนต่อไปด้วยข้อมูลที่ไม่สมบูรณ์")
        raise

    except Exception as e:
        print_header("PIPELINE FAILED")
        print(str(e))
        print("\nระบบหยุดการทำงานแล้ว")
        raise


if __name__ == "__main__":
    main()
