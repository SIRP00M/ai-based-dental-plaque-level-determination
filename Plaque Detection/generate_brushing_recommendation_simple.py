import os
import re

INPUT_ROOT = r"D:/Internship/Plaque Result"
REPORT_FILENAME = "plaque_php_qhpi_report.txt"
OUTPUT_FILENAME = "plaque_brushing_recommendation_simple.txt"


# =========================
# 🦷 แปลงชื่อฟัน
# =========================
def tooth_code_to_thai_name(code):
    mapping = {
        "tooth13.png": "ฟันเขี้ยวบนขวา",
        "tooth12.png": "ฟันตัดข้างบนขวา",
        "tooth11.png": "ฟันตัดกลางบนขวา",
        "tooth21.png": "ฟันตัดกลางบนซ้าย",
        "tooth22.png": "ฟันตัดข้างบนซ้าย",
        "tooth23.png": "ฟันเขี้ยวบนซ้าย",
    }
    return mapping.get(code, code)


# =========================
# ระดับคราบ (ภาษาคน)
# =========================
def zone_severity_text(p, t):
    if t == 0:
        return ""
    r = p / t
    if r >= 0.7:
        return "เยอะ"
    elif r >= 0.3:
        return "ปานกลาง"
    elif r > 0:
        return "เล็กน้อย"
    return ""


def extract_zone_ratios(line):
    data = {}
    matches = re.findall(r'([MICGD])\((\d+)/(\d+)\)', line)
    for z, p, t in matches:
        data[z] = {
            "severity": zone_severity_text(int(p), int(t))
        }
    return data


# =========================
# 🧠 แนะนำรายซี่ (ภาษาคน)
# =========================
def generate_tooth_recommendation(name, php, qh, zone_detail, zone_pixels):
    tips = []
    focus = []

    # จุดที่ควรเน้น
    if zone_detail.get("G", 0):
        focus.append("ขอบเหงือก")

    if zone_detail.get("M", 0) or zone_detail.get("D", 0):
        focus.append("ด้านข้างของฟัน")

    if zone_detail.get("C", 0):
        focus.append("กลางฟัน")

    if zone_detail.get("I", 0):
        focus.append("ปลายฟัน")

    # คำแนะนำแบบเข้าใจง่าย
    if zone_detail.get("G", 0):
        tips.append("ลองเอียงแปรงเล็กน้อยให้โดนขอบเหงือก")

    if zone_detail.get("M", 0) or zone_detail.get("D", 0):
        tips.append("แปรงด้านข้างของฟันให้ทั่วมากขึ้น")

    if zone_detail.get("C", 0):
        tips.append("แปรงตรงกลางฟันช้า ๆ ไม่ต้องรีบ")

    if zone_detail.get("I", 0):
        tips.append("อย่าลืมแปรงปลายฟันด้วย")

    # จาก QHPI
    if qh >= 5:
        tips.append("ซี่นี้ยังมีคราบเยอะมาก ควรแปรงให้สะอาดใหม่ทั้งซี่")
    elif qh >= 4:
        tips.append("ซี่นี้ยังมีคราบเยอะ ควรใช้เวลาแปรงมากขึ้น")
    elif qh == 3:
        tips.append("ยังมีคราบพอสมควร ลองแปรงซ้ำอีกครั้ง")

    # จาก PHP
    if php >= 4:
        tips.append("ซี่นี้ยังแปรงไม่ทั่ว ลองแบ่งแปรงทีละส่วน")

    if not tips:
        tips.append("ซี่นี้สะอาดดีแล้ว")

    return (
        f"{name}\n"
        f"  เน้น: {', '.join(focus) if focus else 'ปกติ'}\n"
        f"  แนะนำ: {' | '.join(tips)}\n"
    )


# =========================
# 🧠 สรุปทั้งเคส (ภาษาคน)
# =========================
def generate_case_recommendation(avg_php, avg_qh, teeth):
    tips = []

    if avg_qh >= 4:
        tips.append("โดยรวมยังมีคราบค่อนข้างเยอะ ควรตั้งใจแปรงมากขึ้น")
    elif avg_qh >= 3:
        tips.append("ยังมีคราบอยู่บ้าง ควรแปรงให้ละเอียดขึ้น")

    if avg_php >= 4:
        tips.append("มีหลายจุดที่แปรงไม่ทั่ว ลองแบ่งแปรงทีละส่วน")

    bad = [tooth_code_to_thai_name(t["name"]) for t in teeth if t["qhpi_score"] >= 4]

    if bad:
        tips.append("ควรเน้นเป็นพิเศษ: " + ", ".join(bad))

    tips.append("แปรงฟันอย่างน้อยวันละ 2 ครั้ง ครั้งละประมาณ 2 นาที")
    tips.append("ถ้าเป็นไปได้ ใช้ไหมขัดฟันร่วมด้วย")

    return "\n".join(f"- {t}" for t in tips)


# =========================
# อ่าน report
# =========================
def parse_report(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    teeth = []
    current = None
    avg_php = None
    avg_qh = None

    for line in lines:
        s = line.strip()

        if not s:
            continue

        if re.match(r"^tooth\d+\.png$", s):
            if current:
                teeth.append(current)
            current = {
                "name": s,
                "php_score": 0,
                "qhpi_score": 0,
                "zone_detail": {},
                "zone_pixels": {}
            }
            continue

        if s.startswith("Average PHP score"):
            avg_php = float(re.search(r'([\d.]+)', s).group(1))
            continue

        if s.startswith("Average QHPI score"):
            avg_qh = float(re.search(r'([\d.]+)', s).group(1))
            continue

        if current is None:
            continue

        if s.startswith("PHP score"):
            current["php_score"] = int(re.search(r'(\d+)/5', s).group(1))

        elif s.startswith("QHPI score"):
            current["qhpi_score"] = int(re.search(r'(\d+)/5', s).group(1))

        elif s.startswith("Zone detail"):
            matches = re.findall(r'([MICGD])=(\d)', s)
            for z, v in matches:
                current["zone_detail"][z] = int(v)

        elif s.startswith("Zone pixels"):
            current["zone_pixels"] = extract_zone_ratios(s)

    if current:
        teeth.append(current)

    return teeth, avg_php, avg_qh


# =========================
# สร้างข้อความ
# =========================
def build_text(case_name, teeth, avg_php, avg_qh):
    out = []
    out.append(f"Case: {case_name}\n")

    out.append("คำแนะนำรายซี่")
    out.append("-"*40)

    for t in teeth:
        name = tooth_code_to_thai_name(t["name"])
        out.append(generate_tooth_recommendation(
            name,
            t["php_score"],
            t["qhpi_score"],
            t["zone_detail"],
            t["zone_pixels"]
        ))

    out.append("\nสรุปภาพรวม")
    out.append("-"*40)
    out.append(f"Average PHP: {avg_php:.2f}")
    out.append(f"Average QHPI: {avg_qh:.2f}\n")

    out.append(generate_case_recommendation(avg_php, avg_qh, teeth))

    return "\n".join(out)


# =========================
# run
# =========================
def main():
    for case in os.listdir(INPUT_ROOT):
        path = os.path.join(INPUT_ROOT, case)
        if not os.path.isdir(path):
            continue

        report = os.path.join(path, REPORT_FILENAME)
        if not os.path.exists(report):
            continue

        teeth, avg_php, avg_qh = parse_report(report)

        text = build_text(case, teeth, avg_php, avg_qh)

        with open(os.path.join(path, OUTPUT_FILENAME), "w", encoding="utf-8") as f:
            f.write(text)

        print("[DONE]", case)


if __name__ == "__main__":
    main()