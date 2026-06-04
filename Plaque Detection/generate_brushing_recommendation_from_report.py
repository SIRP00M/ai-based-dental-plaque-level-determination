import os
import re

INPUT_ROOT = r"D:/Internship/Plaque Result"
REPORT_FILENAME = "plaque_php_qhpi_report.txt"
OUTPUT_FILENAME = "plaque_brushing_recommendation.txt"


# =========================
# Tooth Name Mapping
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
# Severity
# =========================
def zone_severity_text(plaque_pixels, zone_pixels):
    if zone_pixels <= 0:
        return "ไม่มีข้อมูล"
    ratio = plaque_pixels / zone_pixels
    if ratio >= 0.70:
        return "มีคราบมาก"
    elif ratio >= 0.30:
        return "มีคราบปานกลาง"
    elif ratio > 0:
        return "มีคราบเล็กน้อย"
    return "สะอาด"


def extract_zone_ratios(line):
    zone_data = {}
    matches = re.findall(r'([MICGD])\((\d+)/(\d+)\)', line)
    for z, p, t in matches:
        zone_data[z] = {
            "plaque_pixels": int(p),
            "zone_pixels": int(t),
            "severity": zone_severity_text(int(p), int(t))
        }
    return zone_data


# =========================
# Recommendation per tooth
# =========================
def generate_tooth_recommendation(name, php, qh, zone_detail, zone_pixels):
    recs = []
    focus = []

    if zone_detail.get("G", 0):
        focus.append(f"ขอบเหงือก ({zone_pixels.get('G', {}).get('severity', '')})")
        recs.append("แปรงเอียง 45° เข้าหาเหงือก แล้วขยับสั้น ๆ เบา ๆ")

    if zone_detail.get("M", 0):
        focus.append(f"ด้านซ้าย/mesial ({zone_pixels.get('M', {}).get('severity', '')})")

    if zone_detail.get("D", 0):
        focus.append(f"ด้านขวา/distal ({zone_pixels.get('D', {}).get('severity', '')})")

    if zone_detail.get("C", 0):
        focus.append(f"กลางฟัน ({zone_pixels.get('C', {}).get('severity', '')})")
        recs.append("แปรงกลางผิวฟันช้า ๆ ให้ทั่ว ไม่ปัดผ่านเร็วเกินไป")

    if zone_detail.get("I", 0):
        focus.append(f"ปลายฟัน/ขอบตัด ({zone_pixels.get('I', {}).get('severity', '')})")
        recs.append("เก็บบริเวณปลายฟันหรือขอบตัดเพิ่ม")

    if zone_detail.get("M", 0) or zone_detail.get("D", 0):
        recs.append("เพิ่มการแปรงบริเวณด้านข้างของซี่ฟันให้ทั่ว")

    if qh >= 5:
        recs.append("พบคราบปกคลุมเกือบทั้งผิวฟัน ควรแปรงซี่นี้ใหม่อย่างละเอียดทุกด้าน")
    elif qh == 4:
        recs.append("พบคราบมากกว่าหนึ่งในสามของผิวฟัน ควรเพิ่มเวลาและความละเอียดในการแปรง")
    elif qh == 3:
        recs.append("พบคราบค่อนข้างชัดเจน ควรแปรงซ้ำอย่างนุ่มนวลและทั่วถึง")
    elif qh == 2:
        recs.append("พบคราบบางใกล้ขอบเหงือก ควรเน้นการแปรงชิดขอบเหงือก")
    elif qh == 1:
        recs.append("พบคราบเล็กน้อยเฉพาะบางจุด ควรเก็บรายละเอียดการแปรงให้ทั่วขึ้น")

    if php >= 5:
        recs.append("หลายตำแหน่งของซี่นี้ยังมีคราบ ควรแบ่งการแปรงเป็นส่วน ๆ ให้ครบทุกด้าน")
    elif php >= 4:
        recs.append("ยังมีหลายตำแหน่งที่แปรงไม่ทั่ว ควรควบคุมตำแหน่งหัวแปรงให้แม่นขึ้น")

    if not recs:
        recs.append("ซี่นี้สะอาดดีในระดับหนึ่ง ให้คงวิธีการแปรงแบบเดิมและตรวจซ้ำสม่ำเสมอ")

    return (
        f"{name}\n"
        f"  บริเวณที่ควรเน้น: {', '.join(focus) if focus else 'ไม่พบจุดเด่นผิดปกติ'}\n"
        f"  คำแนะนำ: {' '.join(recs)}\n"
    )


# =========================
# Recommendation case
# =========================
def generate_case_recommendation(avg_php, avg_qh, teeth):
    recs = []

    if avg_qh >= 4:
        recs.append("ภาพรวมพบคราบจุลินทรีย์สะสมค่อนข้างมาก ควรเพิ่มความละเอียดในการแปรงฟันทุกครั้ง")
    elif avg_qh >= 3:
        recs.append("ภาพรวมยังพบคราบสะสมระดับปานกลาง ควรปรับเทคนิคการแปรงให้ทั่วถึงมากขึ้น")
    else:
        recs.append("ภาพรวมการสะสมคราบอยู่ในระดับไม่สูงมาก แต่ยังควรรักษาคุณภาพการแปรงให้สม่ำเสมอ")

    if avg_php >= 4:
        recs.append("มีหลายบริเวณของผิวฟันที่ยังทำความสะอาดไม่ทั่ว ควรแบ่งการแปรงเป็นส่วนย่อยและไล่ทีละตำแหน่ง")
    elif avg_php >= 3:
        recs.append("ยังมีบางตำแหน่งที่แปรงไม่ทั่ว ควรขยับหัวแปรงช้าลงและควบคุมทิศทางให้แม่นขึ้น")

    high_qh_teeth = [tooth_code_to_thai_name(t["name"]) for t in teeth if t["qhpi_score"] >= 4]
    gingival_teeth = [tooth_code_to_thai_name(t["name"]) for t in teeth if t["zone_detail"].get("G", 0) == 1]
    side_teeth = [
        tooth_code_to_thai_name(t["name"]) for t in teeth
        if t["zone_detail"].get("M", 0) == 1 or t["zone_detail"].get("D", 0) == 1
    ]
    center_teeth = [tooth_code_to_thai_name(t["name"]) for t in teeth if t["zone_detail"].get("C", 0) == 1]
    cutting_teeth = [tooth_code_to_thai_name(t["name"]) for t in teeth if t["zone_detail"].get("I", 0) == 1]

    if gingival_teeth:
        recs.append(f"ซี่ที่ควรเน้นขอบเหงือกเป็นพิเศษ: {', '.join(gingival_teeth)}")
    if side_teeth:
        recs.append(f"ซี่ที่ควรเพิ่มการแปรงด้านข้างฟัน: {', '.join(side_teeth)}")
    if center_teeth:
        recs.append(f"ซี่ที่ควรแปรงบริเวณกลางผิวฟันให้ทั่ว: {', '.join(center_teeth)}")
    if cutting_teeth:
        recs.append(f"ซี่ที่ควรเก็บบริเวณปลายฟันหรือขอบตัดเพิ่ม: {', '.join(cutting_teeth)}")
    if high_qh_teeth:
        recs.append(f"ซี่ที่ควรเน้นเป็นพิเศษเพราะมีคราบมาก: {', '.join(high_qh_teeth)}")

    recs.append("ควรแปรงฟันอย่างน้อยวันละ 2 ครั้ง ครั้งละประมาณ 2 นาที")
    recs.append("ควรใช้ไหมขัดฟันร่วมด้วย โดยเฉพาะถ้ามีคราบสะสมบริเวณด้านข้างของฟันหลายซี่")
    recs.append("หากยังพบคราบสะสมซ้ำในตำแหน่งเดิม ควรพิจารณาปรับเทคนิคการแปรง เช่น Modified Bass technique")

    return "\n".join(f"- {r}" for r in recs)


# =========================
# Parse Report
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

        if s.startswith("Case:"):
            continue

        if re.match(r"^tooth\d+\.png$", s):
            if current:
                teeth.append(current)
            current = {
                "name": s,
                "php_score": None,
                "qhpi_score": None,
                "zone_detail": {},
                "zone_pixels": {}
            }
            continue

        # average ต้องเช็คก่อน score รายซี่
        if s.startswith("Average PHP score"):
            m = re.search(r":\s*([\d.]+)", s)
            if m:
                avg_php = float(m.group(1))
            continue

        if s.startswith("Average QHPI score"):
            m = re.search(r":\s*([\d.]+)", s)
            if m:
                avg_qh = float(m.group(1))
            continue

        if current is None:
            continue

        if s.startswith("PHP score"):
            m = re.search(r":\s*(\d+)/5", s)
            if m:
                current["php_score"] = int(m.group(1))
            continue

        if s.startswith("QHPI score"):
            m = re.search(r":\s*(\d+)/5", s)
            if m:
                current["qhpi_score"] = int(m.group(1))
            continue

        if s.startswith("Zone detail"):
            matches = re.findall(r'([MICGD])=(\d)', s)
            for z, v in matches:
                current["zone_detail"][z] = int(v)
            continue

        if s.startswith("Zone pixels"):
            current["zone_pixels"] = extract_zone_ratios(s)
            continue

    if current:
        teeth.append(current)

    return teeth, avg_php, avg_qh


# =========================
# Build Text
# =========================
def build_recommendation_text(case_name, teeth, avg_php, avg_qh):
    lines = []
    lines.append(f"Case: {case_name}")
    lines.append("")
    lines.append("คำแนะนำการแปรงฟันรายซี่")
    lines.append("-" * 60)

    for tooth in teeth:
        thai_name = tooth_code_to_thai_name(tooth["name"])
        lines.append(
            generate_tooth_recommendation(
                name=thai_name,
                php=tooth["php_score"] or 0,
                qh=tooth["qhpi_score"] or 0,
                zone_detail=tooth["zone_detail"],
                zone_pixels=tooth["zone_pixels"]
            )
        )

    lines.append("")
    lines.append("สรุปคำแนะนำภาพรวมทั้งเคส")
    lines.append("-" * 60)
    lines.append(f"Average PHP score  : {avg_php:.2f}" if avg_php is not None else "Average PHP score  : N/A")
    lines.append(f"Average QHPI score : {avg_qh:.2f}" if avg_qh is not None else "Average QHPI score : N/A")
    lines.append("")
    lines.append(generate_case_recommendation(avg_php or 0, avg_qh or 0, teeth))
    lines.append("")

    return "\n".join(lines)


# =========================
# Process Case
# =========================
def process_case(folder):
    report_path = os.path.join(folder, REPORT_FILENAME)
    if not os.path.exists(report_path):
        print(f"[SKIP] {os.path.basename(folder)} -> ไม่พบ {REPORT_FILENAME}")
        return

    teeth, avg_php, avg_qh = parse_report(report_path)

    if not teeth:
        print(f"[SKIP] {os.path.basename(folder)} -> ไม่พบข้อมูลฟันใน report")
        return

    text = build_recommendation_text(
        case_name=os.path.basename(folder),
        teeth=teeth,
        avg_php=avg_php,
        avg_qh=avg_qh
    )

    out_path = os.path.join(folder, OUTPUT_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[DONE] {os.path.basename(folder)} -> {out_path}")


# =========================
# Main
# =========================
def main():
    if not os.path.exists(INPUT_ROOT):
        print("INPUT_ROOT not found:", INPUT_ROOT)
        return

    case_folders = [
        os.path.join(INPUT_ROOT, d)
        for d in os.listdir(INPUT_ROOT)
        if os.path.isdir(os.path.join(INPUT_ROOT, d))
    ]

    print("INPUT_ROOT =", INPUT_ROOT)
    print("FOUND CASES =", len(case_folders))

    for case_path in case_folders:
        process_case(case_path)

    print("Done all cases")


if __name__ == "__main__":
    main()