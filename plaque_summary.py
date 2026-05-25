import os
from PIL import Image, ImageDraw, ImageFont, ImageOps

# =========================
# Summary Constants
# =========================
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

SUMMARY_CELL_W = 540        # ความกว้างช่องในภาพ summary
SUMMARY_CELL_H = 430        # ความสูงช่องในภาพ summary
SUMMARY_IMAGE_H = 270       # พื้นที่รูปด้านบนของแต่ละช่อง


# =========================
# Case Summary Helpers
# =========================
def load_summary_font(size=22):
    """โหลดฟอนต์ที่รองรับไทยให้มากที่สุด ถ้าไม่มีจะ fallback เป็น default."""
    candidates = [
        r"C:/Windows/Fonts/leelawad.ttf", # Leelawadee UI (Modern and clean for Thai)
        r"C:/Windows/Fonts/segoeui.ttf",  # Segoe UI (Standard modern UI font)
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
    สร้างภาพรวม 6 ซี่ต่อเคส ในสไตล์ Dental Analysis Dashboard ระดับมืออาชีพ
    ประกอบด้วย:
      - ส่วนหัว (Header Dashboard) สี Slate Blue เข้ม แสดงข้อมูลเคสและค่าเฉลี่ยทั้งหมด
      - บล็อกฟันแต่ละซี่เป็น White Card ดีไซน์สะอาดตา
      - ป้ายสถานะความรุนแรง (Pill Badges) แยกสีตามระดับคราบพลัค
      - หลอดความคืบหน้า (Progress Bar) แสดงคราบพลัคอย่างสวยงามแทนตัวหนังสือ
      - บล็อกคะแนน (Segment Gauges) สำหรับคะแนน PHP และ QHPI
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
    header_h = 160
    margin = 24
    cell_w = 540
    cell_h = 450

    canvas_w = cols * cell_w + margin * 2
    canvas_h = rows * cell_h + header_h + margin * 2

    # Canvas Background: Cool Slate Gray (#F1F5F9)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (241, 245, 249))
    draw = ImageDraw.Draw(canvas)

    # โหลดฟอนต์ขนาดต่าง ๆ
    font_title = load_summary_font(28)
    font_subtitle = load_summary_font(18)
    font_head = load_summary_font(20)
    font_body_bold = load_summary_font(18)
    font_body = load_summary_font(16)
    font_small = load_summary_font(14)
    font_badge = load_summary_font(15)

    # 1. คำนวณค่าเฉลี่ยรวมของเคส
    total_php = sum(r.get("php_score", 0) for r in case_results)
    total_php_max = sum(r.get("php_max", 0) for r in case_results)
    avg_php = (total_php / total_php_max * 5.0) if total_php_max > 0 else 0.0
    
    avg_qhpi = sum(r.get("qhpi_score", 0) for r in case_results) / len(case_results) if case_results else 0.0
    avg_ratio = sum(r.get("ratio", 0.0) for r in case_results) / len(case_results) if case_results else 0.0
    avg_ratio_pct = avg_ratio * 100

    # 2. วาด Header Dashboard Panel (Slate Blue: #1E293B)
    header_box = [margin, margin, canvas_w - margin, margin + header_h]
    draw.rounded_rectangle(header_box, radius=16, fill=(30, 41, 59))

    # ข้อความใน Header (ฝั่งซ้าย)
    draw.text((margin + 28, margin + 25), "รายงานสรุปการวิเคราะห์คราบพลัค (Plaque Summary Dashboard)", fill=(255, 255, 255), font=font_title)
    draw.text((margin + 28, margin + 70), f"ชื่อเคส (Case): {case_name}", fill=(203, 213, 225), font=font_subtitle)
    draw.text((margin + 28, margin + 105), f"จำนวนฟันที่ประเมิน (Teeth Evaluated): {len(case_results)} ซี่", fill=(148, 163, 184), font=font_small)

    # วาดวิดเจ็ตสรุปใน Header (ฝั่งขวา)
    # วิดเจ็ต 1: Avg Plaque Coverage
    widget_w = 260
    widget_h = 100
    widget_y = margin + 30
    
    # Widget 1: Plaque Coverage
    w1_x = canvas_w - margin - widget_w * 3 - 32
    draw.rounded_rectangle([w1_x, widget_y, w1_x + widget_w, widget_y + widget_h], radius=10, fill=(51, 65, 85))
    draw.text((w1_x + 16, widget_y + 12), "พื้นที่คราบฟันเฉลี่ย", fill=(203, 213, 225), font=font_small)
    draw.text((w1_x + 16, widget_y + 32), f"{avg_ratio_pct:.1f}%", fill=(255, 255, 255), font=font_head)
    # mini progress bar
    draw.rounded_rectangle([w1_x + 16, widget_y + 72, w1_x + widget_w - 16, widget_y + 80], radius=4, fill=(71, 85, 105))
    bar_fill_w = int((widget_w - 32) * avg_ratio)
    # Color based on average severity
    avg_color = (16, 185, 129) if avg_ratio_pct < 10 else ((245, 158, 11) if avg_ratio_pct <= 30 else (239, 68, 68))
    if bar_fill_w > 0:
        draw.rounded_rectangle([w1_x + 16, widget_y + 72, w1_x + 16 + bar_fill_w, widget_y + 80], radius=4, fill=avg_color)

    # Widget 2: Avg PHP Score
    w2_x = canvas_w - margin - widget_w * 2 - 16
    draw.rounded_rectangle([w2_x, widget_y, w2_x + widget_w, widget_y + widget_h], radius=10, fill=(51, 65, 85))
    draw.text((w2_x + 16, widget_y + 12), "คะแนน PHP เฉลี่ย", fill=(203, 213, 225), font=font_small)
    draw.text((w2_x + 16, widget_y + 32), f"{avg_php:.2f} / 5.0", fill=(255, 255, 255), font=font_head)
    # Badge level for PHP
    php_badge_text = "น้อย (Good)" if avg_php < 1.0 else ("ปานกลาง (Fair)" if avg_php <= 3.0 else "หนาแน่น (Poor)")
    php_badge_color = (16, 185, 129) if avg_php < 1.0 else ((245, 158, 11) if avg_php <= 3.0 else (239, 68, 68))
    # Draw simple dot indicator
    draw.ellipse([w2_x + 16, widget_y + 74, w2_x + 24, widget_y + 82], fill=php_badge_color)
    draw.text((w2_x + 32, widget_y + 70), php_badge_text, fill=(241, 245, 249), font=font_small)

    # Widget 3: Avg QHPI Score
    w3_x = canvas_w - margin - widget_w
    draw.rounded_rectangle([w3_x, widget_y, w3_x + widget_w, widget_y + widget_h], radius=10, fill=(51, 65, 85))
    draw.text((w3_x + 16, widget_y + 12), "คะแนน QHPI เฉลี่ย", fill=(203, 213, 225), font=font_small)
    draw.text((w3_x + 16, widget_y + 32), f"{avg_qhpi:.2f} / 5.0", fill=(255, 255, 255), font=font_head)
    qhpi_badge_text = "ระดับ 0-1 (น้อย)" if avg_qhpi < 1.5 else ("ระดับ 2 (ปานกลาง)" if avg_qhpi <= 2.5 else "ระดับ 3-5 (หนาแน่น)")
    qhpi_badge_color = (16, 185, 129) if avg_qhpi < 1.5 else ((245, 158, 11) if avg_qhpi <= 2.5 else (239, 68, 68))
    draw.ellipse([w3_x + 16, widget_y + 74, w3_x + 24, widget_y + 82], fill=qhpi_badge_color)
    draw.text((w3_x + 32, widget_y + 70), qhpi_badge_text, fill=(241, 245, 249), font=font_small)

    # 3. วาดฟันแต่ละซี่ (6 ช่องตามผัง)
    for idx, tooth_id in enumerate(SUMMARY_ORDER):
        row = idx // cols
        col = idx % cols

        x = margin + col * cell_w
        # เพิ่มพื้นที่ขยับลงจาก Header บล็อก
        y = header_h + margin + 12 + row * cell_h

        # กล่องการ์ดพื้นหลัง (White Card with Border)
        card_box = [x + 8, y + 8, x + cell_w - 8, y + cell_h - 8]
        draw.rounded_rectangle(
            card_box,
            radius=16,
            fill=(255, 255, 255),
            outline=(226, 232, 240), # Slate 200
            width=2
        )

        r = result_map.get(tooth_id)
        tooth_name = TOOTH_NAME_TH.get(tooth_id, "")

        if r is None:
            # การ์ดกรณีไม่พบข้อมูล (No Data Card)
            draw.text((x + 28, y + 28), f"ซี่ {tooth_id} - {tooth_name}", fill=(148, 163, 184), font=font_head)
            # วาดรูปเป้าหมายแบบโปร่งใสหรือลายประแดด
            placeholder_box = [x + 28, y + 80, x + cell_w - 28, y + cell_h - 28]
            draw.rounded_rectangle(placeholder_box, radius=10, fill=(248, 250, 252), outline=(226, 232, 240), width=1)
            draw_text_center(
                draw,
                placeholder_box,
                "ไม่ได้ตรวจหรือตรวจไม่พบข้อมูลซี่นี้",
                font_subtitle,
                fill=(148, 163, 184)
            )
            continue

        ratio_pct = r.get("ratio", 0.0) * 100
        php_score = r.get("php_score", 0)
        php_max = r.get("php_max", 0)
        qhpi_score = r.get("qhpi_score", 0)

        # 3.1 การจัดหมวดหมู่ระดับความรุนแรง
        # น้อย (< 10%), ปานกลาง (10% - 30%), หนาแน่น (> 30%)
        if ratio_pct < 10.0:
            level_name = "น้อย (Low)"
            badge_bg = (220, 252, 231)  # Emerald-100
            badge_txt = (21, 128, 61)   # Emerald-700
            sev_color = (16, 185, 129)  # Emerald-500
        elif ratio_pct <= 30.0:
            level_name = "ปานกลาง (Medium)"
            badge_bg = (254, 243, 199)  # Amber-100
            badge_txt = (180, 83, 9)    # Amber-700
            sev_color = (245, 158, 11)  # Amber-500
        else:
            level_name = "หนาแน่น (High)"
            badge_bg = (254, 226, 226)  # Rose-100
            badge_txt = (185, 28, 28)   # Rose-700
            sev_color = (239, 68, 68)   # Rose-500

        # 3.2 หัวข้อการ์ด (Tooth ID & Thai Name) และ ป้ายสถานะความรุนแรง
        draw.text(
            (x + 24, y + 24),
            f"ซี่ {tooth_id} - {tooth_name}",
            fill=(15, 23, 42),  # Slate 900
            font=font_head
        )

        # วาด ป้ายสถานะ (Pill Badge) ฝั่งขวา
        badge_w, badge_h = 135, 28
        badge_x1 = x + cell_w - 24 - badge_w
        badge_y1 = y + 22
        draw.rounded_rectangle(
            [badge_x1, badge_y1, badge_x1 + badge_w, badge_y1 + badge_h],
            radius=14,
            fill=badge_bg
        )
        draw_text_center(
            draw,
            (badge_x1, badge_y1, badge_x1 + badge_w, badge_y1 + badge_h),
            level_name,
            font_badge,
            fill=badge_txt
        )

        # 3.3 ส่วนแสดงผลภาพ (PHP vs QHPI Visualizations)
        # พื้นที่แสดงรูปสูง 180px ขยับพื้นที่ให้อ่านง่าย
        img_y1 = y + 66
        img_y2 = y + 250
        gap = 14
        inner_x1 = x + 24
        inner_x2 = x + cell_w - 24
        half_w = (inner_x2 - inner_x1 - gap) // 2

        php_box = (inner_x1, img_y1, inner_x1 + half_w, img_y2)
        qhpi_box = (inner_x1 + half_w + gap, img_y1, inner_x2, img_y2)

        # ดึงภาพจริงมาวาง
        php_vis_path = r.get("php_vis_path")
        qhpi_vis_path = r.get("qhpi_vis_path")

        # วาดกรอบรูปบางๆ
        draw.rounded_rectangle(php_box, radius=8, outline=(241, 245, 249), width=1)
        draw.rounded_rectangle(qhpi_box, radius=8, outline=(241, 245, 249), width=1)

        # แปะภาพ PHP
        if php_vis_path and os.path.exists(php_vis_path):
            try:
                img = Image.open(php_vis_path)
                paste_image_center(canvas, img, php_box)
            except Exception:
                draw_text_center(draw, php_box, "PHP Error", font_small, fill=(239, 68, 68))
        else:
            draw_text_center(draw, php_box, "No PHP Image", font_small, fill=(148, 163, 184))

        # แปะภาพ QHPI
        if qhpi_vis_path and os.path.exists(qhpi_vis_path):
            try:
                img = Image.open(qhpi_vis_path)
                paste_image_center(canvas, img, qhpi_box)
            except Exception:
                draw_text_center(draw, qhpi_box, "QHPI Error", font_small, fill=(239, 68, 68))
        else:
            draw_text_center(draw, qhpi_box, "No QHPI Image", font_small, fill=(148, 163, 184))

        # วาดกล่องข้อความทับขนาดเล็กด้านล่างของรูป
        # PHP tag
        draw.rounded_rectangle([php_box[0] + 6, php_box[3] - 22, php_box[0] + 46, php_box[3] - 6], radius=4, fill=(15, 23, 42, 200))
        draw.text((php_box[0] + 12, php_box[3] - 21), "PHP", fill=(255, 255, 255), font=font_small)
        # QHPI tag
        draw.rounded_rectangle([qhpi_box[0] + 6, qhpi_box[3] - 22, qhpi_box[0] + 50, qhpi_box[3] - 6], radius=4, fill=(15, 23, 42, 200))
        draw.text((qhpi_box[0] + 10, qhpi_box[3] - 21), "QHPI", fill=(255, 255, 255), font=font_small)

        # 3.4 หลอดความคืบหน้า (Plaque Coverage Progress Bar)
        bar_lbl_y = y + 266
        draw.text((x + 24, bar_lbl_y), "ความครอบคลุมคราบพลัค (Plaque Coverage)", fill=(100, 116, 139), font=font_small)
        
        bar_y = y + 288
        bar_h = 12
        bar_w = cell_w - 48 - 90  # หักส่วนตัวเลขขวาออก
        bar_bg_box = [x + 24, bar_y, x + 24 + bar_w, bar_y + bar_h]
        
        # วาด Track พื้นหลังสีเทาอ่อน
        draw.rounded_rectangle(bar_bg_box, radius=bar_h//2, fill=(241, 245, 249))
        
        # วาดส่วนที่ระบายสีตามคราบพลัค
        if ratio_pct > 0:
            fill_len = max(bar_h, int(bar_w * r.get("ratio", 0.0)))
            fill_len = min(fill_len, bar_w)
            draw.rounded_rectangle([x + 24, bar_y, x + 24 + fill_len, bar_y + bar_h], radius=bar_h//2, fill=sev_color)
            
        # ตัวเลขเปอร์เซ็นต์ด้านขวาสุดของหลอด
        pct_text = f"{ratio_pct:.2f}%"
        draw.text((x + 24 + bar_w + 14, bar_y - 3), pct_text, fill=(15, 23, 42), font=font_body_bold)

        # 3.5 ชุดคะแนนแบบบล็อก (Segment Block Gauges) สำหรับ PHP และ QHPI
        scores_y = y + 324
        
        # คอลัมน์ซ้าย: PHP Score Block Gauge
        php_col_x = x + 24
        draw.text((php_col_x, scores_y), f"คะแนน PHP Score: {php_score} / {php_max}", fill=(71, 85, 105), font=font_small)
        
        # วาด 5 บล็อก
        block_y = scores_y + 24
        block_w, block_h = 24, 10
        block_gap = 5
        for i in range(5):
            bx = php_col_x + i * (block_w + block_gap)
            bbox = [bx, block_y, bx + block_w, block_y + block_h]
            if i < php_score:
                draw.rounded_rectangle(bbox, radius=3, fill=sev_color)
            else:
                draw.rounded_rectangle(bbox, radius=3, fill=(226, 232, 240)) # Active off / Gray

        # คอลัมน์ขวา: QHPI Score Block Gauge
        qhpi_col_x = x + 265
        draw.text((qhpi_col_x, scores_y), f"คะแนน QHPI Score: {qhpi_score} / 5", fill=(71, 85, 105), font=font_small)
        
        # วาด 5 บล็อก
        for i in range(5):
            bx = qhpi_col_x + i * (block_w + block_gap)
            bbox = [bx, block_y, bx + block_w, block_y + block_h]
            if i < qhpi_score:
                draw.rounded_rectangle(bbox, radius=3, fill=sev_color)
            else:
                draw.rounded_rectangle(bbox, radius=3, fill=(226, 232, 240)) # Active off / Gray

    # 4. บันทึกรูปภาพผลลัพธ์
    save_path = os.path.join(save_dir, "case_summary_php_qhpi.png")
    canvas.save(save_path)
    return save_path


def main():
    import json
    
    output_root = r"D:/Internship Test/Plaque Result Curves"
    if not os.path.exists(output_root):
        print(f"Error: Output root directory not found: {output_root}")
        return
        
    case_folders = [
        d
        for d in os.listdir(output_root)
        if os.path.isdir(os.path.join(output_root, d))
    ]
    
    print("PLAQUE SUMMARY GENERATOR")
    print(f"OUTPUT_ROOT = {output_root}")
    print(f"FOUND CASES = {len(case_folders)}")
    
    for case_name in case_folders:
        save_dir = os.path.join(output_root, case_name)
        json_path = os.path.join(save_dir, "case_results.json")
        
        if not os.path.exists(json_path):
            print(f"Warning: case_results.json not found for case: {case_name}")
            continue
            
        print(f"\nProcessing case summary for: {case_name}")
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                case_results = json.load(jf)
                
            summary_path = create_case_summary_image(case_name, case_results, save_dir)
            if summary_path:
                print(f"Successfully generated summary image: {summary_path}")
            else:
                print(f"Failed to generate summary image (empty case results) for {case_name}")
        except Exception as e:
            print(f"Error generating summary image for {case_name}: {e}")


if __name__ == "__main__":
    main()

