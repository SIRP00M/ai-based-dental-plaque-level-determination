import cv2
import numpy as np

# นำเข้าฟังก์ชันที่จำเป็นจากการคำนวณ
from plaque_calculation import get_x_on_row, make_shape_based_zone_masks

def draw_dashed_polyline(img, pts, color, thickness=2, dash_length=10, gap_length=6):
    pts = np.asarray(pts, dtype=np.int32)

    if len(pts) < 2:
        return img

    for i in range(len(pts) - 1):
        p1 = pts[i].astype(np.float32)
        p2 = pts[i + 1].astype(np.float32)

        vec = p2 - p1
        dist = np.linalg.norm(vec)

        if dist == 0:
            continue

        direction = vec / dist
        current = 0.0

        while current < dist:
            start = p1 + direction * current
            end = p1 + direction * min(current + dash_length, dist)

            cv2.line(
                img,
                tuple(start.astype(np.int32)),
                tuple(end.astype(np.int32)),
                color,
                thickness,
                cv2.LINE_AA
            )

            current += dash_length + gap_length

    return img

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

def visualize_plaque(image_rgb, plaque_mask, alpha=0.45):
    out = image_rgb.copy().astype(np.float32)

    red = np.zeros_like(out)
    red[:, :, 0] = 255

    mask_bool = plaque_mask > 0
    out[mask_bool] = out[mask_bool] * (1 - alpha) + red[mask_bool] * alpha

    return out.astype(np.uint8)

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
    tooth_id = guide.get("tooth_id", None)

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

        x1 = get_x_on_row(left_x, right_x, y, 1 / 3)
        x2 = get_x_on_row(left_x, right_x, y, 2 / 3)

        if x1 is not None:
            pts_left_mid.append([x1, y])

        if x2 is not None:
            pts_right_mid.append([x2, y])

    if tooth_id == 13:
        if len(pts_right_mid) > 1:
            cv2.polylines(out, [np.array(pts_right_mid, dtype=np.int32)], False, line_color, 2)

    elif tooth_id == 23:
        if len(pts_left_mid) > 1:
            cv2.polylines(out, [np.array(pts_left_mid, dtype=np.int32)], False, line_color, 2)

    else:
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
            x1 = get_x_on_row(left_x, right_x, y, 1 / 3)
            x2 = get_x_on_row(left_x, right_x, y, 2 / 3)

            if x1 is not None and x2 is not None:
                if tooth_id == 13:
                    b_left, b_right = xl, x2
                elif tooth_id == 23:
                    b_left, b_right = x1, xr
                else:
                    b_left, b_right = x1, x2

                if y == y1:
                    row1 = [[x, y] for x in range(b_left, b_right + 1)]
                else:
                    row2 = [[x, y] for x in range(b_left, b_right + 1)]

    if len(row1) > 1:
        cv2.polylines(out, [np.array(row1, dtype=np.int32)], False, line_color, 2)

    if len(row2) > 1:
        cv2.polylines(out, [np.array(row2, dtype=np.int32)], False, line_color, 2)

    zone_masks, _ = make_shape_based_zone_masks(
        tooth_mask,
        tooth_id=tooth_id,
        top_label=top_label
    )

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

def draw_qhpi_visual(image_rgb, tooth_mask, plaque_mask, qh_score, qh_detail):
    out = image_rgb.copy()

    colors = {
        0: (0, 255, 0),
        1: (120, 255, 0),
        2: (255, 255, 0),
        3: (255, 170, 0),
        4: (255, 100, 0),
        5: (255, 0, 0)
    }

    overlay_color = np.array(
        colors.get(qh_score, (255, 255, 255)),
        dtype=np.float32
    )

    out_f = out.astype(np.float32)

    mask_bool = plaque_mask > 0
    alpha = 0.55

    out_f[mask_bool] = (
        out_f[mask_bool] * (1 - alpha)
        + overlay_color * alpha
    )

    out = out_f.astype(np.uint8)

    contours, _ = cv2.findContours(
        (tooth_mask.astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)

    top_edge = qh_detail.get("top_edge")

    if top_edge is not None:
        shift_score2 = qh_detail["shift_score2_band"]
        shift_1 = qh_detail["shift_1"]
        shift_2 = qh_detail["shift_2"]

        pts_score2 = []
        pts1 = []
        pts2 = []

        for x in range(len(top_edge)):
            if top_edge[x] == -1:
                continue

            ys_col = np.where(tooth_mask[:, x])[0]

            if len(ys_col) == 0:
                continue

            y_b = ys_col.max()

            y_score2 = int(round(top_edge[x] + shift_score2))
            y1 = int(round(top_edge[x] + shift_1))
            y2 = int(round(top_edge[x] + shift_2))

            if y_score2 <= y_b:
                pts_score2.append([x, y_score2])

            if y1 <= y_b:
                pts1.append([x, y1])

            if y2 <= y_b:
                pts2.append([x, y2])

        # เส้นเหลือง = ระยะประมาณ 1–2 mm จากขอบเหงือก
        if len(pts_score2) > 1:
            cv2.polylines(
                out,
                [np.array(pts_score2, dtype=np.int32)],
                False,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )

        # เส้นเขียวหลัก = เส้นประ
        if len(pts1) > 1:
            draw_dashed_polyline(
                out,
                pts1,
                color=(0, 255, 0),
                thickness=2,
                dash_length=10,
                gap_length=6
            )

        if len(pts2) > 1:
            draw_dashed_polyline(
                out,
                pts2,
                color=(0, 255, 0),
                thickness=2,
                dash_length=10,
                gap_length=6
            )

    cv2.putText(out, f"QHPI: {qh_score}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return out
