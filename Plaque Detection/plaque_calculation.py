import cv2
import numpy as np


# ==============================================================================
# QHPI SETTINGS
# ==============================================================================

# Approximate gingival band depth for QHPI score 2 region
QHPI_SCORE2_BAND_MM = 0.5


# ==============================================================================
# BASIC MASK HELPERS
# ==============================================================================

def get_tooth_mask_from_rgba(image_rgba):
    """Extract tooth mask from alpha channel."""
    alpha = image_rgba[:, :, 3]
    return alpha > 0


# ==============================================================================
# PLAQUE DETECTION
# ==============================================================================

def detect_plaque_mask(image_rgb, tooth_mask):
    """
    Detect disclosed plaque using HSV + LAB color thresholds.
    """

    # Convert color spaces for color-based plaque filtering
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    A = lab[:, :, 1]
    B = lab[:, :, 2]

    # Pink disclosed plaque
    mask_pink = cv2.inRange(
        hsv,
        np.array([140, 20, 40], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8)
    )

    # Purple disclosed plaque
    mask_purple = cv2.inRange(
        hsv,
        np.array([110, 20, 40], dtype=np.uint8),
        np.array([145, 255, 255], dtype=np.uint8)
    )

    # Blue-purple plaque variation
    mask_bluepurple = cv2.inRange(
        hsv,
        np.array([95, 25, 35], dtype=np.uint8),
        np.array([125, 255, 255], dtype=np.uint8)
    )

    # Combine color and brightness constraints
    color_mask = (mask_pink > 0) | (mask_purple > 0) | (mask_bluepurple > 0)
    sat_mask = S > 25
    val_mask = V > 35
    lab_mask = (A > 120) & (B < 150)

    mask = color_mask & sat_mask & val_mask & lab_mask & tooth_mask
    mask = (mask.astype(np.uint8) * 255)

    # Remove small noise and fill small gaps
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Remove tiny connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    min_area = 20

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned


# ==============================================================================
# TOOTH SHAPE PROFILE
# ==============================================================================

def build_shape_profile(tooth_mask):
    """
    Build left/right tooth boundary profile for each row.
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
    """Get interpolated x-position across tooth width."""
    xl = left_x[y]
    xr = right_x[y]

    if xl < 0 or xr < 0 or xr <= xl:
        return None

    return int(round(xl + frac * (xr - xl)))


# ==============================================================================
# SHAPE-BASED PHP ZONES
# ==============================================================================

def make_shape_based_zone_masks(tooth_mask, tooth_id=None, top_label="I"):
    """
    Create adaptive PHP zones following tooth contour.
    """

    profile = build_shape_profile(tooth_mask)

    if profile is None:
        return None, None

    h, w = tooth_mask.shape

    y_min = profile["y_min"]
    y_max = profile["y_max"]
    left_x = profile["left_x"]
    right_x = profile["right_x"]

    # Split tooth vertically into thirds
    height = y_max - y_min + 1
    y1 = y_min + height // 3
    y2 = y_min + (2 * height) // 3

    active_zones = ["M", top_label, "C", "G", "D"]

    # Canines may use only 4 visible zones
    if tooth_id == 13:
        active_zones = [top_label, "C", "G", "D"]

    elif tooth_id == 23:
        active_zones = ["M", top_label, "C", "G"]

    zone_masks = {}

    for z in active_zones:
        zone_masks[z] = np.zeros((h, w), dtype=np.uint8)

    # Generate zone masks row-by-row
    for y in range(y_min, y_max + 1):
        xl = left_x[y]
        xr = right_x[y]

        if xl < 0 or xr < 0 or xr <= xl:
            continue

        x_l_mid = get_x_on_row(left_x, right_x, y, 1 / 3)
        x_r_mid = get_x_on_row(left_x, right_x, y, 2 / 3)

        if x_l_mid is None or x_r_mid is None:
            continue

        # Side zones
        if tooth_id == 13:
            bound_left = xl
            bound_right = x_r_mid
            zone_masks["D"][y, bound_right:xr + 1] = 1

        elif tooth_id == 23:
            bound_left = x_l_mid
            bound_right = xr
            zone_masks["M"][y, xl:bound_left] = 1

        else:
            bound_left = x_l_mid
            bound_right = x_r_mid
            zone_masks["M"][y, xl:bound_left] = 1
            zone_masks["D"][y, bound_right:xr + 1] = 1

        # Top / center / gingival zones
        if y < y1:
            if top_label in active_zones:
                zone_masks[top_label][y, bound_left:bound_right] = 1

        elif y < y2:
            if "C" in active_zones:
                zone_masks["C"][y, bound_left:bound_right] = 1

        else:
            if "G" in active_zones:
                zone_masks["G"][y, bound_left:bound_right] = 1

    # Keep only valid tooth pixels
    for k in zone_masks:
        zone_masks[k] = (zone_masks[k] > 0) & tooth_mask

    guide = {
        "profile": profile,
        "y1": int(y1),
        "y2": int(y2),
        "top_label": top_label,
        "tooth_id": tooth_id
    }

    return zone_masks, guide


# ==============================================================================
# PHP SCORING
# ==============================================================================

def score_php_from_zone_masks(plaque_mask, zone_masks, min_plaque_pixels=1):
    """
    Compute PHP score from adaptive zone masks.
    """

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

    max_score = len(zone_masks)

    return score, max_score, detail


# ==============================================================================
# QHPI HELPERS
# ==============================================================================

def estimate_pixels_per_mm(tooth_mask):
    """
    Rough pixel-to-mm estimation using crown height heuristic.
    """

    ys, xs = np.where(tooth_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    y_min = ys.min()
    y_max = ys.max()

    tooth_height_px = y_max - y_min + 1

    # Approximate average crown height
    approx_crown_height_mm = 8.0

    px_per_mm = tooth_height_px / approx_crown_height_mm

    return max(px_per_mm, 1.0)


def get_row_segments(binary_row):
    """Split binary row into connected segments."""

    xs = np.where(binary_row)[0]

    if len(xs) == 0:
        return []

    segments = []

    start = xs[0]
    prev = xs[0]

    for x in xs[1:]:

        if x == prev + 1:
            prev = x

        else:
            segments.append((start, prev))
            start = x
            prev = x

    segments.append((start, prev))

    return segments


# ==============================================================================
# GINGIVAL ANALYSIS
# ==============================================================================

def measure_gingival_band_properties(plaque_mask, tooth_mask, px_per_mm):
    """
    Measure gingival plaque depth and continuity heuristics.
    """

    ys, xs = np.where(tooth_mask)

    if len(xs) == 0 or len(ys) == 0:
        return {
            "max_vertical_depth_px": 0,
            "max_band_width_ratio": 0.0,
            "fleck_count": 0
        }

    y_min = ys.min()
    y_max = ys.max()

    height = y_max - y_min + 1

    # Analyze upper gingival region only
    gingival_depth_px = int(round(2.0 * px_per_mm))
    gingival_depth_px = max(1, min(gingival_depth_px, height))

    max_vertical_depth_px = 0
    max_band_width_ratio = 0.0
    fleck_count = 0

    profile = build_shape_profile(tooth_mask)

    left_x = profile["left_x"]
    right_x = profile["right_x"]

    # Analyze horizontal continuity
    for y in range(y_min, y_min + gingival_depth_px):

        if y >= tooth_mask.shape[0]:
            break

        xl = left_x[y]
        xr = right_x[y]

        if xl < 0 or xr < 0 or xr <= xl:
            continue

        tooth_row = tooth_mask[y, xl:xr + 1]
        plaque_row = (plaque_mask[y, xl:xr + 1] > 0) & tooth_row

        row_width = int(np.sum(tooth_row))

        if row_width == 0:
            continue

        segments = get_row_segments(plaque_row)

        # Small isolated plaque spots
        small_flecks = sum(
            1
            for s, e in segments
            if (e - s + 1) <= max(2, int(round(0.15 * row_width)))
        )

        fleck_count += small_flecks

        if segments:
            longest = max((e - s + 1) for s, e in segments)
            max_band_width_ratio = max(max_band_width_ratio, longest / row_width)

    # Analyze vertical depth
    xs_all = np.where(tooth_mask.any(axis=0))[0]

    for x in xs_all:

        col_tooth = tooth_mask[y_min:y_min + gingival_depth_px, x]

        col_plaque = (
            (plaque_mask[y_min:y_min + gingival_depth_px, x] > 0)
            & col_tooth
        )

        if not np.any(col_plaque):
            continue

        ys_local = np.where(col_plaque)[0]

        max_vertical_depth_px = max(
            max_vertical_depth_px,
            int(ys_local.max() + 1)
        )

    return {
        "max_vertical_depth_px": max_vertical_depth_px,
        "max_band_width_ratio": max_band_width_ratio,
        "fleck_count": fleck_count,
    }


# ==============================================================================
# QHPI SCORING
# ==============================================================================

def compute_qhpi_from_mask(plaque_mask, tooth_mask, pixels_per_mm=None):
    """
    Compute QHPI score using plaque depth and coverage heuristics.
    """

    h, w = tooth_mask.shape

    ys, xs = np.where(tooth_mask)

    if len(xs) == 0 or len(ys) == 0:
        return 0, {
            "ratio": 0.0,
            "pixels_per_mm": None,
            "vertical_depth_mm": 0.0,
            "band_width_ratio": 0.0,
            "top_edge": None,
        }

    y_min, y_max = ys.min(), ys.max()

    H = y_max - y_min + 1

    # Estimate scale automatically
    if pixels_per_mm is None:
        pixels_per_mm = estimate_pixels_per_mm(tooth_mask)

    # Top contour of tooth
    top_edge = np.full(w, -1, dtype=np.int32)

    for x in range(w):

        ys_col = np.where(tooth_mask[:, x])[0]

        if len(ys_col) > 0:
            top_edge[x] = ys_col.min()

    # Vertical zone boundaries
    shift_1 = H / 3.0
    shift_2 = 2.0 * H / 3.0

    shift_score2_band = QHPI_SCORE2_BAND_MM * pixels_per_mm
    shift_score2_band = min(shift_score2_band, shift_1 * 0.45)

    zone1_mask = np.zeros_like(tooth_mask)
    zone2_mask = np.zeros_like(tooth_mask)

    gingival_mask = np.zeros_like(tooth_mask)
    middle_mask = np.zeros_like(tooth_mask)
    incisal_mask = np.zeros_like(tooth_mask)

    # Build adaptive QHPI zones
    for x in range(w):

        if top_edge[x] == -1:
            continue

        y_t = top_edge[x]

        y_score2 = int(round(y_t + shift_score2_band))
        y_c1 = int(round(y_t + shift_1))
        y_c2 = int(round(y_t + shift_2))

        ys_col = np.where(tooth_mask[:, x])[0]

        for y in ys_col:

            if y < y_score2:
                zone1_mask[y, x] = 1
                gingival_mask[y, x] = 1

            elif y < y_c1:
                zone2_mask[y, x] = 1
                gingival_mask[y, x] = 1

            elif y < y_c2:
                middle_mask[y, x] = 1

            else:
                incisal_mask[y, x] = 1

    # Gingival plaque measurements
    gingival_props = measure_gingival_band_properties(
        plaque_mask,
        tooth_mask,
        pixels_per_mm
    )

    vertical_depth_mm = (
        gingival_props["max_vertical_depth_px"] / pixels_per_mm
        if pixels_per_mm
        else 0.0
    )

    band_width_ratio = gingival_props["max_band_width_ratio"]

    # Plaque coverage ratio
    plaque_area = int(np.sum((plaque_mask > 0) & tooth_mask))
    tooth_area = int(np.sum(tooth_mask))

    ratio = plaque_area / tooth_area if tooth_area > 0 else 0.0

    # Zone presence checks
    has_zone1 = np.sum((plaque_mask > 0) & zone1_mask) > 0
    has_zone2 = np.sum((plaque_mask > 0) & zone2_mask) > 0
    has_gingival = np.sum((plaque_mask > 0) & gingival_mask) > 0
    has_middle = np.sum((plaque_mask > 0) & middle_mask) > 0
    has_incisal = np.sum((plaque_mask > 0) & incisal_mask) > 0

    # Heuristic QHPI scoring
    if plaque_area == 0:
        score = 0

    elif has_incisal:
        score = 5

    elif has_middle:
        score = 4

    elif has_gingival:

        if vertical_depth_mm > 1.0:
            score = 3

        elif has_zone2 and band_width_ratio >= 0.50:
            score = 2

        elif has_zone1 or has_zone2:
            score = 1

        else:
            score = 0

    else:
        score = 0

    qh_detail = {
        "ratio": ratio,
        "pixels_per_mm": pixels_per_mm,
        "vertical_depth_mm": float(vertical_depth_mm),
        "band_width_ratio": float(band_width_ratio),
        "top_edge": top_edge,
        "shift_score2_band": shift_score2_band,
        "shift_1": shift_1,
        "shift_2": shift_2,
        "score2_band_mm": QHPI_SCORE2_BAND_MM,
        "has_zone1": bool(has_zone1),
        "has_zone2": bool(has_zone2),
        "has_gingival": bool(has_gingival),
        "has_middle": bool(has_middle),
        "has_incisal": bool(has_incisal),
    }

    return score, qh_detail
