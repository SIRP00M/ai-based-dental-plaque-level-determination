import os
import numpy as np
import itertools
from PIL import Image
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

# ==============================================================================
# 1. ตั้งค่า Paths (ตรวจสอบและแก้ไขให้ตรงกับเครื่องของคุณ)
# ==============================================================================
WEIGHTS   = r"D:/Internship/h5/mask_rcnn_tooth_single_class_0097.h5"  
IMG_DIR   = r"D:/Internship/Data"       
OUT_ROOT  = r"D:/Internship/Teeth Segment Result"  

os.makedirs(OUT_ROOT, exist_ok=True)

KEEP_K         = 6              
Y_BAND         = (0.20, 0.65)   
MIN_AREA_RATIO = 0.002          

# ==============================================================================
# 2. Config & Load Model
# ==============================================================================
class PredictionConfig(Config):
    NAME = "tooth_one_class"
    NUM_CLASSES = 1 + 1         
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

print("Loading Model...")
cfg = PredictionConfig()
model = MaskRCNN(mode="inference", model_dir="logs", config=cfg)
model.load_weights(WEIGHTS, by_name=True)

# ==============================================================================
# 3. อัลกอริทึม แม่พิมพ์อัจฉริยะ (Anatomical Template Matching) ***ใหม่ล่าสุด***
# ==============================================================================
def assign_fdi_names_smart(rois, image_width):
    K = rois.shape[0]
    if K == 0: return []

    centers = [(b[1] + b[3]) / 2.0 for b in rois]
    widths  = [b[3] - b[1] for b in rois]
    w_median = np.median(widths)
    
    # 1. ประเมินระยะห่างมาตรฐาน (S) จากฟันที่อยู่ติดกัน
    adjacent_pitches = []
    for i in range(1, K):
        edge_gap = rois[i][1] - rois[i-1][3]
        if edge_gap < w_median * 0.8: # ถ้าอยู่ใกล้กัน ถือว่าติดกัน
            adjacent_pitches.append(centers[i] - centers[i-1])
            
    S = np.median(adjacent_pitches) if adjacent_pitches else (w_median * 1.05)

    # 2. สร้างแม่พิมพ์สัดส่วนฟันจริง (อิงตามค่าเฉลี่ยมิลลิเมตรของฟันมนุษย์)
    # [13, 12, 11, 21, 22, 23] 
    # ฟันหน้ากว้างกว่า ระยะศูนย์กลางจึงห่างกว่า
    template_mults = np.array([-2.2, -1.4, -0.5, 0.5, 1.4, 2.2])
    M = image_width / 2.0 # ยึดกึ่งกลางภาพเป็นกึ่งกลางปาก
    
    FDI_LIST = ["tooth13", "tooth12", "tooth11", "tooth21", "tooth22", "tooth23"]
    
    best_combo = None
    min_error = float('inf')
    num_teeth = min(K, 6)
    
    # 3. สร้าง "ทุกความเป็นไปได้" ในการเอาฟันที่เจอไปใส่ใน 6 ช่อง
    # เช่น เจอ 4 ซี่ อาจจะเป็น (13,12,11,21) หรือ (13,12,22,23)
    valid_combos = list(itertools.combinations(range(6), num_teeth))
    
    for combo in valid_combos:
        # ดึงตำแหน่งช่องที่สมมติขึ้นมา
        combo_offsets = template_mults[list(combo)] * S
        
        # คำนวณหาค่า Offset (O) ที่ทำให้ฟันชุดนี้ลงล็อกแม่พิมพ์ที่สุด
        O = np.mean(np.array(centers[:num_teeth]) - M - combo_offsets)
        
        # คำนวณความคลาดเคลื่อน (ถ้าทายผิดซี่ ตำแหน่งจะเบี้ยว ทำให้ Error สูง)
        predicted_centers = M + O + combo_offsets
        shape_error = np.sum((np.array(centers[:num_teeth]) - predicted_centers)**2)
        
        # บทลงโทษ: ถ้าเลื่อนแม่พิมพ์หนีจากกึ่งกลางภาพมากเกินไป ให้บวก Error เพิ่ม
        # ป้องกันไม่ให้โมเดลเบี้ยวซ้าย/ขวาสุดโต่ง
        offset_penalty = (O ** 2) * 0.5 
        
        total_error = shape_error + offset_penalty
        
        # เก็บรูปแบบที่ Error น้อยที่สุดไว้
        if total_error < min_error:
            min_error = total_error
            best_combo = combo
            
    # 4. ออกชื่อฟันตามรูปแบบที่ดีที่สุด
    final_fdi = []
    for i in range(K):
        if i < 6:
            final_fdi.append(FDI_LIST[best_combo[i]])
        else:
            final_fdi.append(f"extra_tooth_{i}") 
            
    return final_fdi

# ==============================================================================
# 4. Helper Functions
# ==============================================================================
def filter_and_sort_teeth(image, rois, masks, scores):
    H, W = image.shape[:2]
    if rois.size == 0: return rois, masks, scores
    
    y1, x1, y2, x2 = rois[:,0], rois[:,1], rois[:,2], rois[:,3]
    yc, area = (y1 + y2) / 2.0, (y2 - y1) * (x2 - x1)
    
    band_ok = (yc > Y_BAND[0]*H) & (yc < Y_BAND[1]*H)
    size_ok = area > (MIN_AREA_RATIO*H*W)
    keep = np.where(band_ok & size_ok)[0]
    
    if keep.size > 0:
        rois, masks, scores = rois[keep], masks[:,:,keep], scores[keep]
        
    if rois.shape[0] > KEEP_K:
        topk = np.argsort(scores)[-KEEP_K:]
        rois, masks, scores = rois[topk], masks[:,:,topk], scores[topk]
        
    if rois.shape[0] > 0:
        xc = (rois[:,1] + rois[:,3]) / 2.0
        order = np.argsort(xc)
        rois, masks, scores = rois[order], masks[:,:,order], scores[order]
        
    return rois, masks, scores

def tight_crop_by_mask(image_np, m_bool):
    ys, xs = np.where(m_bool)
    if ys.size == 0: return None
    
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    
    crop_rgb  = image_np[y1:y2, x1:x2, :]
    crop_mask = m_bool[y1:y2, x1:x2]
    crop_rgba = np.dstack([crop_rgb, (crop_mask.astype(np.uint8)*255)])
    return Image.fromarray(crop_rgba, mode="RGBA")

# ==============================================================================
# 5. Main Execution
# ==============================================================================
img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"\nFound {len(img_files)} images to process.\n")

for filename in img_files:
    img_path = os.path.join(IMG_DIR, filename)
    name = os.path.splitext(filename)[0]
    
    image = np.array(Image.open(img_path).convert("RGB"))
    H, W = image.shape[:2]
    
    r = model.detect([image], verbose=0)[0]
    rois, masks, scores = filter_and_sort_teeth(image, r['rois'], r['masks'], r.get('scores', np.ones(r['rois'].shape[0])))
    
    K = masks.shape[-1] if masks.size else 0
    if K == 0:
        print(f"[SKIP] {name}: No valid teeth found.")
        continue
        
    fdi_names = assign_fdi_names_smart(rois, W)
    
    img_out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(img_out_dir, exist_ok=True)
    
    Image.fromarray(image).save(os.path.join(img_out_dir, f"{name}_original.jpg"))
    
    union_mask = masks.any(axis=2)
    full_rgba = np.dstack([image, (union_mask.astype(np.uint8)*255)])
    Image.fromarray(full_rgba, mode="RGBA").save(os.path.join(img_out_dir, f"{name}_segmented_teeth_only.png"))
    
    for i in range(K):
        m_bool = masks[:, :, i].astype(bool)
        crop_im = tight_crop_by_mask(image, m_bool)
        
        if crop_im is not None:
            tooth_name = fdi_names[i]
            crop_im.save(os.path.join(img_out_dir, f"{tooth_name}.png"))
            
    print(f"[SUCCESS] {name} -> Detected {K} teeth : {fdi_names}")

print(f"\n==============================================")
print(f"All processing complete! Check results in:\n-> {OUT_ROOT}")
print(f"==============================================")