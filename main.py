import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO
import supervision as sv
import imageio

INPUT_VIDEO = "data/intersection.mp4"
OUTPUT_VIDEO = "data/output_od_matrix.mp4"
OUTPUT_CSV = "data/OD_matrix.csv"

# ضرایب استاندارد مهندسی ترافیک (PCU)
PCU_WEIGHTS = {2: 1.0, 3: 0.5, 5: 2.5, 7: 2.0}
CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

DIRECTIONS = ["North", "South", "East", "West"]
od_counts = {orig: {dest: 0 for dest in DIRECTIONS} for orig in DIRECTIONS}
od_pcu = {orig: {dest: 0.0 for dest in DIRECTIONS} for orig in DIRECTIONS}

turn_counts = {"Straight": 0, "Right-Turn": 0, "Left-Turn": 0}
class_counts = {"Car": 0, "Truck": 0, "Bus": 0, "Motorcycle": 0}

OUT_HEIGHT = 720
VIDEO_WIDTH = 960
DASHBOARD_WIDTH = 480

# مختصات هندسی کالیبره‌شده ۴ زون ورودی و خروجی - دقیقاً منطبق بر پهنای آسفالت
APPROACH_ZONES = {
    "North": np.array([[150, 180], [300, 180], [330, 270], [140, 270]], dtype=np.int32),
    "South": np.array([[220, 560], [600, 560], [600, 710], [220, 710]], dtype=np.int32),
    "East":  np.array([[650, 360], [940, 360], [940, 490], [650, 490]], dtype=np.int32),
    "West":  np.array([[15, 320],  [180, 320], [180, 460], [15, 460]],  dtype=np.int32)
}

ZONE_COLORS = {
    "North": (255, 180, 0),
    "South": (0, 230, 255),
    "East":  (230, 0, 255),
    "West":  (0, 255, 120)
}

def get_zone_at_point(pt):
    x, y = pt
    for name, poly in APPROACH_ZONES.items():
        if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
            return name
    return None

def classify_movement(orig, dest):
    straight_routes = [("North", "South"), ("South", "North"), ("West", "East"), ("East", "West")]
    right_routes = [("North", "West"), ("West", "South"), ("South", "East"), ("East", "North")]
    if (orig, dest) in straight_routes:
        return "Straight"
    elif (orig, dest) in right_routes:
        return "Right-Turn"
    else:
        return "Left-Turn"

track_history = defaultdict(lambda: deque(maxlen=150))
track_origin = {}
track_dest = {}
counted_ids = set()

# مدل با بافر بالا برای حفظ شناسه خودروهای متوقف در صف
model = YOLO("yolov8s.pt")
byte_track = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=120,
    minimum_matching_threshold=0.55,
    frame_rate=30
)

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.45, text_thickness=1)
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=35)

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
cap.release()

print("🚀 در حال پردازش تقاطع و استخراج داده‌های حرکتی...")
writer = imageio.get_writer(OUTPUT_VIDEO, fps=fps, codec='libx264', macro_block_size=None)
frame_generator = sv.get_video_frames_generator(INPUT_VIDEO)

for raw_frame in frame_generator:
    frame = cv2.resize(raw_frame, (VIDEO_WIDTH, OUT_HEIGHT))
    
    results = model(frame, classes=[2, 3, 5, 7], conf=0.25, iou=0.40, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # حذف باکس‌های هم‌پوشان (حل مشکل تفکیک اشتباه بار وانت)
    if len(detections) > 0:
        detections = detections.with_nms(threshold=0.40)
    
    detections = byte_track.update_with_detections(detections)
    bottom_points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    custom_labels = []
    for t_id, class_id, pt_float in zip(detections.tracker_id, detections.class_id, bottom_points):
        pt = (int(pt_float[0]), int(pt_float[1]))
        track_history[t_id].append(pt)

        # ثبت مبدا ورود
        if t_id not in track_origin:
            z = get_zone_at_point(pt)
            if z:
                track_origin[t_id] = z

        origin = track_origin.get(t_id, None)
        c_name = CLASS_NAMES.get(class_id, "Veh")
        lbl = f"#{t_id} {c_name}"

        # بررسی مسیر و ثبت در ماتریس مبدا-مقصد
        if origin and len(track_history[t_id]) >= 10:
            start_pt = track_history[t_id][0]
            displacement = np.linalg.norm(np.array(pt) - np.array(start_pt))
            curr_zone = get_zone_at_point(pt)

            if curr_zone and curr_zone != origin and displacement > 80:
                track_dest[t_id] = curr_zone

                if t_id not in counted_ids:
                    pcu_val = PCU_WEIGHTS.get(class_id, 1.0)

                    od_counts[origin][curr_zone] += 1
                    od_pcu[origin][curr_zone] += pcu_val

                    m_type = classify_movement(origin, curr_zone)
                    turn_counts[m_type] += 1
                    class_counts[c_name] += 1

                    counted_ids.add(t_id)

            if t_id in track_dest:
                m_type = classify_movement(origin, track_dest[t_id])
                lbl += f" [{origin[0]}->{track_dest[t_id][0]} {m_type[:4]}]"

        custom_labels.append(lbl)

    # رسم ۴ زون کالیبره‌شده روی آسفالت
    annotated_frame = frame.copy()
    overlay = annotated_frame.copy()
    for name, poly in APPROACH_ZONES.items():
        cv2.fillPoly(overlay, [poly], ZONE_COLORS[name])
        cv2.polylines(annotated_frame, [poly], isClosed=True, color=ZONE_COLORS[name], thickness=2)
        cx = int(np.mean(poly[:, 0]))
        cy = int(np.mean(poly[:, 1]))
        cv2.putText(annotated_frame, f"Zone: {name}", (cx - 45, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

    cv2.addWeighted(overlay, 0.22, annotated_frame, 0.78, 0, annotated_frame)

    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=custom_labels)

    # --- داشبورد مهندسی ترافیک ---
    dashboard = np.zeros((OUT_HEIGHT, DASHBOARD_WIDTH, 3), dtype=np.uint8)
    cv2.putText(dashboard, "LIVE O-D TRAFFIC MATRIX", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(dashboard, "Automated Turning Movement Counts (TMC)", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    headers = ["O \\ D", "North", "South", "East", "West"]
    start_y = 115
    col_w = 80
    cell_h = 32

    for col_idx, h in enumerate(headers):
        cv2.putText(dashboard, h, (15 + col_idx * col_w, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    for row_idx, orig in enumerate(DIRECTIONS):
        y_pos = start_y + (row_idx + 1) * cell_h
        cv2.putText(dashboard, orig[:5], (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        for col_idx, dest in enumerate(DIRECTIONS):
            count = od_counts[orig][dest]
            color = (255, 255, 255) if count > 0 else (70, 70, 70)
            cv2.putText(dashboard, str(count), (15 + (col_idx + 1) * col_w, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1 if count == 0 else 2)

    # آمار تفکیکی مانورها
    cv2.line(dashboard, (15, 290), (DASHBOARD_WIDTH - 15, 290), (80, 80, 80), 1)
    cv2.putText(dashboard, "TURNING MOVEMENTS (TMC)", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    
    cv2.putText(dashboard, f"• Through (Straight): {turn_counts['Straight']}", (25, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
    cv2.putText(dashboard, f"• Right-Turns:        {turn_counts['Right-Turn']}", (25, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
    cv2.putText(dashboard, f"• Left-Turns:         {turn_counts['Left-Turn']}", (25, 406), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

    # تفکیک وسایل نقلیه
    cv2.line(dashboard, (15, 435), (DASHBOARD_WIDTH - 15, 435), (80, 80, 80), 1)
    cv2.putText(dashboard, "VEHICLE CLASSIFICATION", (20, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(dashboard, f"Cars: {class_counts['Car']} | Trucks: {class_counts['Truck']} | Bus: {class_counts['Bus']} | Moto: {class_counts['Motorcycle']}", 
                (20, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

    # مجموع شاخص‌ها
    total_vehicles = sum(turn_counts.values())
    total_pcu = sum(sum(od_pcu[o].values()) for o in DIRECTIONS)

    cv2.line(dashboard, (15, 525), (DASHBOARD_WIDTH - 15, 525), (80, 80, 80), 1)
    cv2.putText(dashboard, f"Total Trips Logged: {total_vehicles}", (20, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    cv2.putText(dashboard, f"Total Demand: {total_pcu:.1f} PCU", (20, 595), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 120), 2)
    cv2.putText(dashboard, "Standard: Car=1.0 | Truck=2.0 | Bus=2.5 | Moto=0.5", (20, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

    split_screen = np.hstack((annotated_frame, dashboard))
    writer.append_data(cv2.cvtColor(split_screen, cv2.COLOR_BGR2RGB))

writer.close()

df_od = pd.DataFrame(od_counts)
df_od.to_csv(OUTPUT_CSV)
print(f"\n✅ پردازش کامل شد:\n📹 ویدیو: {OUTPUT_VIDEO}\n📊 فایل اکسل: {OUTPUT_CSV}")