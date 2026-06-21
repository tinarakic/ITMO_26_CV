"""
HoneyBee Tracking Interface
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import tempfile
import shutil
import subprocess
import json
import uuid
from collections import defaultdict

from ultralytics import YOLO
import supervision as sv
from trackers import OCSORTTracker
from sklearn.decomposition import PCA

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# ==================== КОНСТАНТЫ ====================

PURPLE = (156, 102, 233)
YELLOW = (255, 255, 0)
ARROW_MIN_DIST = 8


# ==================== ОБРАБОТКА ИЗОБРАЖЕНИЙ ====================

def tile_image(img, tile_size=640, overlap=0.2):
    h, w = img.shape[:2]
    step = max(1, int(tile_size * (1 - overlap)))
    tiles = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            tile = img[y1:y2, x1:x2]
            tiles.append((tile, x1, y1))
            if x2 == w:
                break
        if y2 == h:
            break
    return tiles


def yolo_result_to_xyxy_conf_cls_kpts(result, conf_thres=0.25):
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32), []

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=np.float32)
    cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy), dtype=np.float32)

    keep = conf >= conf_thres
    if keep.sum() == 0:
        return np.empty((0, 6), dtype=np.float32), []

    dets = np.column_stack([xyxy[keep], conf[keep], cls[keep]]).astype(np.float32)

    kpts_list = [None] * len(dets)
    if getattr(result, "keypoints", None) is not None:
        try:
            kdata = result.keypoints.data.cpu().numpy()[keep]
            kpts_list = [k.astype(np.float32) for k in kdata]
        except Exception:
            kpts_list = [None] * len(dets)

    return dets, kpts_list


def nms_keep_indices(dets, iou_thres=0.5):
    if len(dets) == 0:
        return np.empty((0,), dtype=np.int64)
    
    try:
        import torchvision
        boxes = torch.tensor(dets[:, :4], dtype=torch.float32)
        scores = torch.tensor(dets[:, 4], dtype=torch.float32)
        keep = torchvision.ops.nms(boxes, scores, float(iou_thres))
        return keep.cpu().numpy()
    except:
        return simple_nms(dets, iou_thres)


def simple_nms(dets, iou_thres=0.5):
    if len(dets) == 0:
        return np.empty((0,), dtype=np.int64)
    
    boxes = dets[:, :4]
    scores = dets[:, 4]
    order = scores.argsort()[::-1]
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_i + area_rest - inter
        ious = inter / (union + 1e-6)
        order = order[ious <= iou_thres]
    
    return np.array(keep, dtype=np.int64)


def _valid_kpts(kpts):
    if kpts is None:
        return []
    pts = np.asarray(kpts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return []
    vis = pts[:, 2] if pts.shape[1] > 2 else np.ones(len(pts), dtype=np.float32)
    out = []
    for (x, y), v in zip(pts[:, :2], vis):
        if np.isnan(x) or np.isnan(y):
            continue
        if v > 0:
            out.append((float(x), float(y)))
    return out


def _kpt_orientation(kpts, min_dist=ARROW_MIN_DIST):
    valid = _valid_kpts(kpts)
    if len(valid) < 2:
        return None
    p0, p1 = valid[0], valid[1]
    dist = ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5
    if dist < min_dist:
        return None
    v = np.array([p0[0] - p1[0], p0[1] - p1[1]], dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def _draw_pose_marker(img, kpts, color=YELLOW):
    valid = _valid_kpts(kpts)
    if len(valid) == 0:
        return

    if len(valid) == 1:
        x, y = valid[0]
        cv2.circle(img, (int(x), int(y)), 3, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 8, color, 2, lineType=cv2.LINE_AA)
        return

    p0, p1 = valid[0], valid[1]
    dist = ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5

    for x, y in valid:
        cv2.circle(img, (int(x), int(y)), 4, color, -1, lineType=cv2.LINE_AA)

    if dist >= ARROW_MIN_DIST:
        cv2.arrowedLine(img, (int(p1[0]), int(p1[1])), (int(p0[0]), int(p0[1])),
                        color, 2, cv2.LINE_AA, 0, 0.3)
    else:
        cv2.circle(img, (int(p0[0]), int(p0[1])), 15, color, 2, lineType=cv2.LINE_AA)


def match_tracked_to_dets(tracked_xyxy, merged_xyxy, iou_min=0.3):
    out = []
    merged_xyxy = np.asarray(merged_xyxy, dtype=np.float32)
    for tb in tracked_xyxy:
        if len(merged_xyxy) == 0:
            out.append(-1)
            continue
        tb = np.asarray(tb, dtype=np.float32)
        bx1, by1, bx2, by2 = tb[:4]
        ious = []
        for mb in merged_xyxy:
            mx1, my1, mx2, my2 = mb[:4]
            ix1 = max(bx1, mx1)
            iy1 = max(by1, my1)
            ix2 = min(bx2, mx2)
            iy2 = min(by2, my2)
            if ix2 <= ix1 or iy2 <= iy1:
                ious.append(0.0)
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area_b = (bx2 - bx1) * (by2 - by1)
            area_m = (mx2 - mx1) * (my2 - my1)
            union = area_b + area_m - inter + 1e-6
            ious.append(inter / union)
        if not ious:
            out.append(-1)
            continue
        j = int(np.argmax(ious))
        out.append(j if ious[j] >= iou_min else -1)
    return out


def draw_compass(img, direction, angle, polarization, position='bottom-right'):
    h, w = img.shape[:2]
    compass_size = min(w, h) // 12
    margin = 20

    if position == 'bottom-right':
        cx, cy = w - margin - compass_size, h - margin - compass_size
    else:
        cx, cy = w - margin - compass_size, h - margin - compass_size

    padding = 15
    x1 = cx - compass_size - padding
    y1 = cy - compass_size - padding
    x2 = cx + compass_size + padding
    y2 = cy + compass_size + padding + 75

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)

    cv2.circle(img, (cx, cy), compass_size, (240, 240, 240), -1)
    cv2.circle(img, (cx, cy), compass_size, (180, 180, 180), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for angle_deg in range(0, 360, 45):
        rad = np.radians(angle_deg)
        r1 = compass_size - 5
        r2 = compass_size - 12 if angle_deg % 90 == 0 else compass_size - 8
        x1d = cx + r1 * np.cos(rad)
        y1d = cy + r1 * np.sin(rad)
        x2d = cx + r2 * np.cos(rad)
        y2d = cy + r2 * np.sin(rad)
        cv2.line(img, (int(x1d), int(y1d)), (int(x2d), int(y2d)), (150, 150, 150), 1)

    arrow_length = compass_size - 10
    ex_swarm = cx + arrow_length * direction[0]
    ey_swarm = cy + arrow_length * direction[1]

    cv2.arrowedLine(img, (cx, cy), (int(ex_swarm), int(ey_swarm)),
                    (255, 255, 255), 4, tipLength=0.3)
    cv2.arrowedLine(img, (cx, cy), (int(ex_swarm), int(ey_swarm)),
                    (255, 100, 0), 2, tipLength=0.3)

    if len(direction) > 2 and (direction[2] != 0 or direction[3] != 0):
        ex_look = cx + arrow_length * direction[2]
        ey_look = cy + arrow_length * direction[3]
        cv2.arrowedLine(img, (cx, cy), (int(ex_look), int(ey_look)),
                        (255, 255, 255), 4, tipLength=0.3)
        cv2.arrowedLine(img, (cx, cy), (int(ex_look), int(ey_look)),
                        (0, 0, 255), 2, tipLength=0.3)

    cv2.circle(img, (cx, cy), 3, (100, 100, 100), -1)

    info_y = cy + compass_size + 20
    cv2.putText(img, f"{angle:.0f}°", (cx - 18, info_y), font, 0.35, (50, 50, 50), 1)
    cv2.putText(img, f"P:{polarization:.2f}", (cx - 22, info_y + 15), font, 0.35, (50, 50, 50), 1)

    legend_y = info_y + 28
    cv2.arrowedLine(img, (cx - 35, legend_y), (cx - 20, legend_y), (255, 100, 0), 2, tipLength=0.3)
    cv2.putText(img, "Axis", (cx - 15, legend_y + 4), font, 0.3, (50, 50, 50), 1)
    cv2.arrowedLine(img, (cx + 15, legend_y), (cx + 30, legend_y), (0, 0, 255), 2, tipLength=0.3)
    cv2.putText(img, "Look", (cx + 35, legend_y + 4), font, 0.3, (50, 50, 50), 1)


# ==================== ОСНОВНОЙ ПАЙПЛАЙН ====================

def run_pipeline(source, model, tracker, conf=0.3, tile_size=640, overlap=0.2, iou_thres=0.6):
    source = Path(source)
    track_rows = []
    frames = []
    compass_data_list = []

    if source.is_dir():
        files = sorted([p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

        for frame_idx, path in enumerate(files):
            frame = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

            tiles = tile_image(frame, tile_size=tile_size, overlap=overlap)
            all_dets = []
            all_kpts = []

            for tile, ox, oy in tiles:
                r = model.predict(tile, conf=conf, verbose=False)[0]
                dets, kpts_list = yolo_result_to_xyxy_conf_cls_kpts(r, conf_thres=conf)
                if len(dets) == 0:
                    continue
                dets[:, [0, 2]] += ox
                dets[:, [1, 3]] += oy
                for di in range(len(dets)):
                    all_dets.append(dets[di])
                    k = kpts_list[di]
                    if k is not None:
                        k = k.copy()
                        k[:, 0] += ox
                        k[:, 1] += oy
                    all_kpts.append(k)

            if len(all_dets) == 0:
                frames.append((frame_idx, frame, np.empty((0, 6), dtype=np.float32), [], None))
                continue

            merged_dets = np.stack(all_dets, axis=0)
            keep = np.sort(nms_keep_indices(merged_dets, iou_thres=iou_thres))
            merged_dets = merged_dets[keep]
            merged_kpts = [all_kpts[k] for k in keep]

            detections = sv.Detections(
                xyxy=merged_dets[:, :4].astype(np.float32),
                confidence=merged_dets[:, 4].astype(np.float32),
                class_id=merged_dets[:, 5].astype(np.int32),
            )
            tracked = tracker.update(detections=detections)

            if tracked is not None and len(tracked) > 0 and tracked.tracker_id is not None:
                for k in range(len(tracked)):
                    track_rows.append({
                        "frame_idx": frame_idx,
                        "track_id": int(tracked.tracker_id[k]),
                        "x1": float(tracked.xyxy[k][0]),
                        "y1": float(tracked.xyxy[k][1]),
                        "x2": float(tracked.xyxy[k][2]),
                        "y2": float(tracked.xyxy[k][3]),
                        "score": float(tracked.confidence[k]) if tracked.confidence is not None else np.nan,
                    })

            frames.append((frame_idx, frame, merged_dets, merged_kpts, tracked))
            
            compass_info = extract_compass_data(frame, merged_dets, merged_kpts, tracked)
            if compass_info:
                compass_data_list.append(compass_info)

    return pd.DataFrame(track_rows), frames, compass_data_list


def extract_compass_data(frame, merged_dets, kpts_list, tracked):
    merged = np.asarray(merged_dets) if merged_dets is not None else np.empty((0, 6), dtype=np.float32)
    has_tracks = tracked is not None and len(tracked) > 0 and tracked.tracker_id is not None

    centers = []
    orientation_vectors = []

    if has_tracks:
        t_xyxy = tracked.xyxy
        match_idx = match_tracked_to_dets(t_xyxy, merged[:, :4] if len(merged) else np.empty((0, 4)))
        
        for k in range(len(t_xyxy)):
            x1, y1, x2, y2 = t_xyxy[k][:4]
            centers.append([(x1 + x2) / 2, (y1 + y2) / 2])
            di = match_idx[k]
            if di >= 0 and kpts_list is not None and di < len(kpts_list):
                vec = _kpt_orientation(kpts_list[di])
                if vec is not None:
                    orientation_vectors.append(vec)

    if len(centers) >= 3:
        centers_np = np.asarray(centers, dtype=np.float32)
        pca = PCA(n_components=2)
        pca.fit(centers_np)
        swarm_axis = pca.components_[0]

        mean_look_vector = None
        polarization = 0.0
        if len(orientation_vectors) > 0:
            mean_dir = np.mean(orientation_vectors, axis=0)
            mean_norm = np.linalg.norm(mean_dir)
            if mean_norm > 1e-6:
                mean_dir /= mean_norm
                mean_look_vector = mean_dir
                polarization = mean_norm
                if np.dot(swarm_axis, mean_dir) < 0:
                    swarm_axis = -swarm_axis

        swarm_angle = np.degrees(np.arctan2(swarm_axis[1], swarm_axis[0]))
        
        return {
            'angle': swarm_angle,
            'axis': swarm_axis.copy(),
            'look_vector': mean_look_vector.copy() if mean_look_vector is not None else None,
            'polarization': polarization
        }
    
    return None


def get_direction_and_hint(angle_degrees):
    norm = angle_degrees % 360
    
    directions_8 = [
        "СЕВЕР", "СЕВЕРО-ВОСТОК", "ВОСТОК", "ЮГО-ВОСТОК",
        "ЮГ", "ЮГО-ЗАПАД", "ЗАПАД", "СЕВЕРО-ЗАПАД"
    ]
    
    idx = int((norm + 22.5) // 45) % 8
    direction_name = directions_8[idx]
    
    if 315 <= norm or norm < 45:
        hint = "ВОСТОК"
    elif 45 <= norm < 135:
        hint = "ЮГ"
    elif 135 <= norm < 225:
        hint = "ЗАПАД"
    else:
        hint = "СЕВЕР"
    
    return direction_name, hint


def save_annotated_video(frames, out_path, fps=15, save_frames_dir=None, compass_data_list=None):
    if len(frames) == 0:
        raise ValueError("frames пустой.")

    first_frame = frames[0][1].copy()
    h, w = first_frame.shape[:2]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if save_frames_dir is not None:
        save_frames_dir = Path(save_frames_dir)
        save_frames_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(str(out_path.with_suffix('.avi')), fourcc, fps, (w, h))

    for idx, item in enumerate(frames):
        if len(item) == 5:
            frame_idx, frame, merged, kpts_list, tracked = item
        else:
            continue

        img = frame.copy()
        merged = np.asarray(merged) if merged is not None else np.empty((0, 6), dtype=np.float32)
        has_tracks = tracked is not None and len(tracked) > 0 and tracked.tracker_id is not None

        if has_tracks:
            t_xyxy = tracked.xyxy
            t_ids = tracked.tracker_id
            match_idx = match_tracked_to_dets(t_xyxy, merged[:, :4] if len(merged) else np.empty((0, 4)))
        else:
            t_xyxy, t_ids, match_idx = [], [], []

        centers = []
        orientation_vectors = []

        if has_tracks:
            for k in range(len(t_xyxy)):
                x1, y1, x2, y2 = t_xyxy[k][:4]
                centers.append([(x1 + x2) / 2, (y1 + y2) / 2])
                di = match_idx[k]
                if di >= 0 and kpts_list is not None and di < len(kpts_list):
                    vec = _kpt_orientation(kpts_list[di])
                    if vec is not None:
                        orientation_vectors.append(vec)

        if len(centers) >= 3:
            centers_np = np.asarray(centers, dtype=np.float32)
            pca = PCA(n_components=2)
            pca.fit(centers_np)
            swarm_axis = pca.components_[0]

            mean_look_vector = None
            polarization = 0.0
            if len(orientation_vectors) > 0:
                mean_dir = np.mean(orientation_vectors, axis=0)
                mean_norm = np.linalg.norm(mean_dir)
                if mean_norm > 1e-6:
                    mean_dir /= mean_norm
                    mean_look_vector = mean_dir
                    polarization = mean_norm
                    if np.dot(swarm_axis, mean_dir) < 0:
                        swarm_axis = -swarm_axis

            swarm_angle = np.degrees(np.arctan2(swarm_axis[1], swarm_axis[0]))

            if mean_look_vector is not None:
                compass_data = np.array([swarm_axis[0], swarm_axis[1],
                                         mean_look_vector[0], mean_look_vector[1]])
                draw_compass(img, compass_data, swarm_angle, polarization, 'bottom-right')
            else:
                draw_compass(img, np.array([swarm_axis[0], swarm_axis[1], 0, 0]),
                             swarm_angle, polarization, 'bottom-right')

        if has_tracks:
            for k in range(len(t_xyxy)):
                x1, y1, x2, y2 = map(int, t_xyxy[k][:4])
                tid = int(t_ids[k])
                cv2.rectangle(img, (x1, y1), (x2, y2), PURPLE, 2)
                cv2.putText(img, f"ID {tid}", (x1 + 2, y1 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, PURPLE, 1)
                di = match_idx[k]
                if di >= 0 and kpts_list is not None and di < len(kpts_list):
                    _draw_pose_marker(img, kpts_list[di])

        if save_frames_dir is not None:
            cv2.imwrite(str(save_frames_dir / f"{frame_idx:06d}.jpg"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    writer.release()
    
    try:
        h264_path = out_path.parent / f"{out_path.stem}_h264.mp4"
        result = subprocess.run([
            'ffmpeg', '-y', '-i', str(out_path),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            str(h264_path)
        ], check=False, capture_output=True)
        if result.returncode == 0 and h264_path.exists():
            return str(h264_path)
    except:
        pass
    
    return str(out_path)


# ==================== FASTAPI ПРИЛОЖЕНИЕ ====================

app = FastAPI(title="HoneyBee Tracking")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tasks = {}
model = None
MODEL_PATH = Path(r"C:\Users\Nadia\Desktop\Python_ITMO\Neiron\gpu_env\project_team_3\model\bee_pose_yolo11s_640_80ep\weights\best.pt")


def load_model():
    global model
    if model is None:
        if MODEL_PATH.exists():
            model = YOLO(str(MODEL_PATH))
        else:
            model = YOLO("yolo11n-pose.pt")
    return model


# ==================== HTML СТРАНИЦА ====================

HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐝 HoneyBee Tracker</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        
        .header {
            background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
            padding: 25px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .header h1 {
            color: white;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .header p {
            color: white;
            font-size: 1.1em;
            margin-top: 5px;
        }
        
        .container {
            max-width: 1400px;
            margin: 20px auto;
            padding: 0 20px;
            display: grid;
            grid-template-columns: 400px 1fr;
            gap: 20px;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .card h3 {
            color: #555;
            border-bottom: 2px solid #f0f2f5;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
        
        .drop-zone {
            border: 2px dashed #ddd;
            border-radius: 10px;
            padding: 40px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #fafafa;
        }
        .drop-zone:hover {
            border-color: #f6d365;
            background: #fef9e7;
        }
        .drop-zone.drag-over {
            border-color: #f6d365;
            background: #fef9e7;
        }
        .drop-zone .btn {
            display: inline-block;
            padding: 10px 25px;
            background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            margin-top: 10px;
        }
        .drop-zone .btn:hover { transform: scale(1.02); }
        
        .file-info {
            display: none;
            background: #e8f5e9;
            padding: 12px;
            border-radius: 8px;
            margin-top: 10px;
        }
        .file-info .name { font-weight: 600; }
        .file-info .size { color: #666; font-size: 13px; }
        
        .param-group {
            margin-bottom: 15px;
        }
        .param-group label {
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            color: #555;
        }
        .param-group input[type="range"] {
            width: 100%;
            margin-top: 5px;
        }
        .param-group .value {
            background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
            color: white;
            padding: 0 12px;
            border-radius: 12px;
            font-size: 12px;
        }
        
        .btn-primary {
            width: 100%;
            padding: 12px;
            background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-primary:hover { transform: scale(1.02); box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
        .btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }
        
        .video-container {
            display: none;
        }
        .video-container video {
            width: 100%;
            max-height: 400px;
            border-radius: 10px;
            background: #000;
        }
        
        .result-box {
            background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 15px;
        }
        .stat-item {
            background: #f8f9fa;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-item .number {
            font-size: 22px;
            font-weight: 700;
            color: #f6d365;
        }
        .stat-item .label {
            font-size: 12px;
            color: #888;
            margin-top: 4px;
        }
        
        .placeholder {
            text-align: center;
            padding: 60px 20px;
            color: #aaa;
        }
        .placeholder p { margin-top: 15px; }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #f6d365;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .processing-status {
            display: none;
            text-align: center;
            padding: 30px;
        }
        
        #downloadBtn {
            display: inline-block;
            margin-top: 10px;
            padding: 10px 25px;
            background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            text-decoration: none;
            font-weight: bold;
        }
        #downloadBtn:hover { transform: scale(1.02); }
        
        @keyframes fly {
            0% { transform: translate(0, 0); }
            25% { transform: translate(3px, -2px); }
            75% { transform: translate(-2px, 3px); }
            100% { transform: translate(0, 0); }
        }
        .fly-bee {
            display: inline-block;
            animation: fly 0.3s infinite;
        }
        
        @media (max-width: 768px) {
            .container { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🐝 <span class="fly-bee">🐝</span> HoneyBee Tracker <span class="fly-bee">🐝</span> 🐝</h1>
        <p>Система анализа направления движения пчёл. Команда № 3</p>
    </div>
    
    <div class="container">
        <div>
            <div class="card">
                <h3>📤 Загрузка видео</h3>
                <div class="drop-zone" id="dropZone">
                    <div>📁</div>
                    <p>Перетащите видео сюда<br>или нажмите для выбора</p>
                    <input type="file" id="fileInput" accept=".mp4,.avi,.mov,.mkv" style="display:none">
                    <button class="btn" onclick="document.getElementById('fileInput').click()">Выбрать файл</button>
                </div>
                <div class="file-info" id="fileInfo">
                    <div class="name" id="fileName">file.mp4</div>
                    <div class="size" id="fileSize">0 MB</div>
                </div>
                <button class="btn-primary" id="uploadBtn" onclick="uploadVideo()" style="margin-top:10px">
                    ⬆ Загрузить
                </button>
                <div id="uploadProgress" style="display:none;margin-top:10px">
                    <div class="spinner"></div>
                    <p style="text-align:center;color:#888">Загрузка...</p>
                </div>
            </div>
            
            <div class="card">
                <h3>⚙️ Параметры детекции</h3>
                <p style="font-size:12px;color:#888;margin-bottom:10px;">Настройка порогов для детекции пчёл</p>
                
                <div class="param-group">
                    <label>Порог уверенности <span class="value" id="confVal">0.30</span></label>
                    <input type="range" id="confThreshold" min="0.1" max="0.8" step="0.05" value="0.3">
                    <div style="font-size:11px;color:#999;margin-top:2px;">Чем выше, тем точнее</div>
                </div>
                
                <div class="param-group">
                    <label>Размер тайла <span class="value" id="tileVal">640</span></label>
                    <input type="range" id="tileSize" min="320" max="1280" step="64" value="640">
                    <div style="font-size:11px;color:#999;margin-top:2px;">Размер фрагмента для обработки</div>
                </div>
                
                <div class="param-group">
                    <label>Перекрытие тайлов <span class="value" id="overlapVal">0.20</span></label>
                    <input type="range" id="overlap" min="0" max="0.5" step="0.05" value="0.2">
                    <div style="font-size:11px;color:#999;margin-top:2px;">Нахлёст между тайлами</div>
                </div>
                
                <div class="param-group">
                    <label>Порог IoU <span class="value" id="iouVal">0.60</span></label>
                    <input type="range" id="iouThreshold" min="0.3" max="0.8" step="0.05" value="0.6">
                    <div style="font-size:11px;color:#999;margin-top:2px;">Удаление дублей</div>
                </div>
                
                <button class="btn-primary" id="processBtn" onclick="processVideo()">
                    ▶ Запустить обработку
                </button>
            </div>
        </div>
        
        <div>
            <div class="card">
                <h3>📺 Результат</h3>
                <div id="placeholder" class="placeholder">
                    <p>🎬 Загрузите видео для начала работы</p>
                </div>
                <div id="processingStatus" class="processing-status">
                    <div class="spinner"></div>
                    <p>Идет обработка видео...</p>
                    <p id="statusMessage" style="color:#888;font-size:13px"></p>
                </div>
                <div id="videoContainer" class="video-container">
                    <video id="resultVideo" controls></video>
                    <a id="downloadBtn">⬇ Скачать видео</a>
                </div>
                <div id="resultBox" style="display:none" class="result-box">
                    🎯 Преобладающее направление: <span id="directionText">-</span>
                </div>
                <div id="statsContainer" style="display:none">
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="number" id="statFrames">0</div>
                            <div class="label">Кадры</div>
                        </div>
                        <div class="stat-item">
                            <div class="number" id="statDetections">0</div>
                            <div class="label">Детекции</div>
                        </div>
                        <div class="stat-item">
                            <div class="number" id="statTracks">0</div>
                            <div class="label">Треки</div>
                        </div>
                        <div class="stat-item">
                            <div class="number" id="statConfidence">0</div>
                            <div class="label">Сред. уверенность</div>
                        </div>
                    </div>
                    <div id="statsText" style="margin-top:15px;font-size:13px;color:#666;white-space:pre-wrap;background:#f8f9fa;padding:10px;border-radius:8px;"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let taskId = null;
        let uploadedFile = null;
        
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        
        fileInput.addEventListener('click', function(e) {
            e.stopPropagation();
        });
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
        
        dropZone.addEventListener('click', function(e) {
            if (e.target.tagName === 'BUTTON') return;
            if (e.target.closest('button')) return;
            fileInput.click();
        });
        
        fileInput.addEventListener('change', function(e) {
            if (this.files && this.files.length > 0) {
                handleFile(this.files[0]);
            }
            this.value = '';
        });
        
        function handleFile(file) {
            uploadedFile = file;
            document.getElementById('fileName').textContent = file.name;
            document.getElementById('fileSize').textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
            document.getElementById('fileInfo').style.display = 'block';
            document.getElementById('uploadBtn').style.display = 'block';
        }
        
        document.getElementById('confThreshold').addEventListener('input', function() {
            document.getElementById('confVal').textContent = parseFloat(this.value).toFixed(2);
        });
        document.getElementById('overlap').addEventListener('input', function() {
            document.getElementById('overlapVal').textContent = parseFloat(this.value).toFixed(2);
        });
        document.getElementById('iouThreshold').addEventListener('input', function() {
            document.getElementById('iouVal').textContent = parseFloat(this.value).toFixed(2);
        });
        document.getElementById('tileSize').addEventListener('input', function() {
            document.getElementById('tileVal').textContent = this.value;
        });
        
        async function uploadVideo() {
            if (!uploadedFile) { alert('Выберите файл'); return; }
            
            const formData = new FormData();
            formData.append('file', uploadedFile);
            
            document.getElementById('uploadBtn').disabled = true;
            document.getElementById('uploadBtn').textContent = '⏳ Загрузка...';
            document.getElementById('uploadProgress').style.display = 'block';
            
            try {
                const resp = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await resp.json();
                if (resp.ok) {
                    taskId = data.task_id;
                    document.getElementById('uploadBtn').textContent = '✅ Загружено';
                    document.getElementById('uploadBtn').style.background = '#4CAF50';
                } else {
                    throw new Error(data.detail || 'Ошибка');
                }
            } catch (e) {
                alert('Ошибка загрузки: ' + e.message);
                document.getElementById('uploadBtn').textContent = '⬆ Загрузить';
                document.getElementById('uploadBtn').disabled = false;
            }
            document.getElementById('uploadProgress').style.display = 'none';
        }
        
        async function processVideo() {
            if (!taskId) { alert('Сначала загрузите видео'); return; }
            
            const params = {
                conf_threshold: parseFloat(document.getElementById('confThreshold').value),
                tile_size: parseInt(document.getElementById('tileSize').value),
                overlap: parseFloat(document.getElementById('overlap').value),
                iou_threshold: parseFloat(document.getElementById('iouThreshold').value),
                lost_track_buffer: 3,
                min_consecutive_frames: 1
            };
            
            document.getElementById('placeholder').style.display = 'none';
            document.getElementById('videoContainer').style.display = 'none';
            document.getElementById('processingStatus').style.display = 'block';
            document.getElementById('resultBox').style.display = 'none';
            document.getElementById('statsContainer').style.display = 'none';
            document.getElementById('processBtn').disabled = true;
            document.getElementById('processBtn').textContent = '⏳ Обработка...';
            
            try {
                const resp = await fetch(`/api/process/${taskId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params)
                });
                const data = await resp.json();
                if (resp.ok) {
                    showResult(data);
                } else {
                    throw new Error(data.detail || 'Ошибка');
                }
            } catch (e) {
                alert('Ошибка обработки: ' + e.message);
                document.getElementById('processingStatus').style.display = 'none';
                document.getElementById('processBtn').disabled = false;
                document.getElementById('processBtn').textContent = '▶ Запустить обработку';
            }
        }
        
        function showResult(data) {
            document.getElementById('processingStatus').style.display = 'none';
            document.getElementById('videoContainer').style.display = 'block';
            document.getElementById('resultBox').style.display = 'block';
            document.getElementById('statsContainer').style.display = 'block';
            document.getElementById('processBtn').disabled = false;
            document.getElementById('processBtn').textContent = '▶ Запустить обработку';
            
            const video = document.getElementById('resultVideo');
            const url = `/api/download/${taskId}?t=${Date.now()}`;
            video.src = url;
            video.load();
            
            document.getElementById('downloadBtn').href = url;
            document.getElementById('downloadBtn').download = `processed_${taskId}.mp4`;
            
            if (data.direction) {
                document.getElementById('directionText').textContent = data.direction;
            }
            
            if (data.stats) {
                document.getElementById('statFrames').textContent = data.stats.total_frames || 0;
                document.getElementById('statDetections').textContent = data.stats.total_detections || 0;
                document.getElementById('statTracks').textContent = data.stats.unique_tracks || 0;
                document.getElementById('statConfidence').textContent = (data.stats.avg_confidence || 0).toFixed(3);
                
                if (data.stats.details) {
                    document.getElementById('statsText').textContent = data.stats.details;
                }
            }
        }
    </script>
</body>
</html>
"""


# ==================== API ЭНДПОИНТЫ ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(HTML_PAGE)


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(400, "Неподдерживаемый формат файла")
        
        task_id = str(uuid.uuid4())
        task_dir = Path(tempfile.gettempdir()) / "honeybee" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = task_dir / file.filename
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        tasks[task_id] = {
            "id": task_id,
            "status": "uploaded",
            "video_path": str(video_path),
            "filename": file.filename
        }
        
        return {"task_id": task_id, "status": "uploaded"}
    
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/process/{task_id}")
async def process_video(task_id: str, params: dict):
    start_time = time.time()
    
    if task_id not in tasks:
        raise HTTPException(404, "Задача не найдена")
    
    task = tasks[task_id]
    video_path = Path(task["video_path"])
    
    if not video_path.exists():
        raise HTTPException(404, "Видео не найдено")
    
    try:
        tasks[task_id]["status"] = "processing"
        
        frames_dir = video_path.parent / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 15
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(frames_dir / f"{frame_idx:06d}.jpg"), frame)
            frame_idx += 1
        cap.release()
        
        if frame_idx == 0:
            raise Exception("Видео не содержит кадров")
        
        tasks[task_id]["total_frames"] = frame_idx
        tasks[task_id]["progress"] = 30
        
        load_model()
        global model
        
        tracker = OCSORTTracker(
            lost_track_buffer=params.get('lost_track_buffer', 3),
            frame_rate=fps,
            minimum_consecutive_frames=params.get('min_consecutive_frames', 1),
            minimum_iou_threshold=0.2,
            high_conf_det_threshold=0.1,
            delta_t=3,
        )
        
        tasks[task_id]["progress"] = 50
        
        track_df, frames, compass_data_list = run_pipeline(
            source=frames_dir,
            model=model,
            tracker=tracker,
            conf=params.get('conf_threshold', 0.3),
            tile_size=params.get('tile_size', 640),
            overlap=params.get('overlap', 0.2),
            iou_thres=params.get('iou_threshold', 0.6),
        )
        
        tasks[task_id]["progress"] = 70
        
        output_dir = video_path.parent / "output"
        output_dir.mkdir(exist_ok=True)
        video_output = output_dir / "output.mp4"
        
        save_annotated_video(
            frames=frames,
            out_path=video_output,
            fps=fps,
            save_frames_dir=output_dir / "frames",
            compass_data_list=compass_data_list
        )
        
        tasks[task_id]["progress"] = 90
        
        direction_angles = []
        for data in compass_data_list:
            if data and 'angle' in data:
                direction_angles.append(data['angle'])
        
        if direction_angles:
            mean_angle = np.mean(direction_angles)
            direction_name, _ = get_direction_and_hint(mean_angle)
            direction_text = f"{direction_name} ({mean_angle:.1f}°)"
        else:
            direction_text = "Не определено"
            direction_name = "Не определено"
        
        processing_time = time.time() - start_time
        minutes = int(processing_time // 60)
        seconds = int(processing_time % 60)
        time_str = f"{minutes} мин {seconds} сек" if minutes > 0 else f"{seconds} сек"
        fps_processing = len(frames) / processing_time if processing_time > 0 else 0
        
        total_detections = len(track_df) if track_df is not None else 0
        unique_tracks = track_df['track_id'].nunique() if track_df is not None and len(track_df) > 0 else 0
        avg_confidence = float(track_df['score'].mean()) if track_df is not None and len(track_df) > 0 else 0.0
        
        if direction_angles:
            dir_counts = {}
            for a in direction_angles:
                name, _ = get_direction_and_hint(a)
                dir_counts[name] = dir_counts.get(name, 0) + 1
            
            sorted_dirs = sorted(dir_counts.items(), key=lambda x: x[1], reverse=True)
            
            details = f"""
ВРЕМЯ ОБРАБОТКИ
{'=' * 40}
   ⏱️  {time_str}
   📹  {fps_processing:.1f} кадров/сек

АНАЛИЗ НАПРАВЛЕНИЙ ДВИЖЕНИЯ
{'=' * 40}

🎯 ОСНОВНОЕ НАПРАВЛЕНИЕ: {direction_name}
   Угол: {mean_angle:.1f}°
   Разброс: {np.std(direction_angles):.1f}°

РАСПРЕДЕЛЕНИЕ ПО НАПРАВЛЕНИЯМ:
"""
            for name, count in sorted_dirs[:5]:
                pct = count / len(direction_angles) * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                details += f"   {name:12} {bar} {pct:5.1f}%\n"
            
            if np.std(direction_angles) < 10:
                details += "\n✅ Движение стабильное (пчёлы летят вместе)"
            elif np.std(direction_angles) < 30:
                details += "\n🔄 Движение умеренное (небольшой разброс)"
            else:
                details += "\n🌪️ Движение хаотичное (пчёлы разлетаются)"
            
        else:
            details = f"""
ВРЕМЯ ОБРАБОТКИ
{'=' * 40}
   ⏱️  {time_str}
   📹  {fps_processing:.1f} кадров/сек

⚠️ Недостаточно данных для анализа направлений
"""
        
        stats = {
            "total_frames": len(frames),
            "total_detections": total_detections,
            "unique_tracks": unique_tracks,
            "avg_confidence": avg_confidence,
            "frames_with_compass": len(compass_data_list),
            "details": details
        }
        
        tasks[task_id].update({
            "status": "completed",
            "output_path": str(video_output),
            "stats": stats,
            "direction": direction_text,
            "progress": 100
        })
        
        return {
            "task_id": task_id,
            "status": "completed",
            "direction": direction_text,
            "stats": stats
        }
    
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        raise HTTPException(500, str(e))


@app.get("/api/download/{task_id}")
async def download_video(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Задача не найдена")
    
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(400, "Видео еще не обработано")
    
    video_path = Path(task["output_path"])
    if not video_path.exists():
        raise HTTPException(404, "Файл не найден")
    
    return FileResponse(
        path=str(video_path),
        filename=f"processed_{task_id}.mp4",
        media_type="video/mp4"
    )


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Задача не найдена")
    return tasks[task_id]


# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    print("\n" + "🐝" * 50)
    print("HONEYBEE TRACKER")
    print("🐝" * 50 + "\n")
    print("🌐 Откройте: http://127.0.0.1:8000")
    print("🐝" * 50 + "\n")
    
    uvicorn.run(
        "app_interface_clean:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )