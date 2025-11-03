# Fast real-time object detection, tracking and path prediction
# Replaces Faster R-CNN with YOLO for speed, adds simple multi-object tracking and Kalman-based prediction.

import os
import time
import math
import cv2 as cv
import torch
from kalmanfilter import KalmanFilter

# Optional: use Ultralytics YOLO if available for high FPS detection
# pip install ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


def iou_xyxy(a, b):
    # a, b: [x1,y1,x2,y2]
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, tid, box, score, cls_id, fps):
        self.id = tid
        self.box = box  # xyxy
        self.score = score
        self.cls_id = cls_id
        self.kf = KalmanFilter()
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        # initialize with first observation
        self.kf.predict(cx, cy)
        self.last_update = time.time()
        self.fps = max(1.0, fps if fps and fps == fps else 30.0)
        self.trail = []  # recent centers for rendering

    def update(self, box):
        self.box = box
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        self.kf.predict(cx, cy)
        self.trail.append((cx, cy))
        if len(self.trail) > 20:
            self.trail.pop(0)
        self.last_update = time.time()

    def predict_future(self, steps=15):
        # Use constant-velocity Kalman state to project future center positions
        # Here we call the same predict method without measurement; assuming KalmanFilter supports prediction with last state
        # If KalmanFilter requires a measurement, we extrapolate linearly from last two points.
        if len(self.trail) >= 2:
            (x2, y2), (x1, y1) = self.trail[-1], self.trail[-2]
            vx, vy = x2 - x1, y2 - y1
        else:
            (x2, y2) = self.trail[-1] if self.trail else (int((self.box[0]+self.box[2])/2), int((self.box[1]+self.box[3])/2))
            vx, vy = 0, 0
        pts = []
        cx, cy = x2, y2
        for _ in range(steps):
            cx += vx
            cy += vy
            pts.append((int(cx), int(cy)))
        return pts


class SimpleTracker:
    def __init__(self, fps, iou_thresh=0.3, det_conf_thresh=0.25, max_age=30):
        self.fps = fps
        self.iou_thresh = iou_thresh
        self.det_conf_thresh = det_conf_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracks = []
        self.last_seen = {}  # id -> frame_idx

    def step(self, frame_idx, detections):
        # detections: list of (box_xyxy, score, cls_id)
        detections = [d for d in detections if d[1] >= self.det_conf_thresh]
        assigned = set()
        # Greedy IoU matching
        for tr in self.tracks:
            best_iou, best_j = 0.0, -1
            for j, (box, score, cls_id) in enumerate(detections):
                if j in assigned:
                    continue
                if tr.cls_id != cls_id:
                    continue
                iou = iou_xyxy(tr.box, box)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= self.iou_thresh:
                box, score, cls_id = detections[best_j]
                tr.update(box)
                self.last_seen[tr.id] = frame_idx
                assigned.add(best_j)
            else:
                # keep track alive but no update
                pass
        # Create new tracks for unassigned detections
        for j, (box, score, cls_id) in enumerate(detections):
            if j in assigned:
                continue
            tr = Track(self.next_id, box, score, cls_id, self.fps)
            self.tracks.append(tr)
            self.last_seen[tr.id] = frame_idx
            self.next_id += 1
        # Remove stale tracks
        self.tracks = [t for t in self.tracks if (frame_idx - self.last_seen.get(t.id, frame_idx)) <= self.max_age]
        return self.tracks


def load_detector(model_name='yolov8n.pt', device=None, img_size=640):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if not YOLO_AVAILABLE:
        raise RuntimeError('Ultralytics YOLO not installed. Please install with: pip install ultralytics')
    model = YOLO(model_name)
    model.fuse()
    model.to(device)
    return model, device, img_size


def run(input_path='sampleInput2.mp4', output_path='ObjectTrackingOutput2.mp4',
        model_name='yolov8n.pt', img_size=640, conf_thres=0.25, iou_match=0.3,
        predict_horizon_sec=0.5, draw=True):
    model, device, img_size = load_detector(model_name, img_size=img_size)
    vid = cv.VideoCapture(input_path)
    fps = vid.get(cv.CAP_PROP_FPS) or 30.0
    width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    print('video properties:', fps, width, height)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    output_video = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = SimpleTracker(fps=fps, iou_thresh=iou_match, det_conf_thresh=conf_thres, max_age=int(fps))

    frame_idx = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame_idx += 1
        # Inference
        t0 = time.time()
        results = model.predict(source=frame, imgsz=img_size, device=device, conf=conf_thres, verbose=False)
        dets = []
        if len(results):
            r = results[0]
            if hasattr(r, 'boxes') and r.boxes is not None:
                for b in r.boxes:
                    xyxy = b.xyxy[0].detach().cpu().numpy().astype(int).tolist()
                    score = float(b.conf[0].detach().cpu().numpy()) if hasattr(b, 'conf') else 1.0
                    cls_id = int(b.cls[0].detach().cpu().numpy()) if hasattr(b, 'cls') else 0
                    dets.append((xyxy, score, cls_id))
        # Tracking
        tracks = tracker.step(frame_idx, dets)

        # Rendering
        if draw:
            for tr in tracks:
                x1, y1, x2, y2 = [int(v) for v in tr.box]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, f'ID {tr.id}', (x1, max(0, y1-10)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                # draw trail
                for k in range(1, len(tr.trail)):
                    cv.line(frame, tr.trail[k-1], tr.trail[k], (255, 0, 0), 2)
                # future prediction
                steps = max(1, int(predict_horizon_sec * fps))
                future_pts = tr.predict_future(steps=steps)
                for p in future_pts:
                    cv.circle(frame, p, 2, (0, 255, 255), -1)
        output_video.write(frame)

    vid.release()
    output_video.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # Tunables
    run(input_path='sampleInput2.mp4',
        output_path='ObjectTrackingOutput2.mp4',
        model_name='yolov8n.pt',
        img_size=640,
        conf_thres=0.3,
        iou_match=0.3,
        predict_horizon_sec=0.7,
        draw=True)
