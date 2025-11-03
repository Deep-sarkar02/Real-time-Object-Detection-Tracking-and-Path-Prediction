# Real-time Object Detection, Tracking, and Path Prediction (Web Frontend)

A professional, real-time system that detects objects, tracks their movement, and predicts short-term future paths. It provides a modern web frontend to control the source (webcam or external camera) and visualize live annotated video.

## Output Example


<video width="640" height="360" controls>
  <source src="ObjectTrackingOutput2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


## Features

- High-FPS detection using YOLO (Ultralytics) for real-time performance.
- Lightweight multi-object tracker with stable IDs and trails.
- Short-horizon path prediction using constant-velocity extrapolation informed by Kalman updates.
- Professional web UI: start/stop, select source (webcam/URL), choose model, tune thresholds, and view live stream.
- MJPEG streaming for broad browser compatibility.

## Project Structure

- app.py — FastAPI backend serving the UI and streaming annotated frames.
- web/
  - index.html — Frontend UI.
  - styles.css — Professional styling.
  - app.js — Frontend logic for starting/stopping streams.
- fasterRCNN_KLT.py — The updated pipeline with YOLO detector, tracker, and prediction (reused by app.py).
- kalmanfilter.py — Kalman filter used to stabilize and inform predictions.

## Requirements

- Python 3.9+
- OS: Windows/macOS/Linux
- Recommended: NVIDIA GPU with CUDA for real-time speeds.

Install dependencies:

```
pip install ultralytics fastapi uvicorn opencv-python torch torchvision jinja2
```

Note: For CUDA acceleration, install the appropriate PyTorch build from https://pytorch.org/get-started/locally/

## How It Works

1. Frontend requests /start with configuration (source, model, thresholds, etc.).
2. Backend loads the YOLO model and opens the video source (webcam or URL).
3. A background thread captures frames, runs detection, updates the tracker, and draws:
   - Green bounding boxes with red ID labels.
   - Blue trails of past movement.
   - Yellow dots predicting the short-term future path.
4. Encoded JPEG frames are exposed under /stream as an MJPEG stream.
5. Frontend displays the MJPEG stream live.
6. Frontend can send /stop to stop capture and free resources.

### Detection

- Ultralytics YOLO (e.g., yolov8n.pt) runs per frame with selected image size and confidence threshold.
- Model auto-selects GPU (CUDA) if available, otherwise CPU.

### Tracking

- A simple IoU-based data association maintains object IDs and trails.
- Each track uses a Kalman filter update on observed centers; future trajectory is extrapolated from recent motion.
- For more robustness (occlusions/re-ID), the tracker can be upgraded to ByteTrack/OC-SORT later.

### Prediction

- For each track, the recent centers define velocity (dx, dy). A short horizon of future points is produced.
- Steps = horizon_seconds * FPS (estimated), clamped to at least 1 step.

## Usage

Start the backend server:

```
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open the UI in your browser:

```
http://localhost:8000/
```

Select a source:
- Webcam (index 0 by default), or
- External camera URL (RTSP/HTTP path or device path),
then click Start. The live stream will appear with detections and predictions. Click Stop to end.

## Configuration Options

- Source: webcam or URL
- Webcam Index: choose which local camera to use (0, 1, ...)
- Camera URL/Path: RTSP or HTTP URL, or device path string
- Model: yolov8n.pt (fast), yolov8s.pt (balanced)
- Image Size: detector input size (e.g., 640). Lower for more FPS.
- Confidence Threshold: filter detections (e.g., 0.3)
- IoU Match Threshold: tracker association threshold (e.g., 0.3)
- Prediction Horizon (sec): future path duration (e.g., 0.7)

## Notes on Performance

- Use a CUDA-enabled GPU for best results. CPU-only may be limited in FPS.
- Lower image size and slightly higher confidence threshold can improve FPS.
- The first run will download the model weights automatically.

## Extending the System

- Replace SimpleTracker with ByteTrack/OC-SORT for stronger tracking under occlusions.
- Enhance prediction by exposing explicit Kalman predict-steps with dt=1/fps.
- Add class filtering or scene-specific models to reduce clutter and increase accuracy.
- Serve HLS/WebRTC for lower latency; MJPEG is simple and widely compatible but not the lowest-latency option.

## Troubleshooting

- "Failed to load detector": Ensure `pip install ultralytics` and internet access for first-time model download.
- "Cannot open video source": Verify webcam index or RTSP/HTTP URL. Some webcams need `cv.CAP_DSHOW` (Windows).
- Low FPS: Confirm GPU usage, reduce `img_size`, or choose `yolov8n.pt`.

## License

This project is provided as-is for educational and prototyping purposes.
