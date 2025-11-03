# app.py
# This file defines a FastAPI application that serves a professional web frontend and
# provides real-time object detection, multi-object tracking, and short-term path prediction
# from a webcam or external camera URL. The video pipeline runs in a background thread and
# streams annotated frames as an MJPEG stream to the browser. Every line is commented to help novices.

# Import standard library modules for threading and timing
import threading  # To run the video processing loop in a separate thread
import time       # To measure frame intervals and sleep when needed
import io         # To handle in-memory byte buffers for JPEG encoding
import os         # To work with environment variables and file paths
from typing import Optional  # For optional type hints

# Import third-party libraries for the web server and machine learning
from fastapi import FastAPI, Request, Response, status  # FastAPI for HTTP endpoints
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse  # Different response types
from fastapi.staticfiles import StaticFiles  # To serve static frontend files (CSS/JS)
from pydantic import BaseModel  # For request body validation and parsing
import cv2 as cv  # OpenCV for video capture and drawing
import numpy as np  # Numpy for array operations
import torch  # PyTorch for model device checks

# Import our existing pipeline's components for detection and tracking
# We reuse the updated fasterRCNN_KLT run-time utilities and classes.
# If you move/rename the file, update the import accordingly.
from fasterRCNN_KLT import load_detector, SimpleTracker

# Define a data model for the request payload to start the stream
class StartConfig(BaseModel):
    # The source of the video: 'webcam' to use local camera or a string URL/ID for external camera
    source: str = 'webcam'
    # The index of the webcam when source is 'webcam': usually 0 is the default camera
    webcam_index: int = 0
    # The path or URL to a camera stream (e.g., RTSP, HTTP) when using external camera
    camera_url: Optional[str] = None
    # YOLO model name to load (e.g., 'yolov8n.pt', 'yolov8s.pt')
    model_name: str = 'yolov8n.pt'
    # Input image size for the detector (higher may improve accuracy but reduce FPS)
    img_size: int = 640
    # Confidence threshold for filtering detections
    conf_thres: float = 0.3
    # IoU threshold for tracker association
    iou_match: float = 0.3
    # How far into the future to draw predicted path (in seconds)
    predict_horizon_sec: float = 0.7
    # Whether to draw annotations on frames
    draw: bool = True


# Create the FastAPI application instance
app = FastAPI(title="Real-time Object Detection, Tracking, and Path Prediction",
              description="Professional frontend and backend streaming annotated frames in real time",
              version="1.0.0")

# Mount the static directory to serve frontend files (HTML, CSS, JS)
# We will place files in a 'web' folder next to this file
static_dir = os.path.join(os.path.dirname(__file__), 'web')  # Resolve the path to 'web' directory
app.mount("/static", StaticFiles(directory=static_dir), name="static")  # Serve files under /static

# Global state for the streaming system
stream_lock = threading.Lock()  # A lock to protect shared state across threads
stream_running = False          # A flag indicating whether the processing loop is active
stream_thread = None            # The background thread object running the loop
latest_jpeg = None              # The most recent encoded JPEG frame as bytes
cap = None                      # The OpenCV VideoCapture object
tracker = None                  # The tracker instance
model = None                    # The YOLO model
device = None                   # The device string ('cuda' or 'cpu')
img_size = 640                  # Current detector input size
conf_thres = 0.3                # Current detection confidence threshold
iou_match = 0.3                 # Current IoU threshold for tracking
predict_horizon_sec = 0.7       # Current future path horizon in seconds
fps_est = 30.0                  # Estimated frames per second of the capture


def open_source(source: str, webcam_index: int, camera_url: Optional[str]) -> cv.VideoCapture:
    """Open a video source based on parameters and return a VideoCapture.
    - source='webcam' uses the local webcam by index.
    - otherwise it uses camera_url or the provided source string.
    """
    # If the user selected "webcam", use the given webcam index
    if source == 'webcam':
        return cv.VideoCapture(webcam_index)
    # If the user provided a camera URL, open it directly
    if camera_url:
        return cv.VideoCapture(camera_url)
    # Otherwise, try to open whatever was provided as source (file path or device string)
    return cv.VideoCapture(source)


def annotate_and_encode(frame: np.ndarray) -> bytes:
    """Encode the given BGR frame to JPEG and return bytes."""
    # Use OpenCV to encode the frame as JPEG (quality default)
    ok, buf = cv.imencode('.jpg', frame)
    # If encoding failed, return empty bytes to avoid crashing
    if not ok:
        return b''
    # Convert the encoded image (NumPy array) to raw bytes
    return buf.tobytes()


def processing_loop():
    """Background thread function that captures frames, runs detection+tracking,
    draws annotations, and stores the latest JPEG for streaming."""
    # Declare that we will use and modify the global variables defined above
    global latest_jpeg, cap, tracker, model, device, img_size, conf_thres, iou_match, predict_horizon_sec, fps_est

    # Initialize frame counters for FPS estimation
    last_time = time.time()
    frames = 0

    # Keep processing while the global flag says the stream is running
    while True:
        # Acquire the lock to read the current state safely
        with stream_lock:
            running = stream_running
            local_cap = cap
            local_tracker = tracker
            local_model = model
            local_device = device
            local_img_size = img_size
            local_conf = conf_thres
            local_iou = iou_match
            local_pred_hz = predict_horizon_sec
        # If not running or no capture/model, break out to end the thread
        if not running or local_cap is None or local_model is None or local_tracker is None:
            break
        # Read a frame from the capture
        ret, frame = local_cap.read()
        # If read failed, sleep briefly and continue to retry
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        # Run YOLO inference on the frame to get detections
        results = local_model.predict(source=frame, imgsz=local_img_size, device=local_device,
                                      conf=local_conf, verbose=False)
        # Prepare a list of detections as (xyxy, score, cls_id)
        dets = []
        if len(results):
            # Get the first result object
            r = results[0]
            # Ensure boxes exist
            if hasattr(r, 'boxes') and r.boxes is not None:
                # Loop through each detected box
                for b in r.boxes:
                    # Extract the bounding box coordinates as integers
                    xyxy = b.xyxy[0].detach().cpu().numpy().astype(int).tolist()
                    # Extract the confidence score
                    score = float(b.conf[0].detach().cpu().numpy()) if hasattr(b, 'conf') else 1.0
                    # Extract the class id
                    cls_id = int(b.cls[0].detach().cpu().numpy()) if hasattr(b, 'cls') else 0
                    # Append the detection tuple
                    dets.append((xyxy, score, cls_id))
        # Increase the frame counter for FPS estimation
        frames += 1
        # Calculate FPS every 30 frames to update the estimate
        if frames % 30 == 0:
            now = time.time()
            dt = max(1e-6, now - last_time)
            fps_est = 30.0 / dt
            last_time = now
        # Update the tracker with the current detections
        tracks = local_tracker.step(frames, dets)
        # If drawing is enabled, render annotations on the frame
        draw = True
        if draw:
            # Loop through each active track
            for tr in tracks:
                # Unpack the bounding box
                x1, y1, x2, y2 = [int(v) for v in tr.box]
                # Compute the center of the bounding box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                # Draw the bounding box rectangle
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put the track ID label above the box
                cv.putText(frame, f'ID {tr.id}', (x1, max(0, y1-10)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # Draw a small red dot at the center
                cv.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                # Draw the historical trail of the track in blue
                for k in range(1, len(tr.trail)):
                    cv.line(frame, tr.trail[k-1], tr.trail[k], (255, 0, 0), 2)
                # Compute the number of future steps based on the prediction horizon and FPS
                steps = max(1, int(local_pred_hz * (fps_est if fps_est > 0 else 30.0)))
                # Get future predicted points from the track
                future_pts = tr.predict_future(steps=steps)
                # Draw each predicted future point as a small yellow dot
                for p in future_pts:
                    cv.circle(frame, p, 2, (0, 255, 255), -1)
            # Draw a small FPS overlay to show performance
            cv.putText(frame, f'FPS: {fps_est:.1f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        # Encode the annotated frame to JPEG bytes
        jpg_bytes = annotate_and_encode(frame)
        # Acquire the lock to update the global latest frame safely
        with stream_lock:
            latest_jpeg = jpg_bytes
        # Sleep briefly to yield CPU (helps when running without GPU)
        time.sleep(0.001)

    # Cleanup when the loop ends: release the capture safely
    with stream_lock:
        if cap is not None:
            cap.release()
        # Set the capture to None to indicate it's closed
        # Do not reset other global state here; stop endpoint will handle it


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the main HTML page for the frontend UI."""
    # Return the index.html from the static directory as an HTML response
    return FileResponse(os.path.join(static_dir, 'index.html'))


@app.post("/start")
async def start_stream(cfg: StartConfig):
    """Start the background streaming process with the given configuration."""
    # Use global variables to control state
    global stream_running, stream_thread, cap, tracker, model, device, img_size, conf_thres, iou_match, predict_horizon_sec

    # Acquire the lock while we modify shared state
    with stream_lock:
        # If a stream is already running, return a 409 Conflict status
        if stream_running:
            return Response(content="Stream already running", status_code=status.HTTP_409_CONFLICT)
        # Try to load the detector with the desired settings
        try:
            model, device, img_size = load_detector(cfg.model_name, img_size=cfg.img_size)
        except Exception as e:
            # If YOLO is not installed or model can't be loaded, return error
            return Response(content=f"Failed to load detector: {e}", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
        # Open the video source based on the user's configuration
        cap_local = open_source(cfg.source, cfg.webcam_index, cfg.camera_url)
        # If opening failed, return a 400 Bad Request
        if not cap_local or not cap_local.isOpened():
            return Response(content="Cannot open video source", status_code=status.HTTP_400_BAD_REQUEST)
        # Update global capture object
        cap = cap_local
        # Get an initial FPS estimate from the capture (may be 0 for webcams)
        fps = cap.get(cv.CAP_PROP_FPS)
        # Fallback to 30 FPS if unknown
        fps_value = fps if (fps and fps == fps and fps > 0) else 30.0
        # Create the tracker with the provided thresholds and FPS
        tracker = SimpleTracker(fps=fps_value, iou_thresh=cfg.iou_match, det_conf_thresh=cfg.conf_thres, max_age=int(fps_value))
        # Save global thresholds and prediction horizon
        conf_thres = cfg.conf_thres
        iou_match = cfg.iou_match
        predict_horizon_sec = cfg.predict_horizon_sec
        # Mark the stream as running
        stream_running = True
        # Create and start the background processing thread
        stream_thread = threading.Thread(target=processing_loop, daemon=True)
        stream_thread.start()
    # Return a success response
    return {"status": "started"}


@app.post("/stop")
async def stop_stream():
    """Stop the background streaming process and clean up resources."""
    # Use global variables to control state
    global stream_running, stream_thread, cap, tracker, model
    # Acquire lock while modifying state
    with stream_lock:
        # If not running, return 409 Conflict
        if not stream_running:
            return Response(content="Stream not running", status_code=status.HTTP_409_CONFLICT)
        # Set the flag to false so the thread exits gracefully
        stream_running = False
    # If a thread exists, wait a short time for it to exit
    if stream_thread is not None:
        stream_thread.join(timeout=2.0)
    # Cleanup: release capture and reset state
    with stream_lock:
        if cap is not None:
            cap.release()
        cap = None
        tracker = None
        model = None
        stream_thread = None
    # Return a success response
    return {"status": "stopped"}


@app.get("/stream")
async def stream() -> StreamingResponse:
    """Stream the latest annotated frames as an MJPEG stream to the browser."""
    # Define a generator function that yields multipart JPEG frames continuously
    def frame_generator():
        # Use a try/finally to ensure cleanup if needed
        try:
            while True:
                # Read the latest JPEG bytes safely under the lock
                with stream_lock:
                    frame = latest_jpeg
                    running = stream_running
                # If not running, end the stream by breaking the loop
                if not running:
                    break
                # If no frame is available yet, wait briefly and continue
                if frame is None:
                    time.sleep(0.01)
                    continue
                # Yield a properly formatted MJPEG multipart frame
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                # Sleep a tiny amount to avoid busy-waiting too hard
                time.sleep(0.001)
        finally:
            # No special cleanup needed here; the main loop handles resources
            pass
    # Return a StreamingResponse with the correct MIME type for MJPEG
    return StreamingResponse(frame_generator(), media_type='multipart/x-mixed-replace; boundary=frame')


# Health check endpoint for simple monitoring
@app.get("/health")
async def health():
    """Simple health check returning service status."""
    # Report whether the stream is currently running
    return {"status": "ok", "stream_running": stream_running}
