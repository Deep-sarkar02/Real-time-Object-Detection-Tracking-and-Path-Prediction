// web/app.js
// Frontend JavaScript to control the backend, start/stop the stream, and display the MJPEG feed.
// Every line is commented to help novices understand the code.

// Get references to DOM elements by their IDs to interact with the page
const sourceSelect = document.getElementById('source'); // The select element to choose input source
const webcamGroup = document.getElementById('webcam-group'); // The div containing webcam index input
const urlGroup = document.getElementById('url-group'); // The div containing camera URL input
const webcamIndexInput = document.getElementById('webcamIndex'); // The input for webcam index
const cameraUrlInput = document.getElementById('cameraUrl'); // The input for external camera URL
const modelNameSelect = document.getElementById('modelName'); // The select to choose YOLO model
const imgSizeInput = document.getElementById('imgSize'); // The input for image size
const confThresInput = document.getElementById('confThres'); // The input for confidence threshold
const iouMatchInput = document.getElementById('iouMatch'); // The input for IoU threshold
const predHorizonInput = document.getElementById('predHorizon'); // The input for prediction horizon
const startBtn = document.getElementById('startBtn'); // The Start button element
const stopBtn = document.getElementById('stopBtn'); // The Stop button element
const streamImg = document.getElementById('stream'); // The IMG element to display the MJPEG stream
const statusText = document.getElementById('statusText'); // The span showing current status

// Function to update visibility of webcam vs URL input groups based on the selected source
function updateSourceUI() {
  // If 'webcam' is selected, show webcam index input and hide URL input
  if (sourceSelect.value === 'webcam') {
    webcamGroup.classList.remove('hidden'); // Show webcam group
    urlGroup.classList.add('hidden'); // Hide URL group
  } else {
    // Otherwise, show the URL input and hide webcam index
    webcamGroup.classList.add('hidden'); // Hide webcam group
    urlGroup.classList.remove('hidden'); // Show URL group
  }
}

// Attach an event listener to update UI when the source selection changes
sourceSelect.addEventListener('change', updateSourceUI);

// Call the function once at startup to set the initial UI state
updateSourceUI();

// Function to build the configuration object to send to the backend /start endpoint
function buildConfig() {
  // Create a plain JavaScript object with the necessary parameters
  const cfg = {
    source: sourceSelect.value === 'webcam' ? 'webcam' : 'url', // 'webcam' or 'url'
    webcam_index: parseInt(webcamIndexInput.value || '0', 10), // Parse webcam index as integer
    camera_url: sourceSelect.value === 'webcam' ? null : (cameraUrlInput.value || null), // Use URL if not webcam
    model_name: modelNameSelect.value, // Model name from dropdown
    img_size: parseInt(imgSizeInput.value || '640', 10), // Image size as integer
    conf_thres: parseFloat(confThresInput.value || '0.3'), // Confidence threshold as float
    iou_match: parseFloat(iouMatchInput.value || '0.3'), // IoU threshold as float
    predict_horizon_sec: parseFloat(predHorizonInput.value || '0.7'), // Prediction horizon as float
    draw: true // Always draw annotations in this frontend
  };
  // Return the built configuration object
  return cfg;
}

// Function to start the stream by calling the backend /start endpoint
async function startStream() {
  // Build the configuration from current UI values
  const cfg = buildConfig();
  // Update the status text to inform the user
  statusText.textContent = 'Status: Starting...';
  try {
    // Make a POST request to /start with the config as JSON
    const res = await fetch('/start', {
      method: 'POST', // HTTP method
      headers: { 'Content-Type': 'application/json' }, // Tell server we send JSON
      body: JSON.stringify(cfg) // Serialize the configuration object
    });
    // If response is not OK, throw an error with response text
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Failed to start: ${errText}`);
    }
    // If the start succeeded, set the image source to the MJPEG stream endpoint
    streamImg.src = '/stream';
    // Update the status text to running
    statusText.textContent = 'Status: Running';
  } catch (err) {
    // On error, display the error message and clear the stream image
    console.error(err); // Log the error to console for debugging
    statusText.textContent = 'Status: Error - ' + err.message; // Show error to the user
    streamImg.src = ''; // Clear the image source to stop trying to load
  }
}

// Function to stop the stream by calling the backend /stop endpoint
async function stopStream() {
  // Update status to indicate stopping
  statusText.textContent = 'Status: Stopping...';
  try {
    // Make a POST request to /stop without a body
    const res = await fetch('/stop', { method: 'POST' });
    // If the response is not OK, throw an error
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Failed to stop: ${errText}`);
    }
    // On success, clear the MJPEG stream by removing the src
    streamImg.src = '';
    // Update status to idle
    statusText.textContent = 'Status: Idle';
  } catch (err) {
    // On error, show the error and keep the current state
    console.error(err);
    statusText.textContent = 'Status: Error - ' + err.message;
  }
}

// Attach click handlers to Start and Stop buttons
startBtn.addEventListener('click', startStream); // When start is clicked, run startStream()
stopBtn.addEventListener('click', stopStream);   // When stop is clicked, run stopStream()
