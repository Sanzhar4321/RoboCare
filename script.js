const video = document.getElementById('video');
const canvasOverlay = document.getElementById('overlay');
const namePrompt = document.getElementById('name-prompt');
const nameInput = document.getElementById('name');

let labeledDescriptors = [];
let currentDescriptor = null;

async function loadFaceAPI() {
  try {
    const MODEL_URL = 'https://justadudewhohacks.github.io/face-api.js/models';
    
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL)
    ]);

    console.log('Models loaded successfully.');
    startVideo();
  } catch (err) {
    console.error('Error loading models:', err);
  }
}

function startVideo() {
  navigator.mediaDevices
    .getUserMedia({ video: {} })
    .then(stream => {
      video.srcObject = stream;
      console.log('Webcam stream started.');
    })
    .catch(err => console.error('Error accessing webcam:', err));
}

video.addEventListener('play', () => {
  console.log('Video is playing. Starting face detection loop...');
  const displaySize = { width: video.width, height: video.height };

  faceapi.matchDimensions(canvasOverlay, displaySize);

  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors();

    const ctx = canvasOverlay.getContext('2d');
    ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

    const resizedDetections = faceapi.resizeResults(detections, displaySize);

    for (const detection of resizedDetections) {
      const box = detection.detection.box;
      const descriptor = detection.descriptor;

      if (labeledDescriptors.length > 0) {
        const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
        const bestMatch = faceMatcher.findBestMatch(descriptor);

        if (bestMatch.label === 'unknown') {
          promptForNewFace(descriptor);
          drawBox(box, 'Unknown');
        } else {
          drawBox(box, bestMatch.toString());
        }
      } else {
        promptForNewFace(descriptor);
        drawBox(box, 'Unknown');
      }
    }
  }, 200);
});

function promptForNewFace(descriptor) {
  if (namePrompt.style.display === 'none' && !currentDescriptor) {
    currentDescriptor = descriptor;
    namePrompt.style.display = 'flex';
  }
}

function saveNewFace() {
  const name = nameInput.value.trim();
  if (name && currentDescriptor) {
    let existing = labeledDescriptors.find(ld => ld.label === name);

    if (existing) {
      existing.descriptors.push(currentDescriptor);
    } else {
      labeledDescriptors.push(
        new faceapi.LabeledFaceDescriptors(name, [currentDescriptor])
      );
    }
  }

  namePrompt.style.display = 'none';
  nameInput.value = '';
  currentDescriptor = null;
}

function drawBox(box, label) {
  const drawBox = new faceapi.draw.DrawBox(box, { label });
  drawBox.draw(canvasOverlay);
}

loadFaceAPI();
