const API_BASE_URL = 'http://localhost:8000';

async function analyzeVideoWithBackend(file) {
  try {
    showLoadingState();
    
    const formData = new FormData();
    formData.append('file', file);
    
    console.log('Sending request to:', `${API_BASE_URL}/predict?explainability=true`);
    
    const response = await fetch(`${API_BASE_URL}/predict?explainability=true`, {
      method: 'POST',
      body: formData,
    });
    
    console.log('Response status:', response.status);
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Analysis failed');
    }
    
    const result = await response.json();
    console.log('✅ Successfully received result:', result);
    displayAnalysisResults(result);
    
  } catch (error) {
    console.error('❌ Analysis error:', error);
    showError(error.message);
  }
}

function showLoadingState() {
  const resultStatus = document.getElementById("result-status");
  const resultMessage = document.getElementById("result-message");
  const confidenceFill = document.getElementById("confidence-fill");
  const confidenceText = document.getElementById("confidence-text");
  
  console.log('Loading state elements:', {
    resultStatus: !!resultStatus,
    resultMessage: !!resultMessage,
    confidenceFill: !!confidenceFill,
    confidenceText: !!confidenceText
  });
  
  if (!resultStatus || !resultMessage || !confidenceFill || !confidenceText) {
    console.error('❌ Missing required HTML elements for loading state!');
    return;
  }
  
  resultStatus.textContent = "ANALYZING...";
  resultStatus.className = "result-status analyzing";
  resultMessage.textContent = "Processing video frames and analyzing for deepfake patterns...";
  confidenceFill.style.width = "0%";
  confidenceText.textContent = "Processing...";
  
  let progress = 0;
  const loadingInterval = setInterval(() => {
    progress += 2;
    if (progress <= 90) {
      confidenceFill.style.width = progress + "%";
    } else {
      clearInterval(loadingInterval);
    }
  }, 100);
  
  window.loadingInterval = loadingInterval;
}

function displayAnalysisResults(results) {
  if (window.loadingInterval) clearInterval(window.loadingInterval);
  console.log("📊 Raw API response:", results);

  const resultStatus = document.getElementById("result-status");
  const resultMessage = document.getElementById("result-message");
  const confidenceFill = document.getElementById("confidence-fill");
  const confidenceText = document.getElementById("confidence-text");

  if (!results || typeof results.is_fake === "undefined") {
    resultStatus.textContent = "❌ ANALYSIS ERROR";
    resultMessage.textContent = "Invalid response from backend.";
    resultStatus.style.color = "#ff3333";
    return;
  }

const label = results.label ? results.label.toString().toUpperCase() : "";
const isFake =
  results.is_fake === true ||
  results.is_fake === "true" ||
  label === "FAKE" ||
  label.includes("FAKE");

const fakeProb = Number(results.fake_probability) || 0;
const realProb = Number(results.real_probability) || 0;
const confidence = Number(results.confidence) || 0;

console.log("Parsed values:", { label, isFake, fakeProb, realProb, confidence });

if (isFake) {
  resultStatus.textContent = "DEEPFAKE DETECTED";
  resultStatus.style.color = "#ff4d4d";
  resultMessage.textContent = `AI detected manipulation with ${(fakeProb * 100).toFixed(
    1
  )}% fake probability.`;
  confidenceFill.style.background = "#ff4d4d";
} else {
  resultStatus.textContent = "AUTHENTIC MEDIA";
  resultStatus.style.color = "#00ff88";
  resultMessage.textContent = `Media appears genuine with ${(realProb * 100).toFixed(
    1
  )}% authenticity.`;
  confidenceFill.style.background = "#00ff88";
}


  confidenceFill.style.width = (confidence * 100).toFixed(1) + "%";
  confidenceText.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;

  // Optional metrics
  document.getElementById("facial-metric").textContent =
    ((results.facial_consistency || 0) * 100).toFixed(1) + "%";
  document.getElementById("temporal-metric").textContent =
    ((results.temporal_stability || 0) * 100).toFixed(1) + "%";
  document.getElementById("artifact-metric").textContent =
    ((results.artifact_detection || 0) * 100).toFixed(1) + "%";
  document.getElementById("overall-metric").textContent =
    ((results.overall_score || 0) * 100).toFixed(1) + "%";

  // Show heatmaps
  if (Array.isArray(results.explainability) && results.explainability.length) {
    displayExplainabilityHeatmaps(results.explainability);
  }

  console.log("✅ Display updated");
}


function displayExplainabilityHeatmaps(heatmaps) {
  const container = document.createElement('div');
  container.id = 'explainability-container';
  container.style.cssText = 'margin-top: 20px; padding: 20px; background: rgba(0, 255, 136, 0.1); border-radius: 10px; border: 2px solid #00ff88;';
  
  const title = document.createElement('h3');
  title.textContent = 'Grad-CAM Heatmaps (Attention Regions)';
  title.style.cssText = 'color: #00ff88; margin-bottom: 15px; text-align: center;';
  container.appendChild(title);
  
  const gridContainer = document.createElement('div');
  gridContainer.style.cssText = 'display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;';
  
  heatmaps.forEach((base64Image, index) => {
    const imgWrapper = document.createElement('div');
    imgWrapper.style.cssText = 'position: relative;';
    
    const img = document.createElement('img');
    img.src = `data:image/jpeg;base64,${base64Image}`;
    img.style.cssText = 'width: 100%; border-radius: 8px; border: 2px solid #00ff41;';
    img.alt = `Heatmap ${index + 1}`;
    
    const label = document.createElement('div');
    label.textContent = `Frame ${index + 1}`;
    label.style.cssText = 'position: absolute; bottom: 5px; left: 5px; background: rgba(0,0,0,0.7); color: #00ff88; padding: 2px 8px; border-radius: 4px; font-size: 12px;';
    
    imgWrapper.appendChild(img);
    imgWrapper.appendChild(label);
    gridContainer.appendChild(imgWrapper);
  });
  
  container.appendChild(gridContainer);
  
  const existingContainer = document.getElementById('explainability-container');
  if (existingContainer) {
    existingContainer.remove();
  }
  
  const resultMessage = document.getElementById("result-message");
  if (resultMessage && resultMessage.parentElement) {
    resultMessage.parentElement.appendChild(container);
  }
}

function showError(message) {
  const resultStatus = document.getElementById("result-status");
  const resultMessage = document.getElementById("result-message");
  const confidenceFill = document.getElementById("confidence-fill");
  const confidenceText = document.getElementById("confidence-text");
  
  if (!resultStatus || !resultMessage || !confidenceFill || !confidenceText) {
    console.error('❌ Cannot show error - missing HTML elements');
    alert(`Error: ${message}`);
    return;
  }
  
  resultStatus.textContent = "ANALYSIS ERROR";
  resultStatus.className = "result-status error";
  resultMessage.textContent = `Error: ${message}. Please try again with a different video.`;
  confidenceFill.style.width = "0%";
  confidenceFill.className = "confidence-fill error-fill";
  confidenceText.textContent = "Error";
}

function handleFileUpload(file) {
  const validVideoTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/x-matroska'];
  const validImageTypes = ['image/jpeg', 'image/png', 'image/jpg'];
  
  if (!validVideoTypes.includes(file.type) && !validImageTypes.includes(file.type)) {
    showError('Invalid file type. Please upload MP4, AVI, MOV, MKV, JPG, or PNG files.');
    return;
  }
  
  const maxSize = 500 * 1024 * 1024;
  if (file.size > maxSize) {
    showError('File too large. Maximum size is 500MB.');
    return;
  }
  
  const reader = new FileReader();
  reader.onload = (e) => {
    const result = e.target.result;
    let mediaElement;

    if (file.type.startsWith("image/")) {
      mediaElement = document.createElement("img");
      mediaElement.src = result;
      mediaElement.className = "preview-media glitch-image";
      mediaElement.id = "uploaded-media";
    } else if (file.type.startsWith("video/")) {
      mediaElement = document.createElement("video");
      mediaElement.src = result;
      mediaElement.className = "preview-media";
      mediaElement.controls = true;
      mediaElement.id = "uploaded-media";
    }

    if (mediaElement) {
      const previewContent = document.getElementById("preview-content");
      const previewArea = document.getElementById("preview-area");
      
      previewContent.innerHTML = "";
      previewContent.appendChild(mediaElement);
      previewArea.style.display = "block";

      setTimeout(() => {
        new Glitch("uploaded-media", "high");
      }, 100);

      setTimeout(() => {
        goToSection(3);
        
        if (file.type.startsWith("video/")) {
          setTimeout(() => {
            analyzeVideoWithBackend(file);
          }, 500);
        } else {
          setTimeout(() => {
            simulateAnalysis();
          }, 500);
        }
      }, 1000);
    }
  };
  reader.readAsDataURL(file);
}

const additionalStyles = `
  <style>
    .result-status.analyzing {
      color: #00b4d8;
      animation: pulse 1.5s infinite;
    }
    
    .result-status.uncertain {
      color: #ffa500;
    }
    
    .result-status.error {
      color: #ff6b6b;
    }
    
    .confidence-fill.uncertain-fill {
      background: linear-gradient(90deg, #ffa500, #ffcc00);
    }
    
    .confidence-fill.error-fill {
      background: linear-gradient(90deg, #ff6b6b, #ff8787);
    }
    
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.6; }
    }
  </style>
`;

document.head.insertAdjacentHTML('beforeend', additionalStyles);

class Glitch {
  constructor(imageId, intensity = "medium") {
    this.img = document.getElementById(imageId);
    this.intensity = intensity;
    this.setupGlitch();
  }

  setupGlitch() {
    if (!this.img) return;

    this.img.addEventListener("mouseenter", () => {
      this.startGlitch();
    });

    this.img.addEventListener("mouseleave", () => {
      this.stopGlitch();
    });
  }

  startGlitch() {
    this.img.style.filter = "hue-rotate(90deg) saturate(150%)";
    this.img.style.animation = "glitchEffect 0.3s infinite";
    this.addGlitchStyles();
  }

  stopGlitch() {
    this.img.style.filter = "none";
    this.img.style.animation = "none";
  }

  addGlitchStyles() {
    if (!document.getElementById("glitch-styles")) {
      const style = document.createElement("style");
      style.id = "glitch-styles";
      style.textContent = `
        @keyframes glitchShift {
          0% { transform: translate(0, 0); clip-path: inset(0 0 0 0); }
          2% { transform: translate(-8px, 0); clip-path: inset(10px 0 80px 0); }
          4% { transform: translate(3px, -2px); clip-path: inset(70px 0 20px 0); }
          100% { transform: translate(0, 0); clip-path: inset(0 0 0 0); }
        }
        .glitch-image { position: relative; }
        .glitch-image:hover { animation: glitchShift 0.3s infinite; }
      `;
      document.head.appendChild(style);
    }
  }
}

function animateTextWords(element) {
  const text = element.textContent;
  const words = text.split(" ");
  element.innerHTML = words
    .map((word) => `<span class="word">${word}</span>`)
    .join(" ");

  const wordElements = element.querySelectorAll(".word");
  wordElements.forEach((word, index) => {
    word.style.animationDelay = `${index * 0.1}s`;
  });
}

document.addEventListener("DOMContentLoaded", function () {
  const video = document.getElementById("main-video");
  const navButtons = document.querySelectorAll(".nav-btn");
  const allSections = document.querySelectorAll(".content");

  const sections = [
    { title: "Digital Deception", desc: "Welcome to the Digital Truth Lab forensics headquarters", start: 0, pause: 3 },
    { title: "Truth vs Deception", desc: "Understanding the difference between authentic and synthetic media", start: 3, pause: 7 },
    { title: "Evidence Analysis Lab", desc: "Submit your media evidence for forensic examination", start: 7, pause: 11 },
    { title: "Investigation Results", desc: "Detailed analysis of submitted evidence", start: 11, pause: 15 },
    { title: "Model Performance", desc: "Technical specifications and accuracy metrics", start: 15, pause: 19 },
    { title: "Suspects", desc: "Meet the forensic investigators behind this project", start: 19, pause: 24 },
  ];

  let currentSection = 0;
  let pauseTimeout = null;
  let isScrolling = false;
  let scrollTimeout;
  let performanceChart = null;
  let isTransitioning = false;

  new Glitch("glitch-img-1", "medium");

  setTimeout(() => {
    const descElement = document.getElementById("main-description");
    if (descElement) {
      descElement.classList.add("show");
    }
  }, 1500);

  function updateActiveButton(index) {
    navButtons.forEach((btn, i) => {
      btn.classList.toggle("active", i === index);
    });
  }

  function showSection(index) {
    allSections.forEach((section) => {
      section.classList.remove("active");
    });
    setTimeout(() => {
      allSections[index].classList.add("active");
    }, 2000);
  }

  // Replace your current video control section with this:

function playToPausePoint(startTime, pauseTime) {
  if (pauseTimeout) clearTimeout(pauseTimeout);
  isTransitioning = true;

  video.currentTime = startTime;
  
  // Better error handling for video playback
  const playPromise = video.play();
  
  if (playPromise !== undefined) {
    playPromise.catch(error => {
      console.log('Video play failed, trying again:', error);
      // Auto-play might be blocked, but we'll still proceed
      isTransitioning = false;
    });
  }

  const timeUntilPause = (pauseTime - startTime) * 1000;
  pauseTimeout = setTimeout(() => {
    // Only pause if we're still at the expected section
    if (!isTransitioning || Math.abs(video.currentTime - pauseTime) < 2) {
      video.pause();
    }
    isTransitioning = false;
  }, timeUntilPause);
}

function goToSection(index) {
  if (index < 0 || index >= sections.length || isTransitioning) return;
  
  // Ensure video is ready
  if (video.readyState < 2) {
    console.log('Video not ready, waiting...');
    setTimeout(() => goToSection(index), 100);
    return;
  }
  
  currentSection = index;
  updateActiveButton(index);
  showSection(index);
  
  // Add a small delay to ensure DOM updates complete
  setTimeout(() => {
    playToPausePoint(sections[index].start, sections[index].pause);
  }, 50);

  if (index === 4) {
    setTimeout(() => createPerformanceChart(), 600);
  }
}

function nextSection() {
  if (currentSection < sections.length - 1) {
    goToSection(currentSection + 1);
  }
}

  window.goToSection = goToSection;
  window.nextSection = nextSection;

  function handleScroll(event) {
    if (isScrolling || isTransitioning) return;
    isScrolling = true;
    const delta = Math.sign(event.deltaY);
    if (delta > 0 && currentSection < sections.length - 1) {
      goToSection(currentSection + 1);
    } else if (delta < 0 && currentSection > 0) {
      goToSection(currentSection - 1);
    }
    clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(() => {
      isScrolling = false;
    }, 800);
  }

  const fileInput = document.getElementById("file-input");
  const uploadArea = document.getElementById("upload-area");
  const previewArea = document.getElementById("preview-area");
  const previewContent = document.getElementById("preview-content");

  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
  });

  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
  });

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  function createPerformanceChart() {
    if (performanceChart) {
      performanceChart.destroy();
    }

    const ctx = document.getElementById("performance-chart").getContext("2d");
    performanceChart = new Chart(ctx, {
      type: "radar",
      data: {
        labels: ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "AUC-ROC"],
        datasets: [{
          label: "Model Performance",
          data: [94.7, 92.3, 96.1, 94.2, 91.8, 98.7],
          borderColor: "#00ff88",
          backgroundColor: "rgba(0, 255, 136, 0.2)",
          borderWidth: 3,
          pointBackgroundColor: "#00ff88",
          pointBorderColor: "#ffffff",
          pointBorderWidth: 2,
          pointRadius: 6,
          pointHoverRadius: 8,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            beginAtZero: true,
            max: 100,
            ticks: { color: "rgba(255, 255, 255, 0.7)", stepSize: 20 },
            grid: { color: "rgba(0, 255, 136, 0.3)" },
            pointLabels: { color: "#ffffff", font: { size: 13, weight: "bold" } },
          },
        },
      },
    });
  }

  video.addEventListener("loadedmetadata", function () {
    goToSection(0);
  });

  navButtons.forEach((btn, index) => {
    btn.addEventListener("click", function () {
      if (!isTransitioning) {
        goToSection(index);
      }
    });
  });

  window.addEventListener("wheel", handleScroll, { passive: true });

  document.body.addEventListener("click", function () {
    if (video.readyState >= 2) {
      video.play().catch(() => {});
    }
  }, { once: true });
});