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
                0% { 
                  transform: translate(0, 0);
                  clip-path: inset(0 0 0 0);
                }
                2% { 
                  transform: translate(-8px, 0);
                  clip-path: inset(10px 0 80px 0);
                }
                4% { 
                  transform: translate(3px, -2px);
                  clip-path: inset(70px 0 20px 0);
                }
                6% { 
                  transform: translate(-5px, 4px);
                  clip-path: inset(30px 0 60px 0);
                }
                8% { 
                  transform: translate(7px, -1px);
                  clip-path: inset(5px 0 85px 0);
                }
                10% { 
                  transform: translate(-2px, 3px);
                  clip-path: inset(90px 0 5px 0);
                }
                15% { 
                  transform: translate(4px, -3px);
                  clip-path: inset(40px 0 50px 0);
                }
                20% { 
                  transform: translate(-6px, 1px);
                  clip-path: inset(75px 0 15px 0);
                }
                25% { 
                  transform: translate(2px, -4px);
                  clip-path: inset(15px 0 75px 0);
                }
                30% { 
                  transform: translate(-3px, 2px);
                  clip-path: inset(60px 0 30px 0);
                }
                50% { 
                  transform: translate(0, 0);
                  clip-path: inset(0 0 0 0);
                }
                52% { 
                  transform: translate(5px, -2px);
                  clip-path: inset(25px 0 65px 0);
                }
                54% { 
                  transform: translate(-4px, 3px);
                  clip-path: inset(80px 0 10px 0);
                }
                60% { 
                  transform: translate(6px, 0);
                  clip-path: inset(45px 0 45px 0);
                }
                80% { 
                  transform: translate(0, 0);
                  clip-path: inset(0 0 0 0);
                }
                85% { 
                  transform: translate(-7px, 2px);
                  clip-path: inset(20px 0 70px 0);
                }
                90% { 
                  transform: translate(3px, -1px);
                  clip-path: inset(85px 0 5px 0);
                }
                95% { 
                  transform: translate(-1px, 4px);
                  clip-path: inset(35px 0 55px 0);
                }
                100% { 
                  transform: translate(0, 0);
                  clip-path: inset(0 0 0 0);
                }
              }
              
              @keyframes digitalNoise {
                0% { 
                  clip-path: polygon(0 0, 100% 0, 100% 100%, 0 100%);
                }
                5% { 
                  clip-path: polygon(0 0, 100% 0, 100% 15%, 85% 15%, 85% 25%, 100% 25%, 100% 100%, 0 100%);
                }
                10% { 
                  clip-path: polygon(0 0, 15% 0, 15% 45%, 0 45%, 0 55%, 100% 55%, 100% 100%, 0 100%);
                }
                15% { 
                  clip-path: polygon(0 0, 100% 0, 100% 35%, 70% 35%, 70% 70%, 100% 70%, 100% 100%, 0 100%);
                }
                20% { 
                  clip-path: polygon(0 0, 100% 0, 100% 100%, 0 100%);
                }
                25% { 
                  clip-path: polygon(0 0, 25% 0, 25% 80%, 0 80%, 0 85%, 100% 85%, 100% 100%, 0 100%);
                }
                50% { 
                  clip-path: polygon(0 0, 100% 0, 100% 100%, 0 100%);
                }
                55% { 
                  clip-path: polygon(0 0, 100% 0, 100% 40%, 60% 40%, 60% 50%, 100% 50%, 100% 100%, 0 100%);
                }
                80% { 
                  clip-path: polygon(0 0, 100% 0, 100% 100%, 0 100%);
                }
                85% { 
                  clip-path: polygon(0 0, 40% 0, 40% 25%, 0 25%, 0 75%, 100% 75%, 100% 100%, 0 100%);
                }
                100% { 
                  clip-path: polygon(0 0, 100% 0, 100% 100%, 0 100%);
                }
              }
              
              .glitch-image {
                position: relative;
              }
              
              .glitch-image:hover {
                animation: glitchShift 0.3s infinite;
              }
              
              .glitch-image:hover::before {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: inherit;
                background-size: inherit;
                background-position: inherit;
                background-repeat: inherit;
                background: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGRlZnM+CjxmaWx0ZXIgaWQ9Im5vaXNlIj4KPGZ0dXJidWxlbmNlIGJhc2VGcmVxdWVuY3k9IjAuOSIgbnVtT2N0YXZlcz0iNCIgc2VlZD0iMiIgLz4KPGNvbG9yTWF0cml4IHZhbHVlcz0iMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDAgMSAwIiAvPgo8L2ZpbHRlcj4KPC9kZWZzPgo8cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWx0ZXI9InVybCgjbm9pc2UpIiBvcGFjaXR5PSIwLjEiIC8+Cjwvc3ZnPg==);
                mix-blend-mode: overlay;
                animation: digitalNoise 0.5s infinite;
                z-index: 1;
                pointer-events: none;
              }
              
              .glitch-image:hover::after {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: 
                  linear-gradient(90deg, transparent 78%, rgba(255,0,0,0.4) 79%, rgba(255,0,0,0.4) 81%, transparent 82%),
                  linear-gradient(90deg, transparent 35%, rgba(0,255,0,0.3) 36%, rgba(0,255,0,0.3) 37%, transparent 38%),
                  linear-gradient(90deg, transparent 60%, rgba(0,0,255,0.4) 61%, rgba(0,0,255,0.4) 62%, transparent 63%);
                transform: translateX(-100%);
                animation: scanLine 0.8s infinite;
                z-index: 2;
                pointer-events: none;
              }
              
              @keyframes scanLine {
                0% { transform: translateX(-100%); }
                50% { transform: translateX(100%); }
                100% { transform: translateX(100%); }
              }
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
          {
            title: "Digital Deception",
            desc: "Welcome to the Digital Truth Lab forensics headquarters",
            start: 0,
            pause: 3,
          },
          {
            title: "Truth vs Deception",
            desc: "Understanding the difference between authentic and synthetic media",
            start: 3,
            pause: 7,
          },
          {
            title: "Evidence Analysis Lab",
            desc: "Submit your media evidence for forensic examination",
            start: 7,
            pause: 11,
          },
          {
            title: "Investigation Results",
            desc: "Detailed analysis of submitted evidence",
            start: 11,
            pause: 15,
          },
          {
            title: "Model Performance",
            desc: "Technical specifications and accuracy metrics",
            start: 15,
            pause: 19,
          },
          {
            title: "Suspects",
            desc: "Meet the forensic investigators behind this project",
            start: 19,
            pause: 24,
          },
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

        function playToPausePoint(startTime, pauseTime) {
          if (pauseTimeout) clearTimeout(pauseTimeout);
          isTransitioning = true;

          video.currentTime = startTime;
          video.play().catch(() => {});

          const timeUntilPause = (pauseTime - startTime) * 1000;
          pauseTimeout = setTimeout(() => {
            video.pause();
            isTransitioning = false;
          }, timeUntilPause);
        }

        function goToSection(index) {
          if (index < 0 || index >= sections.length || isTransitioning) return;
          currentSection = index;
          updateActiveButton(index);
          showSection(index);
          playToPausePoint(sections[index].start, sections[index].pause);

          if (index === 4) {
            setTimeout(() => createPerformanceChart(), 600);
          }

          if (index === 3) {
            setTimeout(() => simulateAnalysis(), 500);
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

        function handleFileUpload(file) {
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
              previewContent.innerHTML = "";
              previewContent.appendChild(mediaElement);
              previewArea.style.display = "block";

              setTimeout(() => {
                new Glitch("uploaded-media", "high");
              }, 100);

              setTimeout(() => {
                goToSection(3);
              }, 2000);
            }
          };
          reader.readAsDataURL(file);
        }

        function simulateAnalysis() {
          const isDeepfake = Math.random() > 0.5;
          const confidence = Math.random() * 30 + 70;

          const resultStatus = document.getElementById("result-status");
          const resultMessage = document.getElementById("result-message");
          const confidenceFill = document.getElementById("confidence-fill");
          const confidenceText = document.getElementById("confidence-text");

          if (isDeepfake) {
            resultStatus.textContent = "DEEPFAKE DETECTED";
            resultStatus.className = "result-status deepfake-detected";
            resultMessage.textContent =
              "Our analysis indicates this media contains synthetic elements. Artificial manipulation patterns were detected in facial features and temporal consistency.";
            confidenceFill.className = "confidence-fill";
          } else {
            resultStatus.textContent = "AUTHENTIC MEDIA";
            resultStatus.className = "result-status authentic-detected";
            resultMessage.textContent =
              "Analysis confirms this media appears to be genuine. No significant artificial manipulation patterns were detected.";
            confidenceFill.className = "confidence-fill authentic-fill";
          }

          setTimeout(() => {
            confidenceFill.style.width = confidence + "%";
            confidenceText.textContent = `Confidence: ${confidence.toFixed(
              1
            )}%`;
          }, 500);

          setTimeout(() => {
            document.getElementById("facial-metric").textContent =
              (Math.random() * 20 + 80).toFixed(1) + "%";
            document.getElementById("temporal-metric").textContent =
              (Math.random() * 15 + 85).toFixed(1) + "%";
            document.getElementById("artifact-metric").textContent =
              (Math.random() * 25 + 75).toFixed(1) + "%";
            document.getElementById("overall-metric").textContent =
              confidence.toFixed(1) + "%";
          }, 1000);
        }

        function createPerformanceChart() {
          if (performanceChart) {
            performanceChart.destroy();
          }

          const ctx = document
            .getElementById("performance-chart")
            .getContext("2d");

          performanceChart = new Chart(ctx, {
            type: "radar",
            data: {
              labels: [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "Specificity",
                "AUC-ROC",
              ],
              datasets: [
                {
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
                },
              ],
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              interaction: {
                intersect: false,
                mode: "point",
              },
              plugins: {
                legend: {
                  labels: {
                    color: "#ffffff",
                    font: {
                      size: 14,
                      weight: "bold",
                    },
                  },
                },
                tooltip: {
                  enabled: true,
                  backgroundColor: "rgba(0, 0, 0, 0.8)",
                  titleColor: "#00ff88",
                  bodyColor: "#ffffff",
                  borderColor: "#00ff88",
                  borderWidth: 1,
                  displayColors: false,
                  callbacks: {
                    label: function (context) {
                      return (
                        context.dataset.label + ": " + context.parsed.r + "%"
                      );
                    },
                  },
                },
              },
              scales: {
                r: {
                  beginAtZero: true,
                  max: 100,
                  ticks: {
                    color: "rgba(255, 255, 255, 0.7)",
                    backdropColor: "transparent",
                    font: {
                      size: 12,
                    },
                    stepSize: 20,
                  },
                  grid: {
                    color: "rgba(0, 255, 136, 0.3)",
                  },
                  angleLines: {
                    color: "rgba(0, 255, 136, 0.3)",
                  },
                  pointLabels: {
                    color: "#ffffff",
                    font: {
                      size: 13,
                      weight: "bold",
                    },
                  },
                },
              },
              animation: {
                duration: 1000,
                easing: "easeInOutQuart",
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

        document.body.addEventListener(
          "click",
          function () {
            if (video.readyState >= 2) {
              video.play().catch(() => {});
            }
          },
          { once: true }
        );
      });