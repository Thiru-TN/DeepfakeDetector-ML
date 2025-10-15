import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tempfile
import shutil
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Deepfake Detection API")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: ["http://localhost:3000", "https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
# Mount frontend folder if it exists
if os.path.exists("../frontend"):
    app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Configuration
class Config:
    MODEL_PATH = "../models/EfficientNet.pth"
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
    FRAME_SAMPLE_RATE = 10  # Extract every 10th frame
    MAX_FRAMES = 50  # Maximum frames to analyze
    FACE_MIN_CONFIDENCE = 0.9
    INPUT_SIZE = (224, 224)

config = Config()

# Response Models
class AnalysisResponse(BaseModel):
    success: bool
    message: str
    results: Optional[Dict] = None
    error: Optional[str] = None

class FrameResult(BaseModel):
    frame_id: int
    timestamp: float
    fake_score: float
    confidence: float
    face_detected: bool

# Global model loader
class ModelLoader:
    def __init__(self):
        self.model = None
        self.device = None
        self.face_detector = None
    
    def load_model(self):
        """Load EfficientNet model"""
        if self.model is not None:
            return self.model
        
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Import EfficientNet
            from efficientnet_pytorch import EfficientNet
            
            # Load model architecture
            self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
            
            # Load trained weights
            if os.path.exists(config.MODEL_PATH):
                checkpoint = torch.load(config.MODEL_PATH, map_location=self.device)
                
                # Handle different save formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    # If saved as entire model
                    self.model = checkpoint
                
                logger.info("Model weights loaded successfully")
            else:
                logger.warning(f"Model file not found at {config.MODEL_PATH}, using pretrained weights")
            
            self.model.to(self.device)
            self.model.eval()
            logger.info("EfficientNet model ready")
            
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    
    def load_face_detector(self):
        """Load face detector"""
        if self.face_detector is not None:
            return self.face_detector
        
        try:
            # Try MTCNN first
            from facenet_pytorch import MTCNN
            self.face_detector = MTCNN(
                keep_all=False,
                device=self.device if self.device else 'cpu',
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7]
            )
            logger.info("MTCNN face detector loaded")
            return self.face_detector
        except ImportError:
            # Fallback to Haar Cascade
            logger.warning("MTCNN not available, using Haar Cascade")
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            return self.face_detector

model_loader = ModelLoader()

# Utility Functions
class VideoValidator:
    """Validate video file"""
    
    @staticmethod
    def validate_file(file: UploadFile) -> tuple[bool, str]:
        """Check if file is valid"""
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in config.SUPPORTED_FORMATS:
            return False, f"Unsupported format. Allowed: {', '.join(config.SUPPORTED_FORMATS)}"
        
        # Check file size
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size == 0:
            return False, "File is empty"
        
        if file_size > config.MAX_FILE_SIZE:
            return False, f"File too large. Max size: {config.MAX_FILE_SIZE / (1024*1024):.0f}MB"
        
        return True, "Valid"
    
    @staticmethod
    def check_video_corruption(video_path: str) -> tuple[bool, str, Dict]:
        """Check if video is corrupted and get metadata"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return False, "Cannot open video file - possibly corrupted", {}
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            if frame_count == 0:
                cap.release()
                return False, "Video has no frames", {}
            
            if width < 100 or height < 100:
                cap.release()
                return False, "Video resolution too low", {}
            
            # Try to read first frame
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                return False, "Cannot read video frames - corrupted", {}
            
            cap.release()
            
            metadata = {
                'frame_count': frame_count,
                'fps': round(fps, 2),
                'width': width,
                'height': height,
                'duration': round(duration, 2)
            }
            
            return True, "Video is valid", metadata
            
        except Exception as e:
            return False, f"Error validating video: {str(e)}", {}

class FrameExtractor:
    """Extract and preprocess frames from video"""
    
    def __init__(self, face_detector):
        self.face_detector = face_detector
        self.detector_type = 'mtcnn' if hasattr(face_detector, 'detect') else 'haar'
    
    def extract_frames(self, video_path: str) -> List[Dict]:
        """Extract frames from video with face detection"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Cannot open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames_data = []
        frame_idx = 0
        extracted_count = 0
        
        # Calculate sampling rate
        sample_rate = max(1, total_frames // config.MAX_FRAMES)
        
        logger.info(f"Extracting frames: total={total_frames}, sample_rate={sample_rate}")
        
        while extracted_count < config.MAX_FRAMES:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Sample frames
            if frame_idx % sample_rate == 0:
                timestamp = frame_idx / fps if fps > 0 else 0
                
                # Quality check
                if self._is_frame_usable(frame):
                    # Detect face
                    face_data = self._detect_face(frame)
                    
                    if face_data['detected']:
                        frames_data.append({
                            'frame_id': frame_idx,
                            'timestamp': round(timestamp, 2),
                            'frame': face_data['face_crop'],
                            'bbox': face_data['bbox'],
                            'face_confidence': face_data['confidence']
                        })
                        extracted_count += 1
                    else:
                        # Include frame without face if we have few frames
                        if extracted_count < 10:
                            frames_data.append({
                                'frame_id': frame_idx,
                                'timestamp': round(timestamp, 2),
                                'frame': frame,
                                'bbox': None,
                                'face_confidence': 0.0
                            })
                            extracted_count += 1
            
            frame_idx += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames_data)} frames with faces")
        
        if len(frames_data) == 0:
            raise ValueError("No usable frames with faces detected")
        
        return frames_data
    
    def _is_frame_usable(self, frame: np.ndarray) -> bool:
        """Check frame quality"""
        if frame is None or frame.size == 0:
            return False
        
        h, w = frame.shape[:2]
        if h < 100 or w < 100:
            return False
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Brightness check
        brightness = np.mean(gray)
        if brightness < 20 or brightness > 235:
            return False
        
        # Blur check
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 50:
            return False
        
        return True
    
    def _detect_face(self, frame: np.ndarray) -> Dict:
        """Detect face in frame"""
        if self.detector_type == 'mtcnn':
            return self._detect_face_mtcnn(frame)
        else:
            return self._detect_face_haar(frame)
    
    def _detect_face_mtcnn(self, frame: np.ndarray) -> Dict:
        """MTCNN detection"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            boxes, probs = self.face_detector.detect(pil_img)
            
            if boxes is not None and len(boxes) > 0:
                best_idx = np.argmax(probs)
                box = boxes[best_idx].astype(int)
                x, y, x2, y2 = box
                
                # Add padding
                padding = int(max(x2-x, y2-y) * 0.1)
                x = max(0, x - padding)
                y = max(0, y - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)
                
                face_crop = frame[y:y2, x:x2]
                
                return {
                    'detected': True,
                    'confidence': float(probs[best_idx]),
                    'bbox': {'x': int(x), 'y': int(y), 'w': int(x2-x), 'h': int(y2-y)},
                    'face_crop': face_crop
                }
        except Exception as e:
            logger.error(f"MTCNN detection error: {e}")
        
        return {'detected': False, 'confidence': 0.0, 'bbox': None, 'face_crop': None}
    
    def _detect_face_haar(self, frame: np.ndarray) -> Dict:
        """Haar Cascade detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_crop = frame[y:y+h, x:x+w]
            
            return {
                'detected': True,
                'confidence': 1.0,
                'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'face_crop': face_crop
            }
        
        return {'detected': False, 'confidence': 0.0, 'bbox': None, 'face_crop': None}

class DeepfakePredictor:
    """Run deepfake prediction on frames"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for EfficientNet"""
        # Resize
        frame_resized = cv2.resize(frame, config.INPUT_SIZE)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize (ImageNet stats)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame_normalized = (frame_normalized - mean) / std
        
        # Convert to tensor (C, H, W)
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).float()
        
        # Add batch dimension
        frame_tensor = frame_tensor.unsqueeze(0)
        
        return frame_tensor
    
    def predict(self, frame: np.ndarray) -> Dict:
        """Predict if frame is deepfake"""
        try:
            # Preprocess
            frame_tensor = self.preprocess_frame(frame).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(frame_tensor)
                
                # Handle different output formats
                if output.shape[-1] == 1:
                    prob = torch.sigmoid(output)
                    fake_score = prob[0][0].item()
                else:
                    prob = torch.nn.functional.softmax(output, dim=-1)
                    fake_score = prob[0][1].item()
                
                confidence = max(prob[0]).item()
            
            return {
                'fake_score': round(fake_score, 4),
                'confidence': round(confidence, 4),
                'label': 'FAKE' if fake_score > 0.5 else 'REAL'
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'fake_score': 0.5,
                'confidence': 0.0,
                'label': 'ERROR',
                'error': str(e)
            }
    
    def analyze_frames(self, frames_data: List[Dict]) -> Dict:
        """Analyze all extracted frames"""
        frame_results = []
        total_fake_score = 0
        faces_detected = 0
        
        for frame_info in frames_data:
            frame = frame_info['frame']
            
            # Predict
            pred = self.predict(frame)
            
            total_fake_score += pred['fake_score']
            if frame_info.get('bbox') is not None:
                faces_detected += 1
            
            frame_results.append({
                'frame_id': frame_info['frame_id'],
                'timestamp': frame_info['timestamp'],
                'fake_score': pred['fake_score'],
                'confidence': pred['confidence'],
                'label': pred['label'],
                'face_detected': frame_info.get('bbox') is not None,
                'face_confidence': frame_info.get('face_confidence', 0.0)
            })
        
        # Aggregate results
        avg_fake_score = total_fake_score / len(frames_data)
        avg_confidence = np.mean([f['confidence'] for f in frame_results])
        
        # Temporal analysis
        scores = [f['fake_score'] for f in frame_results]
        score_volatility = np.std(scores)
        
        # Final verdict
        final_label = 'FAKE' if avg_fake_score > 0.5 else 'REAL'
        
        if avg_confidence < 0.5:
            final_label = 'UNCERTAIN'
        
        return {
            'final_verdict': final_label,
            'avg_fake_score': round(avg_fake_score, 3),
            'avg_confidence': round(avg_confidence, 3),
            'score_volatility': round(score_volatility, 3),
            'frames_analyzed': len(frames_data),
            'frames_with_faces': faces_detected,
            'face_detection_rate': round(faces_detected / len(frames_data), 2),
            'frame_results': frame_results
        }

# API Endpoints
@app.get("/")
async def root():
    """Serve frontend or API info"""
    # If frontend exists, serve index.html
    frontend_path = Path("../frontend/index.html")
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    
    # Otherwise return API info
    return {
        "status": "online",
        "service": "Deepfake Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST)",
            "batch": "/analyze-batch (POST)"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        model = model_loader.load_model()
        face_detector = model_loader.load_face_detector()
        
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "face_detector_loaded": face_detector is not None,
            "device": str(model_loader.device)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(file: UploadFile = File(...)):
    """
    Main endpoint to analyze uploaded video for deepfakes
    """
    temp_video_path = None
    
    try:
        logger.info(f"Received video: {file.filename}")
        
        # 1. Validate file
        is_valid, message = VideoValidator.validate_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # 2. Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            temp_video_path = tmp_file.name
        
        logger.info(f"Video saved to: {temp_video_path}")
        
        # 3. Check video corruption and get metadata
        is_valid, message, metadata = VideoValidator.check_video_corruption(temp_video_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        logger.info(f"Video metadata: {metadata}")
        
        # 4. Load models
        model = model_loader.load_model()
        face_detector = model_loader.load_face_detector()
        
        # 5. Extract frames with face detection
        extractor = FrameExtractor(face_detector)
        frames_data = extractor.extract_frames(temp_video_path)
        
        if len(frames_data) == 0:
            raise HTTPException(status_code=400, detail="No faces detected in video")
        
        # 6. Run deepfake prediction
        predictor = DeepfakePredictor(model, model_loader.device)
        results = predictor.analyze_frames(frames_data)
        
        # 7. Combine with metadata
        final_results = {
            **results,
            'video_metadata': metadata,
            'filename': file.filename
        }
        
        logger.info(f"Analysis complete: {results['final_verdict']}")
        
        return AnalysisResponse(
            success=True,
            message="Analysis completed successfully",
            results=final_results
        )
    
    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return AnalysisResponse(
            success=False,
            message="Analysis failed",
            error=str(e)
        )
    finally:
        # Cleanup
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")

@app.post("/analyze-batch")
async def analyze_batch(files: List[UploadFile] = File(...)):
    """
    Analyze multiple videos in batch
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 videos per batch")
    
    results = []
    for file in files:
        try:
            result = await analyze_video(file)
            results.append({
                'filename': file.filename,
                'result': result
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return {
        'success': True,
        'total_videos': len(files),
        'results': results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)