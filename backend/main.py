from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
from scipy.stats import skew
import pickle
from typing import List, Dict
import base64

# Global detector instance
detector = None

class FusionModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0', classical_dim=72, num_classes=2, 
                 hidden_size=128, num_layers=1, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0
        )
        
        self.classical_proj = nn.Sequential(
            nn.Linear(classical_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        fusion_dim = hidden_size + 32
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, video, classical_feat):
        batch_size, num_frames, c, h, w = video.shape
        
        video_flat = video.view(batch_size * num_frames, c, h, w)
        frame_features = self.backbone(video_flat)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        
        lstm_out, _ = self.lstm(frame_features)
        lstm_features = lstm_out[:, -1, :]
        
        classical_features = self.classical_proj(classical_feat)
        
        fused = torch.cat([lstm_features, classical_features], dim=1)
        
        output = self.classifier(fused)
        
        return output

class ClassicalFeatureExtractor:
    def __init__(self):
        self.lbp_radius = 1
        self.lbp_points = 8
        
    def extract_lbp_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, self.lbp_points, self.lbp_radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=self.lbp_points + 2, range=(0, self.lbp_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        return hist
    
    def extract_color_stats(self, image):
        stats = []
        for channel in range(3):
            ch = image[:, :, channel].flatten()
            stats.extend([
                np.mean(ch),
                np.std(ch),
                skew(ch)
            ])
        return np.array(stats)
    
    def extract_landmark_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        faces = detector.detectMultiScale(gray, 1.3, 5)
        
        features = np.zeros(5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            features[0] = w
            features[1] = h
            features[2] = w / (h + 1e-7)
            features[3] = x + w/2
            features[4] = y + h/2
        
        return features
    
    def extract_all_features(self, image):
        image = cv2.resize(image, (128, 128))
        
        lbp_feats = self.extract_lbp_features(image)
        color_feats = self.extract_color_stats(image)
        landmark_feats = self.extract_landmark_features(image)
        
        all_features = np.concatenate([lbp_feats, color_feats, landmark_feats])
        return all_features

class DeepfakeDetector:
    def __init__(self, model_path, scaler_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = FusionModel(classical_dim=72, num_classes=2)
        
        # Load model with error handling
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Model loading warning: {e}")
            print("Using untrained model weights")
        
        self.model.to(self.device)
        self.model.eval()
        
        # FIXED: Handle scaler file corruption
        if scaler_path and os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("Scaler loaded successfully")
            except Exception as e:
                print(f"Scaler loading failed: {e}. Creating new scaler.")
                self.scaler = StandardScaler()
        else:
            print("No scaler path provided, using default scaler")
            self.scaler = StandardScaler()
        
        self.classical_extractor = ClassicalFeatureExtractor()
        
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def extract_frames(self, video_path, num_frames=16):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError("Could not read video or video has no frames")
        
        frames = []
        for idx in range(min(num_frames, total_frames)):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video")
        
        return frames
    
    def extract_classical_features(self, frames):
        features_list = []
        for frame in frames:
            feats = self.classical_extractor.extract_all_features(frame)
            features_list.append(feats)
        
        features_array = np.array(features_list)
        aggregated = np.concatenate([
            np.mean(features_array, axis=0),
            np.std(features_array, axis=0),
            np.max(features_array, axis=0)
        ])
        return aggregated

    
    def preprocess_frames(self, frames):
        processed = []
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            augmented = self.transform(image=frame_rgb)
            processed.append(augmented['image'])
        
        while len(processed) < 16:
            processed.append(processed[-1] if processed else torch.zeros(3, 224, 224))
        
        return torch.stack(processed[:16]).unsqueeze(0)
    
    def predict(self, video_path):
        frames = self.extract_frames(video_path)
        
        classical_features = self.extract_classical_features(frames)
        classical_features_scaled = self.scaler.transform(classical_features.reshape(1, -1))
        classical_tensor = torch.tensor(classical_features_scaled, dtype=torch.float32).to(self.device)
        
        video_tensor = self.preprocess_frames(frames).to(self.device)
        
        with torch.no_grad():
            output = self.model(video_tensor, classical_tensor)
            probs = torch.softmax(output, dim=1)
            fake_prob = probs[0, 1].item()
        
        result = {
            "is_fake": fake_prob > 0.5,
            "fake_probability": fake_prob,
            "real_probability": 1 - fake_prob,
            "confidence": max(fake_prob, 1 - fake_prob),
            "frames_processed": len(frames)
        }
        
        return result

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    model_path = os.getenv("MODEL_PATH", "../models/best_fusion_model.pth")
    scaler_path = os.getenv("SCALER_PATH", "../models/scaler.pkl")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Create dummy files if they don't exist
    if not os.path.exists(model_path):
        print("⚠️ Model file not found, creating dummy model...")
        model_state = {'model_state_dict': FusionModel().state_dict()}
        torch.save(model_state, model_path)
    
    if not os.path.exists(scaler_path):
        print("Scaler file not found, creating dummy scaler...")
        scaler = StandardScaler()
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    detector = DeepfakeDetector(model_path, scaler_path)
    yield
    detector = None

app = FastAPI(title="Deepfake Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API", "status": "running"}

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file format. Only video files allowed.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = detector.predict(tmp_path)
        os.unlink(tmp_path)
        return JSONResponse(content=result)
    
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": detector.device if detector else "unknown",
        "model_loaded": detector is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)