from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
from scipy.stats import skew
import pickle
import base64

detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    model_path = os.getenv("MODEL_PATH", "../models_all/best_fusion_model.pth")
    scaler_path = os.getenv("SCALER_PATH", "../models_all/scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
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

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    
    def forward(self, video, classical_feat, return_features=False):
        batch_size, num_frames, c, h, w = video.shape
        
        video_flat = video.view(batch_size * num_frames, c, h, w)
        frame_features = self.backbone(video_flat)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        
        lstm_out, _ = self.lstm(frame_features)
        lstm_features = lstm_out[:, -1, :]
        
        classical_features = self.classical_proj(classical_feat)
        
        fused = torch.cat([lstm_features, classical_features], dim=1)
        
        output = self.classifier(fused)
        
        if return_features:
            return output, lstm_features, classical_features, frame_features
        
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
        gray_small = cv2.resize(gray, (128, 128))
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        faces = detector.detectMultiScale(gray_small, 1.3, 3)
        
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

class FaceDetector:
    def __init__(self, method='haar'):
        self.method = method
        if method == 'haar':
            self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_and_crop(self, frame):
        if self.method == 'haar':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                return frame[y:y+h, x:x+w]
        
        return frame

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, output, target_class):
        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        if cam.ndim > 2:
            cam = np.mean(cam, axis=0)
        
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

class DeepfakeDetector:
    def __init__(self, model_path, scaler_path=None):
        self.model = FusionModel(classical_dim=72, num_classes=2)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = StandardScaler()
        
        self.classical_extractor = ClassicalFeatureExtractor()
        self.face_detector = FaceDetector(method='haar')
        
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
            raise ValueError("Could not read video")
        
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames / num_frames
            indices = [int(i * step) for i in range(num_frames)]
        
        frames = []
        face_crops = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                face = self.face_detector.detect_and_crop(frame)
                frames.append(frame)
                face_crops.append(face)
        
        cap.release()
        
        if len(face_crops) == 0:
            raise ValueError("No faces detected in video")
        
        return frames, face_crops
    
    def extract_classical_features(self, face_crops):
        features_list = []
        for face in face_crops:
            feats = self.classical_extractor.extract_all_features(face)
            features_list.append(feats)
        
        features_array = np.array(features_list)
        aggregated = np.concatenate([
            np.mean(features_array, axis=0),
            np.std(features_array, axis=0),
            np.max(features_array, axis=0)
        ])
        
        return aggregated
    
    def preprocess_frames(self, face_crops):
        processed = []
        for face in face_crops:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            augmented = self.transform(image=face_rgb)
            processed.append(augmented['image'])
        
        while len(processed) < 16:
            processed.append(processed[-1])
        
        return torch.stack(processed[:16]).unsqueeze(0)
    
    def generate_explainability(self, frames, face_crops, prediction):
        heatmaps = []
        
        gradcam = GradCAM(self.model, self.model.backbone.blocks[-1][-1])
        
        video_tensor = self.preprocess_frames(face_crops).to(device)
        classical_feat = torch.zeros(1, 72).to(device)
        
        with torch.enable_grad():
            output, _, _, _ = self.model(video_tensor, classical_feat, return_features=True)
            target_class = 1 if prediction > 0.5 else 0
            cam = gradcam.generate(output, target_class)
        
        for idx, (frame, face) in enumerate(zip(frames[:8], face_crops[:8])):
            cam_resized = cv2.resize(cam, (face.shape[1], face.shape[0]))
            cam_uint8 = np.uint8(255 * cam_resized)
            
            if cam_uint8.ndim == 2:
                heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            else:
                cam_gray = cv2.cvtColor(cam_uint8, cv2.COLOR_BGR2GRAY) if cam_uint8.shape[2] == 3 else cam_uint8[:,:,0]
                heatmap = cv2.applyColorMap(cam_gray, cv2.COLORMAP_JET)
            
            overlay = cv2.addWeighted(face, 0.6, heatmap, 0.4, 0)
            
            _, buffer = cv2.imencode('.jpg', overlay)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            heatmaps.append(img_base64)
        
        return heatmaps
    
    def predict(self, video_path, return_explainability=False):
        frames, face_crops = self.extract_frames(video_path)

        classical_features = self.extract_classical_features(face_crops)
        classical_features_scaled = self.scaler.transform(classical_features.reshape(1, -1))
        classical_tensor = torch.tensor(classical_features_scaled, dtype=torch.float32).to(device)

        video_tensor = self.preprocess_frames(face_crops).to(device)

        with torch.no_grad():
            output = self.model(video_tensor, classical_tensor)
            probs = torch.softmax(output, dim=1)
            fake_prob = float(probs[0, 1].item())

        real_prob = 1 - fake_prob

        result = {
            "label": "FAKE" if fake_prob > 0.5 else "REAL",
            "is_fake": bool(fake_prob > 0.5),
            "fake_probability": round(fake_prob, 2),
            "real_probability": round(real_prob, 2),
            "confidence": round(max(fake_prob, real_prob), 2),
            "facial_consistency": round(real_prob, 2),
            "temporal_stability": round(real_prob, 2),
            "artifact_detection": round(real_prob, 2),
            "overall_score": round(real_prob, 2)
        }

        if return_explainability:
            heatmaps = self.generate_explainability(frames, face_crops, fake_prob)
            result["explainability"] = heatmaps

        return result



@app.get("/")
async def root():
    return {"message": "Deepfake Detection API", "status": "running"}

@app.post("/predict")
async def predict_video(file: UploadFile = File(...), explainability: bool = False):
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file format. Only video files allowed.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = detector.predict(tmp_path, return_explainability=explainability)
        os.unlink(tmp_path)
        return JSONResponse(content=result)
    
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": detector is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)