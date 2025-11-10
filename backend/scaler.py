# In your scaler creation script
import pickle
from sklearn.preprocessing import StandardScaler

with open('D:/Coding/Projects/DeepfakeDetector-ML/models/classical_features.pkl', 'rb') as f:
    data = pickle.load(f)

scaler = StandardScaler()
scaler.fit(data['train_features'])

# Use pickle to save
with open('D:/Coding/Projects/DeepfakeDetector-ML/models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"Scaler saved with pickle! Shape: {data['train_features'].shape}")