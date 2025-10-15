"""
Test script for the Deepfake Detection API
Usage: python test_api.py <video_path>
"""

import requests
import sys
from pathlib import Path
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def analyze_video(video_path: str):
    """Test video analysis endpoint"""
    video_file = Path(video_path)
    
    if not video_file.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    print(f"\n📹 Analyzing video: {video_file.name}")
    print(f"File size: {video_file.stat().st_size / (1024*1024):.2f} MB")
    
    with open(video_file, 'rb') as f:
        files = {'file': (video_file.name, f, 'video/mp4')}
        
        print("\n⏳ Uploading and analyzing... (this may take a while)")
        response = requests.post(f"{API_URL}/analyze", files=files)
    
    print(f"\n📊 Response Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Analysis Results:")
        print("=" * 50)
        
        if result['success']:
            data = result['results']
            
            # Video metadata
            print("\n📹 Video Information:")
            metadata = data.get('video_metadata', {})
            print(f"  Duration: {metadata.get('duration', 0):.2f}s")
            print(f"  Resolution: {metadata.get('width', 0)}x{metadata.get('height', 0)}")
            print(f"  FPS: {metadata.get('fps', 0):.2f}")
            print(f"  Total Frames: {metadata.get('frame_count', 0)}")
            
            # Analysis results
            print("\n🔍 Detection Results:")
            print(f"  Final Verdict: {data['final_verdict']} {'🚨' if data['final_verdict'] == 'FAKE' else '✅'}")
            print(f"  Fake Score: {data['avg_fake_score']:.1%}")
            print(f"  Confidence: {data['avg_confidence']:.1%}")
            print(f"  Frames Analyzed: {data['frames_analyzed']}")
            print(f"  Faces Detected: {data['frames_with_faces']} ({data['face_detection_rate']:.0%})")
            print(f"  Score Volatility: {data['score_volatility']:.3f}")
            
            # Frame-by-frame results (first 5)
            print("\n📊 Frame-by-Frame Results (first 5):")
            for i, frame_result in enumerate(data['frame_results'][:5]):
                face_icon = "👤" if frame_result['face_detected'] else "❌"
                fake_icon = "🚨" if frame_result['label'] == 'FAKE' else "✅"
                print(f"  Frame {frame_result['frame_id']} ({frame_result['timestamp']:.1f}s): "
                      f"{fake_icon} {frame_result['label']} "
                      f"({frame_result['fake_score']:.1%}) "
                      f"{face_icon}")
            
            if len(data['frame_results']) > 5:
                print(f"  ... and {len(data['frame_results']) - 5} more frames")
            
            # Recommendation
            print("\n💡 Recommendation:")
            if data['final_verdict'] == 'FAKE':
                print("  ⚠️  This video shows signs of manipulation.")
                print("  Consider manual verification for important decisions.")
            elif data['final_verdict'] == 'REAL':
                print("  ✅ This video appears to be authentic.")
                print("  No significant manipulation detected.")
            else:
                print("  ⚠️  Unable to make confident determination.")
                print("  Manual review recommended.")
        else:
            print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
    else:
        print(f"\n❌ Error: {response.text}")

def main():
    print("=" * 50)
    print("🎭 Deepfake Detection API Test")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("\n❌ API is not healthy. Make sure the server is running.")
        print("Run: python backend/main.py")
        return
    
    # Analyze video if path provided
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        analyze_video(video_path)
    else:
        print("\n📝 Usage: python test_api.py <video_path>")
        print("Example: python test_api.py sample_video.mp4")

if __name__ == "__main__":
    main()