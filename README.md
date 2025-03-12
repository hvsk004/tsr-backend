A FastAPI-based backend service for real-time traffic sign detection and recognition using YOLOv8 models. This system can process both images and video streams, including real-time webcam feed.

## Features

- 🚦 Traffic sign detection using GTSDB (German Traffic Sign Detection Benchmark) model
- 🎯 Sign classification using GTSRB (German Traffic Sign Recognition Benchmark) model
- 📷 Support for image uploads
- 🎥 Video file processing
- 💻 Real-time webcam feed analysis
- 🔄 WebSocket support for live streaming
- ⚡ Fast processing with optimized model inference
- 🔒 Rate limiting for API endpoints

## Prerequisites

- Python 3.8+
- OpenCV
- FastAPI
- Ultralytics YOLOv8
- CUDA-capable GPU (recommended for better performance)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hvsk004/tsr-backend
cd tsr-backend
```

2. Create and activate a virtual environment:
```bash
python -m venv tsr
source tsr/bin/activate  # On Windows: .\tsr\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the model files:
- Place `gtsdb.pt` in the root directory (GTSDB detection model)
- Place `gtsrb.pt` in the root directory (GTSRB classification model)

## Environment Variables

Create a `.env` file in the root directory with the following options:
```env
CORS_ORIGINS=["http://localhost:5173","http://localhost:5174","http://localhost:5500"]
CONFIDENCE_THRESHOLD=0.6
WEBCAM_FPS=20.0
WEBCAM_DEVICE=0
CLEANUP_HOURS=24
```

## API Endpoints

### REST Endpoints

- `POST /predict`
  - Upload image/video files for processing
  - Supports various formats (jpg, jpeg, png, mp4, avi)
  - Returns annotated media with detected signs

- `POST /predict/webcam`
  - Captures and processes webcam feed
  - Specify duration and processing parameters
  - Returns annotated video file

### WebSocket Endpoint

- `WS /ws/predict`
  - Real-time webcam streaming and processing
  - Receives frames as binary data
  - Returns base64 encoded annotated frames with detection results

## Usage Examples

### Image Processing
```python
import requests

url = 'http://localhost:8000/predict'
files = {'file': open('traffic_sign.jpg', 'rb')}
data = {'mode': 'both', 'conf_threshold': 0.6}
response = requests.post(url, files=files, data=data)
```

### WebSocket Streaming
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/predict');
ws.binaryType = 'arraybuffer';

// Send frame
ws.send(frameData);

// Receive processed frame
ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    if (response.hasDetections) {
        // Handle detected signs
    }
};
```

## Development

1. Start the development server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
tsr-backend/
├── main.py                 # FastAPI application and endpoints
├── predict_module.py       # Traffic sign detection logic
├── requirements.txt        # Project dependencies
├── .env                   # Environment variables
├── inputs/                # Input media storage
├── outputs/               # Processed media storage
└── temp/                  # Temporary files
```

## Rate Limiting

- `/predict`: 10 requests per minute
- `/predict/webcam`: 5 requests per minute
- WebSocket connections are not rate-limited

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
