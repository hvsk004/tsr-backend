import aiofiles
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse
import shutil
import os
from predict_module import predict_on_frame  # Ensure this module is correctly referenced
import uuid
import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import time  # Add at the top with other imports
from typing import List
from pydantic_settings import BaseSettings
from fastapi.middleware import Middleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import contextlib
import base64
import json

class Settings(BaseSettings):
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:5174",  # Adding the new origin
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ]
    CONFIDENCE_THRESHOLD: float = 0.6
    WEBCAM_FPS: float = 20.0
    WEBCAM_DEVICE: int = 0
    CLEANUP_HOURS: int = 24
    
    class Config:
        env_file = ".env"

settings = Settings()
limiter = Limiter(key_func=get_remote_address)

# Replace on_event with lifespan context manager
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Clean old files on startup"""
    current_time = time.time()
    for directory in [OUTPUTS_DIR, TEMP_DIR]:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path) and (current_time - os.path.getmtime(file_path)) > 24 * 3600:
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
    yield

app = FastAPI(lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Update CORS middleware with settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Detection-Results"],
)

@app.get("/")
async def root():
    return {"message": "Traffic Sign Recognition API"}

# Define directory paths
INPUTS_DIR = 'inputs'
INPUTS_IMAGES_DIR = os.path.join(INPUTS_DIR, 'images')
INPUTS_VIDEOS_DIR = os.path.join(INPUTS_DIR, 'videos', 'webcam')
OUTPUTS_DIR = 'outputs'
OUTPUTS_IMAGES_DIR = os.path.join(OUTPUTS_DIR, 'images')
OUTPUTS_VIDEOS_DIR = os.path.join(OUTPUTS_DIR, 'videos', 'webcam')
TEMP_DIR = 'temp'

# Create necessary directories if they do not exist
os.makedirs(INPUTS_IMAGES_DIR, exist_ok=True)
os.makedirs(INPUTS_VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_VIDEOS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def get_media_type(file_extension):
    """Helper function to determine media type based on file extension."""
    return {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo'
    }.get(file_extension)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form('both'),
    conf_threshold: float = Form(settings.CONFIDENCE_THRESHOLD),
    debug: bool = Form(False)
):
    """
    Endpoint to handle prediction requests for images and videos.

    :param file: Uploaded image or video file.
    :param mode: Which model(s) to use for labeling. Options: 'gtsdb', 'gtsrb', 'both'.
    :param conf_threshold: Minimum confidence threshold to filter predictions.
    :param debug: If True, include debug information.
    :return: Annotated image/video file.
    """
    # Determine the file type based on extension
    filename = file.filename
    file_extension = os.path.splitext(filename)[1].lower()

    if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        input_subdir = INPUTS_IMAGES_DIR
        output_subdir = OUTPUTS_IMAGES_DIR
        output_extension = '.jpg'
    elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        input_subdir = INPUTS_VIDEOS_DIR
        output_subdir = OUTPUTS_VIDEOS_DIR
        output_extension = '.mp4'
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an image or video file.")

    # Generate a unique filename to prevent conflicts
    unique_id = uuid.uuid4().hex
    input_filename = f"{unique_id}_{filename}"
    input_path = os.path.join(input_subdir, input_filename)

    # Save the uploaded file to the designated input directory
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Define the output path with the same unique identifier
    output_filename = f"{unique_id}_{os.path.splitext(filename)[0]}"
    output_path = os.path.join(output_subdir, output_filename)

    try:
        # Call the prediction function
        detection_results = predict_on_frame(
            input_path=input_path,
            output_path=os.path.join(output_subdir, os.path.splitext(output_filename)[0]),
            mode=mode,
            conf_threshold=conf_threshold,
            debug=debug
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Determine the response file path based on the type
    if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        response_path = output_path + '.jpg'
        # For images, return both the file and detection results
        if not os.path.exists(response_path):
            raise HTTPException(status_code=500, detail="Output file was not created.")
        
        # Return the file with custom headers containing the detection results
        return FileResponse(
            response_path,
            media_type=get_media_type(file_extension) or 'application/octet-stream',
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(response_path)}",
                "X-Detection-Results": json.dumps(detection_results)
            }
        )
    else:
        response_path = output_path + '.mp4'
        # For videos, just return the file as before
        if not os.path.exists(response_path):
            raise HTTPException(status_code=500, detail="Output file was not created.")
        
        return FileResponse(
            response_path,
            media_type=get_media_type(file_extension) or 'application/octet-stream',
            headers={"Content-Disposition": f"attachment; filename={os.path.basename(response_path)}"}
        )

@app.post("/predict/webcam")
@limiter.limit("5/minute")
async def predict_webcam(
    request: Request,  # Add Request parameter
    duration: int = Form(10),  # Duration to capture in seconds
    mode: str = Form('both'),
    conf_threshold: float = Form(settings.CONFIDENCE_THRESHOLD),
    debug: bool = Form(False)
):
    """
    Endpoint to handle prediction on live webcam feed.

    :param duration: Duration to capture the webcam feed in seconds.
    :param mode: Which model(s) to use for labeling. Options: 'gtsdb', 'gtsrb', 'both'.
    :param conf_threshold: Minimum confidence threshold to filter predictions.
    :param debug: If True, include debug information.
    :return: Annotated webcam video file.
    """
    # Define paths for webcam capture
    input_subdir = INPUTS_VIDEOS_DIR
    output_subdir = OUTPUTS_VIDEOS_DIR

    # Generate a unique filename
    unique_id = uuid.uuid4().hex
    input_filename = f"webcam_{unique_id}.mp4"
    input_path = os.path.join(input_subdir, input_filename)

    output_filename = f"webcam_{unique_id}"
    output_path = os.path.join(output_subdir, output_filename)

    # Initialize webcam capture
    cap = cv2.VideoCapture(settings.WEBCAM_DEVICE)  # 0 is typically the default webcam
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot access the webcam.")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = settings.WEBCAM_FPS  # You can adjust this as needed
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(input_path, fourcc, fps, (frame_width, frame_height))

    import time
    start_time = time.time()

    try:
        while int(time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
    except Exception as e:
        cap.release()
        out.release()
        raise HTTPException(status_code=500, detail=f"Error capturing webcam feed: {str(e)}")

    # Release the webcam and VideoWriter
    cap.release()
    out.release()

    try:
        # Perform prediction on the captured webcam video
        predict_on_frame(
            input_path=input_path,
            output_path=os.path.join(output_subdir, output_filename),
            mode=mode,
            conf_threshold=conf_threshold,
            debug=debug
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    response_path = output_path + '.mp4'

    # Clean up the input webcam video file
    os.remove(input_path)

    # Check if the output file was created
    if not os.path.exists(response_path):
        raise HTTPException(status_code=500, detail="Output webcam file was not created.")

    # Return the annotated webcam video
    return FileResponse(response_path, media_type='application/octet-stream', filename=os.path.basename(response_path))

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """
    WebSocket endpoint for live webcam predictions.
    Sends base64 encoded images with detection status information.
    """
    await websocket.accept()
    print("WebSocket connection accepted.")

    try:
        while True:
            try:
                print("Waiting for frame from client...")
                # Await for bytes data from the frontend
                data = await websocket.receive_bytes()
                print(f"Received frame from client. Size: {len(data)} bytes")

                # Convert bytes to NumPy array
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    print("Error: Received invalid frame data")
                    await websocket.send_json({
                        "error": "Received invalid image data.",
                        "hasDetections": False
                    })
                    continue

                print(f"Successfully decoded frame. Shape: {frame.shape}")

                # Generate unique filenames
                unique_id = uuid.uuid4().hex
                temp_input_path = os.path.join(TEMP_DIR, f"frame_{unique_id}.jpg")
                temp_output_path = os.path.join(TEMP_DIR, f"annotated_{unique_id}")

                # Save the received frame
                _, encoded_image = cv2.imencode('.jpg', frame)
                async with aiofiles.open(temp_input_path, 'wb') as out_file:
                    await out_file.write(encoded_image.tobytes())

                # Get the results from prediction
                has_detections = False
                try:
                    results = predict_on_frame(
                        input_path=temp_input_path,
                        output_path=temp_output_path,
                        mode='both',
                        conf_threshold=0.6,
                        debug=False
                    )
                    
                    # Check if there were any detections
                    if os.path.exists(temp_output_path + '.jpg'):
                        annotated_frame = cv2.imread(temp_output_path + '.jpg')
                        original_frame = cv2.imread(temp_input_path)
                        # If frames are different, it means detections were drawn
                        has_detections = not np.array_equal(original_frame, annotated_frame)
                        
                except Exception as e:
                    await websocket.send_json({
                        "error": f"Prediction failed: {str(e)}",
                        "hasDetections": False
                    })
                    os.remove(temp_input_path)
                    continue

                annotated_path = temp_output_path + '.jpg'
                if not os.path.exists(annotated_path):
                    await websocket.send_json({
                        "error": "Annotated frame was not created.",
                        "hasDetections": False
                    })
                    os.remove(temp_input_path)
                    continue

                # Read and encode the annotated image as base64
                async with aiofiles.open(annotated_path, 'rb') as annotated_file:
                    annotated_bytes = await annotated_file.read()
                    base64_image = base64.b64encode(annotated_bytes).decode('utf-8')

                # Send as JSON with base64 data URL and detection status
                try:
                    await websocket.send_json({
                        "image": f"data:image/jpeg;base64,{base64_image}",
                        "hasDetections": has_detections
                    })
                except WebSocketDisconnect:
                    print("WebSocket disconnected while sending.")
                    break
                except Exception as send_error:
                    print(f"Failed to send annotated frame: {send_error}")

                # Clean up temporary files
                os.remove(temp_input_path)
                os.remove(annotated_path)

            except WebSocketDisconnect:
                print("WebSocket connection closed by client.")
                break
            except Exception as e:
                error_detail = f"Unexpected error: {str(e)}"
                try:
                    await websocket.send_json({
                        "error": error_detail,
                        "hasDetections": False
                    })
                except Exception as send_exception:
                    print(f"Failed to send error message: {send_exception}")
                print(error_detail)
                break

    except Exception as outer_exception:
        print(f"WebSocket handler encountered an error: {outer_exception}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass  # Ignore any errors during close
        print("WebSocket connection closed.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)