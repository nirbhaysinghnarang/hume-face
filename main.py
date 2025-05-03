import asyncio
import cv2
import numpy as np
import os
import time
from hume import AsyncHumeClient
from hume.expression_measurement.stream import Config
from hume.expression_measurement.stream.socket_client import StreamConnectOptions
from hume.expression_measurement.stream.types import StreamFace
import json
from datetime import datetime

# Configuration
OUTPUT_DIR = "expression_results"
API_KEY = "<enter here>"
CAMERA_ID = 0  # Usually 0 for built-in webcam, try other integers if not working
FRAME_INTERVAL = 0.5  # Seconds between analyzed frames

# Image compression settings
COMPRESSION_QUALITY = 40  # JPEG quality (0-100, lower means more compression)
RESIZE_FACTOR = 0.4  # Resize factor (0.5 = half size, 0.75 = 75% of original size)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compress_image(image):
    """
    Compress the image by resizing and JPEG compression
    """
    # Resize the image
    h, w = image.shape[:2]
    new_h, new_w = int(h * RESIZE_FACTOR), int(w * RESIZE_FACTOR)
    resized = cv2.resize(image, (new_w, new_h))
    
    # Compress using JPEG encoding parameters
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, COMPRESSION_QUALITY]
    _, compressed = cv2.imencode('.jpg', resized, encode_params)
    
    # Decode back to image
    return cv2.imdecode(compressed, cv2.IMREAD_COLOR)

async def analyze_face_with_hume(face_image):
    """
    Send a compressed frame to Hume API and get expression analysis
    """
    # Compress the image before sending
    compressed_image = compress_image(face_image)
    
    client = AsyncHumeClient(api_key=API_KEY)
    model_config = Config(face=StreamFace())
    stream_options = StreamConnectOptions(config=model_config)
    
    # Create a temporary file for the compressed image
    temp_frame_path = os.path.join(OUTPUT_DIR, "temp_frame.jpg")
    cv2.imwrite(temp_frame_path, compressed_image)
    
    # Get file size for logging
    file_size = os.path.getsize(temp_frame_path) / 1024  # Size in KB
    
    try:
        print(f"Sending compressed image ({file_size:.2f} KB)")
        async with client.expression_measurement.stream.connect(options=stream_options) as socket:
            result = await socket.send_file(temp_frame_path)
            return result
    except Exception as e:
        print(f"Error in Hume API call: {e}")
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)

async def process_webcam():
    """
    Process webcam feed in real-time and analyze expressions
    """
    # Open the webcam
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with ID {CAMERA_ID}")
        return
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Webcam initialized: {width}x{height} @ {fps}fps")
    print(f"Analyzing frames every {FRAME_INTERVAL} seconds")
    print(f"Compression settings: resize to {RESIZE_FACTOR*100}%, JPEG quality {COMPRESSION_QUALITY}")
    print("Press 'q' to quit")
    
    # Prepare results storage
    session_results = {
        "session_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frames": []
    }
    
    last_analysis_time = 0
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Display the frame
            cv2.imshow('Webcam Feed', frame)
            
            # Check if it's time for a new analysis
            current_time = time.time()
            if (current_time - last_analysis_time) >= FRAME_INTERVAL:
                print(f"Analyzing frame {frame_count}")
                start = time.time()                
                # Analyze the frame with Hume API
                hume_result = await analyze_face_with_hume(frame)
                print(f"Latency: {time.time()-start:.3f} seconds")
                
                if hume_result:
                    frame_result = {
                        "frame_idx": frame_count,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        "expression_data": hume_result
                    }
                    session_results["frames"].append(frame_result)
                    
                    # Print a summary of the results
                    try:
                        faces = hume_result.face.predictions
                        if faces:
                            for i, face in enumerate(faces):
                                emotions = face.emotions
                                top_emotion = max(emotions, key=lambda x: x.score)
                                print(f"Face {i+1}: Top emotion - {top_emotion.name} ({top_emotion.score:.2f})")
                        else:
                            print("No faces detected in this frame")
                    except (AttributeError, IndexError) as e:
                        print(f"Error parsing results: {e}")
                
                last_analysis_time = current_time
                frame_count += 1
                
                # Save the current frame with timestamp (using original, not compressed)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                frame_filename = os.path.join(OUTPUT_DIR, f"frame_{timestamp}.jpg")
                cv2.imwrite(frame_filename, frame)
            
            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Save the session results to a JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(OUTPUT_DIR, f"session_results_{timestamp}.json")
        with open(results_file, "w") as f:
            json.dump(session_results, f, indent=2)
        
        print(f"Session complete. Results saved to {results_file}")

if __name__ == "__main__":
    try:
        asyncio.run(process_webcam())
    except KeyboardInterrupt:
        print("Process interrupted by user")
