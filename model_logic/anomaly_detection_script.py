import cv2
import numpy as np
import time
import tensorflow as tf
import os
from flask import Flask, Response, jsonify, send_file
import threading
from collections import deque 
from werkzeug.utils import secure_filename
import csv
import datetime
import hashlib
from web3 import Web3 
import queue
from hexbytes import HexBytes
import sys # Added for robustness

# --- CONFIGURATION ---
RTSP_URL = 'rtsp://192.162.235.162:554/live/ch00_1' 
MODEL_PATH = '/home/anomalyproject/anomaly/models/new_3DCAE_model.tflite'
ANOMALY_LOG_FILE = '/home/anomalyproject/anomaly/anomaly_history.csv' 
ANOMALY_FRAME_DIR = '/home/anomalyproject/anomaly/anomaly_frames/'
os.makedirs(ANOMALY_FRAME_DIR, exist_ok=True) 

SEQ_LENGTH = 16
IMG_SIZE = 128
NUM_THREADS = 4 
THRESHOLD = 0.045 
FONT = cv2.FONT_HERSHEY_SIMPLEX
FLASK_PORT = 5000 

# GStreamer Pipeline for efficient RTSP decoding on RPi
GSTREAMER_PIPELINE = (
    f'rtspsrc location="{RTSP_URL}" latency=0 ! '
    f'rtph264depay ! h264parse ! avdec_h264 ! '
    f'videoconvert ! videorate ! '
    f'appsink'
)

# --- BLOCKCHAIN CONFIGURATION (Sepolia Testnet) ---
INFURA_URL = "https://sepolia.infura.io/v3/8742554fd5c94c549cb8b4117b076e7a"
CONTRACT_ADDRESS = "0x279FcACc1eB244BBD7Be138D34F3f562Da179dd5"
WALLET_ADDRESS = "0xa8824b2E3b176bBc530a6a6B54f08beb0447C21e"
PRIVATE_KEY = os.getenv('ETH_PRIVATE_KEY') 

# ABI for logAnomaly (string folder, uint256 frameIdx, string error)
CONTRACT_ABI = [
    {"inputs": [{"internalType": "string", "name": "_folder", "type": "string"}, 
                {"internalType": "uint256", "name": "_frameIdx", "type": "uint256"}, 
                {"internalType": "string", "name": "_error", "type": "string"}], 
     "name": "logAnomaly", "outputs": [], "stateMutability": "nonpayable", "type": "function"}
]

# --- FLASK APP INITIALIZATION (FIX for NameError: name 'app' is not defined) ---
app = Flask(__name__)
# Push context for use with Web3/TFLite if required
app.app_context().push() 
# ---------------------------------------------------------------------------------


# --- Initialize Web3 and Contract (Global Objects) ---
try:
    w3 = Web3(Web3.HTTPProvider(INFURA_URL))
    contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
    print(f"Web3 connected: {w3.is_connected()}")
except Exception as e:
    print(f"Web3 initialization failed: {e}")
    w3 = None
    contract = None
# ---------------------------------------------------

# --- GLOBAL STATE (Shared between threads) ---
lock = threading.Lock() 
global_anomaly_score = 0.0
global_anomaly_detected = False
global_frame = None 
inference_buffer = deque(maxlen=SEQ_LENGTH) 
TRANSACTION_QUEUE = queue.Queue() 


# --- TFLITE INITIALIZATION ---
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter._interpreter.SetNumThreads(NUM_THREADS)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    INPUT_SHAPE = tuple(input_details[0]['shape']) 
    print("TFLite model loaded.")
except Exception as e:
    print(f"FATAL WARNING: TFLite setup failed ({e}). Check model path and TensorFlow Lite installation. Error: {e}")
    # Exit gracefully if TFLite cannot be initialized
    sys.exit(1)


# --- HELPER FUNCTIONS ---
def preprocess_frame(frame):
    """Resizes, converts to float32, and normalizes (0-1)."""
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_float = frame_resized.astype(np.float32)
    return frame_float / 255.0

def calculate_mse(original_seq, reconstructed_seq):
    """Calculates Mean Squared Error (MSE) over the sequence."""
    return np.mean((original_seq - reconstructed_seq) ** 2)


# --- BLOCKCHAIN TRANSACTION FUNCTION ---
def log_to_blockchain(timestamp, score, frame_hash):
    """Logs the anomaly event to the blockchain."""
    if not w3 or not w3.is_connected():
        print("Blockchain Tx Failed: Web3 not connected.")
        return

    if not PRIVATE_KEY:
        print("Blockchain Tx Failed: Private key not set (ETH_PRIVATE_KEY environment variable missing).")
        return

    try:
        # Convert hash to a large integer (uint256) for the _frameIdx field
        frame_hash_int = int(frame_hash[:64], 16) 
        
        # Build the transaction
        tx = contract.functions.logAnomaly(
            timestamp,         # Mapped to _folder (string)
            frame_hash_int,    # Mapped to _frameIdx (uint256) 
            score              # Mapped to _error (string)
        ).build_transaction({
            'from': WALLET_ADDRESS,
            'nonce': w3.eth.get_transaction_count(WALLET_ADDRESS),
            'gas': 2000000, 
            'gasPrice': w3.eth.gas_price,
            'chainId': w3.eth.chain_id
        })
        
        # Sign and Send
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        print(f"Blockchain Tx SUCCESS: {HexBytes(tx_hash).hex()}")
        
    except Exception as e:
        print(f"Blockchain Tx FAILED: {e}")


# --- BLOCKCHAIN THREAD LOOP (Consumer) ---
def blockchain_logging_loop():
    """Continuously pulls anomaly data from the queue and logs it to the blockchain."""
    print("\nStarting Blockchain Logging Loop (Consumer)...")
    while True:
        try:
            # Block until an item is available (timeout=5s)
            timestamp, score, frame_hash = TRANSACTION_QUEUE.get(timeout=5)
            
            log_to_blockchain(timestamp, score, frame_hash)
            
            TRANSACTION_QUEUE.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in Blockchain Loop: {e}")
            # Do not call task_done if we failed to process the item entirely
            time.sleep(1)


# --- 1. FAST CAPTURE/STREAMING LOOP ---
def capture_and_stream_loop():
    global global_frame, global_anomaly_score, global_anomaly_detected, inference_buffer
    
    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER) 
    
    if not cap.isOpened():
        print("FATAL: Could not open RTSP stream. Trying default FFmpeg.")
        cap = cv2.VideoCapture(RTSP_URL)
        if not cap.isOpened():
            print("FATAL: Could not open RTSP stream via default backend either."); return

    print("\nStarting FAST Capture/Stream Loop...")

    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(1); continue
        
        frame_h, frame_w = frame.shape[:2]
        
        inference_buffer.append(preprocess_frame(frame))
        
        with lock:
            anomaly_detected = global_anomaly_detected
            anomaly_score = global_anomaly_score

        # Overlay Generation
        if anomaly_detected:
            color = (0, 0, 255)  # RED
            status_text = "ANOMALY DETECTED!"
            cv2.rectangle(frame, (0, 0), (frame_w, frame_h), color, 10)
        else:
            color = (0, 255, 0)  # GREEN
            status_text = "NORMAL"

        score_text = f"Score: {anomaly_score:.4f}"
        
        cv2.putText(frame, score_text, (20, 50), FONT, 1, color, 3, cv2.LINE_AA)
        cv2.putText(frame, status_text, (20, frame_h - 20), FONT, 1.2, color, 4, cv2.LINE_AA)
        
        with lock:
            global_frame = frame.copy()


# --- 2. SLOW INFERENCE LOOP (PRODUCER) ---
def inference_loop():
    global global_anomaly_score, global_anomaly_detected, inference_buffer

    print("\nStarting SLOW TFLite Inference Loop (Producer)...")

    while True:
        if len(inference_buffer) == SEQ_LENGTH:
            start_time = time.time()
            
            input_sequence = np.array(inference_buffer, dtype=np.float32)
            input_data = np.expand_dims(input_sequence, axis=0) 

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            reconstructed_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Calculate Score
            reconstructed_sequence = reconstructed_data[0]
            current_score = calculate_mse(input_sequence, reconstructed_sequence)
            
            anomaly_detected = current_score > THRESHOLD

            if anomaly_detected:
                # LOGGING AND SAVING ANOMALY DATA
                current_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                frame_filename = f"{current_ts.replace(':', '-')}.jpg"
                frame_path = os.path.join(ANOMALY_FRAME_DIR, frame_filename)
                
                with lock:
                    cv2.imwrite(frame_path, global_frame.copy())
                
                with open(frame_path, 'rb') as f:
                    frame_hash = hashlib.sha256(f.read()).hexdigest()
                    
                # Log to CSV (include the hash)
                with open(ANOMALY_LOG_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([current_ts, f"{current_score:.6f}", "ANOMALY", frame_path, frame_hash])

                # Enqueue the transaction data (THIS IS INSTANTANEOUS)
                tx_data = (current_ts, f"{current_score:.6f}", frame_hash)
                TRANSACTION_QUEUE.put(tx_data)
                
            # Update global status variables safely
            with lock:
                global_anomaly_score = current_score
                global_anomaly_detected = anomaly_detected

            # Log performance 
            inference_time = time.time() - start_time
            status_text = "ANOMALY DETECTED!" if global_anomaly_detected else "NORMAL"
            print(f"[{time.strftime('%H:%M:%S')}] Inference Time: {inference_time:.2f}s | Score: {global_anomaly_score:.6f} | Status: {status_text}")
        else:
            time.sleep(0.1)
            

# --- FLASK DATA ENDPOINTS (API for Render Dashboard) ---
@app.route("/api/rpi/history")
def rpi_history():
    """Returns the anomaly history CSV data as JSON."""
    history = []
    try:
        with open(ANOMALY_LOG_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append(row)
        return jsonify({
            'success': True,
            'history': history
        })
    except FileNotFoundError:
        return jsonify({'success': False, 'error': 'History file not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/api/rpi/image/<filename>")
def rpi_image(filename):
    """Serves a single anomaly image from the RPi."""
    safe_filename = secure_filename(filename)
    image_path = os.path.join(ANOMALY_FRAME_DIR, safe_filename)
    
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404


# --- FLASK STREAMING SETUP ---
def generate():
    global global_frame, lock
    
    while True:
        with lock:
            if global_frame is None:
                time.sleep(0.01)
                continue

            (flag, encodedImage) = cv2.imencode(".jpg", global_frame)

            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    return "Anomaly Detector Core Running. Use /video_feed or Render Dashboard."


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    try:
        # Start all three threads
        t_capture = threading.Thread(target=capture_and_stream_loop)
        t_inference = threading.Thread(target=inference_loop)
        t_blockchain = threading.Thread(target=blockchain_logging_loop)

        t_capture.daemon = True
        t_inference.daemon = True
        t_blockchain.daemon = True
        
        t_capture.start()
        t_inference.start()
        t_blockchain.start()
        
        print(f"\n🎥 Detector Core and Streaming Started.")
        
        # Run Flask
        app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, threaded=False)
        
    except KeyboardInterrupt:
        print("\nScript terminated by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Detector core stopped.")
