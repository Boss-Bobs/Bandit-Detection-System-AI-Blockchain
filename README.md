# Development of Anomaly-Detection Based Surveillance System for Bandit Attack Detection
A Final Year Project using Deep Learning and Computer Vision. <br>

![Project Demo](https://github.com/Boss-Bobs/Bandit-Detection-System-AI-Blockchain/blob/main/demo.gif?raw=true)

## 🛡️ Project Goal<br>
The objective of this system is to automate the detection of suspicious activities associated with bandit attacks in surveillance footage. By using Deep Learning, the system identifies "anomalies" (atypical behaviors or objects) that deviate from normal environmental patterns, providing a real-time visualization dashboard for security monitoring.

## 🚀 Key Features<br>
Deep Learning Inference: Uses a CNN-based architecture to classify frames as 'Normal' or 'Anomaly'.<br>
Real-time Dashboard: A web-based interface (hosted on Render) to visualize detection alerts.<br>
Data Pipeline: Custom scripts for data augmentation (datagen.py) to handle low-light or low-quality surveillance footage.<br>

## 📊 Dataset & Training Environment<br>
Because of the high computational requirements for analyzing surveillance video frames, the model was developed and trained on Kaggle using their cloud-based GPU infrastructure.<br>
Dataset Source: https://www.kaggle.com/datasets/bossbobs/futminna-dataset<br>
Data Characteristics: The dataset consists of [e.g., 2,000+] frames categorized into "Normal" surveillance footage and "Anomaly" (Bandit activity) scenarios.<br>
Preprocessing: I used data_gen.py to perform image normalization and augmentation (rotating, flipping, and brightness adjustment) to ensure the model can detect attacks in various lighting conditions.<br>

## 🛠️ Technical Stack & Architecture<br>
The system is divided into two main components: <br>
the Inference Engine (Python/AI) and the Monitoring Interface (Web Dashboard). <br>
Deep Learning Framework: TensorFlow / Keras (used for building and training the Anomaly Detection model). <br>
Computer Vision: OpenCV (used for frame extraction and real-time image processing). <br>
Backend Logic: Python (handles the scripts for training, data generation, and prediction). <br>
Frontend Dashboard: HTML5, CSS3, and JavaScript (hosted on Render for real-time visualization). <br>
Data Analysis:   NumPy and Pandas for handling array-based image data and logs. <br>
Blockchain: Solidity, Web3.py, Sepolia Testnet, MetaMask.

## ⚙️ Installation & Setup <br>
To run the analysis scripts or the dashboard locally, follow these steps: <br>
Clone the Repository: git clone https://github.com/Boss-Bobs/anomaly-detection-dashboard-2.git
cd anomaly-detection-dashboard-2 <br>
Install Dependencies: Make sure you have Python installed, then run: pip install -r requirements.txt
Download the Model: Ensure the trained model file (e.g., model.h5) is placed in the model_logic/ folder. <br>
Run Inference: To test the detection on a video sample: python model_logic/predict.py

## 🔗 Blockchain Integration (Data Integrity)
To ensure that surveillance logs are tamper-proof, this system integrates Blockchain technology. <br>
Network: Ethereum Sepolia Testnet. <br>
Purpose: When a "Bandit Attack" is detected, the system automatically generates a transaction hash. <br>
Immutable Logging: The timestamp, device ID, and detection confidence are stored on-chain, ensuring that security records cannot be altered or deleted by unauthorized parties. <br>
Smart Contract: [Optional: Link to your Etherscan/Sepolia transaction or contract address here].

## Web3 Integration
The system utilizes the Web3.py library to interact with an Ethereum Smart Contract. Upon detection of an anomaly, a signed transaction is sent to the Sepolia Testnet, ensuring that security logs are immutable and verifiable by stakeholders.

## 🔌 Hardware Implementation
To simulate a real-world surveillance environment, the system was deployed on edge hardware.

* **Processor:** Raspberry Pi 4B (4GB RAM)
* **Camera:** V380 IP Security Camera
* **Model Format:** TensorFlow Lite (.tflite) for edge optimization.
* **Inference Speed:** ~3-4 seconds per detection cycle.

> **Note on Performance:** The 3-4s inference time allows the system to analyze frames periodically, making it suitable for monitoring slow-moving security threats or "banditry" indicators in a controlled surveillance area.
> The total system latency includes the AI inference on the edge (RPi 4B) and the time required to broadcast the detection event to the Sepolia Testnet for immutable logging.

## 🔗 Smart Contract Details <br>
Contract Address: 0x279FcACc1eB244BBD7Be138D34F3f562Da179dd5 <br>
Network: Ethereum Sepolia Testnet <br>
Explorer: [View on Sepolia Etherscan] (https://sepolia.etherscan.io/address/0x279FcACc1eB244BBD7Be138D34F3f562Da179dd5) <br>
Key Functions: <br>
logAnomaly(string _folder, uint256 _frameIdx, string _error): This function records the specific folder, frame number, and error message of a bandit attack permanently on the blockchain.

### 🔗 Immutable Security Audit Trail
Every detection event is hashed and sent to the Ethereum Sepolia Testnet. 

**Contract Logic:**
- `logAnomaly`: Records the frame index and type of bandit activity.
- `block.timestamp`: Automatically records the exact time of the attack from the blockchain clock (cannot be faked by the user).
- **Security:** Only the system can write logs, but anyone with the contract address can verify them.

## 📊 Model Performance <br>
The system was evaluated based on its ability to distinguish between normal activity and simulated banditry scenarios. <br>
Accuracy: 79% <br>
Inference Speed: 3-4, making it suitable for real-time surveillance. <br>
Confusion Matrix: The model shows high precision in detecting [mention specific objects like weapons or unauthorized vehicles if applicable]. <br>

## 🚀 Future Improvements <br>
To make this system "production-ready" for real-world deployment, I plan to: <br>
Edge Deployment: Optimize the model using TensorFlow Lite for deployment on Raspberry Pi or Jetson Nano (to be used directly on cameras). <br>
SMS/Email Alerts: Integrate an API (like Twilio) to send instant notifications to security personnel when an attack is detected. <br>
Night Vision Optimization: Fine-tune the model with more infrared (IR) footage to improve detection in total darkness. <br>

## 👨‍🎓 Author
**Destiny Omojo Onoja**, **Jedidiah Ngbede**,  <br>
Final Year Project, <br>
Computer Engineering, <br>
Federal University of Technology, Minna (FUTMINNA), <br>
2025.
