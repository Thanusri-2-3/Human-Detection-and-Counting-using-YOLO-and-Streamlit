# Human-Detection-and-Counting-using-YOLO-and-Streamlit
This project demonstrates real-time human detection and counting in video streams using YOLO (You Only Look Once) object detection and an interactive Streamlit web interface.
🚀 Features
Real-time person detection using YOLOv4
Easy-to-use web interface with Streamlit
Supports:
Webcam
Image upload
Video upload
Highlights and counts detected persons in the frame
🖥️ Tech Stack
Python
OpenCV (cv2)
NumPy
Streamlit
YOLOv4 model files (.cfg, .weights, .names)
🛠️ Setup Instructions
1. Clone the repository
git clone https://github.com/yourusername/yolo-human-detection.git
cd yolo-human-detection
2. Install dependencies
pip install -r requirements.txt
3. Download YOLOv4 Files
Please download the following files and place them in your project directory:

yolov4.weights 👉 Download from official YOLO GitHub https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights

yolov4.cfg 👉 View/download from here https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights (Click "Raw" → Right-click → "Save As")

coco.names (class labels) https://github.com/pjreddie/darknet/blob/master/data/coco.names 👉 Download coco.namess)

4. Run the app
streamlit run human_streamlit.py

📝 Notes
The app will need access to your webcam for real-time detection.
Detection is focused on people only using class ID 0 from COCO dataset.
For large video files, processing may take some time depending on hardware.
