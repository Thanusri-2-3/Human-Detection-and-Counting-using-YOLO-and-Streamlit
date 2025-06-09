# Human Detection and Counting using YOLOv4 and Streamlit

This project demonstrates real-time human detection and counting using the **YOLOv4** object detection model integrated with a **Streamlit** web application. It supports live webcam feeds, image uploads, and video file processing.

## 🚀 Features

- Real-time human detection using **YOLOv4**
- Count number of people in the frame
- Streamlit web interface with three modes:
  - **Webcam**
  - **Image Upload**
  - **Video Upload**
- Visual output with bounding boxes and labels
- Non-Maximum Suppression (NMS) to eliminate duplicate detections

---

## 🖼️ Example Output

You’ll see output frames with green bounding boxes around each detected person and a label like “Person 1”, “Person 2”, etc.

---

## 🛠️ Requirements

Install the dependencies using pip:

```bash
pip install streamlit opencv-python numpy
Make sure you also have Python 3.7+ installed.

structure:

├── human_streamlit.py         # Main Streamlit app
├── yolov4.weights             # YOLOv4 weights file
├── yolov4.cfg                 # YOLOv4 configuration file
├── coco.names                 # COCO labels (contains "person")
└── README.md                  # Project documentation

Make sure the paths in human_streamlit.py match your file locations.

To start the Streamlit app, use:

streamlit run human_streamlit.py

The Streamlit interface will open in your browser. Choose your desired input method from the sidebar:

Webcam: Starts your camera for live detection.
Image: Upload an image (JPG, PNG).
Video: Upload a video (MP4, AVI, MOV).

🔍 How It Works:

The YOLOv4 model is loaded with pre-trained COCO weights.
Only detections of class "person" are used for bounding boxes and counting.
Uses OpenCV’s cv2.dnn module for running detections.
Non-Maximum Suppression ensures cleaner results.
Frames are rendered in real-time using Streamlit’s st.image.

🧠 Model Info:

Model: YOLOv4 (You Only Look Once)
Dataset: COCO
Detected Class: Only person (class_id = 0)

🙌 Acknowledgments:

YOLOv4 by AlexeyAB
Streamlit
OpenCV

📃 License
This project is licensed under the MIT License.
