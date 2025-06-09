import streamlit as st
import cv2
import numpy as np

# Load YOLO model files
weights_path = r"C:\Users\ssman\OneDrive\Desktop\mlt_project\yolov4.weights"  # Download from: https://pjreddie.com/media/files/yolov4.weights
config_path = r"C:\Users\ssman\OneDrive\Desktop\mlt_project\yolov4 (1).cfg"  # Download from: https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg
labels_path = r"C:\Users\ssman\OneDrive\Desktop\mlt_project\coco.names" 
# Load class labels
with open(labels_path, "r") as f:
    labels = f.read().strip().split("\n")

# Load YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def detect_and_count(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == 0 and confidence > 0.5:
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, box_width, box_height) = box.astype("int")
                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))
                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    count = 0

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {count + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        count += 1

    return frame, count

st.title("Human Detection using YOLO")
option = st.sidebar.selectbox("Choose Input Source", ["Webcam", "Image", "Video"])

if option == "Webcam":
    st.write("Using Webcam")
    start = st.button("Start Webcam")

    if start:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        if not cap.isOpened():
            st.error("Error: Unable to access the webcam.")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.warning("No frames to read from webcam.")
                    break

                processed_frame, count = detect_and_count(frame)
                stframe.image(
                    cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                    caption=f"Total Persons: {count}",
                    channels="RGB",
                )

        cap.release()

elif option == "Image":
    st.write("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        processed_frame, count = detect_and_count(frame)

        st.image(
            cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
            caption=f"Detected {count} persons",
            channels="RGB",
        )

elif option == "Video":
    st.write("Upload a Video")
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        if not cap.isOpened():
            st.error("Error: Unable to read video file.")
        else:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, count = detect_and_count(frame)
                stframe.image(
                    cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                    caption=f"Total Persons: {count}",
                    channels="RGB",
                )

        cap.release()
#streamlit run human_streamlit.py