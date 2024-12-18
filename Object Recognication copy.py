import cv2
import numpy as np
import time
from twilio.rest import Client
import streamlit as st

# Access secrets securely
account_sid = st.secrets["twilio"]["account_sid"]
auth_token = st.secrets["twilio"]["auth_token"]
twilio_number = st.secrets["twilio"]["twilio_number"]
target_number = st.secrets["twilio"]["target_number"]  # Ensure the target number includes the country code

client = Client(account_sid, auth_token)

# Load the object detection model
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Read class labels
classLabels = []
file_name = 'Label.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Parameters
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
last_sent_time = 0
send_interval = 5 * 60  # 5 minutes in seconds
animal_classes = [16, 17, 18, 19, 20, 21, 22, 23, 24]  # Example class IDs for animals

# Streamlit app
st.title("Live Object Detection")

# Start the webcam feed
st.text("Accessing webcam...")
video_feed = st.camera_input("Capture live feed")

if video_feed is not None:
    # Convert the uploaded image to a NumPy array
    bytes_data = video_feed.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Perform object detection
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd == 1:  # Class ID for person
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, font_scale, (0, 255, 0), 3)

                current_time = time.time()
                if current_time - last_sent_time > send_interval:
                    # Send SMS
                    message = client.messages.create(
                        body="Person detected!",
                        from_=twilio_number,
                        to=target_number
                    )
                    st.write(f"SMS sent: {message.sid}")
                    last_sent_time = current_time
            elif ClassInd in animal_classes:  # Check if the detected object is an animal
                cv2.rectangle(frame, boxes, (0, 255, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, font_scale, (255, 0, 0), 3)

                current_time = time.time()
                if current_time - last_sent_time > send_interval:
                    # Send SMS
                    message = client.messages.create(
                        body=f"Animal detected!",
                        from_=twilio_number,
                        to=target_number
                    )
                    st.write(f"SMS sent: {message.sid}")
                    last_sent_time = current_time

    # Display the processed frame
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
