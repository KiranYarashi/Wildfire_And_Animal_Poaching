import cv2
import matplotlib.pyplot as plt
from twilio.rest import Client
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, VideoFrame

# Twilio credentials
account_sid = st.secrets["twilio"]["account_sid"]
auth_token = st.secrets["twilio"]["auth_token"]
twilio_number = st.secrets["twilio"]["twilio_number"]
target_number = st.secrets["twilio"]["target_number"]  # Ensure the target number includes the country code

client = Client(account_sid, auth_token)

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = 'Label.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

send_interval = 5 * 60  # 5 minutes in seconds
animal_classes = {16, 17, 18, 19, 20, 21, 22, 23, 24}  # Example class IDs for animals

def send_sms(body):
    message = client.messages.create(
        body=body,
        from_=twilio_number,
        to=target_number
    )
    print(f"SMS sent: {message.sid}")

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_sent_time = 0

    def transform(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.55)

        current_time = time.time()
        if len(ClassIndex) != 0:
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                label = classLabels[ClassInd - 1]
                if ClassInd == 1:  # Class ID for person
                    cv2.rectangle(img, boxes, (255, 0, 0), 2)
                    cv2.putText(img, label, (boxes[0] + 10, boxes[1] + 40), font, font_scale, (0, 255, 0), 3)
                    if current_time - self.last_sent_time > send_interval:
                        send_sms("Person detected!")
                        self.last_sent_time = current_time
                elif ClassInd in animal_classes:  # Check if the detected object is an animal
                    cv2.rectangle(img, boxes, (0, 255, 0), 2)
                    cv2.putText(img, label, (boxes[0] + 10, boxes[1] + 40), font, font_scale, (255, 0, 0), 3)
                    if current_time - self.last_sent_time > send_interval:
                        send_sms("Animal detected!")
                        self.last_sent_time = current_time

        return VideoFrame.from_ndarray(img, format="bgr24")

st.title("Object Detection with Twilio Notifications")

# Check if running in a local environment or deployed
if st.secrets.get("is_deployed", False):
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
    else:
        frame_placeholder = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image")
                break

            frame = VideoTransformer().transform(VideoFrame.from_ndarray(frame, format="bgr24")).to_ndarray(format="bgr24")
            frame_placeholder.image(frame, channels="BGR")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
