import streamlit as st
import cv2
import av
import time
from twilio.rest import Client
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Access secrets securely
account_sid = st.secrets["twilio"]["account_sid"]
auth_token = st.secrets["twilio"]["auth_token"]
twilio_number = st.secrets["twilio"]["twilio_number"]
target_number = st.secrets["twilio"]["target_number"]  # Ensure the target number includes the country code

client = Client(account_sid, auth_token)

# Detection settings
st.title("Animal and Person Detection System")
st.sidebar.header("Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.55)
notification_interval = st.sidebar.number_input("Notification Interval (minutes)", 1, 60, 5) * 60  # Convert to seconds

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
with open('Label.txt', 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

animal_classes = [16, 17, 18, 19, 20, 21, 22, 23, 24]  # Example class IDs for animals

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_sent_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        ClassIndex, confidence, bbox = model.detect(img, confThreshold=confidence_threshold)

        if len(ClassIndex) != 0:
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                if ClassInd == 1:  # Class ID for person
                    cv2.rectangle(img, boxes, (255, 0, 0), 2)
                    cv2.putText(img, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40),
                                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

                    current_time = time.time()
                    if current_time - self.last_sent_time > notification_interval:
                        # Send SMS
                        message = client.messages.create(
                            body="Person detected!",
                            from_=twilio_number,
                            to=target_number
                        )
                        print(f"SMS sent: {message.sid}")
                        self.last_sent_time = current_time
                elif ClassInd in animal_classes:
                    cv2.rectangle(img, boxes, (0, 255, 0), 2)
                    cv2.putText(img, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40),
                                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

                    current_time = time.time()
                    if current_time - self.last_sent_time > notification_interval:
                        # Send SMS
                        message = client.messages.create(
                            body="Animal detected!",
                            from_=twilio_number,
                            to=target_number
                        )
                        print(f"SMS sent: {message.sid}")
                        self.last_sent_time = current_time

        return av.VideoFrame.from_ndarray(img, format='bgr24')

webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False}
)
