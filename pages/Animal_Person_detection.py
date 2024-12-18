import logging
import av
import cv2
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from twilio.rest import Client

# Enable detailed logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Access secrets securely (make sure you've added your Twilio details in streamlit secrets)
account_sid = st.secrets["twilio"]["account_sid"]
auth_token = st.secrets["twilio"]["auth_token"]
twilio_number = st.secrets["twilio"]["twilio_number"]
target_number = st.secrets["twilio"]["target_number"]  # Ensure target number includes the country code

# Initialize Twilio client
client = Client(account_sid, auth_token)

# Streamlit UI setup
st.title("Animal and Person Detection System")
st.sidebar.header("Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.55)
notification_interval = st.sidebar.number_input("Notification Interval (minutes)", 1, 60, 5) * 60  # Convert to seconds

# Load pre-trained model (SSD MobileNet for object detection)
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Load class labels
with open('Label.txt', 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Class IDs for animals (modify as per your requirements)
animal_classes = [16, 17, 18, 19, 20, 21, 22, 23, 24]

# Define the video transformer class
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_sent_time = 0

    def send_notification(self, body):
        """Send an SMS notification using Twilio."""
        try:
            message = client.messages.create(
                body=body,
                from_=twilio_number,
                to=target_number
            )
            logger.debug(f"SMS sent: {message.sid}")
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")

    def recv(self, frame):
        """Process each video frame."""
        img = frame.to_ndarray(format="bgr24")
        
        # Resize frame for faster processing
        img = cv2.resize(img, (640, 480))
        ClassIndex, confidence, bbox = model.detect(img, confThreshold=confidence_threshold)

        if len(ClassIndex) > 0:
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                if conf > confidence_threshold:
                    # Detection for person
                    if ClassInd == 1:
                        cv2.rectangle(img, boxes, (255, 0, 0), 2)
                        cv2.putText(img, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40),
                                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                        
                        # Send notification
                        current_time = time.time()
                        if current_time - self.last_sent_time > notification_interval:
                            self.send_notification("Person detected!")
                            self.last_sent_time = current_time

                    # Detection for animals
                    elif ClassInd in animal_classes:
                        cv2.rectangle(img, boxes, (0, 255, 0), 2)
                        cv2.putText(img, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40),
                                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

                        # Send notification
                        current_time = time.time()
                        if current_time - self.last_sent_time > notification_interval:
                            self.send_notification("Animal detected!")
                            self.last_sent_time = current_time

        return av.VideoFrame.from_ndarray(img, format='bgr24')

# WebRTC streamer setup
webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoTransformer,
    async_processing=True,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]  # Google's public STUN server
    },
)

# Display instructions for the user
if webrtc_ctx.state.playing:
    st.write("Streaming started. Enable your camera to view the feed.")
else:
    st.write("Click the 'Start' button above to begin streaming.")

# Optional debugging information
st.write("Logs will be displayed in the console for debugging.")
