import streamlit as st
import subprocess
import os
import sys
import time

def start_detection():
    st.title("Animal and Person Detection System")
    
    # Detection settings
    st.sidebar.header("Detection Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.55)
    notification_interval = st.sidebar.number_input("Notification Interval (minutes)", 1, 60, 5)
    
    # Status indicator
    status_placeholder = st.empty()
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_button = st.button('Start Detection')
    with col2:
        stop_button = st.button('Stop Detection')
    
    if start_button:
        try:
            status_placeholder.info("Starting detection system...")
            
            # Create command with parameters
            cmd = [
                'python',
                os.path.join(os.path.dirname(__file__), '..', 'Object Recognication copy.py'),
                '--confidence', str(confidence_threshold),
                '--interval', str(notification_interval * 60)
            ]
            
            # Start process
            process = subprocess.Popen(cmd)
            
            st.session_state['detection_running'] = True
            st.session_state['process'] = process
            
            status_placeholder.success("Detection system is running")
            
        except Exception as e:
            status_placeholder.error(f"Error starting detection: {str(e)}")
    
    if stop_button and 'process' in st.session_state:
        try:
            st.session_state['process'].terminate()
            status_placeholder.warning("Detection system stopped")
            st.session_state['detection_running'] = False
            del st.session_state['process']
        except Exception as e:
            status_placeholder.error(f"Error stopping detection: {str(e)}")
    
    # Show current status
    if 'detection_running' in st.session_state and st.session_state['detection_running']:
        st.success("System Status: Active")
        st.info(f"""
        Current Settings:
        - Confidence Threshold: {confidence_threshold}
        - Notification Interval: {notification_interval} minutes
        """)
    else:
        st.warning("System Status: Inactive")

if __name__ == "__main__":
    start_detection()