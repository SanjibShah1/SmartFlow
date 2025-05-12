import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import torch
from config import COUNTING_LINE_Y, SPEED_LIMIT, TRACKING_CONFIG
from utils import put_text_with_background
from tracking import vehicle_tracker, update_vehicle_tracking, lane_incoming, lane_outgoing, overspeeding_vehicles, overspeeding_images, displayed_overspeeding_ids
from ui import setup_sidebar, setup_counters, display_overspeeding_vehicles, create_traffic_light, display_traffic_lights
import time

# Adaptive traffic signal control algorithm
def adaptive_signal_control(lane_incoming, lane_outgoing):
    """
    Adjusts traffic signal timings based on the number of vehicles in each lane.
    """
    total_incoming = lane_incoming['two_wheeler'] + lane_incoming['four_wheeler']
    total_outgoing = lane_outgoing['two_wheeler'] + lane_outgoing['four_wheeler']

    # Base green time (minimum 10 seconds, maximum 60 seconds)
    base_green_time = 10
    max_green_time = 60

    # Adaptive logic
    if total_incoming > total_outgoing:
        green_time_incoming = min(base_green_time + total_incoming * 2, max_green_time)
        green_time_outgoing = max(base_green_time, max_green_time - green_time_incoming)
    else:
        green_time_outgoing = min(base_green_time + total_outgoing * 2, max_green_time)
        green_time_incoming = max(base_green_time, max_green_time - green_time_outgoing)

    return green_time_incoming, green_time_outgoing

# Main script logic
if __name__ == "__main__":
    # Load YOLO model
    device = "cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    print(f"Using device: {device}")
    model = YOLO("best.pt")
    model.to(device)

    # Setup Streamlit UI
    option, confidence_threshold, iou_threshold = setup_sidebar()

    # Initialize video capture based on sidebar input
    uploaded_file = None
    camera_url = None
    camera_index = 0

    if option == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    elif option == "Use Webcam":
        st.sidebar.markdown("""
        **To use Iriun Webcam:**
        1. Install the Iriun Webcam app on your phone and computer.
        2. Connect both devices to the same Wi-Fi network.
        3. Launch the Iriun Webcam app on both devices.
        4. Select "Use Webcam" in the sidebar.
        """)
        camera_index = st.sidebar.selectbox("Select Camera Index", [0, 1, 2])
    elif option == "Use External Camera":
        camera_url = st.sidebar.text_input("Enter Camera URL or IP Address (e.g., http://192.168.x.x:4747/video)")

    # Initialize video capture
    cap = None
    if option == "Upload Video" and uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
    elif option == "Use Webcam":
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            st.error("Error: Could not open webcam. Ensure Iriun Webcam is running.")
            cap = None
    elif option == "Use External Camera" and camera_url:
        cap = cv2.VideoCapture(camera_url)
    else:
        cap = None

    if cap is not None and cap.isOpened():
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        frame_time = 1 / fps  # Time between frames in seconds

        # Create a two-column layout: video on the left, traffic lights on the right
        col_video, col_lights = st.columns([3, 1])

        with col_video:
            video_placeholder = st.empty()

        # Placeholder for vehicle counts below the video
        incoming_two_wheeler, incoming_four_wheeler, incoming_total, outgoing_two_wheeler, outgoing_four_wheeler, outgoing_total = setup_counters()

        # Add a section for overspeeding vehicles
        st.markdown("🚨 Overspeeding Vehicles ID (>50 km/h)", unsafe_allow_html=True)
        overspeeding_placeholder = st.empty()

        # Placeholder for overspeeding vehicle images
        st.subheader("Overspeeding Vehicle Snapshots")
        overspeeding_images_placeholder = st.empty()

        # Initialize traffic light placeholders
        with col_lights:
            st.markdown("### Traffic Lights")
            incoming_light_placeholder = st.empty()
            outgoing_light_placeholder = st.empty()

        # Main loop to process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform tracking on the current frame with confidence and IoU thresholds
            results = model.track(
                frame,
                persist=True,
                tracker=str(TRACKING_CONFIG),
                conf=confidence_threshold,  # Use the slider value
                iou=iou_threshold          # Use the slider value
            )

            # Draw the counting line on the frame
            cv2.line(frame, (0, COUNTING_LINE_Y), (width, COUNTING_LINE_Y), (0, 0, 255), 2)

            # Process each detected object
            for box in results[0].boxes:
                if box.id is None:
                    continue

                track_id = int(box.id.item())
                xyxy = box.xyxy[0].cpu().numpy()
                x_center = (xyxy[0] + xyxy[2]) / 2
                y_center = (xyxy[1] + xyxy[3]) / 2
                cls = int(box.cls.item())  # Class ID (0: Four Wheeler, 1: Two Wheeler)

                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, xyxy)

                # Draw bounding box
                box_color = (18, 227, 227)  #  color for bounding boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # Prepare class name and ID text
                class_name = "four_wheeler" if cls == 0 else "two_wheeler"
                id_text = f"ID: {track_id} ({class_name})"

                # Put text with background above the bounding box
                put_text_with_background(
                    img=frame,
                    text=id_text,
                    position=(x1, y1 - 10),  # Position text above the bounding box
                    font_scale=0.6,
                    text_color=(0, 255, 0),  # Green text
                    bg_color=(0, 0, 0),      # Black background
                    thickness=1
                )

                # Update vehicle tracking
                lane_incoming, lane_outgoing, overspeeding_vehicles, overspeeding_images = update_vehicle_tracking(
                    track_id, box, frame_time, COUNTING_LINE_Y, width, SPEED_LIMIT, overspeeding_vehicles, overspeeding_images, frame
                )

            # Adaptive traffic signal control
            green_time_incoming, green_time_outgoing = adaptive_signal_control(lane_incoming, lane_outgoing)

            # Update traffic lights dynamically
            display_traffic_lights(
                "green" if green_time_incoming > 0 else "red",
                "red" if green_time_incoming > 0 else "green",
                incoming_light_placeholder,
                outgoing_light_placeholder
            )

            # Convert frame from BGR to RGB and display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame, channels="RGB", use_container_width=True)

            # Update the compact counts below the video
            incoming_two_wheeler.markdown(f"Two Wheeler: {lane_incoming['two_wheeler']}", unsafe_allow_html=True)
            incoming_four_wheeler.markdown(f"Four Wheeler: {lane_incoming['four_wheeler']}", unsafe_allow_html=True)
            incoming_total.markdown(f"Total: {lane_incoming['two_wheeler'] + lane_incoming['four_wheeler']}", unsafe_allow_html=True)
            outgoing_two_wheeler.markdown(f"Two Wheeler: {lane_outgoing['two_wheeler']}", unsafe_allow_html=True)
            outgoing_four_wheeler.markdown(f"Four Wheeler: {lane_outgoing['four_wheeler']}", unsafe_allow_html=True)
            outgoing_total.markdown(f"Total: {lane_outgoing['two_wheeler'] + lane_outgoing['four_wheeler']}", unsafe_allow_html=True)

            # Update overspeeding vehicles list and total count
            overspeeding_placeholder.markdown(
                f"{', '.join([f'ID: {k}, Speed: {v:.2f} km/h' for k, v in overspeeding_vehicles.items()]) if overspeeding_vehicles else 'None'}", 
                unsafe_allow_html=True)

            # Display overspeeding vehicle images in real-time
            display_overspeeding_vehicles(overspeeding_vehicles, overspeeding_images, displayed_overspeeding_ids)

        # Release video capture when finished
        cap.release()

        # Optionally, display final counts after processing ends
        st.markdown("", unsafe_allow_html=True)
        st.subheader("Final Counts")
        st.write(f"Total Incoming Vehicles: {lane_incoming['two_wheeler'] + lane_incoming['four_wheeler']}")
        st.write(f"Total Outgoing Vehicles: {lane_outgoing['two_wheeler'] + lane_outgoing['four_wheeler']}")
        st.write(f"Overspeeding Vehicles: {', '.join([f'ID: {k}, Speed: {v:.2f} km/h' for k, v in overspeeding_vehicles.items()]) if overspeeding_vehicles else 'None'}")
        st.write(f"Total Overspeeding Vehicles: {len(overspeeding_vehicles)}")
    else:
        st.error("No video source selected. Please select a valid source from the sidebar.")