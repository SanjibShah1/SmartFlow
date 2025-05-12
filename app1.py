import streamlit as st
import cv2
import tempfile
from collections import defaultdict
from ultralytics import YOLO
import torch
import os
from pathlib import Path
import math

# Suppress OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Set CUDA_VISIBLE_DEVICES (adjust as needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Dynamically set the device
device = "cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
print(f"Using device: {device}")
# Load YOLO model (update the model path as needed)
model = YOLO("best.pt")
model.to(device)

# Global variables for vehicle tracking and counting
vehicle_tracker = defaultdict(lambda: {
    'positions': [],
    'counted': False,
    'lane': None,
    'direction': None,
    'speed': 0.0,
    'overspeeding': False
})
lane_outgoing = {'two_wheeler': 0, 'four_wheeler': 0}  # For vehicles on the right (Outgoing)
lane_incoming = {'two_wheeler': 0, 'four_wheeler': 0}  # For vehicles on the left (Incoming)
counting_line_y = 350  # Adjust this based on your video
speed_limit = 40  # Speed limit in km/h
overspeeding_vehicles = {}  # Track IDs and speeds of overspeeding vehicles
overspeeding_images = {}  # Store images of overspeeding vehicles
displayed_overspeeding_ids = set()  # Track IDs of displayed overspeeding vehicles

# Helper function to draw text with a white background
def put_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                             font_scale=0.6, text_color=(0, 255, 0),
                             bg_color=(255, 255, 255), thickness=2, padding=5):
    """
    Draws text on an image with a background rectangle.
    """
    x, y = position
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # Calculate coordinates for the background rectangle
    rect_top_left = (x - padding, y - text_height - padding)
    rect_bottom_right = (x + text_width + padding, y + baseline + padding)
    # Draw filled rectangle
    cv2.rectangle(img, rect_top_left, rect_bottom_right, bg_color, -1)
    # Put the text on top of the rectangle
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

# SIDEBAR FOR VIDEO SOURCE SELECTION
st.sidebar.header("Video Source Options")
option = st.sidebar.radio("Choose a video source:", ("Upload Video", "Use Webcam", "Use External Camera"))

# Add sliders for confidence and IoU thresholds
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05)

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

# Initialize video capture based on sidebar input
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

    # Create a two-column layout: video on the left, compact counter panel on the right.
    col_video, col_counters = st.columns([2, 1])

    with col_video:
        video_placeholder = st.empty()

    with col_counters:
        # Create two subcolumns for Incoming and Outgoing counts
        inc_col, out_col = st.columns(2)
        inc_col.markdown("🚗 Incoming Vehicles", unsafe_allow_html=True)
        incoming_two_wheeler = inc_col.empty()
        incoming_four_wheeler = inc_col.empty()
        incoming_total = inc_col.empty()

        out_col.markdown("🚗 Outgoing Vehicles", unsafe_allow_html=True)
        outgoing_two_wheeler = out_col.empty()
        outgoing_four_wheeler = out_col.empty()
        outgoing_total = out_col.empty()

    # Add a section for overspeeding vehicles
    st.markdown("🚨 Overspeeding Vehicles ID (>50 km/h)", unsafe_allow_html=True)
    overspeeding_placeholder = st.empty()

    # Define the tracker configuration file path (update as needed)
    TRACKER_CONFIGS = Path(os.getenv("YOLO_TRACKER_DIR", "trackers"))
    tracking_config = TRACKER_CONFIGS / "botsort.yaml"

    # Placeholder for overspeeding vehicle images
    st.subheader("Overspeeding Vehicle Snapshots")
    overspeeding_images_placeholder = st.empty()

    # Main loop to process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform tracking on the current frame with confidence and IoU thresholds
        results = model.track(
            frame,
            persist=True,
            tracker=str(tracking_config),
            conf=confidence_threshold,  # Use the slider value
            iou=iou_threshold          # Use the slider value
        )

        # Draw the counting line on the frame
        cv2.line(frame, (0, counting_line_y), (width, counting_line_y), (0, 0, 255), 2)

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
            box_color = (0, 255, 0)  # Green color for bounding boxes
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

            # Update vehicle tracking history
            vehicle = vehicle_tracker[track_id]
            vehicle['positions'].append((x_center, y_center))
            if len(vehicle['positions']) > 5:
                vehicle['positions'].pop(0)

            # Calculate speed if there are at least 2 positions
            if len(vehicle['positions']) >= 2:
                prev_x, prev_y = vehicle['positions'][-2]
                curr_x, curr_y = vehicle['positions'][-1]
                distance = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)  # Distance in pixels
                speed_pixels_per_second = distance / frame_time  # Speed in pixels per second
                speed_kmh = speed_pixels_per_second * 0.036  # Convert to km/h (adjust scaling factor as needed)
                vehicle['speed'] = speed_kmh

                # Check for overspeeding
                if speed_kmh > speed_limit:
                    vehicle['overspeeding'] = True
                    overspeeding_vehicles[track_id] = speed_kmh  # Store ID and speed

                    # Capture the image of the overspeeding vehicle immediately
                    if track_id not in overspeeding_images:
                        vehicle_img = frame[y1:y2, x1:x2]
                        if vehicle_img.size != 0:  # Ensure we don't try to save empty images
                            resized_img = cv2.resize(vehicle_img, (100, 70))  # Smaller size
                            overspeeding_images[track_id] = resized_img
                else:
                    vehicle['overspeeding'] = False

                # Display speed on the frame
                speed_text = f"Speed: {vehicle['speed']:.2f} km/h"
                put_text_with_background(
                    img=frame,
                    text=speed_text,
                    position=(x1, y2 + 20),  # Position text below the bounding box
                    font_scale=0.6,
                    text_color=(0, 0, 255) if vehicle['overspeeding'] else (0, 255, 0),  # Red for overspeeding, green otherwise
                    bg_color=(0, 0, 0),  # Black background
                    thickness=1
                )

            # Determine lane based on horizontal position
            lane = "left" if x_center < width / 2 else "right"
            vehicle['lane'] = lane

            # Determine direction based on previous position
            if len(vehicle['positions']) >= 2:
                prev_y = vehicle['positions'][-2][1]
                direction = "outgoing" if y_center < prev_y else "incoming"
                vehicle['direction'] = direction

                # Check if the vehicle crosses the counting line
                if (prev_y > counting_line_y and y_center <= counting_line_y) or \
                   (prev_y < counting_line_y and y_center >= counting_line_y):
                    if not vehicle['counted']:
                        if lane == "right":  # Outgoing vehicles (right lane)
                            if cls == 0:
                                lane_outgoing['four_wheeler'] += 1
                            elif cls == 1:
                                lane_outgoing['two_wheeler'] += 1
                        else:  # Incoming vehicles (left lane)
                            if cls == 0:
                                lane_incoming['four_wheeler'] += 1
                            elif cls == 1:
                                lane_incoming['two_wheeler'] += 1
                        vehicle['counted'] = True

        # In-Video Overlays (all with white background)
        # Overlay totals at the top
        put_text_with_background(frame, f"In Total: {lane_incoming['two_wheeler'] + lane_incoming['four_wheeler']}", 
                                 (20, 30))
        put_text_with_background(frame, f"Out Total: {lane_outgoing['two_wheeler'] + lane_outgoing['four_wheeler']}", 
                                 (width - 200, 30))

        # Overlay detailed incoming counts
        incoming_texts = [
            f"Two Wheeler: {lane_incoming['two_wheeler']}",
            f"Four Wheeler: {lane_incoming['four_wheeler']}"
        ]
        for i, text in enumerate(incoming_texts):
            put_text_with_background(frame, text, (20, 60 + i * 30))

        # Overlay detailed outgoing counts
        outgoing_texts = [
            f"Two Wheeler: {lane_outgoing['two_wheeler']}",
            f"Four Wheeler: {lane_outgoing['four_wheeler']}"
        ]
        for i, text in enumerate(outgoing_texts):
            put_text_with_background(frame, text, (width - 200, 60 + i * 30))

        # Convert frame from BGR to RGB and display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame, channels="RGB", use_container_width=True)

        # Update the compact counts in the right panel (markdown text in white)
        with inc_col:
            incoming_two_wheeler.markdown(
                f"Two Wheeler: {lane_incoming['two_wheeler']}", 
                unsafe_allow_html=True)
            incoming_four_wheeler.markdown(
                f"Four Wheeler: {lane_incoming['four_wheeler']}", 
                unsafe_allow_html=True)
            incoming_total.markdown(
                f"Total: {lane_incoming['two_wheeler'] + lane_incoming['four_wheeler']}", 
                unsafe_allow_html=True)

        with out_col:
            outgoing_two_wheeler.markdown(
                f"Two Wheeler: {lane_outgoing['two_wheeler']}", 
                unsafe_allow_html=True)
            outgoing_four_wheeler.markdown(
                f"Four Wheeler: {lane_outgoing['four_wheeler']}", 
                unsafe_allow_html=True)
            outgoing_total.markdown(
                f"Total: {lane_outgoing['two_wheeler'] + lane_outgoing['four_wheeler']}", 
                unsafe_allow_html=True)

        # Update overspeeding vehicles list and total count
        overspeeding_placeholder.markdown(
            f"{', '.join([f'ID: {k}, Speed: {v:.2f} km/h' for k, v in overspeeding_vehicles.items()]) if overspeeding_vehicles else 'None'}", 
            unsafe_allow_html=True)

        # Display overspeeding vehicle images in real-time
        if overspeeding_images:
            # Create a grid layout with 4 columns per row
            cols_per_row = 4
            image_keys = list(overspeeding_images.keys())
            rows = []
            
            for i in range(0, len(image_keys), cols_per_row):
                row_images = []
                for j in range(cols_per_row):
                    if i + j < len(image_keys):
                        track_id = image_keys[i + j]
                        if track_id not in displayed_overspeeding_ids:  # Only display new images
                            row_images.append((track_id, overspeeding_images[track_id]))
                            displayed_overspeeding_ids.add(track_id)  # Mark as displayed
                rows.append(row_images)

            # Display images in the grid
            for row in rows:
                cols = st.columns(cols_per_row)
                for idx, (track_id, img) in enumerate(row):
                    with cols[idx]:
                        st.image(
                            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                            caption=f"ID: {track_id} | Speed: {overspeeding_vehicles[track_id]:.1f} km/h",
                            use_container_width=True,
                            width=150  # Control display size
                        )

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
    st.error("Error opening video source. Please select a valid source from the sidebar.")