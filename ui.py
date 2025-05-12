import streamlit as st
import cv2
from PIL import Image, ImageDraw

st.header("Vehicle Tracking, Counting, and Speed Estimation System")

def setup_sidebar():
    st.sidebar.header("Video Source Options")
    option = st.sidebar.radio("Choose a video source:", ("Upload Video", "Use Webcam", "Use External Camera"))
    confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    iou_threshold = st.sidebar.slider("IoU Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    return option, confidence_threshold, iou_threshold

def setup_counters():
    """
    Creates placeholders for vehicle counts below the video preview.
    """
    st.markdown("### Vehicle Counts")
    inc_col, out_col = st.columns(2)
    inc_col.markdown("🚗 Incoming Vehicles", unsafe_allow_html=True)
    incoming_two_wheeler = inc_col.empty()
    incoming_four_wheeler = inc_col.empty()
    incoming_total = inc_col.empty()

    out_col.markdown("🚗 Outgoing Vehicles", unsafe_allow_html=True)
    outgoing_two_wheeler = out_col.empty()
    outgoing_four_wheeler = out_col.empty()
    outgoing_total = out_col.empty()

    return incoming_two_wheeler, incoming_four_wheeler, incoming_total, outgoing_two_wheeler, outgoing_four_wheeler, outgoing_total

def display_overspeeding_vehicles(overspeeding_vehicles, overspeeding_images, displayed_overspeeding_ids):
    """
    Displays overspeeding vehicle images in real-time.
    """
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

def create_traffic_light(color):
    """
    Creates a traffic light image with the given color overlay.
    """
    # Base image (traffic light frame)
    base_image = Image.new("RGB", (100, 300), "black")
    draw = ImageDraw.Draw(base_image)

    # Draw the red light
    draw.ellipse([25, 25, 75, 75], fill="red" if color == "red" else "gray")

    # Draw the yellow light
    draw.ellipse([25, 125, 75, 175], fill="yellow" if color == "yellow" else "gray")

    # Draw the green light
    draw.ellipse([25, 225, 75, 275], fill="green" if color == "green" else "gray")

    return base_image



def display_traffic_lights(incoming_state, outgoing_state, incoming_light_placeholder, outgoing_light_placeholder):
    """
    Updates traffic light images dynamically with smaller size.
    """
    incoming_light = create_traffic_light(incoming_state)
    outgoing_light = create_traffic_light(outgoing_state)

    # Resize images to make them smaller
    target_size = (25, 50)  # Adjust size as needed
    incoming_light = incoming_light.resize(target_size, Image.LANCZOS)
    outgoing_light = outgoing_light.resize(target_size, Image.LANCZOS)

    # Use st.columns() to place them side by side
    col1, col2 = st.columns(2)

    with col1:
        incoming_light_placeholder.image(incoming_light, caption="Incoming", use_container_width=False)

    with col2:
        outgoing_light_placeholder.image(outgoing_light, caption="Outgoing", use_container_width=False)


