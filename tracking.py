from collections import defaultdict
import math
import cv2

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
overspeeding_vehicles = {}  # Track IDs and speeds of overspeeding vehicles
overspeeding_images = {}  # Store images of overspeeding vehicles
displayed_overspeeding_ids = set()  # Track IDs of displayed overspeeding vehicles

def update_vehicle_tracking(track_id, box, frame_time, counting_line_y, width, speed_limit, overspeeding_vehicles, overspeeding_images, frame):
    """
    Updates the vehicle tracking dictionary with new positions, speed, and overspeeding status.
    """
    xyxy = box.xyxy[0].cpu().numpy()
    x_center = (xyxy[0] + xyxy[2]) / 2
    y_center = (xyxy[1] + xyxy[3]) / 2
    cls = int(box.cls.item())  # Class ID (0: Four Wheeler, 1: Two Wheeler)

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
                x1, y1, x2, y2 = map(int, xyxy)
                vehicle_img = frame[y1:y2, x1:x2]
                if vehicle_img.size != 0:  # Ensure we don't try to save empty images
                    resized_img = cv2.resize(vehicle_img, (100, 70))  # Smaller size
                    overspeeding_images[track_id] = resized_img
        else:
            vehicle['overspeeding'] = False

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

    return lane_incoming, lane_outgoing, overspeeding_vehicles, overspeeding_images