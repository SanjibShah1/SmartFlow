import cv2

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