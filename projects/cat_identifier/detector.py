import cv2
from ultralytics import YOLO

# Replace with your RTSP URL
rtsp_url = "rtsp://potato:ilovetonic@192.168.68.133/live"

# Load the YOLO model (lightweight version)
model = YOLO('yolov8n.pt')  # Nano version for speed

# Classes to detect (filter out others)
allowed_classes = ['person', 'car', 'truck', 'bus']

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Unable to access the RTSP stream.")
    exit()

# Control variables
resolution_scale = 50  # Default resolution scale as a percentage (100% original resolution)
frame_rate_skip = 10     # Process every Nth frame to control frame rate

frame_count = 0  # Counter for skipping frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame. Retrying...")
        continue

    # Skip frames to control frame rate
    frame_count += 1
    if frame_count % frame_rate_skip != 0:
        continue

    # Scale resolution dynamically
    height, width, _ = frame.shape
    scaled_width = int(width * (resolution_scale / 100))
    scaled_height = int(height * (resolution_scale / 100))
    frame_resized = cv2.resize(frame, (scaled_width, scaled_height))

    # Perform object detection
    results = model(frame_resized)
    detections = results[0].boxes  # Access bounding boxes

    for box in detections:
        # Extract box coordinates, confidence, and class ID
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]  # Confidence score
        cls = int(box.cls[0])  # Class ID
        class_name = model.names[cls]  # Map ID to class name

        # classify everything
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Filter detections based on allowed classes
        # if class_name in allowed_classes:
        #     # Draw bounding box and label
        #     cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     label = f"{class_name} {conf:.2f}"
        #     cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the video feed with detections
    cv2.imshow("Wyze Camera Stream with Detection", frame_resized)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
