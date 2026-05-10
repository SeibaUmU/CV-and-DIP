import cv2

from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"D:\TaiLieu\XLA\XLA_TH\CV-and-DIP\Buoi1_2\YOLO\runs\detect\traffic_sign_result_fast\weights\best.pt")

# Open the video file
video_path = 1  # Use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame, device=0, conf=0.5, iou=0.45, augment=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()