import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open webcam")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    h, w, _ = frame.shape  # Get frame dimensions

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Initialize depth map as grayscale
    depth_map = np.zeros((h, w), dtype=np.uint8)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = int((1 - landmark.z) * 255)  # Flip depth for better contrast

                # Bounds check
                if 0 <= x < w and 0 <= y < h:
                    depth_map[y, x] = np.clip(z, 0, 255)

    # Enhance visibility using heatmap
    depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

    # Show outputs
    cv2.imshow("Depth Map", depth_map_colored)
    cv2.imshow("Face Landmarks", frame)

    # Debugging: Print first few landmark positions
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark[0]
        print(f"Landmark sample: x={lm.x:.2f}, y={lm.y:.2f}, z={lm.z:.2f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
