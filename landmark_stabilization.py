import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Initialize Kalman Filters for 3D landmarks
class KalmanFilter3D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 3)  # 6 state variables (x, y, z, vx, vy, vz) and 3 measured (x, y, z)

        # State transition matrix (Assumes constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix (Only observe position, not velocity)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        # Process noise covariance
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2  # Adjust for smoothness

        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1  # Adjust for stability

        # Error covariance
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

        # Initial state
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)

    def update(self, measurement):
        """Predicts the next state and corrects it with a new measurement"""
        self.kf.predict()
        return self.kf.correct(np.array(measurement, dtype=np.float32).reshape(3, 1))


# Create Kalman Filters for all 468 landmarks
kalman_filters = [KalmanFilter3D() for _ in range(468)]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Face Mesh
    results = face_mesh.process(rgb_frame)

    # If face landmarks detected, stabilize them
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            smoothed_landmarks = []
            for i, landmark in enumerate(face_landmarks.landmark):
                # Get current position
                measurement = [landmark.x, landmark.y, landmark.z]

                # Stabilize using Kalman filter
                filtered_state = kalman_filters[i].update(measurement)
                smoothed_landmarks.append([
                    filtered_state[0, 0],  # x
                    filtered_state[1, 0],  # y
                    filtered_state[2, 0]   # z
                ])

            # Convert to NumPy array for visualization
            smoothed_landmarks = np.array(smoothed_landmarks)

            # Draw stabilized landmarks
            for x, y, _ in smoothed_landmarks:
                x, y = int(x * frame.shape[1]), int(y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Show output
    cv2.imshow("Stabilized Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
