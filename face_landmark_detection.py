import cv2
import mediapipe as mp
import open3d as o3d
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam

# Open3D Setup
vis = o3d.visualization.Visualizer()
vis.create_window()

# Create an empty point cloud object
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

# Add a coordinate frame for reference
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
vis.add_geometry(axis)

# Set camera view to look at the origin
ctr = vis.get_view_control()
ctr.set_front([0, 0, -1])  # Camera faces the -Z direction
ctr.set_lookat([0, 0, 0])  # Center the view
ctr.set_up([0, -1, 0])  # Adjust up direction
ctr.set_zoom(0.5)  # Adjust zoom level

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with FaceMesh
    results = face_mesh.process(rgb_frame)

    # Create a list to store 3D landmarks
    landmark_points = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:  # Iterate over detected faces
            for landmark in face_landmarks.landmark:  # Iterate over 468 landmarks
                landmark_points.append([landmark.x - 0.5, -(landmark.y - 0.5), -landmark.z])  # Adjust coordinates

    # Update the 3D point cloud visualization
    if landmark_points:
        points = np.array(landmark_points)

        # Normalize and scale the points for better visualization
        points = (points - np.mean(points, axis=0)) * 2  # Center and scale

        # Update Open3D point cloud
        pcd.points = o3d.utility.Vector3dVector(points)

        # ✅ Ensure Open3D updates properly
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    # Show the detected landmarks on the video feed (optional)
    cv2.imshow("Face Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
