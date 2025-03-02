import cv2
import mediapipe as mp
import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Open3D Setup for real-time visualization
vis = o3d.visualization.Visualizer()
vis.create_window()

# Create empty point cloud and mesh
pcd = o3d.geometry.PointCloud()
mesh = o3d.geometry.TriangleMesh()
vis.add_geometry(pcd)
vis.add_geometry(mesh)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    landmark_points = []

    # Extract landmarks
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                landmark_points.append([landmark.x, landmark.y, landmark.z])

    # Convert to NumPy array
    if len(landmark_points) > 0:
        points = np.array(landmark_points)

        # Normalize points for Open3D visualization
        points -= np.mean(points, axis=0)  # Center the face
        points *= 2  # Scale up

        # Create a Delaunay triangulation
        tri = Delaunay(points[:, :2])  # Use only x, y for triangulation

        # Update Open3D objects
        pcd.points = o3d.utility.Vector3dVector(points)
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
        mesh.compute_vertex_normals()

        # Update visualization
        vis.update_geometry(pcd)
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()

    # Show landmarks on video (for debugging)
    for x, y, _ in landmark_points:
        x, y = int(x * frame.shape[1]), int(y * frame.shape[0])
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Face Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
