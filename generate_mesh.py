import open3d as o3d
import numpy as np
import cv2
import mediapipe as mp
from scipy.spatial import Delaunay

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Initialize Open3D visualization
vis = o3d.visualization.Visualizer()
vis.create_window(width=600, height=600)
mesh = o3d.geometry.TriangleMesh()
vis.add_geometry(mesh)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    landmark_points = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                landmark_points.append([landmark.x, landmark.y, landmark.z])
    
    if landmark_points:
        points = np.array(landmark_points)
        
        # Normalize to real-world proportions
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
        width = max_x - min_x
        height = max_y - min_y
        aspect_ratio = width / height
        
        # Adjust scaling based on aspect ratio
        points[:, 0] -= (min_x + max_x) / 2  # Center horizontally
        points[:, 1] -= (min_y + max_y) / 2  # Center vertically
        points[:, 2] *= aspect_ratio * 5  # Adjust depth scaling
        
        points *= 20  # Scale for visibility
        
        # Perform Delaunay Triangulation in 2D (ignoring depth for now)
        tri = Delaunay(points[:, :2])
        
        # Create Open3D mesh
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
        mesh.compute_vertex_normals()
        
        # Apply Laplacian smoothing
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)
        mesh.compute_vertex_normals()
        
        vis.clear_geometries()
        vis.add_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
    
    # Show landmarks in OpenCV window
    cv2.imshow("Face Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()