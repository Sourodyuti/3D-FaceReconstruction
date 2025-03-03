import cv2
import mediapipe as mp
import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Initialize MediaPipe Face Mesh
logger.info("Initializing MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize OpenCV video capture
logger.info("Initializing OpenCV video capture...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Could not open webcam")
    raise Exception("Could not open webcam")

# Open3D Setup
logger.info("Initializing Open3D...")
vis = o3d.visualization.Visualizer()
vis.create_window(width=640, height=480)

# Create TriangleMesh with a large, simple triangle first
mesh = o3d.geometry.TriangleMesh()
simple_vertices = np.array([
    [-10.0, -10.0, 0.0],  # Large triangle to ensure visibility
    [10.0, -10.0, 0.0],
    [0.0, 10.0, 0.0]
], dtype=np.float64)
mesh.vertices = o3d.utility.Vector3dVector(simple_vertices)
mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]], dtype=np.int32))
mesh.compute_vertex_normals()
vis.add_geometry(mesh)

# Disable backface culling
render_option = vis.get_render_option()
render_option.mesh_show_back_face = True
logger.info("Backface culling disabled")

# Adjust camera view
ctr = vis.get_view_control()
ctr.set_front([0, 0, -1])  # Look from negative Z
ctr.set_lookat([0, 0, 0])  # Center at origin
ctr.set_up([0, 1, 0])      # Y up
ctr.set_zoom(0.1)          # Zoom out far

# Flag to set triangles once
triangles_set = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        logger.warning("Failed to read from camera")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    landmark_points = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for i, landmark in enumerate(face_landmarks.landmark):
                if len(face_landmarks.landmark) == 478 and i >= 468:
                    continue
                x = (landmark.x - 0.5) * 10.0  # Larger scale
                y = -(landmark.y - 0.5) * 10.0  # Inverted Y-axis, larger scale
                z = landmark.z * 10.0
                landmark_points.append([x, y, z])

    if len(landmark_points) == 468:
        points = np.array(landmark_points, dtype=np.float64)
        logger.info(f"Points shape: {points.shape}, min: {np.min(points)}, max: {np.max(points)}")

        # Compute Delaunay triangulation once and switch to face mesh
        if not triangles_set:
            xy_points = points[:, :2]
            tri = Delaunay(xy_points)
            triangles = tri.simplices
            mesh.vertices = o3d.utility.Vector3dVector(points)  # Switch to face data
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            logger.info(f"Triangulation set with {len(triangles)} triangles")
            triangles_set = True

        # Update mesh
        try:
            mesh.vertices = o3d.utility.Vector3dVector(points)
            mesh.compute_vertex_normals()
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()
            logger.info("Mesh updated")
        except Exception as e:
            logger.error(f"Error updating mesh: {str(e)}")
            continue

        # Draw landmarks on video feed
        for i, landmark in enumerate(results.multi_face_landmarks[0].landmark[:468]):
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Face Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
face_mesh.close()