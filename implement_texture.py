import cv2
import mediapipe as mp
import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay

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

# Open3D Setup
vis = o3d.visualization.Visualizer()
vis.create_window(width=640, height=480)

# Create TriangleMesh
mesh = o3d.geometry.TriangleMesh()
dummy_vertices = np.zeros((468, 3), dtype=np.float64)
mesh.vertices = o3d.utility.Vector3dVector(dummy_vertices)
mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]], dtype=np.int32))
mesh.compute_vertex_normals()
vis.add_geometry(mesh)

# Disable backface culling
render_option = vis.get_render_option()
render_option.mesh_show_back_face = True  # Show back faces

# Add coordinate frame for reference
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
vis.add_geometry(axis)

# Adjust camera view
ctr = vis.get_view_control()
ctr.set_front([0, 0, 1])    # Camera looks along positive Z
ctr.set_lookat([0, 0, 0])   # Center of the mesh
ctr.set_up([0, 1, 0])       # Positive Y is up for camera
ctr.set_zoom(0.8)           # Slightly zoomed in

print("Starting face mesh rendering...")

# Flag to set triangles once
triangles_set = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    landmark_points = []
    uv_coords = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for i, landmark in enumerate(face_landmarks.landmark):
                if len(face_landmarks.landmark) == 478 and i >= 468:
                    continue
                x = (landmark.x - 0.5) * 0.1
                y = -(landmark.y - 0.5) * 0.1  # Invert Y-axis
                z = landmark.z * 0.1
                landmark_points.append([x, y, z])
                uv_coords.append([landmark.x, landmark.y])

    if len(landmark_points) == 468:
        points = np.array(landmark_points, dtype=np.float64)
        uvs = np.array(uv_coords, dtype=np.float64)

        # Compute Delaunay triangulation once
        if not triangles_set:
            xy_points = points[:, :2]
            tri = Delaunay(xy_points)
            triangles = tri.simplices
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            print(f"Triangulation set with {len(triangles)} triangles")
            triangles_set = True

        # Update mesh
        try:
            mesh.vertices = o3d.utility.Vector3dVector(points)
            mesh.compute_vertex_normals()

            # Apply texture
            texture = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mesh.textures = [o3d.geometry.Image(texture)]
            mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs[triangles.flatten()])
            mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(triangles), dtype=np.int32))

            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()
        except Exception as e:
            print(f"Error updating mesh: {str(e)}")
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