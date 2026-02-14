import open3d as o3d
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class MeshVisualizer:
    """
    Real-time 3D mesh visualization using Open3D
    """
    
    def __init__(self, 
                 window_name: str = "3D Face Reconstruction",
                 width: int = 800,
                 height: int = 600,
                 background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)):
        """
        Initialize the visualizer
        
        Args:
            window_name: Window title
            width: Window width
            height: Window height
            background_color: RGB background color (0-1 range)
        """
        self.window_name = window_name
        self.width = width
        self.height = height
        self.background_color = background_color
        
        # Create visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=width, height=height)
        
        # Geometry objects
        self.mesh = None
        self.point_cloud = None
        self.coordinate_frame = None
        self.wireframe = None
        
        # Setup render options
        self._setup_render_options()
        
        # Setup camera
        self._setup_camera()
        
        logger.info(f"Visualizer initialized: {window_name} ({width}x{height})")
    
    def _setup_render_options(self):
        """Configure rendering options"""
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array(self.background_color)
        render_option.mesh_show_back_face = True
        render_option.mesh_show_wireframe = False
        render_option.point_size = 2.0
        render_option.line_width = 1.0
    
    def _setup_camera(self):
        """Configure camera view"""
        ctr = self.vis.get_view_control()
        ctr.set_front([0, 0, -1])  # Look from positive Z
        ctr.set_lookat([0, 0, 0])  # Look at origin
        ctr.set_up([0, -1, 0])  # Y-axis down
        ctr.set_zoom(0.8)
    
    def add_mesh(self, mesh: o3d.geometry.TriangleMesh, reset_view: bool = False):
        """
        Add or update mesh in the visualizer
        
        Args:
            mesh: Triangle mesh to display
            reset_view: Whether to reset camera view
        """
        if self.mesh is None:
            self.mesh = mesh
            self.vis.add_geometry(mesh)
            if reset_view:
                self.vis.reset_view_point(True)
        else:
            self.mesh.vertices = mesh.vertices
            self.mesh.triangles = mesh.triangles
            if mesh.has_vertex_normals():
                self.mesh.vertex_normals = mesh.vertex_normals
            if mesh.has_vertex_colors():
                self.mesh.vertex_colors = mesh.vertex_colors
            self.vis.update_geometry(self.mesh)
    
    def add_point_cloud(self, points: np.ndarray, colors: Optional[np.ndarray] = None):
        """
        Add or update point cloud
        
        Args:
            points: Points array with shape (N, 3)
            colors: Optional colors array with shape (N, 3)
        """
        if self.point_cloud is None:
            self.point_cloud = o3d.geometry.PointCloud()
            self.point_cloud.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
            self.vis.add_geometry(self.point_cloud)
        else:
            self.point_cloud.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.point_cloud)
    
    def add_coordinate_frame(self, size: float = 1.0, origin: np.ndarray = None):
        """
        Add coordinate frame for reference
        
        Args:
            size: Size of the coordinate frame
            origin: Origin position (default: [0, 0, 0])
        """
        if self.coordinate_frame is None:
            self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
            if origin is not None:
                self.coordinate_frame.translate(origin)
            self.vis.add_geometry(self.coordinate_frame)
    
    def set_mesh_color(self, color: Tuple[float, float, float]):
        """
        Set uniform color for the mesh
        
        Args:
            color: RGB color (0-1 range)
        """
        if self.mesh is not None:
            self.mesh.paint_uniform_color(color)
            self.vis.update_geometry(self.mesh)
    
    def toggle_wireframe(self, show: bool):
        """
        Toggle wireframe display
        
        Args:
            show: Whether to show wireframe
        """
        render_option = self.vis.get_render_option()
        render_option.mesh_show_wireframe = show
    
    def update(self) -> bool:
        """
        Update the visualizer
        
        Returns:
            False if window is closed, True otherwise
        """
        self.vis.poll_events()
        self.vis.update_renderer()
        return not self.vis.poll_events()
    
    def clear(self):
        """Clear all geometries"""
        self.vis.clear_geometries()
        self.mesh = None
        self.point_cloud = None
        self.coordinate_frame = None
    
    def capture_screen(self, filename: str):
        """Capture screenshot"""
        self.vis.capture_screen_image(filename, do_render=True)
        logger.info(f"Screenshot saved: {filename}")
    
    def destroy(self):
        """Close the visualizer window"""
        self.vis.destroy_window()
        logger.info("Visualizer window closed")


class VideoVisualizer:
    """
    Visualization utilities for video frames with overlays
    """
    
    @staticmethod
    def draw_landmarks(frame: np.ndarray, 
                      landmarks: np.ndarray,
                      color: Tuple[int, int, int] = (0, 255, 0),
                      radius: int = 2,
                      thickness: int = -1) -> np.ndarray:
        """
        Draw landmarks on video frame
        
        Args:
            frame: Input BGR image
            landmarks: Normalized landmarks (0-1 range) with shape (N, 3)
            color: BGR color for landmarks
            radius: Circle radius
            thickness: Circle thickness (-1 for filled)
        
        Returns:
            Frame with landmarks drawn
        """
        h, w = frame.shape[:2]
        output = frame.copy()
        
        for landmark in landmarks:
            x = int(landmark[0] * w)
            y = int(landmark[1] * h)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(output, (x, y), radius, color, thickness)
        
        return output
    
    @staticmethod
    def draw_connections(frame: np.ndarray,
                        landmarks: np.ndarray,
                        connections: List[Tuple[int, int]],
                        color: Tuple[int, int, int] = (255, 255, 255),
                        thickness: int = 1) -> np.ndarray:
        """
        Draw connections between landmarks
        
        Args:
            frame: Input BGR image
            landmarks: Normalized landmarks (0-1 range)
            connections: List of (start_idx, end_idx) tuples
            color: BGR color for lines
            thickness: Line thickness
        
        Returns:
            Frame with connections drawn
        """
        h, w = frame.shape[:2]
        output = frame.copy()
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                x1, y1 = int(start[0] * w), int(start[1] * h)
                x2, y2 = int(end[0] * w), int(end[1] * h)
                
                if (0 <= x1 < w and 0 <= y1 < h and 
                    0 <= x2 < w and 0 <= y2 < h):
                    cv2.line(output, (x1, y1), (x2, y2), color, thickness)
        
        return output
    
    @staticmethod
    def draw_face_box(frame: np.ndarray,
                     landmarks: np.ndarray,
                     color: Tuple[int, int, int] = (0, 255, 255),
                     thickness: int = 2) -> np.ndarray:
        """
        Draw bounding box around face
        
        Args:
            frame: Input BGR image
            landmarks: Normalized landmarks
            color: BGR color for box
            thickness: Line thickness
        
        Returns:
            Frame with bounding box
        """
        h, w = frame.shape[:2]
        output = frame.copy()
        
        # Compute bounding box
        x_coords = landmarks[:, 0] * w
        y_coords = landmarks[:, 1] * h
        
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        
        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), color, thickness)
        
        return output
    
    @staticmethod
    def draw_info_panel(frame: np.ndarray,
                       info_dict: Dict[str, str],
                       position: Tuple[int, int] = (10, 30),
                       font_scale: float = 0.6,
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 1) -> np.ndarray:
        """
        Draw information panel on frame
        
        Args:
            frame: Input BGR image
            info_dict: Dictionary of key-value pairs to display
            position: Starting position (x, y)
            font_scale: Font scale
            color: BGR text color
            thickness: Text thickness
        
        Returns:
            Frame with info panel
        """
        output = frame.copy()
        x, y = position
        line_height = int(30 * font_scale)
        
        for key, value in info_dict.items():
            text = f"{key}: {value}"
            cv2.putText(output, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, color, thickness, cv2.LINE_AA)
            y += line_height
        
        return output
    
    @staticmethod
    def draw_orientation_axes(frame: np.ndarray,
                            center: Tuple[int, int],
                            rotation_matrix: np.ndarray,
                            axis_length: int = 50) -> np.ndarray:
        """
        Draw 3D orientation axes on frame
        
        Args:
            frame: Input BGR image
            center: Center point (x, y)
            rotation_matrix: 3x3 rotation matrix
            axis_length: Length of axes in pixels
        
        Returns:
            Frame with orientation axes
        """
        output = frame.copy()
        
        # Define axis directions
        axes = np.array([
            [axis_length, 0, 0],  # X-axis (red)
            [0, axis_length, 0],  # Y-axis (green)
            [0, 0, axis_length]   # Z-axis (blue)
        ])
        
        # Rotate axes
        rotated_axes = rotation_matrix @ axes.T
        
        # Project to 2D (simple orthographic projection)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: Red, Green, Blue
        
        for i, color in enumerate(colors):
            end_point = (
                int(center[0] + rotated_axes[0, i]),
                int(center[1] - rotated_axes[1, i])  # Invert Y for screen coordinates
            )
            cv2.arrowedLine(output, center, end_point, color, 2, tipLength=0.3)
        
        return output


class DepthMapVisualizer:
    """
    Visualization for depth maps
    """
    
    @staticmethod
    def colorize_depth(depth_map: np.ndarray,
                      colormap: int = cv2.COLORMAP_JET,
                      normalize: bool = True) -> np.ndarray:
        """
        Convert depth map to colored visualization
        
        Args:
            depth_map: Depth values (2D array)
            colormap: OpenCV colormap
            normalize: Whether to normalize to [0, 255]
        
        Returns:
            Colored depth map (BGR)
        """
        if normalize:
            # Normalize to 0-255 range
            depth_min = depth_map[depth_map > 0].min() if (depth_map > 0).any() else 0
            depth_max = depth_map.max()
            if depth_max > depth_min:
                depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)
        else:
            depth_normalized = depth_map.astype(np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(depth_normalized, colormap)
        
        return colored
    
    @staticmethod
    def create_side_by_side(rgb_frame: np.ndarray,
                           depth_frame: np.ndarray,
                           labels: Optional[Tuple[str, str]] = None) -> np.ndarray:
        """
        Create side-by-side visualization of RGB and depth
        
        Args:
            rgb_frame: RGB image
            depth_frame: Depth image (colored)
            labels: Optional labels for each frame
        
        Returns:
            Combined side-by-side image
        """
        # Resize depth to match RGB if needed
        if depth_frame.shape[:2] != rgb_frame.shape[:2]:
            depth_frame = cv2.resize(depth_frame, (rgb_frame.shape[1], rgb_frame.shape[0]))
        
        # Concatenate horizontally
        combined = np.hstack([rgb_frame, depth_frame])
        
        # Add labels if provided
        if labels is not None:
            label_rgb, label_depth = labels
            h, w = rgb_frame.shape[:2]
            cv2.putText(combined, label_rgb, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined, label_depth, (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        return combined


class MultiViewVisualizer:
    """
    Multi-view visualization combining 2D and 3D views
    """
    
    def __init__(self):
        self.mesh_vis = None
        self.window_name = "Multi-View Face Reconstruction"
    
    def create_layout(self, 
                     rgb_frame: np.ndarray,
                     depth_frame: np.ndarray,
                     mesh_screenshot: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create multi-view layout
        
        Args:
            rgb_frame: RGB camera frame
            depth_frame: Depth visualization
            mesh_screenshot: 3D mesh view screenshot
        
        Returns:
            Combined layout image
        """
        h, w = rgb_frame.shape[:2]
        
        # Resize frames to consistent size
        target_size = (w // 2, h // 2)
        rgb_small = cv2.resize(rgb_frame, target_size)
        depth_small = cv2.resize(depth_frame, target_size)
        
        if mesh_screenshot is not None:
            mesh_small = cv2.resize(mesh_screenshot, target_size)
            # Create 2x2 grid
            top_row = np.hstack([rgb_small, depth_small])
            bottom_row = np.hstack([mesh_small, np.zeros_like(mesh_small)])
            layout = np.vstack([top_row, bottom_row])
        else:
            # Create 1x2 layout
            layout = np.hstack([rgb_small, depth_small])
        
        return layout


def create_demo_visualization():
    """Create a demo visualization for testing"""
    print("Creating demo visualization...")
    
    # Create sample mesh
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.9])
    
    # Create visualizer
    vis = MeshVisualizer(window_name="Demo", width=800, height=600)
    vis.add_coordinate_frame(size=0.5)
    vis.add_mesh(mesh, reset_view=True)
    
    print("Press 'Q' in the OpenCV window to exit...")
    
    # Visualization loop
    import time
    for i in range(100):
        # Rotate mesh
        R = mesh.get_rotation_matrix_from_xyz((0, np.pi / 100, 0))
        mesh.rotate(R, center=mesh.get_center())
        
        vis.add_mesh(mesh)
        vis.update()
        time.sleep(0.03)
    
    vis.destroy()
    print("Demo completed!")


if __name__ == "__main__":
    # Run demo
    create_demo_visualization()
