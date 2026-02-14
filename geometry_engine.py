import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class GeometryEngine:
    """
    Core geometric operations for 3D face reconstruction
    Handles mesh generation, depth estimation, and transformations
    """
    
    # MediaPipe Face Mesh topology (predefined connections)
    # These are the standard connections from MediaPipe's tesselation
    FACE_MESH_TESSELATION = [
        # This is a simplified version. Full tesselation would have 468+ connections
        # We'll use Delaunay triangulation as a more robust alternative
    ]
    
    def __init__(self, use_delaunay: bool = True, smoothing_iterations: int = 3):
        """
        Initialize geometry engine
        
        Args:
            use_delaunay: Use Delaunay triangulation (True) or predefined topology (False)
            smoothing_iterations: Number of smoothing iterations for mesh
        """
        self.use_delaunay = use_delaunay
        self.smoothing_iterations = smoothing_iterations
        self.triangles = None
        self.triangles_computed = False
        logger.info(f"GeometryEngine initialized (Delaunay: {use_delaunay}, Smoothing: {smoothing_iterations})")
    
    def create_mesh_from_landmarks(self, 
                                  landmarks: np.ndarray,
                                  compute_normals: bool = True) -> o3d.geometry.TriangleMesh:
        """
        Create a triangle mesh from facial landmarks
        
        Args:
            landmarks: Array of 3D points with shape (N, 3)
            compute_normals: Whether to compute vertex normals
        
        Returns:
            Open3D TriangleMesh object
        """
        if landmarks.shape[0] < 3:
            raise ValueError("Need at least 3 landmarks to create a mesh")
        
        # Create mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(landmarks)
        
        # Generate triangles
        if not self.triangles_computed or self.triangles is None:
            self.triangles = self._generate_triangulation(landmarks)
            self.triangles_computed = True
        
        mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
        
        # Compute normals
        if compute_normals:
            mesh.compute_vertex_normals()
        
        return mesh
    
    def _generate_triangulation(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Generate triangle connectivity using Delaunay triangulation
        
        Args:
            landmarks: 3D points with shape (N, 3)
        
        Returns:
            Triangle indices array with shape (M, 3)
        """
        if self.use_delaunay:
            # Use 2D Delaunay on X-Y plane
            try:
                xy_points = landmarks[:, :2]
                tri = Delaunay(xy_points)
                triangles = tri.simplices
                logger.debug(f"Generated {len(triangles)} triangles using Delaunay")
                return triangles
            except Exception as e:
                logger.error(f"Delaunay triangulation failed: {e}")
                return self._generate_simple_triangulation(len(landmarks))
        else:
            return self._generate_simple_triangulation(len(landmarks))
    
    def _generate_simple_triangulation(self, num_points: int) -> np.ndarray:
        """
        Generate simple triangulation as fallback
        
        Args:
            num_points: Number of vertices
        
        Returns:
            Basic triangle indices
        """
        # Create a simple fan triangulation from first vertex
        triangles = []
        for i in range(1, num_points - 1):
            triangles.append([0, i, i + 1])
        return np.array(triangles, dtype=np.int32)
    
    def smooth_mesh(self, 
                   mesh: o3d.geometry.TriangleMesh,
                   method: str = 'laplacian',
                   iterations: int = None) -> o3d.geometry.TriangleMesh:
        """
        Smooth mesh to reduce noise and improve quality
        
        Args:
            mesh: Input mesh
            method: Smoothing method ('laplacian', 'taubin', or 'simple')
            iterations: Number of iterations (uses default if None)
        
        Returns:
            Smoothed mesh
        """
        if iterations is None:
            iterations = self.smoothing_iterations
        
        if iterations <= 0:
            return mesh
        
        try:
            if method == 'laplacian':
                smoothed = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
            elif method == 'taubin':
                smoothed = mesh.filter_smooth_taubin(number_of_iterations=iterations)
            elif method == 'simple':
                smoothed = mesh.filter_smooth_simple(number_of_iterations=iterations)
            else:
                logger.warning(f"Unknown smoothing method '{method}', using laplacian")
                smoothed = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
            
            smoothed.compute_vertex_normals()
            return smoothed
        except Exception as e:
            logger.error(f"Mesh smoothing failed: {e}")
            return mesh
    
    def enhance_depth(self, 
                     landmarks: np.ndarray,
                     depth_scale: float = 1.0,
                     use_curvature: bool = True) -> np.ndarray:
        """
        Enhance depth information for better 3D reconstruction
        
        Args:
            landmarks: Input 3D landmarks
            depth_scale: Scaling factor for depth
            use_curvature: Apply curvature-based depth enhancement
        
        Returns:
            Enhanced landmarks with improved depth
        """
        enhanced = landmarks.copy()
        
        # Scale depth
        enhanced[:, 2] *= depth_scale
        
        if use_curvature:
            # Enhance depth based on facial curvature
            # Apply stronger depth to central features (nose, eyes, mouth)
            center_indices = self._get_central_feature_indices(len(landmarks))
            for idx in center_indices:
                if idx < len(enhanced):
                    enhanced[idx, 2] *= 1.2  # Enhance central features
        
        return enhanced
    
    def _get_central_feature_indices(self, num_landmarks: int) -> List[int]:
        """
        Get indices of central facial features (nose, eyes, mouth)
        Based on MediaPipe's 468-point model
        """
        if num_landmarks < 468:
            return []
        
        # Key feature indices in MediaPipe 468-point model
        nose_indices = list(range(1, 9))  # Nose tip and bridge
        left_eye_indices = list(range(33, 42))  # Left eye
        right_eye_indices = list(range(263, 272))  # Right eye
        mouth_indices = list(range(61, 68)) + list(range(291, 298))  # Mouth
        
        return nose_indices + left_eye_indices + right_eye_indices + mouth_indices
    
    def compute_face_normals(self, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
        """
        Compute face normals for the mesh
        
        Args:
            mesh: Input mesh
        
        Returns:
            Face normals array
        """
        mesh.compute_triangle_normals()
        return np.asarray(mesh.triangle_normals)
    
    def refine_mesh_quality(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """
        Refine mesh quality by removing degenerate triangles and duplicates
        
        Args:
            mesh: Input mesh
        
        Returns:
            Refined mesh
        """
        # Remove degenerate triangles
        mesh.remove_degenerate_triangles()
        
        # Remove duplicated triangles
        mesh.remove_duplicated_triangles()
        
        # Remove duplicated vertices
        mesh.remove_duplicated_vertices()
        
        # Remove non-manifold edges if possible
        mesh.remove_non_manifold_edges()
        
        # Recompute normals
        mesh.compute_vertex_normals()
        
        logger.debug(f"Refined mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        return mesh
    
    def apply_subdivision(self, 
                         mesh: o3d.geometry.TriangleMesh,
                         iterations: int = 1,
                         method: str = 'loop') -> o3d.geometry.TriangleMesh:
        """
        Apply mesh subdivision for higher resolution
        
        Args:
            mesh: Input mesh
            iterations: Number of subdivision iterations
            method: Subdivision method ('loop' or 'midpoint')
        
        Returns:
            Subdivided mesh
        """
        try:
            subdivided = mesh
            for _ in range(iterations):
                if method == 'loop':
                    subdivided = subdivided.subdivide_loop(number_of_iterations=1)
                elif method == 'midpoint':
                    subdivided = subdivided.subdivide_midpoint(number_of_iterations=1)
                else:
                    logger.warning(f"Unknown subdivision method '{method}'")
                    break
            
            subdivided.compute_vertex_normals()
            logger.debug(f"Subdivided mesh: {len(subdivided.vertices)} vertices")
            return subdivided
        except Exception as e:
            logger.error(f"Mesh subdivision failed: {e}")
            return mesh


class DepthEstimator:
    """
    Estimates and enhances depth information for facial landmarks
    """
    
    def __init__(self, baseline_depth: float = 0.1):
        """
        Initialize depth estimator
        
        Args:
            baseline_depth: Baseline depth value for normalization
        """
        self.baseline_depth = baseline_depth
    
    def estimate_relative_depth(self, 
                               landmarks: np.ndarray,
                               reference_points: Optional[List[int]] = None) -> np.ndarray:
        """
        Estimate relative depth based on landmark positions
        
        Args:
            landmarks: 3D landmarks
            reference_points: Indices of reference points (e.g., ears, forehead)
        
        Returns:
            Depth map with enhanced values
        """
        depth_map = landmarks[:, 2].copy()
        
        if reference_points is None:
            # Use facial plane as reference
            reference_depth = np.median(depth_map)
        else:
            # Use specific points as reference
            reference_depth = np.mean([depth_map[i] for i in reference_points if i < len(depth_map)])
        
        # Normalize relative to reference
        depth_map = depth_map - reference_depth
        
        return depth_map
    
    def create_depth_map_image(self, 
                              landmarks: np.ndarray,
                              image_shape: Tuple[int, int],
                              colormap: int = 2) -> np.ndarray:
        """
        Create a depth map visualization
        
        Args:
            landmarks: Landmarks with normalized coordinates [0-1] for x,y and depth for z
            image_shape: Output image shape (height, width)
            colormap: OpenCV colormap ID (default: COLORMAP_JET)
        
        Returns:
            Colored depth map image
        """
        import cv2
        
        h, w = image_shape
        depth_map = np.zeros((h, w), dtype=np.float32)
        
        # Project landmarks onto image plane
        for landmark in landmarks:
            x = int(landmark[0] * w)
            y = int(landmark[1] * h)
            z = landmark[2]
            
            if 0 <= x < w and 0 <= y < h:
                depth_map[y, x] = z
        
        # Normalize depth map
        depth_min, depth_max = depth_map[depth_map > 0].min(), depth_map.max()
        depth_normalized = np.clip((depth_map - depth_min) / (depth_max - depth_min + 1e-6), 0, 1)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        
        # Apply colormap
        colored_depth = cv2.applyColorMap(depth_uint8, colormap)
        
        return colored_depth


class MeshTextureMapper:
    """
    Handles UV mapping and texture application for 3D meshes
    """
    
    def __init__(self):
        self.uv_coords = None
    
    def compute_uv_coordinates(self, 
                             landmarks: np.ndarray,
                             method: str = 'cylindrical') -> np.ndarray:
        """
        Compute UV texture coordinates for mesh vertices
        
        Args:
            landmarks: 3D mesh vertices
            method: UV mapping method ('planar', 'cylindrical', or 'spherical')
        
        Returns:
            UV coordinates with shape (N, 2)
        """
        if method == 'planar':
            return self._planar_mapping(landmarks)
        elif method == 'cylindrical':
            return self._cylindrical_mapping(landmarks)
        elif method == 'spherical':
            return self._spherical_mapping(landmarks)
        else:
            logger.warning(f"Unknown UV mapping method '{method}', using planar")
            return self._planar_mapping(landmarks)
    
    def _planar_mapping(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Simple planar UV mapping (project onto XY plane)
        """
        xy = landmarks[:, :2].copy()
        
        # Normalize to [0, 1] range
        min_vals = xy.min(axis=0)
        max_vals = xy.max(axis=0)
        uv = (xy - min_vals) / (max_vals - min_vals + 1e-6)
        
        return uv
    
    def _cylindrical_mapping(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Cylindrical UV mapping around Y axis
        """
        x, y, z = landmarks[:, 0], landmarks[:, 1], landmarks[:, 2]
        
        # Compute cylindrical coordinates
        theta = np.arctan2(x, z)
        u = (theta + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
        
        # Normalize v based on y coordinate
        v = (y - y.min()) / (y.max() - y.min() + 1e-6)
        
        return np.column_stack([u, v])
    
    def _spherical_mapping(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Spherical UV mapping
        """
        x, y, z = landmarks[:, 0], landmarks[:, 1], landmarks[:, 2]
        
        # Compute spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-6
        theta = np.arctan2(x, z)
        phi = np.arccos(np.clip(y / r, -1, 1))
        
        u = (theta + np.pi) / (2 * np.pi)
        v = phi / np.pi
        
        return np.column_stack([u, v])
    
    def apply_texture_to_mesh(self,
                             mesh: o3d.geometry.TriangleMesh,
                             texture_image: np.ndarray,
                             uv_coords: np.ndarray) -> o3d.geometry.TriangleMesh:
        """
        Apply texture image to mesh using UV coordinates
        
        Args:
            mesh: Input mesh
            texture_image: RGB texture image
            uv_coords: UV coordinates for vertices
        
        Returns:
            Textured mesh
        """
        try:
            # Convert texture image to Open3D Image
            texture = o3d.geometry.Image(texture_image)
            
            # Set texture
            mesh.textures = [texture]
            
            # Set UV coordinates for each triangle vertex
            triangles = np.asarray(mesh.triangles)
            triangle_uvs = uv_coords[triangles.flatten()]
            mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
            
            # Set material IDs (all triangles use the same texture)
            mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(triangles), dtype=np.int32))
            
            logger.debug("Texture applied to mesh successfully")
            return mesh
        except Exception as e:
            logger.error(f"Failed to apply texture: {e}")
            return mesh


if __name__ == "__main__":
    # Test geometry engine
    print("Testing GeometryEngine...")
    
    # Create sample landmarks
    np.random.seed(42)
    test_landmarks = np.random.rand(468, 3) * 10
    
    # Initialize engine
    engine = GeometryEngine(use_delaunay=True, smoothing_iterations=3)
    
    # Create mesh
    mesh = engine.create_mesh_from_landmarks(test_landmarks)
    print(f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    
    # Smooth mesh
    smoothed = engine.smooth_mesh(mesh)
    print(f"Smoothed mesh created")
    
    # Test depth estimator
    depth_estimator = DepthEstimator()
    depth_map = depth_estimator.estimate_relative_depth(test_landmarks)
    print(f"Depth map computed: min={depth_map.min():.3f}, max={depth_map.max():.3f}")
    
    print("\nGeometryEngine tests completed successfully!")
