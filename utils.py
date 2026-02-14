import numpy as np
import cv2
import yaml
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Config:
    """Configuration management for the 3D Face Reconstruction system"""
    
    DEFAULT_CONFIG = {
        'camera': {
            'device_id': 0,
            'width': 640,
            'height': 480,
            'fps': 30
        },
        'mediapipe': {
            'max_num_faces': 1,
            'refine_landmarks': True,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5
        },
        'mesh': {
            'scale_factor': 10.0,
            'depth_scale': 10.0,
            'smoothing_iterations': 3,
            'enable_smoothing': True
        },
        'visualization': {
            'window_width': 640,
            'window_height': 480,
            'show_wireframe': True,
            'show_landmarks': True,
            'show_axes': True,
            'background_color': [0.1, 0.1, 0.1]
        },
        'export': {
            'output_dir': './output',
            'default_format': 'obj',
            'texture_size': 1024
        },
        'stabilization': {
            'enable_kalman': True,
            'process_noise': 1e-2,
            'measurement_noise': 1e-1
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or use defaults"""
        self.config = self.DEFAULT_CONFIG.copy()
        if config_path and Path(config_path).exists():
            self.load(config_path)
    
    def load(self, config_path: str):
        """Load configuration from YAML or JSON file"""
        path = Path(config_path)
        try:
            with open(path, 'r') as f:
                if path.suffix in ['.yaml', '.yml']:
                    loaded_config = yaml.safe_load(f)
                elif path.suffix == '.json':
                    loaded_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {path.suffix}")
                
                self._deep_update(self.config, loaded_config)
                logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
    
    def save(self, config_path: str):
        """Save current configuration to file"""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w') as f:
                if path.suffix in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False)
                elif path.suffix == '.json':
                    json.dump(self.config, f, indent=4)
                else:
                    raise ValueError(f"Unsupported config format: {path.suffix}")
                logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionary"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'camera.width')"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value


class CoordinateTransformer:
    """Handle coordinate system transformations"""
    
    @staticmethod
    def normalize_landmarks(landmarks: np.ndarray, 
                          center: bool = True, 
                          scale: float = 1.0) -> np.ndarray:
        """
        Normalize landmark coordinates
        
        Args:
            landmarks: Array of shape (N, 3) with x, y, z coordinates
            center: Whether to center the landmarks at origin
            scale: Scale factor to apply
        
        Returns:
            Normalized landmarks
        """
        normalized = landmarks.copy()
        
        if center:
            centroid = np.mean(normalized, axis=0)
            normalized -= centroid
        
        if scale != 1.0:
            normalized *= scale
        
        return normalized
    
    @staticmethod
    def mediapipe_to_3d(landmarks: np.ndarray, 
                       scale_xy: float = 10.0,
                       scale_z: float = 10.0,
                       invert_y: bool = True) -> np.ndarray:
        """
        Convert MediaPipe normalized coordinates to 3D space
        
        Args:
            landmarks: MediaPipe landmarks (normalized 0-1 range)
            scale_xy: Scaling factor for x and y
            scale_z: Scaling factor for z (depth)
            invert_y: Whether to invert y-axis (for proper orientation)
        
        Returns:
            3D coordinates in world space
        """
        points = landmarks.copy()
        
        # Center around 0.5 and scale
        points[:, 0] = (points[:, 0] - 0.5) * scale_xy
        points[:, 1] = (points[:, 1] - 0.5) * scale_xy
        points[:, 2] = points[:, 2] * scale_z
        
        if invert_y:
            points[:, 1] = -points[:, 1]
        
        return points
    
    @staticmethod
    def compute_bounding_box(landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 3D bounding box for landmarks
        
        Returns:
            min_point, max_point
        """
        min_point = np.min(landmarks, axis=0)
        max_point = np.max(landmarks, axis=0)
        return min_point, max_point
    
    @staticmethod
    def compute_face_dimensions(landmarks: np.ndarray) -> Dict[str, float]:
        """
        Compute face dimensions (width, height, depth)
        
        Returns:
            Dictionary with width, height, depth, and aspect_ratio
        """
        min_point, max_point = CoordinateTransformer.compute_bounding_box(landmarks)
        dimensions = max_point - min_point
        
        return {
            'width': dimensions[0],
            'height': dimensions[1],
            'depth': dimensions[2],
            'aspect_ratio': dimensions[0] / dimensions[1] if dimensions[1] > 0 else 1.0
        }


class ImagePreprocessor:
    """Image preprocessing utilities"""
    
    @staticmethod
    def enhance_image(image: np.ndarray, 
                     denoise: bool = False,
                     enhance_contrast: bool = False) -> np.ndarray:
        """
        Enhance image quality for better landmark detection
        
        Args:
            image: Input BGR image
            denoise: Apply denoising
            enhance_contrast: Apply contrast enhancement
        
        Returns:
            Enhanced image
        """
        enhanced = image.copy()
        
        if denoise:
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        if enhance_contrast:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    @staticmethod
    def resize_with_aspect_ratio(image: np.ndarray, 
                                 target_width: int = None,
                                 target_height: int = None) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: Input image
            target_width: Target width (if None, calculated from height)
            target_height: Target height (if None, calculated from width)
        
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        
        if target_width is None and target_height is None:
            return image
        
        if target_width is None:
            ratio = target_height / h
            target_width = int(w * ratio)
        elif target_height is None:
            ratio = target_width / w
            target_height = int(h * ratio)
        
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    @staticmethod
    def extract_face_region(image: np.ndarray, 
                          landmarks: np.ndarray,
                          padding: float = 0.2) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract face region from image based on landmarks
        
        Args:
            image: Input image
            landmarks: Normalized landmarks (0-1 range) with shape (N, 3)
            padding: Padding ratio around face
        
        Returns:
            Cropped face image and bounding box (x, y, w, h)
        """
        h, w = image.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        x_coords = landmarks[:, 0] * w
        y_coords = landmarks[:, 1] * h
        
        # Compute bounding box
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        
        # Add padding
        width = x_max - x_min
        height = y_max - y_min
        x_min = max(0, int(x_min - width * padding))
        y_min = max(0, int(y_min - height * padding))
        x_max = min(w, int(x_max + width * padding))
        y_max = min(h, int(y_max + height * padding))
        
        # Crop image
        cropped = image[y_min:y_max, x_min:x_max]
        return cropped, (x_min, y_min, x_max - x_min, y_max - y_min)


class MathUtils:
    """Mathematical utility functions"""
    
    @staticmethod
    def compute_face_orientation(landmarks: np.ndarray) -> Dict[str, float]:
        """
        Compute face orientation (yaw, pitch, roll) in degrees
        
        Args:
            landmarks: 3D landmarks array
        
        Returns:
            Dictionary with yaw, pitch, roll angles
        """
        # Use key facial landmarks for orientation
        # Approximate indices: nose tip, chin, left eye, right eye
        if landmarks.shape[0] < 468:
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
        
        nose_tip = landmarks[1]
        chin = landmarks[152]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        # Calculate yaw (left-right rotation)
        eye_center = (left_eye + right_eye) / 2
        yaw = np.arctan2(nose_tip[0] - eye_center[0], nose_tip[2] - eye_center[2])
        yaw = np.degrees(yaw)
        
        # Calculate pitch (up-down rotation)
        pitch = np.arctan2(nose_tip[1] - chin[1], nose_tip[2] - chin[2])
        pitch = np.degrees(pitch)
        
        # Calculate roll (tilt)
        roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        roll = np.degrees(roll)
        
        return {'yaw': yaw, 'pitch': pitch, 'roll': roll}
    
    @staticmethod
    def interpolate_landmarks(landmarks1: np.ndarray, 
                            landmarks2: np.ndarray, 
                            alpha: float) -> np.ndarray:
        """
        Interpolate between two landmark sets
        
        Args:
            landmarks1: First landmark set
            landmarks2: Second landmark set
            alpha: Interpolation factor (0.0 to 1.0)
        
        Returns:
            Interpolated landmarks
        """
        return landmarks1 * (1 - alpha) + landmarks2 * alpha
    
    @staticmethod
    def smooth_landmarks_exponential(landmarks: np.ndarray,
                                    previous_landmarks: Optional[np.ndarray],
                                    smoothing_factor: float = 0.5) -> np.ndarray:
        """
        Apply exponential smoothing to landmarks
        
        Args:
            landmarks: Current landmarks
            previous_landmarks: Previous frame landmarks
            smoothing_factor: Smoothing strength (0.0 to 1.0)
        
        Returns:
            Smoothed landmarks
        """
        if previous_landmarks is None:
            return landmarks
        
        return smoothing_factor * landmarks + (1 - smoothing_factor) * previous_landmarks


def create_output_directory(output_dir: str) -> Path:
    """Create output directory if it doesn't exist"""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {path.absolute()}")
    return path


def generate_timestamp() -> str:
    """Generate timestamp string for file naming"""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    # Test configuration
    config = Config()
    print("Default configuration loaded")
    print(f"Camera device: {config.get('camera.device_id')}")
    print(f"Mesh scale: {config.get('mesh.scale_factor')}")
    
    # Test coordinate transformer
    test_landmarks = np.random.rand(468, 3)
    transformed = CoordinateTransformer.mediapipe_to_3d(test_landmarks)
    print(f"\nTransformed landmarks shape: {transformed.shape}")
    
    # Test face dimensions
    dims = CoordinateTransformer.compute_face_dimensions(transformed)
    print(f"Face dimensions: {dims}")
