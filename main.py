#!/usr/bin/env python3
"""
3D Face Reconstruction - Main Entry Point
Unified pipeline for real-time monocular 3D face reconstruction
"""

import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple
import sys

# Import custom modules
from utils import (
    Config, 
    CoordinateTransformer, 
    ImagePreprocessor, 
    MathUtils,
    create_output_directory,
    generate_timestamp
)
from geometry_engine import (
    GeometryEngine, 
    DepthEstimator, 
    MeshTextureMapper
)
from visualization import (
    MeshVisualizer, 
    VideoVisualizer, 
    DepthMapVisualizer
)
from export_mesh import MeshExporter, TextureGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaceReconstruction3D:
    """
    Main class for 3D Face Reconstruction pipeline
    Orchestrates landmark detection, mesh generation, and visualization
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the 3D Face Reconstruction pipeline
        
        Args:
            config_path: Path to configuration file (YAML/JSON)
        """
        # Load configuration
        self.config = Config(config_path)
        logger.info("Configuration loaded")
        
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=self.config.get('mediapipe.max_num_faces', 1),
            refine_landmarks=self.config.get('mediapipe.refine_landmarks', True),
            min_detection_confidence=self.config.get('mediapipe.min_detection_confidence', 0.5),
            min_tracking_confidence=self.config.get('mediapipe.min_tracking_confidence', 0.5)
        )
        logger.info("MediaPipe Face Mesh initialized")
        
        # Initialize geometry engine
        self.geometry_engine = GeometryEngine(
            use_delaunay=True,
            smoothing_iterations=self.config.get('mesh.smoothing_iterations', 3)
        )
        
        # Initialize depth estimator
        self.depth_estimator = DepthEstimator()
        
        # Initialize texture mapper
        self.texture_mapper = MeshTextureMapper()
        
        # Initialize coordinate transformer
        self.coord_transformer = CoordinateTransformer()
        
        # Initialize video visualizer
        self.video_viz = VideoVisualizer()
        
        # Initialize mesh exporter
        output_dir = self.config.get('export.output_dir', './output')
        self.mesh_exporter = MeshExporter(output_dir)
        
        # State variables
        self.current_mesh = None
        self.current_landmarks = None
        self.previous_landmarks = None
        self.frame_count = 0
        
        logger.info("3D Face Reconstruction pipeline initialized")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[o3d.geometry.TriangleMesh]]:
        """
        Process a single frame to extract landmarks and generate 3D mesh
        
        Args:
            frame: Input BGR image from camera
        
        Returns:
            Tuple of (landmarks_3d, mesh) or (None, None) if no face detected
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, None
        
        # Extract landmarks (use first face only)
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array
        landmarks_normalized = np.array([
            [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark[:468]
        ])
        
        # Apply smoothing if enabled
        if self.config.get('stabilization.enable_kalman', True) and self.previous_landmarks is not None:
            landmarks_normalized = MathUtils.smooth_landmarks_exponential(
                landmarks_normalized,
                self.previous_landmarks,
                smoothing_factor=0.7
            )
        
        self.previous_landmarks = landmarks_normalized.copy()
        
        # Transform to 3D world coordinates
        landmarks_3d = self.coord_transformer.mediapipe_to_3d(
            landmarks_normalized,
            scale_xy=self.config.get('mesh.scale_factor', 10.0),
            scale_z=self.config.get('mesh.depth_scale', 10.0),
            invert_y=True
        )
        
        # Enhance depth
        landmarks_3d = self.geometry_engine.enhance_depth(
            landmarks_3d,
            depth_scale=1.0,
            use_curvature=True
        )
        
        # Generate mesh
        mesh = self.geometry_engine.create_mesh_from_landmarks(
            landmarks_3d,
            compute_normals=True
        )
        
        # Apply smoothing if enabled
        if self.config.get('mesh.enable_smoothing', True):
            mesh = self.geometry_engine.smooth_mesh(mesh, method='laplacian')
        
        # Refine mesh quality
        mesh = self.geometry_engine.refine_mesh_quality(mesh)
        
        # Store current state
        self.current_mesh = mesh
        self.current_landmarks = landmarks_normalized
        self.frame_count += 1
        
        return landmarks_3d, mesh
    
    def run_live_mode(self, show_depth: bool = False, show_mesh: bool = True):
        """
        Run live reconstruction from webcam
        
        Args:
            show_depth: Show depth map visualization
            show_mesh: Show 3D mesh visualization
        """
        logger.info("Starting live mode...")
        
        # Initialize camera
        camera_id = self.config.get('camera.device_id', 0)
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('camera.width', 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('camera.height', 480))
        cap.set(cv2.CAP_PROP_FPS, self.config.get('camera.fps', 30))
        
        # Initialize 3D visualizer if needed
        mesh_vis = None
        if show_mesh:
            mesh_vis = MeshVisualizer(
                window_name="3D Face Mesh",
                width=self.config.get('visualization.window_width', 640),
                height=self.config.get('visualization.window_height', 480),
                background_color=self.config.get('visualization.background_color', [0.1, 0.1, 0.1])
            )
            if self.config.get('visualization.show_axes', True):
                mesh_vis.add_coordinate_frame(size=2.0)
        
        logger.info("Press 'q' to quit, 's' to save snapshot, 'd' to toggle depth, 'm' to toggle mesh")
        
        show_depth_map = show_depth
        show_mesh_window = show_mesh
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                # Process frame
                landmarks_3d, mesh = self.process_frame(frame)
                
                # Visualize on video frame
                display_frame = frame.copy()
                
                if landmarks_3d is not None:
                    # Draw landmarks
                    if self.config.get('visualization.show_landmarks', True):
                        display_frame = self.video_viz.draw_landmarks(
                            display_frame,
                            self.current_landmarks,
                            color=(0, 255, 0),
                            radius=1
                        )
                    
                    # Draw face box
                    display_frame = self.video_viz.draw_face_box(
                        display_frame,
                        self.current_landmarks,
                        color=(0, 255, 255)
                    )
                    
                    # Compute and display face orientation
                    orientation = MathUtils.compute_face_orientation(landmarks_3d)
                    info = {
                        'FPS': f"{self.frame_count / (cv2.getTickCount() / cv2.getTickFrequency()):.1f}",
                        'Yaw': f"{orientation['yaw']:.1f}°",
                        'Pitch': f"{orientation['pitch']:.1f}°",
                        'Roll': f"{orientation['roll']:.1f}°",
                        'Landmarks': len(landmarks_3d)
                    }
                    display_frame = self.video_viz.draw_info_panel(
                        display_frame,
                        info,
                        position=(10, 30)
                    )
                    
                    # Update 3D mesh visualization
                    if show_mesh_window and mesh_vis is not None and mesh is not None:
                        mesh_vis.add_mesh(mesh)
                        mesh_vis.update()
                    
                    # Show depth map if enabled
                    if show_depth_map:
                        h, w = frame.shape[:2]
                        depth_map_img = self.depth_estimator.create_depth_map_image(
                            self.current_landmarks,
                            (h, w),
                            colormap=cv2.COLORMAP_JET
                        )
                        cv2.imshow("Depth Map", depth_map_img)
                
                # Display video frame
                cv2.imshow("3D Face Reconstruction - Live", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quitting...")
                    break
                elif key == ord('s'):
                    self._save_snapshot(frame, mesh)
                elif key == ord('d'):
                    show_depth_map = not show_depth_map
                    logger.info(f"Depth map: {'ON' if show_depth_map else 'OFF'}")
                    if not show_depth_map:
                        cv2.destroyWindow("Depth Map")
                elif key == ord('m'):
                    show_mesh_window = not show_mesh_window
                    logger.info(f"3D mesh: {'ON' if show_mesh_window else 'OFF'}")
                    if not show_mesh_window and mesh_vis is not None:
                        mesh_vis.destroy()
                        mesh_vis = None
                    elif show_mesh_window and mesh_vis is None:
                        mesh_vis = MeshVisualizer(window_name="3D Face Mesh")
                        mesh_vis.add_coordinate_frame(size=2.0)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if mesh_vis is not None:
                mesh_vis.destroy()
            logger.info("Live mode terminated")
    
    def _save_snapshot(self, frame: np.ndarray, mesh: Optional[o3d.geometry.TriangleMesh]):
        """Save current frame and mesh as snapshot"""
        timestamp = generate_timestamp()
        output_dir = create_output_directory(self.config.get('export.output_dir', './output'))
        
        # Save frame
        frame_path = output_dir / f"frame_{timestamp}.png"
        cv2.imwrite(str(frame_path), frame)
        logger.info(f"Frame saved: {frame_path}")
        
        # Save mesh if available
        if mesh is not None:
            mesh_path = self.mesh_exporter.export_mesh(
                mesh,
                f"snapshot_{timestamp}",
                format=self.config.get('export.default_format', 'obj'),
                include_texture=False
            )
            logger.info(f"Mesh saved: {mesh_path}")
    
    def run_capture_mode(self, output_filename: str = "captured_face"):
        """
        Capture a single frame and export the 3D model
        
        Args:
            output_filename: Base name for output files
        """
        logger.info("Starting capture mode...")
        
        # Initialize camera
        camera_id = self.config.get('camera.device_id', 0)
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return
        
        logger.info("Press SPACE to capture, 'q' to quit")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                landmarks_3d, mesh = self.process_frame(frame)
                
                # Visualize
                display_frame = frame.copy()
                if self.current_landmarks is not None:
                    display_frame = self.video_viz.draw_landmarks(
                        display_frame,
                        self.current_landmarks,
                        color=(0, 255, 0)
                    )
                    display_frame = self.video_viz.draw_face_box(
                        display_frame,
                        self.current_landmarks
                    )
                
                cv2.putText(display_frame, "Press SPACE to capture", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Capture Mode", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Space bar
                    if mesh is not None:
                        self._export_captured_model(frame, mesh, output_filename)
                        break
                    else:
                        logger.warning("No face detected. Try again.")
                elif key == ord('q'):
                    logger.info("Capture cancelled")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _export_captured_model(self, frame: np.ndarray, mesh: o3d.geometry.TriangleMesh, filename: str):
        """Export captured model in multiple formats"""
        logger.info("Exporting captured model...")
        
        # Generate texture
        if self.current_landmarks is not None:
            uv_coords = self.texture_mapper.compute_uv_coordinates(
                np.asarray(mesh.vertices),
                method='cylindrical'
            )
            texture_image = TextureGenerator.create_uv_texture(
                frame,
                self.current_landmarks,
                uv_coords,
                texture_size=(1024, 1024)
            )
            texture_image = TextureGenerator.enhance_texture(texture_image)
        else:
            texture_image = None
        
        # Export to multiple formats
        formats = ['obj', 'ply', 'stl']
        results = self.mesh_exporter.batch_export(
            mesh,
            filename,
            formats=formats,
            texture_image=texture_image
        )
        
        logger.info("Export complete:")
        for fmt, path in results.items():
            if path:
                logger.info(f"  {fmt.upper()}: {path}")
    
    def process_image_file(self, image_path: str, output_filename: str = "processed"):
        """
        Process a single image file and export 3D model
        
        Args:
            image_path: Path to input image
            output_filename: Base name for output files
        """
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        
        # Process frame
        landmarks_3d, mesh = self.process_frame(frame)
        
        if mesh is None:
            logger.error("No face detected in image")
            return
        
        # Export model
        self._export_captured_model(frame, mesh, output_filename)
        logger.info("Image processing complete")
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        logger.info("Cleanup complete")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="3D Face Reconstruction - Monocular 3D Face Reconstruction from RGB Images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run live reconstruction with default settings
  python main.py live
  
  # Run live mode with depth map visualization
  python main.py live --show-depth
  
  # Capture mode to save a single model
  python main.py capture --output my_face_model
  
  # Process an image file
  python main.py image --input photo.jpg --output face_model
  
  # Use custom configuration
  python main.py live --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['live', 'capture', 'image'],
        help='Operation mode: live (realtime), capture (single frame), image (process file)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output filename for captured/processed model'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input image file path (for image mode)'
    )
    
    parser.add_argument(
        '--show-depth',
        action='store_true',
        help='Show depth map visualization (live mode)'
    )
    
    parser.add_argument(
        '--no-mesh',
        action='store_true',
        help='Disable 3D mesh visualization (live mode)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Print banner
    print("=" * 70)
    print(" 3D FACE RECONSTRUCTION SYSTEM")
    print(" Monocular 3D Face Reconstruction from RGB Images")
    print("=" * 70)
    print()
    
    # Initialize pipeline
    try:
        pipeline = FaceReconstruction3D(config_path=args.config)
        
        # Run appropriate mode
        if args.mode == 'live':
            pipeline.run_live_mode(
                show_depth=args.show_depth,
                show_mesh=not args.no_mesh
            )
        
        elif args.mode == 'capture':
            output_name = args.output if args.output else "captured_face"
            pipeline.run_capture_mode(output_filename=output_name)
        
        elif args.mode == 'image':
            if not args.input:
                logger.error("--input is required for image mode")
                sys.exit(1)
            output_name = args.output if args.output else "processed_face"
            pipeline.process_image_file(args.input, output_filename=output_name)
        
        # Cleanup
        pipeline.cleanup()
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print(" Thank you for using 3D Face Reconstruction System!")
    print("=" * 70)


if __name__ == "__main__":
    main()
