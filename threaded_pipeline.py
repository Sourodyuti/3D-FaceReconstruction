#!/usr/bin/env python3
"""
Threaded Pipeline Module
Multi-threaded processing pipeline for improved real-time performance
"""

import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
from typing import Optional, Tuple, Dict
import threading
import queue
import time
import logging
from dataclasses import dataclass
from enum import Enum

from utils import CoordinateTransformer, MathUtils
from geometry_engine import GeometryEngine
from visualization import VideoVisualizer

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Pipeline state enumeration"""
    STOPPED = 0
    RUNNING = 1
    PAUSED = 2
    STOPPING = 3


@dataclass
class FrameData:
    """Data container for frame processing"""
    frame_id: int
    timestamp: float
    frame: np.ndarray
    landmarks: Optional[np.ndarray] = None
    landmarks_3d: Optional[np.ndarray] = None
    mesh: Optional[o3d.geometry.TriangleMesh] = None
    processed: bool = False


class CameraThread(threading.Thread):
    """
    Dedicated thread for camera capture
    Ensures consistent frame capture without blocking
    """
    
    def __init__(self, camera_id: int, frame_queue: queue.Queue, config: Dict):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.frame_queue = frame_queue
        self.config = config
        self.state = PipelineState.STOPPED
        self.cap = None
        self.frame_count = 0
        
    def run(self):
        """Main camera capture loop"""
        logger.info(f"Camera thread starting (device {self.camera_id})")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('width', 640))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('height', 480))
        self.cap.set(cv2.CAP_PROP_FPS, self.config.get('fps', 30))
        
        self.state = PipelineState.RUNNING
        
        while self.state == PipelineState.RUNNING:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                time.sleep(0.01)
                continue
            
            # Create frame data
            frame_data = FrameData(
                frame_id=self.frame_count,
                timestamp=time.time(),
                frame=frame
            )
            
            # Try to add to queue (non-blocking)
            try:
                self.frame_queue.put_nowait(frame_data)
                self.frame_count += 1
            except queue.Full:
                # Queue full, skip this frame
                pass
        
        # Cleanup
        if self.cap:
            self.cap.release()
        logger.info("Camera thread stopped")
    
    def stop(self):
        """Stop the camera thread"""
        self.state = PipelineState.STOPPED


class ProcessingThread(threading.Thread):
    """
    Dedicated thread for face detection and 3D reconstruction
    Handles compute-intensive operations
    """
    
    def __init__(self, 
                 input_queue: queue.Queue, 
                 output_queue: queue.Queue,
                 config: Dict):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.config = config
        self.state = PipelineState.STOPPED
        
        # Initialize MediaPipe
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize geometry engine
        self.geometry_engine = GeometryEngine(
            use_delaunay=True,
            smoothing_iterations=2  # Reduced for speed
        )
        
        # Initialize coordinate transformer
        self.coord_transformer = CoordinateTransformer()
        
        # Previous landmarks for smoothing
        self.prev_landmarks = None
        
        # Statistics
        self.processed_count = 0
        self.failed_count = 0
        
    def run(self):
        """Main processing loop"""
        logger.info("Processing thread starting")
        self.state = PipelineState.RUNNING
        
        while self.state == PipelineState.RUNNING:
            try:
                # Get frame data with timeout
                frame_data = self.input_queue.get(timeout=0.1)
                
                # Process frame
                self._process_frame(frame_data)
                
                # Send to output queue
                try:
                    self.output_queue.put_nowait(frame_data)
                except queue.Full:
                    # Output queue full, drop oldest
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(frame_data)
                    except queue.Empty:
                        pass
                
            except queue.Empty:
                # No frame available, continue
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
                self.failed_count += 1
        
        # Cleanup
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        logger.info(f"Processing thread stopped (processed: {self.processed_count}, failed: {self.failed_count})")
    
    def _process_frame(self, frame_data: FrameData):
        """Process a single frame"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            frame_data.processed = False
            self.failed_count += 1
            return
        
        # Extract landmarks
        face_landmarks = results.multi_face_landmarks[0]
        landmarks_normalized = np.array([
            [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark[:468]
        ])
        
        # Apply exponential smoothing
        if self.prev_landmarks is not None:
            landmarks_normalized = MathUtils.smooth_landmarks_exponential(
                landmarks_normalized,
                self.prev_landmarks,
                smoothing_factor=0.7
            )
        self.prev_landmarks = landmarks_normalized.copy()
        
        # Transform to 3D
        landmarks_3d = self.coord_transformer.mediapipe_to_3d(
            landmarks_normalized,
            scale_xy=10.0,
            scale_z=10.0,
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
        
        # Light smoothing (for speed)
        mesh = self.geometry_engine.smooth_mesh(mesh, method='laplacian', iterations=1)
        
        # Update frame data
        frame_data.landmarks = landmarks_normalized
        frame_data.landmarks_3d = landmarks_3d
        frame_data.mesh = mesh
        frame_data.processed = True
        self.processed_count += 1
    
    def stop(self):
        """Stop the processing thread"""
        self.state = PipelineState.STOPPED


class VisualizationThread(threading.Thread):
    """
    Dedicated thread for rendering and display
    Handles OpenCV and Open3D visualization
    """
    
    def __init__(self, 
                 input_queue: queue.Queue,
                 config: Dict,
                 show_mesh: bool = True):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.config = config
        self.show_mesh = show_mesh
        self.state = PipelineState.STOPPED
        
        # Initialize video visualizer
        self.video_viz = VideoVisualizer()
        
        # Statistics
        self.display_count = 0
        self.fps = 0.0
        self.last_time = time.time()
        
    def run(self):
        """Main visualization loop"""
        logger.info("Visualization thread starting")
        self.state = PipelineState.RUNNING
        
        while self.state == PipelineState.RUNNING:
            try:
                # Get processed frame with timeout
                frame_data = self.input_queue.get(timeout=0.1)
                
                # Display frame
                self._display_frame(frame_data)
                self.display_count += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.fps = self.display_count / (current_time - self.last_time)
                    self.display_count = 0
                    self.last_time = current_time
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.state = PipelineState.STOPPING
                    break
                
            except queue.Empty:
                # No frame available, check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.state = PipelineState.STOPPING
                    break
                continue
        
        # Cleanup
        cv2.destroyAllWindows()
        logger.info("Visualization thread stopped")
    
    def _display_frame(self, frame_data: FrameData):
        """Display a processed frame"""
        display_frame = frame_data.frame.copy()
        
        if frame_data.processed and frame_data.landmarks is not None:
            # Draw landmarks
            display_frame = self.video_viz.draw_landmarks(
                display_frame,
                frame_data.landmarks,
                color=(0, 255, 0),
                radius=1
            )
            
            # Draw face box
            display_frame = self.video_viz.draw_face_box(
                display_frame,
                frame_data.landmarks,
                color=(0, 255, 255)
            )
            
            # Compute orientation
            if frame_data.landmarks_3d is not None:
                orientation = MathUtils.compute_face_orientation(frame_data.landmarks_3d)
                
                # Draw info panel
                info = {
                    'FPS': f"{self.fps:.1f}",
                    'Frame': frame_data.frame_id,
                    'Yaw': f"{orientation['yaw']:.1f}°",
                    'Pitch': f"{orientation['pitch']:.1f}°",
                    'Roll': f"{orientation['roll']:.1f}°"
                }
                display_frame = self.video_viz.draw_info_panel(
                    display_frame,
                    info,
                    position=(10, 30)
                )
        else:
            # No face detected
            cv2.putText(display_frame, "No face detected",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"FPS: {self.fps:.1f}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display
        cv2.imshow("Threaded Pipeline - 3D Face Reconstruction", display_frame)
    
    def stop(self):
        """Stop the visualization thread"""
        self.state = PipelineState.STOPPED


class ThreadedPipeline:
    """
    Multi-threaded pipeline for real-time 3D face reconstruction
    Separates capture, processing, and visualization for optimal performance
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize threaded pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Queues for inter-thread communication
        self.capture_queue = queue.Queue(maxsize=5)
        self.processing_queue = queue.Queue(maxsize=5)
        
        # Threads
        self.camera_thread = None
        self.processing_thread = None
        self.visualization_thread = None
        
        # State
        self.running = False
        
        logger.info("ThreadedPipeline initialized")
    
    def start(self, camera_id: int = 0, show_mesh: bool = True):
        """
        Start the threaded pipeline
        
        Args:
            camera_id: Camera device ID
            show_mesh: Whether to show 3D mesh visualization
        """
        if self.running:
            logger.warning("Pipeline already running")
            return
        
        logger.info("Starting threaded pipeline...")
        
        # Camera configuration
        camera_config = {
            'width': self.config.get('camera.width', 640),
            'height': self.config.get('camera.height', 480),
            'fps': self.config.get('camera.fps', 30)
        }
        
        # Create and start threads
        self.camera_thread = CameraThread(camera_id, self.capture_queue, camera_config)
        self.processing_thread = ProcessingThread(self.capture_queue, self.processing_queue, self.config)
        self.visualization_thread = VisualizationThread(self.processing_queue, self.config, show_mesh)
        
        self.camera_thread.start()
        self.processing_thread.start()
        self.visualization_thread.start()
        
        self.running = True
        logger.info("Threaded pipeline started")
        
        # Wait for visualization thread to finish (user pressed 'q')
        self.visualization_thread.join()
        
        # Stop other threads
        self.stop()
    
    def stop(self):
        """Stop the threaded pipeline"""
        if not self.running:
            return
        
        logger.info("Stopping threaded pipeline...")
        
        # Stop threads
        if self.camera_thread:
            self.camera_thread.stop()
        if self.processing_thread:
            self.processing_thread.stop()
        if self.visualization_thread:
            self.visualization_thread.stop()
        
        # Wait for threads to finish (with timeout)
        timeout = 2.0
        if self.camera_thread:
            self.camera_thread.join(timeout)
        if self.processing_thread:
            self.processing_thread.join(timeout)
        if self.visualization_thread:
            self.visualization_thread.join(timeout)
        
        self.running = False
        logger.info("Threaded pipeline stopped")
    
    def get_stats(self) -> Dict:
        """
        Get pipeline statistics
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'running': self.running,
            'camera_frames': self.camera_thread.frame_count if self.camera_thread else 0,
            'processed_frames': self.processing_thread.processed_count if self.processing_thread else 0,
            'failed_frames': self.processing_thread.failed_count if self.processing_thread else 0,
            'display_fps': self.visualization_thread.fps if self.visualization_thread else 0.0,
            'capture_queue_size': self.capture_queue.qsize(),
            'processing_queue_size': self.processing_queue.qsize()
        }
        return stats


if __name__ == "__main__":
    # Test threaded pipeline
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print(" THREADED PIPELINE TEST")
    print(" Multi-threaded 3D Face Reconstruction")
    print("=" * 70)
    print()
    print("Press 'Q' to quit")
    print()
    
    # Initialize pipeline
    config = {
        'camera.width': 640,
        'camera.height': 480,
        'camera.fps': 30
    }
    
    pipeline = ThreadedPipeline(config)
    
    try:
        # Start pipeline
        camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        pipeline.start(camera_id=camera_id, show_mesh=False)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pipeline.stop()
        
        # Print final stats
        stats = pipeline.get_stats()
        print("\n" + "=" * 70)
        print(" FINAL STATISTICS")
        print("=" * 70)
        print(f"  Camera frames captured: {stats['camera_frames']}")
        print(f"  Frames processed: {stats['processed_frames']}")
        print(f"  Frames failed: {stats['failed_frames']}")
        print(f"  Average FPS: {stats['display_fps']:.1f}")
        print("=" * 70)
