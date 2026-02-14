#!/usr/bin/env python3
"""
Video Processor Module
Process video files for 4D face reconstruction with temporal coherence
"""

import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import logging
from tqdm import tqdm
import json
from datetime import datetime

from utils import CoordinateTransformer, MathUtils
from geometry_engine import GeometryEngine, DepthEstimator
from export_mesh import MeshExporter

logger = logging.getLogger(__name__)


class TemporalSmoother:
    """
    Temporal smoothing for landmark sequences
    Ensures smooth transitions between frames
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize temporal smoother
        
        Args:
            window_size: Number of frames to use for smoothing
        """
        self.window_size = window_size
        self.landmark_history = []
    
    def smooth(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing to landmarks
        
        Args:
            landmarks: Current frame landmarks (N, 3)
        
        Returns:
            Smoothed landmarks
        """
        # Add to history
        self.landmark_history.append(landmarks.copy())
        
        # Keep only recent history
        if len(self.landmark_history) > self.window_size:
            self.landmark_history.pop(0)
        
        # Apply moving average
        if len(self.landmark_history) >= 2:
            smoothed = np.mean(self.landmark_history, axis=0)
            return smoothed
        else:
            return landmarks
    
    def reset(self):
        """Reset the smoother state"""
        self.landmark_history.clear()


class VideoProcessor:
    """
    Process video files for 3D face reconstruction
    Supports frame-by-frame processing with temporal coherence
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize video processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize MediaPipe Face Mesh
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
            smoothing_iterations=3
        )
        
        # Initialize coordinate transformer
        self.coord_transformer = CoordinateTransformer()
        
        # Initialize temporal smoother
        self.temporal_smoother = TemporalSmoother(window_size=5)
        
        # Initialize mesh exporter
        self.mesh_exporter = MeshExporter('./output')
        
        logger.info("VideoProcessor initialized")
    
    def process_video(
        self,
        video_path: str,
        output_dir: str = './output/video',
        export_frames: bool = False,
        export_meshes: bool = True,
        export_animation: bool = False,
        skip_frames: int = 0,
        max_frames: Optional[int] = None
    ) -> Dict:
        """
        Process a video file
        
        Args:
            video_path: Path to input video
            output_dir: Directory for output files
            export_frames: Export processed frame images
            export_meshes: Export mesh for each frame
            export_animation: Export as animated sequence
            skip_frames: Skip N frames between processing
            max_frames: Maximum frames to process (None = all)
        
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing video: {video_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video info: {width}x{height} @ {fps} FPS, {frame_count} frames")
        
        # Determine frames to process
        if max_frames:
            frames_to_process = min(max_frames, frame_count)
        else:
            frames_to_process = frame_count
        
        # Reset temporal smoother
        self.temporal_smoother.reset()
        
        # Storage for results
        results = {
            'video_path': video_path,
            'fps': fps,
            'frame_count': frame_count,
            'processed_frames': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'output_dir': str(output_path),
            'meshes': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Process frames
        frame_idx = 0
        processed_count = 0
        
        with tqdm(total=frames_to_process, desc="Processing video") as pbar:
            while cap.isOpened() and processed_count < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed
                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue
                
                # Process frame
                try:
                    landmarks_3d, mesh = self._process_frame(frame, frame_idx)
                    
                    if landmarks_3d is not None and mesh is not None:
                        results['successful_frames'] += 1
                        
                        # Export frame image
                        if export_frames:
                            frame_path = output_path / f"frame_{frame_idx:06d}.png"
                            cv2.imwrite(str(frame_path), frame)
                        
                        # Export mesh
                        if export_meshes:
                            mesh_path = self.mesh_exporter.export_mesh(
                                mesh,
                                f"frame_{frame_idx:06d}",
                                format='ply',
                                include_texture=False
                            )
                            results['meshes'].append({
                                'frame': frame_idx,
                                'path': mesh_path
                            })
                    else:
                        results['failed_frames'] += 1
                
                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx}: {e}")
                    results['failed_frames'] += 1
                
                frame_idx += 1
                processed_count += 1
                results['processed_frames'] = processed_count
                pbar.update(1)
        
        cap.release()
        
        # Export animation if requested
        if export_animation and results['successful_frames'] > 0:
            self._export_animation(results, output_path)
        
        # Save metadata
        metadata_path = output_path / 'processing_results.json'
        with open(metadata_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Video processing complete: {results['successful_frames']}/{results['processed_frames']} frames successful")
        return results
    
    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int
    ) -> Tuple[Optional[np.ndarray], Optional[o3d.geometry.TriangleMesh]]:
        """
        Process a single video frame
        
        Args:
            frame: Input BGR frame
            frame_idx: Frame index
        
        Returns:
            Tuple of (landmarks_3d, mesh) or (None, None) if failed
        """
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, None
        
        # Extract landmarks
        face_landmarks = results.multi_face_landmarks[0]
        landmarks_normalized = np.array([
            [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark[:468]
        ])
        
        # Apply temporal smoothing
        landmarks_normalized = self.temporal_smoother.smooth(landmarks_normalized)
        
        # Transform to 3D coordinates
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
        
        # Smooth mesh
        mesh = self.geometry_engine.smooth_mesh(mesh, method='laplacian', iterations=2)
        
        return landmarks_3d, mesh
    
    def _export_animation(self, results: Dict, output_path: Path):
        """
        Export meshes as animation sequence
        
        Args:
            results: Processing results dictionary
            output_path: Output directory
        """
        logger.info("Exporting animation sequence...")
        
        # Create animation metadata
        animation_data = {
            'fps': results['fps'],
            'frame_count': results['successful_frames'],
            'meshes': results['meshes']
        }
        
        animation_path = output_path / 'animation.json'
        with open(animation_path, 'w') as f:
            json.dump(animation_data, f, indent=4)
        
        logger.info(f"Animation metadata saved: {animation_path}")
    
    def extract_key_frames(
        self,
        video_path: str,
        num_keyframes: int = 10,
        method: str = 'uniform'
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Extract key frames from video
        
        Args:
            video_path: Path to video file
            num_keyframes: Number of key frames to extract
            method: Extraction method ('uniform', 'difference', 'quality')
        
        Returns:
            List of (frame_index, frame_image) tuples
        """
        logger.info(f"Extracting {num_keyframes} key frames using '{method}' method")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keyframes = []
        
        if method == 'uniform':
            # Uniform sampling
            step = max(1, frame_count // num_keyframes)
            indices = list(range(0, frame_count, step))[:num_keyframes]
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    keyframes.append((idx, frame))
        
        elif method == 'difference':
            # Extract frames with highest inter-frame difference
            prev_frame = None
            differences = []
            
            # Calculate differences
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if prev_frame is not None:
                    diff = cv2.absdiff(frame, prev_frame)
                    diff_score = np.mean(diff)
                    differences.append((frame_idx, diff_score, frame))
                
                prev_frame = frame
                frame_idx += 1
            
            # Sort by difference and select top N
            differences.sort(key=lambda x: x[1], reverse=True)
            keyframes = [(idx, frame) for idx, _, frame in differences[:num_keyframes]]
            keyframes.sort(key=lambda x: x[0])  # Sort by frame index
        
        cap.release()
        logger.info(f"Extracted {len(keyframes)} key frames")
        return keyframes
    
    def create_preview_video(
        self,
        input_video: str,
        output_video: str,
        show_landmarks: bool = True,
        show_info: bool = True,
        max_frames: Optional[int] = None
    ):
        """
        Create preview video with overlays
        
        Args:
            input_video: Input video path
            output_video: Output video path
            show_landmarks: Show landmarks on video
            show_info: Show info overlay
            max_frames: Max frames to process
        """
        logger.info(f"Creating preview video: {output_video}")
        
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        frames_to_process = min(max_frames, frame_count) if max_frames else frame_count
        
        with tqdm(total=frames_to_process, desc="Creating preview") as pbar:
            frame_idx = 0
            while cap.isOpened() and frame_idx < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks and show_landmarks:
                    # Draw landmarks
                    for face_landmarks in results.multi_face_landmarks:
                        for landmark in face_landmarks.landmark[:468]:
                            x = int(landmark.x * width)
                            y = int(landmark.y * height)
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                if show_info:
                    # Draw info
                    cv2.putText(frame, f"Frame: {frame_idx}/{frame_count}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if results.multi_face_landmarks:
                        cv2.putText(frame, "Face Detected",
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                out.write(frame)
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        logger.info(f"Preview video created: {output_video}")
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        logger.info("VideoProcessor cleanup complete")


if __name__ == "__main__":
    # Test video processor
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_processor.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Initialize processor
    processor = VideoProcessor()
    
    # Process video
    results = processor.process_video(
        video_path,
        output_dir='./output/video_test',
        export_frames=False,
        export_meshes=True,
        skip_frames=5,  # Process every 5th frame
        max_frames=100  # Limit to 100 frames for testing
    )
    
    print(f"\nProcessing complete!")
    print(f"Successful frames: {results['successful_frames']}")
    print(f"Failed frames: {results['failed_frames']}")
    print(f"Output directory: {results['output_dir']}")
    
    processor.cleanup()
