#!/usr/bin/env python3
"""
Unit tests for utils module
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import CoordinateTransformer, MathUtils, Config


class TestCoordinateTransformer:
    """Test CoordinateTransformer class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.transformer = CoordinateTransformer()
    
    def test_mediapipe_to_3d(self):
        """Test MediaPipe to 3D coordinate transformation"""
        # Create sample normalized landmarks
        landmarks = np.array([
            [0.5, 0.5, 0.0],
            [0.3, 0.3, 0.1],
            [0.7, 0.7, -0.1]
        ])
        
        result = self.transformer.mediapipe_to_3d(
            landmarks,
            scale_xy=10.0,
            scale_z=10.0,
            invert_y=True
        )
        
        # Check shape
        assert result.shape == landmarks.shape
        
        # Check scaling applied
        assert np.abs(result[0, 0] - 5.0) < 0.01  # x scaled
        assert np.abs(result[0, 1] + 5.0) < 0.01  # y scaled and inverted
        assert np.abs(result[0, 2] - 0.0) < 0.01  # z scaled
    
    def test_normalize_landmarks(self):
        """Test landmark normalization"""
        landmarks = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        normalized = self.transformer.normalize_landmarks(landmarks)
        
        # Check centering
        assert np.abs(np.mean(normalized, axis=0)).sum() < 0.01


class TestMathUtils:
    """Test MathUtils class"""
    
    def test_smooth_landmarks_exponential(self):
        """Test exponential smoothing"""
        current = np.array([[1.0, 1.0, 1.0]])
        previous = np.array([[0.0, 0.0, 0.0]])
        
        smoothed = MathUtils.smooth_landmarks_exponential(
            current, previous, smoothing_factor=0.5
        )
        
        # Should be between current and previous
        assert np.all(smoothed >= 0.0)
        assert np.all(smoothed <= 1.0)
    
    def test_compute_distance(self):
        """Test distance computation"""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([3.0, 4.0, 0.0])
        
        distance = MathUtils.compute_distance(p1, p2)
        
        # Should be 5.0 (3-4-5 triangle)
        assert np.abs(distance - 5.0) < 0.01
    
    def test_compute_face_orientation(self):
        """Test face orientation computation"""
        # Create simple face landmarks
        landmarks = np.random.rand(468, 3) * 10
        
        orientation = MathUtils.compute_face_orientation(landmarks)
        
        # Check all keys present
        assert 'yaw' in orientation
        assert 'pitch' in orientation
        assert 'roll' in orientation
        
        # Check values in reasonable range
        assert -180 <= orientation['yaw'] <= 180
        assert -180 <= orientation['pitch'] <= 180
        assert -180 <= orientation['roll'] <= 180


class TestConfig:
    """Test Config class"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = Config()
        
        # Should have default values
        value = config.get('camera.device_id', -1)
        assert value != -1  # Should have a default
    
    def test_nested_get(self):
        """Test nested configuration access"""
        config = Config()
        
        # Test nested key access
        value = config.get('camera.width', 0)
        assert value > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
