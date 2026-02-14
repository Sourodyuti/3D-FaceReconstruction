#!/usr/bin/env python3
"""
Unit tests for video_processor module
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_processor import TemporalSmoother, VideoProcessor


class TestTemporalSmoother:
    """Test TemporalSmoother class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.smoother = TemporalSmoother(window_size=3)
    
    def test_smooth_single_frame(self):
        """Test smoothing with single frame"""
        landmarks = np.random.rand(468, 3)
        
        smoothed = self.smoother.smooth(landmarks)
        
        # First frame should be unchanged
        assert np.allclose(smoothed, landmarks)
    
    def test_smooth_multiple_frames(self):
        """Test smoothing with multiple frames"""
        # Add several frames
        for i in range(5):
            landmarks = np.random.rand(468, 3)
            smoothed = self.smoother.smooth(landmarks)
        
        # Should have history
        assert len(self.smoother.landmark_history) <= 3  # window_size
    
    def test_reset(self):
        """Test smoother reset"""
        # Add some frames
        for i in range(3):
            landmarks = np.random.rand(468, 3)
            self.smoother.smooth(landmarks)
        
        # Reset
        self.smoother.reset()
        
        # History should be empty
        assert len(self.smoother.landmark_history) == 0


class TestVideoProcessor:
    """Test VideoProcessor class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = VideoProcessor()
    
    def test_initialization(self):
        """Test processor initialization"""
        assert self.processor is not None
        assert hasattr(self.processor, 'face_mesh')
        assert hasattr(self.processor, 'geometry_engine')
    
    def test_cleanup(self):
        """Test cleanup"""
        # Should not raise exception
        self.processor.cleanup()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
