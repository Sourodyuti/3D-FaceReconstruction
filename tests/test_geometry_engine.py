#!/usr/bin/env python3
"""
Unit tests for geometry_engine module
"""

import pytest
import numpy as np
import open3d as o3d
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geometry_engine import GeometryEngine, DepthEstimator, MeshTextureMapper


class TestGeometryEngine:
    """Test GeometryEngine class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.engine = GeometryEngine(use_delaunay=True, smoothing_iterations=2)
    
    def test_create_mesh_from_landmarks(self):
        """Test mesh creation from landmarks"""
        # Create sample 3D landmarks
        landmarks = np.random.rand(468, 3) * 10
        
        mesh = self.engine.create_mesh_from_landmarks(
            landmarks,
            compute_normals=True
        )
        
        # Check mesh properties
        assert mesh is not None
        assert len(mesh.vertices) > 0
        assert len(mesh.triangles) > 0
        
        # Check if manifold (optional, might fail for complex geometries)
        # is_manifold = mesh.is_vertex_manifold()
    
    def test_smooth_mesh(self):
        """Test mesh smoothing"""
        # Create simple mesh
        landmarks = np.random.rand(100, 3) * 5
        mesh = self.engine.create_mesh_from_landmarks(landmarks)
        
        original_vertex_count = len(mesh.vertices)
        
        # Smooth mesh
        smoothed = self.engine.smooth_mesh(
            mesh,
            method='laplacian',
            iterations=2
        )
        
        # Vertex count should remain same
        assert len(smoothed.vertices) == original_vertex_count
    
    def test_enhance_depth(self):
        """Test depth enhancement"""
        landmarks = np.random.rand(468, 3) * 10
        
        enhanced = self.engine.enhance_depth(
            landmarks,
            depth_scale=1.5,
            use_curvature=True
        )
        
        # Shape should be preserved
        assert enhanced.shape == landmarks.shape
    
    def test_refine_mesh_quality(self):
        """Test mesh quality refinement"""
        landmarks = np.random.rand(200, 3) * 10
        mesh = self.engine.create_mesh_from_landmarks(landmarks)
        
        refined = self.engine.refine_mesh_quality(mesh)
        
        # Should still have vertices and triangles
        assert len(refined.vertices) > 0
        assert len(refined.triangles) > 0


class TestDepthEstimator:
    """Test DepthEstimator class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.estimator = DepthEstimator()
    
    def test_estimate_depth_from_landmarks(self):
        """Test depth estimation"""
        landmarks = np.random.rand(468, 3)
        
        depth_values = self.estimator.estimate_depth_from_landmarks(
            landmarks,
            method='relative'
        )
        
        # Should return depth values
        assert depth_values is not None
        assert len(depth_values) == len(landmarks)


class TestMeshTextureMapper:
    """Test MeshTextureMapper class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mapper = MeshTextureMapper()
    
    def test_compute_uv_coordinates(self):
        """Test UV coordinate computation"""
        vertices = np.random.rand(100, 3) * 10
        
        uv_coords = self.mapper.compute_uv_coordinates(
            vertices,
            method='cylindrical'
        )
        
        # Check UV shape
        assert uv_coords.shape[0] == vertices.shape[0]
        assert uv_coords.shape[1] == 2
        
        # UV coordinates should be in [0, 1] range
        assert np.all(uv_coords >= 0.0)
        assert np.all(uv_coords <= 1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
