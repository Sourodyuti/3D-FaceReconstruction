import open3d as o3d
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class MeshExporter:
    """
    Export 3D face meshes to various file formats
    Supports: OBJ, PLY, STL with textures and metadata
    """
    
    SUPPORTED_FORMATS = ['obj', 'ply', 'stl', 'gltf', 'glb']
    
    def __init__(self, output_dir: str = './output'):
        """
        Initialize mesh exporter
        
        Args:
            output_dir: Directory for exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"MeshExporter initialized. Output: {self.output_dir.absolute()}")
    
    def export_mesh(self,
                   mesh: o3d.geometry.TriangleMesh,
                   filename: str,
                   format: str = 'obj',
                   include_texture: bool = True,
                   texture_image: Optional[np.ndarray] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Export mesh to file
        
        Args:
            mesh: Triangle mesh to export
            filename: Output filename (without extension)
            format: Export format ('obj', 'ply', 'stl', 'gltf', 'glb')
            include_texture: Whether to export texture
            texture_image: Texture image (if available)
            metadata: Additional metadata to save
        
        Returns:
            Path to exported file
        """
        format = format.lower()
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{format}'. Supported: {self.SUPPORTED_FORMATS}")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename}_{timestamp}"
        
        # Export based on format
        if format == 'obj':
            output_path = self._export_obj(mesh, base_filename, include_texture, texture_image)
        elif format == 'ply':
            output_path = self._export_ply(mesh, base_filename, include_texture)
        elif format == 'stl':
            output_path = self._export_stl(mesh, base_filename)
        elif format in ['gltf', 'glb']:
            output_path = self._export_gltf(mesh, base_filename, format, texture_image)
        else:
            raise ValueError(f"Format {format} not implemented")
        
        # Save metadata if provided
        if metadata is not None:
            self._save_metadata(base_filename, metadata)
        
        logger.info(f"Mesh exported: {output_path}")
        return str(output_path)
    
    def _export_obj(self,
                   mesh: o3d.geometry.TriangleMesh,
                   filename: str,
                   include_texture: bool,
                   texture_image: Optional[np.ndarray]) -> Path:
        """
        Export mesh to OBJ format with optional texture
        
        Args:
            mesh: Triangle mesh
            filename: Base filename
            include_texture: Whether to save texture
            texture_image: Texture image
        
        Returns:
            Path to OBJ file
        """
        obj_path = self.output_dir / f"{filename}.obj"
        mtl_path = self.output_dir / f"{filename}.mtl"
        texture_path = self.output_dir / f"{filename}_texture.png"
        
        # Prepare mesh for export
        export_mesh = mesh
        
        # Handle texture
        if include_texture and texture_image is not None:
            # Save texture image
            cv2.imwrite(str(texture_path), texture_image)
            logger.debug(f"Texture saved: {texture_path}")
            
            # Create MTL file
            self._create_mtl_file(mtl_path, texture_path.name)
            
            # Write OBJ with texture reference
            success = o3d.io.write_triangle_mesh(str(obj_path), export_mesh, 
                                                write_ascii=True,
                                                write_vertex_normals=True,
                                                write_vertex_colors=False,
                                                write_triangle_uvs=True)
            
            if success:
                # Add MTL reference to OBJ file
                self._add_mtl_reference_to_obj(obj_path, mtl_path.name)
        else:
            # Export without texture
            success = o3d.io.write_triangle_mesh(str(obj_path), export_mesh,
                                                write_ascii=True,
                                                write_vertex_normals=True,
                                                write_vertex_colors=True)
        
        if not success:
            raise RuntimeError(f"Failed to export OBJ: {obj_path}")
        
        return obj_path
    
    def _create_mtl_file(self, mtl_path: Path, texture_filename: str):
        """Create MTL material file for OBJ"""
        mtl_content = f"""# Material file generated by 3D Face Reconstruction
newmtl material_0
Ka 1.0 1.0 1.0
Kd 1.0 1.0 1.0
Ks 0.0 0.0 0.0
Ns 10.0
illum 2
map_Kd {texture_filename}
"""
        with open(mtl_path, 'w') as f:
            f.write(mtl_content)
        logger.debug(f"MTL file created: {mtl_path}")
    
    def _add_mtl_reference_to_obj(self, obj_path: Path, mtl_filename: str):
        """Add MTL reference to OBJ file"""
        with open(obj_path, 'r') as f:
            content = f.read()
        
        # Add mtllib and usemtl directives
        lines = content.split('\n')
        new_lines = []
        added_mtl = False
        
        for line in lines:
            new_lines.append(line)
            # Add after first comment or vertex line
            if not added_mtl and (line.startswith('#') or line.startswith('v ')):
                if line.startswith('v '):
                    # Insert before first vertex
                    new_lines.insert(-1, f"mtllib {mtl_filename}")
                    new_lines.insert(-1, "usemtl material_0")
                    added_mtl = True
        
        with open(obj_path, 'w') as f:
            f.write('\n'.join(new_lines))
    
    def _export_ply(self,
                   mesh: o3d.geometry.TriangleMesh,
                   filename: str,
                   include_texture: bool) -> Path:
        """
        Export mesh to PLY format
        
        Args:
            mesh: Triangle mesh
            filename: Base filename
            include_texture: Whether to include vertex colors
        
        Returns:
            Path to PLY file
        """
        ply_path = self.output_dir / f"{filename}.ply"
        
        success = o3d.io.write_triangle_mesh(str(ply_path), mesh,
                                            write_ascii=False,
                                            write_vertex_normals=True,
                                            write_vertex_colors=include_texture)
        
        if not success:
            raise RuntimeError(f"Failed to export PLY: {ply_path}")
        
        return ply_path
    
    def _export_stl(self,
                   mesh: o3d.geometry.TriangleMesh,
                   filename: str) -> Path:
        """
        Export mesh to STL format (no texture support)
        
        Args:
            mesh: Triangle mesh
            filename: Base filename
        
        Returns:
            Path to STL file
        """
        stl_path = self.output_dir / f"{filename}.stl"
        
        success = o3d.io.write_triangle_mesh(str(stl_path), mesh,
                                            write_ascii=False)
        
        if not success:
            raise RuntimeError(f"Failed to export STL: {stl_path}")
        
        return stl_path
    
    def _export_gltf(self,
                    mesh: o3d.geometry.TriangleMesh,
                    filename: str,
                    format: str,
                    texture_image: Optional[np.ndarray]) -> Path:
        """
        Export mesh to GLTF/GLB format
        
        Args:
            mesh: Triangle mesh
            filename: Base filename
            format: 'gltf' or 'glb'
            texture_image: Optional texture image
        
        Returns:
            Path to GLTF/GLB file
        """
        extension = format
        output_path = self.output_dir / f"{filename}.{extension}"
        
        try:
            success = o3d.io.write_triangle_mesh(str(output_path), mesh)
            if not success:
                raise RuntimeError(f"Failed to export {format.upper()}: {output_path}")
        except Exception as e:
            logger.warning(f"GLTF export not fully supported in this Open3D version: {e}")
            logger.info("Falling back to OBJ export")
            return self._export_obj(mesh, filename, True, texture_image)
        
        return output_path
    
    def _save_metadata(self, filename: str, metadata: Dict[str, Any]):
        """
        Save metadata as JSON file
        
        Args:
            filename: Base filename
            metadata: Metadata dictionary
        """
        meta_path = self.output_dir / f"{filename}_metadata.json"
        
        # Add timestamp
        metadata['export_time'] = datetime.now().isoformat()
        metadata['exporter_version'] = '1.0.0'
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.debug(f"Metadata saved: {meta_path}")
    
    def export_point_cloud(self,
                          points: np.ndarray,
                          filename: str,
                          colors: Optional[np.ndarray] = None,
                          normals: Optional[np.ndarray] = None,
                          format: str = 'ply') -> str:
        """
        Export point cloud to file
        
        Args:
            points: Point coordinates (N, 3)
            filename: Output filename
            colors: Optional point colors (N, 3)
            normals: Optional point normals (N, 3)
            format: Export format ('ply', 'pcd', 'xyz')
        
        Returns:
            Path to exported file
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{filename}_{timestamp}.{format}"
        
        success = o3d.io.write_point_cloud(str(output_path), pcd)
        
        if not success:
            raise RuntimeError(f"Failed to export point cloud: {output_path}")
        
        logger.info(f"Point cloud exported: {output_path}")
        return str(output_path)
    
    def batch_export(self,
                    mesh: o3d.geometry.TriangleMesh,
                    filename: str,
                    formats: list = None,
                    texture_image: Optional[np.ndarray] = None) -> Dict[str, str]:
        """
        Export mesh to multiple formats at once
        
        Args:
            mesh: Triangle mesh
            filename: Base filename
            formats: List of formats to export (default: ['obj', 'ply'])
            texture_image: Optional texture image
        
        Returns:
            Dictionary mapping format to output path
        """
        if formats is None:
            formats = ['obj', 'ply']
        
        results = {}
        for fmt in formats:
            try:
                output_path = self.export_mesh(mesh, filename, format=fmt,
                                             texture_image=texture_image)
                results[fmt] = output_path
            except Exception as e:
                logger.error(f"Failed to export as {fmt}: {e}")
                results[fmt] = None
        
        return results


class TextureGenerator:
    """
    Generate texture maps for 3D face meshes
    """
    
    @staticmethod
    def create_uv_texture(frame: np.ndarray,
                         landmarks: np.ndarray,
                         uv_coords: np.ndarray,
                         texture_size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
        """
        Create texture image from video frame using UV coordinates
        
        Args:
            frame: Input BGR image
            landmarks: 3D landmarks in normalized space
            uv_coords: UV coordinates (N, 2) in range [0, 1]
            texture_size: Output texture size (width, height)
        
        Returns:
            Texture image
        """
        h, w = frame.shape[:2]
        tex_w, tex_h = texture_size
        
        # Create empty texture
        texture = np.zeros((tex_h, tex_w, 3), dtype=np.uint8)
        
        # Map frame pixels to texture using UV coordinates
        for i, (u, v) in enumerate(uv_coords):
            # Get pixel from frame
            if i < len(landmarks):
                frame_x = int(landmarks[i, 0] * w)
                frame_y = int(landmarks[i, 1] * h)
                
                if 0 <= frame_x < w and 0 <= frame_y < h:
                    # Get texture coordinates
                    tex_x = int(u * tex_w)
                    tex_y = int(v * tex_h)
                    
                    if 0 <= tex_x < tex_w and 0 <= tex_y < tex_h:
                        texture[tex_y, tex_x] = frame[frame_y, frame_x]
        
        # Fill gaps using inpainting
        mask = (texture.sum(axis=2) == 0).astype(np.uint8) * 255
        if mask.sum() > 0:
            texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)
        
        return texture
    
    @staticmethod
    def enhance_texture(texture: np.ndarray) -> np.ndarray:
        """
        Enhance texture quality
        
        Args:
            texture: Input texture image
        
        Returns:
            Enhanced texture
        """
        # Apply bilateral filter to smooth while preserving edges
        enhanced = cv2.bilateralFilter(texture, 9, 75, 75)
        
        # Enhance contrast
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced


if __name__ == "__main__":
    # Test exporter
    print("Testing MeshExporter...")
    
    # Create sample mesh
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh.compute_vertex_normals()
    
    # Create exporter
    exporter = MeshExporter(output_dir='./test_output')
    
    # Export to multiple formats
    results = exporter.batch_export(mesh, "test_mesh", formats=['obj', 'ply', 'stl'])
    
    print("\nExported files:")
    for fmt, path in results.items():
        print(f"  {fmt.upper()}: {path}")
    
    print("\nMeshExporter test completed!")
