# 3D Face Reconstruction Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green.svg)](https://mediapipe.dev/)

## 📌 Project Overview

This repository hosts a **complete, production-ready pipeline** for **Monocular 3D Face Reconstruction**. By leveraging advanced computer vision techniques and deep geometric learning, this system is capable of inferring high-fidelity 3D facial topology from a single 2D RGB input in real-time.

The framework bridges the gap between planar imagery and volumetric representation, offering a scalable solution for applications in biometrics, digital avatar creation, AR/VR, gaming, and medical visualization. It focuses on maintaining strict topological consistency while optimizing for real-time inference latency.

### ✨ What's New (February 2026)

- ✅ **Complete unified pipeline** with `main.py` orchestrator
- ✅ **Multiple operation modes:** Live, Capture, and Image processing
- ✅ **Advanced mesh export** supporting OBJ, PLY, STL, GLTF formats
- ✅ **Real-time visualization** with Open3D integration
- ✅ **Configuration system** for easy customization
- ✅ **Comprehensive documentation** and examples

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Sourodyuti/3D-FaceReconstruction.git
cd 3D-FaceReconstruction

# Install dependencies
pip install -r requirements.txt

# Run live reconstruction
python main.py live
```

### Basic Usage

```bash
# Live mode with real-time reconstruction
python main.py live

# Capture mode to save a single 3D model
python main.py capture --output my_face

# Process an existing image
python main.py image --input photo.jpg --output face_model

# Show depth map visualization
python main.py live --show-depth
```

**Interactive Controls (Live Mode):**
- `Q` - Quit
- `S` - Save snapshot
- `D` - Toggle depth map
- `M` - Toggle 3D mesh window

## ⚙️ Technical Architecture

The core engine operates through a multi-stage pipeline designed to maximize geometric accuracy and texture retention.

### Pipeline Stages

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Camera    │───▶│   MediaPipe  │───▶│  Geometry   │───▶│    Export    │
│   Input     │    │   Face Mesh  │    │   Engine    │    │  OBJ/PLY/STL │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
       │                   │                    │                  │
       │                   ▼                    ▼                  ▼
       │          ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
       │          │ 468 Landmark │    │ 3D Mesh +   │    │  Texture +   │
       └─────────▶│  Detection   │───▶│  Smoothing  │───▶│   Metadata   │
                  └──────────────┘    └─────────────┘    └──────────────┘
```

### 1. High-Dimensional Feature Extraction

The system initiates by ingesting raw visual data and performing a granular analysis of the input tensor. It utilizes MediaPipe's lightweight convolutional backbone to identify key facial regions, filtering out noise and environmental artifacts to isolate the Region of Interest (ROI) with pixel-perfect precision.

### 2. Dense Landmark Regression

Moving beyond simple bounding boxes, the algorithm projects a dense point cloud (468 discrete vertices) onto the identified face. This stage employs a specialized neural regressor to estimate the spatial coordinates of facial features (eyes, nose, lips) in a normalized coordinate space, effectively "anchoring" the geometry to the image data.

### 3. Orthogonal Depth Inference

This is the critical transformation phase. The system extrapolates the Z-axis (depth) data by analyzing relative landmark distances and learned canonical face models. It mathematically reconstructs the curvature of the face, converting the 2D planar map into a 3D mesh structure that respects human facial anthropometry.

### 4. Mesh Generation & Refinement

Delaunay triangulation creates a manifold surface mesh from the landmark points. The mesh undergoes Laplacian smoothing to reduce noise while preserving geometric features. Quality refinement removes degenerate triangles and ensures topological consistency.

### 5. Texture Mapping & Export

The pipeline extracts pixel data corresponding to mesh vertices to generate UV texture maps. This aligns visual texture with geometric structure, producing textured 3D objects ready for rendering engines or further geometric analysis.

## 🎯 Key Features

### Core Capabilities

- **Real-Time Volumetric Inference:** Optimized algorithms allow for frame-by-frame 3D reconstruction with minimal latency (30+ FPS)
- **6-DoF Head Pose Estimation:** Accurately calculates Yaw, Pitch, and Roll to track head orientation in three-dimensional space
- **Robust Occlusion Handling:** Mesh generation remains stable even when parts of the face are partially obscured
- **Dynamic Mesh Deformations:** Topology adapts instantly to facial micro-expressions, maintaining mesh integrity during movement
- **Multi-Format Export:** OBJ (with MTL/textures), PLY, STL, GLTF/GLB support
- **Configurable Pipeline:** YAML/JSON-based configuration for easy customization

### Advanced Features

- **Kalman Filtering:** Landmark stabilization for smooth tracking
- **Exponential Smoothing:** Reduces jitter in real-time applications
- **Depth Enhancement:** Curvature-based depth amplification for prominent features
- **UV Mapping:** Multiple methods (planar, cylindrical, spherical)
- **Batch Processing:** Process multiple images programmatically
- **Texture Generation:** Automatic texture extraction and enhancement

## 📂 Project Structure

```
3D-FaceReconstruction/
├── main.py                          # Primary entry point and CLI
├── config.yaml                      # Default configuration file
├── requirements.txt                 # Python dependencies
│
├── Core Modules:
├── utils.py                         # Configuration, transformations, preprocessing
├── geometry_engine.py               # Mesh generation, depth estimation, texturing
├── visualization.py                 # Real-time 3D and 2D visualization
├── export_mesh.py                   # Multi-format mesh export
│
├── Legacy Scripts (for reference):
├── face_landmark_detection.py       # Basic landmark detection
├── face_mesh_generation.py          # Simple mesh generation
├── generate_depth_maps.py           # Depth map visualization
├── landmark_stabilization.py        # Kalman filter implementation
└── implement_texture.py             # Texture mapping experiments
│
├── Documentation:
├── README.md                        # This file
├── USAGE.md                         # Comprehensive usage guide
└── LICENSE                          # MIT License
```

## 🛠️ Module Breakdown

### Core Pipeline

| Module | Description | Key Classes |
|--------|-------------|-------------|
| **`main.py`** | Unified entry point with CLI interface | `FaceReconstruction3D` |
| **`geometry_engine.py`** | Geometric operations and mesh processing | `GeometryEngine`, `DepthEstimator`, `MeshTextureMapper` |
| **`visualization.py`** | Real-time visualization utilities | `MeshVisualizer`, `VideoVisualizer`, `DepthMapVisualizer` |
| **`utils.py`** | Shared utilities and configuration | `Config`, `CoordinateTransformer`, `ImagePreprocessor` |
| **`export_mesh.py`** | Multi-format export functionality | `MeshExporter`, `TextureGenerator` |

## 📦 Installation & Requirements

### System Requirements

- **Python:** 3.8 or higher
- **OS:** Windows, macOS, or Linux
- **Camera:** Webcam or USB camera (for live mode)
- **RAM:** 4GB minimum, 8GB recommended
- **GPU:** Optional (CPU-only operation supported)

### Dependencies

Key libraries:
- `mediapipe` >= 0.10.0 - Face landmark detection
- `opencv-python` >= 4.5.0 - Image processing
- `open3d` >= 0.18.0 - 3D visualization and mesh operations
- `numpy` >= 1.21.0 - Numerical computations
- `scipy` >= 1.7.0 - Scientific computing (Delaunay triangulation)
- `pyyaml` >= 6.0 - Configuration file parsing

See [`requirements.txt`](requirements.txt) for complete list.

### Installation Steps

1. **Clone repository:**
   ```bash
   git clone https://github.com/Sourodyuti/3D-FaceReconstruction.git
   cd 3D-FaceReconstruction
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python main.py --help
   ```

## 📖 Documentation

- **[USAGE.md](USAGE.md)** - Comprehensive usage guide with examples
- **[config.yaml](config.yaml)** - Configuration reference
- **Code Documentation** - Inline docstrings and type hints

## 🎮 Usage Examples

### Example 1: Live Reconstruction

```bash
python main.py live --show-depth
```

Opens webcam with real-time 3D reconstruction, depth map visualization, and face orientation tracking.

### Example 2: Capture High-Quality Model

```bash
python main.py capture --output john_face --config high_quality.yaml
```

Capture a single frame and export as multi-format 3D model with custom configuration.

### Example 3: Batch Image Processing

```python
from main import FaceReconstruction3D
import glob

pipeline = FaceReconstruction3D()

for img_path in glob.glob("images/*.jpg"):
    pipeline.process_image_file(img_path, output_filename=f"model_{i}")

pipeline.cleanup()
```

### Example 4: Custom Configuration

```yaml
# custom_config.yaml
mesh:
  scale_factor: 15.0
  smoothing_iterations: 5

export:
  default_format: ply
  texture_size: 2048
```

```bash
python main.py live --config custom_config.yaml
```

## 🔮 Roadmap & Future Scope

We are committed to pushing the boundaries of monocular reconstruction. Planned developments:

### Short-Term (Q1-Q2 2026)

- [ ] **High-Res Texture Baking:** Automated generation with baked normal and displacement maps
- [ ] **Improved Depth Estimation:** Integration of monocular depth estimation networks
- [ ] **Multi-Face Support:** Simultaneous tracking and reconstruction of multiple faces
- [ ] **Expression Transfer:** Retarget facial animations to rigged 3D characters

### Medium-Term (Q3-Q4 2026)

- [ ] **Neural Rendering Integration:** NeRF-based photorealistic lighting estimation
- [ ] **Mesh Subdivision:** Adaptive mesh refinement for higher detail
- [ ] **GPU Acceleration:** CUDA/OpenCL optimization for real-time HD processing
- [ ] **Mobile Deployment:** iOS/Android apps with on-device reconstruction

### Long-Term (2027+)

- [ ] **Edge Optimization:** Model quantization for embedded IoT devices
- [ ] **4D Reconstruction:** Temporal coherence for video-based reconstruction
- [ ] **Physically-Based Rendering:** PBR material estimation and export
- [ ] **Cloud API:** RESTful API for scalable reconstruction services

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/Sourodyuti/3D-FaceReconstruction.git
cd 3D-FaceReconstruction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests (coming soon)
python -m pytest tests/
```

## 📄 License

This project is open-source and available under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe** by Google for robust facial landmark detection
- **Open3D** for powerful 3D geometry processing
- **OpenCV** community for computer vision tools
- All contributors and users of this project

## 📧 Contact

For questions, suggestions, or collaboration:

- **GitHub Issues:** [Report bugs or request features](https://github.com/Sourodyuti/3D-FaceReconstruction/issues)
- **Pull Requests:** Contributions welcome!

---

**Made with ❤️ for the computer vision and 3D graphics community**

⭐ Star this repository if you find it useful!
