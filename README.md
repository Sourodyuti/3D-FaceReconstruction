# 3D Face Reconstruction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green.svg)](https://mediapipe.dev/)

A real-time pipeline that takes a 2D image of a face (from a webcam, video, or photo) and reconstructs a full 3D surface mesh from it using only a single camera. No depth sensors required.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Core Concepts Explained](#core-concepts-explained)
- [How the Full Pipeline Works](#how-the-full-pipeline-works)
- [Module Reference](#module-reference)
- [Getting Started](#getting-started)
- [Running the Program](#running-the-program)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Roadmap](#roadmap)

---

## What This Project Does

Given a face (live webcam, a video file, or a still image), this system:

1. Detects 468 specific landmark points on the face using MediaPipe.
2. Converts those 2D points into 3D coordinates by inferring depth.
3. Builds a triangle mesh (a surface made of connected triangles) from those 3D points.
4. Smooths and refines the mesh.
5. Maps the original face texture onto the mesh.
6. Exports the final 3D model in standard formats (OBJ, PLY, STL, GLTF).

The result is a textured 3D face model that can be opened in Blender, Unity, or any 3D viewer.

---

## Core Concepts Explained

### What is Monocular 3D Reconstruction?

"Monocular" means a single camera. Stereo cameras (like human eyes) can measure depth by comparing two slightly different views. A monocular camera has only one view, so depth must be *inferred* mathematically from clues like relative distances between face features. This project does exactly that: it estimates depth from a single RGB image.

### What are Facial Landmarks?

A facial landmark is a specific, meaningful point on a face -- the corner of the left eye, the tip of the nose, the edge of the lip. This project uses **MediaPipe Face Mesh**, a neural network by Google, which detects **468 landmarks** per face. Each landmark has three coordinates: X (horizontal), Y (vertical), and Z (depth).

```
Front view of face with landmark points:

         *   *   *       <-- forehead landmarks
       *           *
      * [L eye] [R eye] *
       *           *
          *     *        <-- nose bridge
            * *          <-- nose tip
         *       *
        * [mouth] *
         *       *
           * * *         <-- chin

Total: 468 points, each with (X, Y, Z)
```

### What is a Triangle Mesh?

A 3D object in computer graphics is represented as a collection of triangles. Each triangle connects three 3D points (called vertices). When thousands of small triangles are stitched together, they form a smooth-looking surface -- just like the face of a character in a video game.

```
     Vertices (dots)         Triangulated mesh

      .   .   .                /\ /\ /\
        .   .               /\/  \/  \/\
          .               /  /\  /\  /\ \
                         /__/  \/__\/__\ \

  Raw 3D points  -->  Connected triangle surface
```

### What is Delaunay Triangulation?

When you have a cloud of 3D points (the 468 landmarks), you need to decide which points to connect with triangles. Delaunay triangulation is an algorithm that connects points in such a way that:
- No point falls inside any triangle's circumcircle (avoids very thin/spiky triangles).
- Triangles are as equilateral as possible, giving a clean, stable mesh.

This project projects the 3D points onto the 2D X-Y plane and runs Delaunay triangulation there, then lifts the connectivity back to 3D.

### What is Laplacian Smoothing?

The raw mesh generated from 468 landmark points can have bumpy or jagged surfaces because landmark detection is not pixel-perfect. Laplacian smoothing fixes this by moving each vertex toward the average position of its neighbors, repeated a few times. The result is a smoother, more natural-looking surface.

```
Before smoothing:       After smoothing:

  *    *                  *    *
    *                       *
  *    *                  *    *
 (jagged)               (smoothed)
```

### What are Vertex Normals?

A normal is a vector (an arrow) that points perpendicular to a surface at a given point. Normals are used by rendering engines to compute lighting -- how light bounces off a surface. Computing vertex normals makes the 3D model look correctly lit when loaded in Blender or Unity.

### What is UV Mapping / Texture Mapping?

A 3D mesh has no color by itself. UV mapping is the process of projecting the 3D mesh onto a 2D "unwrapped" image (the UV map), then painting the face's original photo pixels onto that image. When the renderer wraps the UV image back onto the 3D mesh, the face looks photorealistic.

This project supports three UV unwrapping methods:
- **Planar**: Straightforward projection onto the XY plane. Fast but distorts edges.
- **Cylindrical**: Wraps the mesh around an imaginary cylinder. Better for faces viewed from the front.
- **Spherical**: Wraps using spherical coordinates. Best for fully 360-degree models.

### What is a Depth Map?

A depth map is a grayscale (or false-color) image where the brightness of each pixel represents how far that point is from the camera. Bright pixels are close; dark pixels are far away. This project can generate and display a real-time depth map of the detected face using a JET colormap (red = close, blue = far).

### What is Kalman Filtering / Exponential Smoothing?

In live webcam mode, landmark positions jump around slightly from frame to frame due to detection noise. Kalman filtering and exponential smoothing are mathematical techniques that track a signal (the landmark position) over time and filter out these tiny random jumps, producing a stable, smooth animation.

### What is 6-DoF Head Pose Estimation?

DoF stands for Degrees of Freedom. A head in 3D space has 6 degrees of freedom:
- **Translation**: X, Y, Z position (where the head is)
- **Rotation**: Yaw (left-right), Pitch (up-down), Roll (tilt)

This project computes Yaw, Pitch, and Roll from the 3D landmarks and displays them in real time, which is useful for gaze tracking, AR face filters, and animation.

### What are OBJ, PLY, STL, GLTF formats?

| Format | Used for |
|---|---|
| OBJ | Universal 3D format; supported by Blender, Maya, Unity. Can include textures via a .MTL file |
| PLY | Stanford format; stores geometry and often color per vertex. Common in research |
| STL | Used for 3D printing; stores only surface geometry, no color |
| GLTF/GLB | Modern web and game engine format; supports textures, animations, PBR materials |

---

## How the Full Pipeline Works

### High-Level Flow

```
+-------------------+
| Input Source      |
| (Webcam / Image / |
|  Video file)      |
+--------+----------+
         |
         v
+--------+----------+
| Frame Capture     |  OpenCV reads each frame
| (cv2.VideoCapture)|  and converts BGR -> RGB
+--------+----------+
         |
         v
+--------+----------+
| MediaPipe         |  Neural network detects
| Face Mesh         |  468 facial landmarks
| (468 landmarks)   |  with X, Y, Z coordinates
+--------+----------+
         |
         v
+--------+----------+
| Landmark          |  Exponential smoothing removes
| Stabilization     |  frame-to-frame jitter
+--------+----------+
         |
         v
+--------+----------+
| Coordinate        |  Normalized [0-1] coords
| Transformation    |  scaled to world space (e.g. 10x)
+--------+----------+
         |
         v
+--------+----------+
| Depth Enhancement |  Z-axis enhanced using
|                   |  curvature heuristics for
|                   |  nose, eyes, mouth regions
+--------+----------+
         |
         v
+--------+----------+
| Delaunay          |  468 points connected into
| Triangulation     |  triangle mesh (approx 900 triangles)
+--------+----------+
         |
         v
+--------+----------+
| Mesh Refinement   |  Laplacian smoothing,
|                   |  remove degenerate triangles,
|                   |  compute vertex normals
+--------+----------+
         |
         v
+---+----+----+-----+
    |         |
    v         v
+-------+  +----------+
| 3D    |  | Texture   |
| View  |  | Mapping   |  UV coordinates computed
| Open3D|  | (UV map)  |  and face photo applied
+-------+  +-----+----+
                 |
                 v
          +------+------+
          | Export      |
          | OBJ/PLY/STL |
          | /GLTF       |
          +-------------+
```

### Frame Processing Detail (process_frame)

Every single frame (from webcam or video) goes through this exact sequence inside `main.py`:

```
Raw BGR Frame
    |
    +--> Convert BGR to RGB (MediaPipe requires RGB)
    |
    +--> mediapipe.face_mesh.process(rgb_frame)
    |         |
    |         +--> Returns 468 landmarks as (x, y, z) in [0, 1] range
    |
    +--> [If previous frame exists] Apply exponential smoothing
    |       new_pos = 0.7 * current + 0.3 * previous
    |
    +--> CoordinateTransformer.mediapipe_to_3d()
    |       Scales x, y by 10.0; flips Y axis; scales z by 10.0
    |
    +--> GeometryEngine.enhance_depth()
    |       Boosts Z values for nose/eye/mouth landmarks by 1.2x
    |
    +--> GeometryEngine.create_mesh_from_landmarks()
    |       Delaunay triangulation on XY plane
    |       Lifts result to 3D, computes vertex normals
    |
    +--> GeometryEngine.smooth_mesh(method='laplacian')
    |       Smooths over N iterations (default: 3)
    |
    +--> GeometryEngine.refine_mesh_quality()
            Removes degenerate/duplicate triangles
            Removes non-manifold edges

Output: (landmarks_3d, mesh)  -- or (None, None) if no face found
```

---

## Module Reference

| File | Role | Key Classes / Functions |
|---|---|---|
| `main.py` | Entry point and CLI orchestrator | `FaceReconstruction3D`, `main()` |
| `geometry_engine.py` | All mesh math | `GeometryEngine`, `DepthEstimator`, `MeshTextureMapper` |
| `visualization.py` | All display logic | `MeshVisualizer`, `VideoVisualizer`, `DepthMapVisualizer` |
| `utils.py` | Config, coordinate math, image preprocessing | `Config`, `CoordinateTransformer`, `MathUtils` |
| `export_mesh.py` | Save mesh to disk | `MeshExporter`, `TextureGenerator` |
| `threaded_pipeline.py` | Parallel processing for performance | Thread-safe queues for frame capture and mesh building |
| `video_processor.py` | Process full video files frame-by-frame | `VideoProcessor` |
| `error_handler.py` | Centralized error management | Custom exception classes and recovery strategies |
| `config.yaml` | Default settings for all pipeline parameters | (YAML config file) |

### Legacy / Experimental Scripts

These were written before the unified pipeline and are kept for reference:

| File | What it explores |
|---|---|
| `face_landmark_detection.py` | Basic MediaPipe landmark detection |
| `face_mesh_generation.py` | Simple mesh generation experiment |
| `face_mesh_texture_refined.py` | Refined texture mapping experiment |
| `generate_depth_maps.py` | Standalone depth map visualization |
| `generate_mesh.py` | Standalone mesh generation |
| `landmark_stabilization.py` | Kalman filter implementation for landmarks |
| `implement_texture.py` | UV texture mapping experiments |

---

## Getting Started

### System Requirements

- Python 3.8 or higher
- Webcam (for live and capture modes)
- 4 GB RAM minimum; 8 GB recommended
- GPU is optional; the pipeline runs on CPU

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Sourodyuti/3D-FaceReconstruction.git
cd 3D-FaceReconstruction

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Verify the installation
python main.py --help
```

---

## Running the Program

The program has three operation modes, selected via the first argument:

### Mode 1: Live (Webcam)

Opens your webcam and performs real-time 3D reconstruction on every frame.

```bash
python main.py live

# Also show a depth map window
python main.py live --show-depth

# Disable the 3D mesh window (lighter on CPU)
python main.py live --no-mesh
```

Keyboard controls while running:

| Key | Action |
|---|---|
| Q | Quit |
| S | Save snapshot (frame + mesh) |
| D | Toggle depth map window |
| M | Toggle 3D mesh window |

### Mode 2: Capture (Single Frame)

Opens the webcam but waits for you to press SPACE. At that moment it captures the frame, builds the 3D model, and exports it.

```bash
python main.py capture --output my_face_model
```

Exported files will be saved to the `./output/` directory in OBJ, PLY, and STL formats.

### Mode 3: Image File

Processes a single image file and exports the 3D model without opening a webcam.

```bash
python main.py image --input photo.jpg --output face_3d
```

### Using a Custom Config

```bash
python main.py live --config custom_config.yaml
```

---

## Configuration

All pipeline parameters are controlled by `config.yaml`. You can copy it and modify values:

```yaml
camera:
  device_id: 0          # Webcam index (0 = default camera)
  width: 640
  height: 480
  fps: 30

mediapipe:
  max_num_faces: 1
  refine_landmarks: true
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5

mesh:
  scale_factor: 10.0          # How much to scale X and Y
  depth_scale: 10.0           # How much to scale Z (depth)
  smoothing_iterations: 3     # Laplacian smoothing passes
  enable_smoothing: true

stabilization:
  enable_kalman: true         # Enable landmark smoothing over time

export:
  output_dir: ./output
  default_format: obj

visualization:
  show_landmarks: true
  show_axes: true
  window_width: 640
  window_height: 480
  background_color: [0.1, 0.1, 0.1]
```

---

## File Structure

```
3D-FaceReconstruction/
|
|-- main.py                       # Entry point and CLI (start here)
|-- config.yaml                   # Pipeline configuration
|-- requirements.txt              # All Python dependencies
|
|-- Core Modules:
|-- utils.py                      # Config loader, coordinate math, image utilities
|-- geometry_engine.py            # Triangulation, mesh build/smooth/refine, UV mapping
|-- visualization.py              # OpenCV + Open3D display utilities
|-- export_mesh.py                # Save mesh to OBJ / PLY / STL / GLTF
|-- threaded_pipeline.py          # Multi-threaded pipeline for performance
|-- video_processor.py            # Full video file processing
|-- error_handler.py              # Centralized error handling
|
|-- Legacy Scripts (reference):
|-- face_landmark_detection.py
|-- face_mesh_generation.py
|-- face_mesh_texture_refined.py
|-- generate_depth_maps.py
|-- generate_mesh.py
|-- landmark_stabilization.py
|-- implement_texture.py
|
|-- tests/                        # Unit tests
|-- test_setup.py                 # Setup verification script
|
|-- Documentation:
|-- README.md                     # This file
|-- USAGE.md                      # Extended usage guide
|-- COMPREHENSIVE_ANALYSIS_REPORT.md
|-- CONTRIBUTING.md
|-- LICENSE
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `mediapipe` | Google's neural network for detecting 468 facial landmarks per face |
| `opencv-python` | Camera capture, image I/O, drawing, color conversions |
| `open3d` | 3D mesh data structures, Laplacian smoothing, visualization, export |
| `numpy` | Fast numerical array operations for all coordinate math |
| `scipy` | Delaunay triangulation algorithm (`scipy.spatial.Delaunay`) |
| `pyyaml` | Parsing the `config.yaml` configuration file |
| `torch` (optional) | Enables MiDaS deep learning depth estimation if installed |

Install all:

```bash
pip install -r requirements.txt
```

---

## Roadmap

### Short-Term (Q1-Q2 2026)
- High-resolution texture baking with normal and displacement maps
- Multi-face simultaneous reconstruction
- Integration of monocular depth estimation networks (MiDaS)
- Expression transfer to rigged 3D characters

### Medium-Term (Q3-Q4 2026)
- NeRF-based neural rendering for photorealistic lighting
- Adaptive mesh subdivision for higher-detail models
- CUDA/OpenCL GPU acceleration
- Mobile deployment (iOS/Android)

### Long-Term (2027+)
- 4D reconstruction with temporal coherence across video
- Physically-based rendering (PBR) material estimation
- Cloud REST API for scalable reconstruction
- Edge deployment on embedded IoT devices

---

## Contributing

Contributions are welcome. Please open an issue before submitting a large pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) by Google for facial landmark detection
- [Open3D](http://www.open3d.org/) for 3D geometry processing and visualization
- [OpenCV](https://opencv.org/) for image and video processing
