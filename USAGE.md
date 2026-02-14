# 3D Face Reconstruction - Usage Guide

Comprehensive guide for using the 3D Face Reconstruction system.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Modes](#usage-modes)
- [Configuration](#configuration)
- [Command-Line Interface](#command-line-interface)
- [Output Formats](#output-formats)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Live Mode

```bash
python main.py live
```

### 3. Interactive Controls (Live Mode)

- **Q**: Quit the application
- **S**: Save snapshot (frame + 3D model)
- **D**: Toggle depth map visualization
- **M**: Toggle 3D mesh window
- **SPACE**: Pause/Resume (in capture mode)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- Windows, macOS, or Linux

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sourodyuti/3D-FaceReconstruction.git
   cd 3D-FaceReconstruction
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python main.py --help
   ```

---

## Usage Modes

### 1. Live Mode (Real-Time Reconstruction)

Real-time 3D face reconstruction from webcam feed.

```bash
# Basic live mode
python main.py live

# With depth map visualization
python main.py live --show-depth

# Without 3D mesh window (video only)
python main.py live --no-mesh

# With custom configuration
python main.py live --config my_config.yaml
```

**Features:**
- Real-time landmark detection
- Live 3D mesh visualization
- Face orientation tracking (yaw, pitch, roll)
- FPS counter
- Snapshot capture (press 'S')

### 2. Capture Mode (Single Frame Export)

Capture a single frame and export the 3D model.

```bash
# Basic capture
python main.py capture

# With custom output name
python main.py capture --output my_face_model
```

**Process:**
1. Webcam window opens
2. Position your face in frame
3. Press **SPACE** to capture
4. Model automatically exported in multiple formats

### 3. Image Mode (Process Image File)

Process an existing image file and generate 3D model.

```bash
# Process single image
python main.py image --input photo.jpg --output face_model

# With custom configuration
python main.py image --input selfie.png --output model --config config.yaml
```

**Supported image formats:** JPG, PNG, BMP, TIFF

---

## Configuration

### Configuration File Structure

The system uses YAML or JSON configuration files. Default: `config.yaml`

```yaml
camera:
  device_id: 0              # Camera ID
  width: 640                # Resolution width
  height: 480               # Resolution height
  fps: 30                   # Target FPS

mediapipe:
  max_num_faces: 1          # Max faces to detect
  refine_landmarks: true    # Enable refinement
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5

mesh:
  scale_factor: 10.0        # XY scale
  depth_scale: 10.0         # Z scale (depth)
  smoothing_iterations: 3   # Mesh smoothing
  enable_smoothing: true

visualization:
  window_width: 640
  window_height: 480
  show_wireframe: false
  show_landmarks: true
  show_axes: true
  background_color: [0.1, 0.1, 0.1]

export:
  output_dir: ./output      # Export directory
  default_format: obj       # obj, ply, stl, gltf
  texture_size: 1024        # Texture resolution

stabilization:
  enable_kalman: true       # Landmark smoothing
  process_noise: 0.01
  measurement_noise: 0.1
```

### Creating Custom Configuration

1. **Copy default config:**
   ```bash
   cp config.yaml my_config.yaml
   ```

2. **Edit parameters:**
   ```yaml
   # Increase mesh quality
   mesh:
     scale_factor: 15.0
     smoothing_iterations: 5
   
   # Change export settings
   export:
     output_dir: ./my_models
     default_format: ply
     texture_size: 2048
   ```

3. **Use custom config:**
   ```bash
   python main.py live --config my_config.yaml
   ```

---

## Command-Line Interface

### Complete Command Reference

```bash
python main.py <mode> [options]
```

### Modes

- `live` - Real-time webcam reconstruction
- `capture` - Single frame capture and export
- `image` - Process image file

### Options

| Option | Description | Example |
|--------|-------------|----------|
| `--config PATH` | Custom configuration file | `--config custom.yaml` |
| `--output NAME` | Output filename | `--output my_model` |
| `--input PATH` | Input image file (image mode) | `--input photo.jpg` |
| `--show-depth` | Show depth map (live mode) | `--show-depth` |
| `--no-mesh` | Disable 3D visualization | `--no-mesh` |
| `--log-level LEVEL` | Logging level | `--log-level DEBUG` |
| `--help` | Show help message | `--help` |

### Examples

```bash
# Live mode with all visualizations
python main.py live --show-depth

# Capture with custom output
python main.py capture --output john_face --config high_quality.yaml

# Process batch of images (use script)
for img in *.jpg; do
    python main.py image --input "$img" --output "${img%.jpg}_model"
done

# Debug mode with verbose logging
python main.py live --log-level DEBUG
```

---

## Output Formats

### Supported Export Formats

| Format | Extension | Features | Use Case |
|--------|-----------|----------|----------|
| **OBJ** | `.obj` + `.mtl` + `.png` | Texture support, widely supported | 3D modeling, rendering |
| **PLY** | `.ply` | Vertex colors, efficient | Point cloud processing |
| **STL** | `.stl` | Simple geometry, no texture | 3D printing |
| **GLTF** | `.gltf` / `.glb` | Modern format, animations | Web/AR/VR applications |

### Output File Structure

When you export a model, the following files are created:

```
output/
├── my_model_20260214_151023.obj        # Main mesh file
├── my_model_20260214_151023.mtl        # Material file (OBJ)
├── my_model_20260214_151023_texture.png # Texture image
├── my_model_20260214_151023.ply        # PLY format
├── my_model_20260214_151023.stl        # STL format
└── my_model_20260214_151023_metadata.json # Metadata
```

### Viewing 3D Models

**Online Viewers:**
- [3D Viewer Online](https://3dviewer.net/)
- [Clara.io](https://clara.io/)

**Desktop Software:**
- MeshLab (Free, cross-platform)
- Blender (Free, open-source)
- Autodesk Fusion 360 (Free for hobbyists)

---

## Troubleshooting

### Common Issues

#### 1. Camera Not Opening

**Error:** `Failed to open camera 0`

**Solutions:**
- Check if camera is connected
- Try different device ID: `--config` with `camera.device_id: 1`
- Close other applications using the camera
- On Linux, check permissions: `sudo usermod -a -G video $USER`

#### 2. No Face Detected

**Error:** `No face detected in image`

**Solutions:**
- Ensure good lighting
- Face the camera directly
- Move closer to camera
- Lower detection confidence in config:
  ```yaml
  mediapipe:
    min_detection_confidence: 0.3
  ```

#### 3. Poor Mesh Quality

**Problem:** Mesh looks rough or distorted

**Solutions:**
- Increase smoothing iterations:
  ```yaml
  mesh:
    smoothing_iterations: 5
  ```
- Improve lighting conditions
- Keep face still during capture
- Use higher resolution camera

#### 4. Low FPS / Laggy Performance

**Problem:** Slow frame rate in live mode

**Solutions:**
- Disable 3D mesh window: `--no-mesh`
- Reduce camera resolution in config
- Disable depth map visualization
- Close other applications
- Reduce smoothing iterations

#### 5. Import Errors

**Error:** `ModuleNotFoundError: No module named 'mediapipe'`

**Solution:**
```bash
pip install -r requirements.txt
```

#### 6. Open3D Visualization Issues

**Problem:** Black screen or no mesh visible

**Solutions:**
- Update graphics drivers
- Try software rendering: `export LIBGL_ALWAYS_SOFTWARE=1` (Linux)
- Check Open3D compatibility: `python -c "import open3d; print(open3d.__version__)"`

### Getting Help

If you encounter issues:

1. **Check logs:** Run with `--log-level DEBUG`
2. **Check GitHub Issues:** [Repository Issues](https://github.com/Sourodyuti/3D-FaceReconstruction/issues)
3. **Create new issue:** Include error message, OS, Python version

---

## Performance Tips

### For Best Quality

```yaml
mesh:
  scale_factor: 15.0
  depth_scale: 15.0
  smoothing_iterations: 5
  enable_smoothing: true

export:
  texture_size: 2048
```

### For Best Performance

```yaml
camera:
  width: 320
  height: 240
  fps: 15

mesh:
  smoothing_iterations: 1
  enable_smoothing: false

visualization:
  show_landmarks: false
```

---

## Advanced Usage

### Batch Processing Script

```python
#!/usr/bin/env python3
import os
import glob
from main import FaceReconstruction3D

# Initialize pipeline
pipeline = FaceReconstruction3D(config_path="config.yaml")

# Process all images in directory
for img_path in glob.glob("images/*.jpg"):
    basename = os.path.splitext(os.path.basename(img_path))[0]
    print(f"Processing {img_path}...")
    pipeline.process_image_file(img_path, output_filename=basename)

pipeline.cleanup()
print("Batch processing complete!")
```

### Integration Example

```python
from main import FaceReconstruction3D
import cv2

# Initialize
pipeline = FaceReconstruction3D()

# Process single frame
frame = cv2.imread("photo.jpg")
landmarks, mesh = pipeline.process_frame(frame)

if mesh is not None:
    # Export mesh
    pipeline.mesh_exporter.export_mesh(mesh, "output", format="obj")
    print("Export successful!")

pipeline.cleanup()
```

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

**Need more help?** Check the [README.md](README.md) or open an issue on GitHub!
