# Quick Start Guide - 3D Face Reconstruction Web App

Get up and running with the 3D Face Reconstruction web application in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Installation

```bash
# 1. Navigate to the web_app directory
cd web_app

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install main project dependencies (if not already installed)
cd ..
pip install -r requirements.txt
```

## Running the Server

```bash
# From the project root directory
python web_app/server.py
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Using the Web Application

1. **Open your browser** and navigate to:
   ```
   http://localhost:8000/app
   ```

2. **Upload an image**:
   - Drag and drop an image onto the upload area
   - Or click "Browse Files" to select an image
   - Or click "Use Camera" to capture from webcam

3. **Adjust settings** (optional):
   - Quality: Low/Medium/High
   - Smoothing: 0-10 iterations
   - Scale Factor: 5-20
   - Export Formats: OBJ, PLY, STL, GLTF

4. **Generate 3D Model**:
   - Click "Generate 3D Model" button
   - Wait for processing to complete
   - View the 3D model in the viewer

5. **Interact with the 3D Model**:
   - **Rotate**: Left-click and drag
   - **Pan**: Right-click and drag
   - **Zoom**: Mouse scroll wheel
   - **Wireframe**: Click wireframe button
   - **Reset**: Click reset button
   - **Screenshot**: Click camera button

6. **Download the Model**:
   - Generated files appear in the "Output Files" panel
   - Click the download button to save

## Troubleshooting

### Server won't start

```bash
# Check if port 8000 is in use
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill the process or use a different port
# Edit server.py and change port=8000 to another port
```

### Camera not working

- Ensure browser has camera permissions
- Check if another application is using the camera
- Try using HTTPS (camera requires secure context in some browsers)

### 3D model not loading

- Check browser console for errors (F12)
- Ensure Three.js CDN is accessible
- Verify CORS settings in server.py

### Processing is slow

- Reduce quality setting to "low"
- Decrease smoothing iterations
- Use smaller image files (< 2MP)

## API Usage

### Process Image via API

```bash
curl -X POST "http://localhost:8000/api/process/image" \
  -F "file=@path/to/image.jpg" \
  -F "quality=medium" \
  -F "smoothing=3" \
  -F "scale_factor=10.0" \
  -F "export_formats=obj,ply"
```

### List Output Files

```bash
curl "http://localhost:8000/api/outputs"
```

### Download File

```bash
curl -O "http://localhost:8000/api/download/reconstruction_20260218_014645.obj"
```

## Next Steps

- Read the full documentation: `web_app/README.md`
- Explore the comprehensive analysis: `COMPREHENSIVE_ANALYSIS_REPORT.md`
- Check the main project README: `README.md`

## Support

For issues and questions:
- GitHub Issues: https://github.com/Sourodyuti/3D-FaceReconstruction/issues
- Documentation: See `web_app/README.md`

---

**Happy 3D modeling! 🎨**