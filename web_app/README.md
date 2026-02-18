# 3D Face Reconstruction - Web Application

A modern web-based interface for monocular 3D face reconstruction using FastAPI backend and Three.js frontend.

## 🌟 Features

- **Interactive 3D Viewer**: Real-time 3D mesh visualization with Three.js
- **Drag & Drop Upload**: Easy image upload with drag-and-drop support
- **Camera Integration**: Capture images directly from webcam
- **Multiple Export Formats**: OBJ, PLY, STL, GLTF support
- **Real-time Processing**: WebSocket support for live updates
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Quality Controls**: Adjustable smoothing, scale, and quality settings

## 📋 Prerequisites

- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Webcam (for camera feature)

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to the web_app directory
cd web_app

# Install dependencies
pip install -r requirements.txt

# Install main project dependencies (if not already installed)
cd ..
pip install -r requirements.txt
```

### 2. Run the Server

```bash
# From the project root directory
python web_app/server.py
```

The server will start at `http://localhost:8000`

### 3. Open in Browser

Navigate to `http://localhost:8000/app` in your web browser.

## 🎯 Usage

### Upload and Process Image

1. **Upload Image**:
   - Drag and drop an image onto the upload area
   - Or click "Browse Files" to select an image
   - Or click "Use Camera" to capture from webcam

2. **Adjust Settings**:
   - **Quality**: Low (fast), Medium, High (best)
   - **Smoothing**: 0-10 iterations for mesh smoothing
   - **Scale Factor**: 5-20 for mesh size
   - **Export Formats**: Select OBJ, PLY, STL, or GLTF

3. **Generate 3D Model**:
   - Click "Generate 3D Model" button
   - Wait for processing to complete
   - View the 3D model in the viewer

### 3D Viewer Controls

- **Rotate**: Left-click and drag
- **Pan**: Right-click and drag
- **Zoom**: Mouse scroll wheel
- **Toggle Wireframe**: Click wireframe button
- **Reset Camera**: Click reset button
- **Screenshot**: Click camera button to save image

### Download Models

Generated models appear in the "Output Files" panel. Click the download button to save.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Browser (Client)                        │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │   HTML/CSS   │  │  Three.js    │  │   JavaScript  │   │
│  │   (UI)       │  │  (3D Viewer) │  │   (Logic)     │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/WebSocket
┌───────────────────────────┴─────────────────────────────────┐
│                   FastAPI Server (Backend)                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │  API Routes  │  │  WebSocket   │  │  File Server  │   │
│  │  (REST)      │  │  (Real-time) │  │  (Downloads)  │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────┐
│              3D Face Reconstruction Pipeline                │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │ MediaPipe    │  │ Geometry     │  │  Mesh Export  │   │
│  │ (Landmarks)  │  │ Engine       │  │  (OBJ/PLY...) │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
web_app/
├── __init__.py              # Package initialization
├── server.py                # FastAPI server and API endpoints
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── static/                 # Static files
│   ├── index.html         # Main HTML page
│   └── app.js             # Frontend JavaScript
└── outputs/               # Generated 3D models (auto-created)
```

## 🔌 API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/api/process/image` | POST | Process uploaded image |
| `/api/download/{filename}` | GET | Download generated file |
| `/api/outputs` | GET | List all output files |
| `/api/outputs/{filename}` | DELETE | Delete output file |
| `/app` | GET | Serve frontend application |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws/process` | Real-time processing updates |

## 🎨 Customization

### Modify Color Scheme

Edit `static/index.html` CSS variables:

```css
:root {
    --primary-color: #4e73df;
    --secondary-color: #858796;
    --success-color: #1cc88a;
    /* ... */
}
```

### Adjust 3D Viewer Settings

Edit `static/app.js`:

```javascript
// Camera position
camera.position.z = 5;

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);

// Auto-rotation
controls.autoRotate = true;
controls.autoRotateSpeed = 1.0;
```

### Change Server Port

Edit `server.py`:

```python
uvicorn.run(
    "web_app.server:app",
    host="0.0.0.0",
    port=8000,  # Change this
    reload=True
)
```

## 🔧 Configuration

### Server Configuration

Edit `server.py` to modify:

- CORS settings
- Upload file size limits
- Processing parameters
- Output directory

### Processing Parameters

Default parameters can be set in `server.py` or passed via API:

```python
{
    "quality": "medium",      # low, medium, high
    "smoothing": 3,           # 0-10 iterations
    "scale_factor": 10.0,     # 5-20
    "export_formats": "obj,ply"  # obj, ply, stl, gltf
}
```

## 🚢 Production Deployment

### Using Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn web_app.server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "web_app.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t face-reconstruction-web .
docker run -p 8000:8000 face-reconstruction-web
```

### Nginx Reverse Proxy

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## 🐛 Troubleshooting

### Server won't start

```bash
# Check if port 8000 is in use
lsof -i :8000

# Use a different port
python web_app/server.py  # Edit server.py to change port
```

### Camera not working

- Ensure browser has camera permissions
- Check if another application is using the camera
- Try HTTPS (camera requires secure context in some browsers)

### 3D model not loading

- Check browser console for errors
- Ensure Three.js CDN is accessible
- Verify CORS settings in server.py

### Processing is slow

- Reduce quality setting to "low"
- Decrease smoothing iterations
- Use smaller image files

## 📊 Performance Tips

### For Better Performance

1. **Reduce Image Resolution**: Use images < 2MP
2. **Lower Quality**: Set quality to "low" or "medium"
3. **Fewer Formats**: Export only needed formats
4. **Disable Auto-rotation**: In `app.js`, set `controls.autoRotate = false`

### For Better Quality

1. **High Quality Images**: Use well-lit, high-resolution photos
2. **High Quality Setting**: Set quality to "high"
3. **More Smoothing**: Increase smoothing to 5-10 iterations
4. **Multiple Formats**: Export in OBJ and PLY for compatibility

## 🔒 Security Considerations

### Production Deployment

1. **Enable HTTPS**: Use SSL/TLS certificates
2. **CORS Configuration**: Restrict allowed origins
3. **File Size Limits**: Set maximum upload size
4. **Rate Limiting**: Implement API rate limiting
5. **Authentication**: Add user authentication if needed

### Example CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],  # Specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📈 Monitoring

### Enable Logging

Server logs are displayed in the console. For production:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Performance Monitoring

Consider adding:
- Request/response timing
- Processing time tracking
- Error rate monitoring
- Resource usage tracking

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe** by Google for face landmark detection
- **Three.js** for 3D visualization
- **FastAPI** for the web framework
- **Open3D** for 3D mesh operations

## 📧 Support

For issues and questions:
- GitHub Issues: [Report bugs or request features](https://github.com/Sourodyuti/3D-FaceReconstruction/issues)
- Documentation: See main project README.md

---

**Made with ❤️ for the computer vision and 3D graphics community**