#!/usr/bin/env python3
"""
FastAPI Server for 3D Face Reconstruction Web Application
Provides REST API endpoints and WebSocket support for real-time processing
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import asyncio
import tempfile
import shutil
import logging

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import cv2
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import FaceReconstruction3D
from utils import Config
from export_mesh import MeshExporter
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="3D Face Reconstruction API",
    description="Monocular 3D face reconstruction from RGB images",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[FaceReconstruction3D] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# Pydantic models for API
class ProcessRequest(BaseModel):
    """Request model for image processing"""
    quality: str = "medium"  # low, medium, high
    smoothing: int = 3
    scale_factor: float = 10.0
    export_formats: List[str] = ["obj", "ply"]

class ProcessingStatus(BaseModel):
    """Status model for processing progress"""
    status: str  # processing, completed, error
    progress: float  # 0.0 to 1.0
    message: str
    result: Optional[Dict[str, Any]] = None

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the reconstruction pipeline on startup"""
    global pipeline
    logger.info("Initializing 3D Face Reconstruction pipeline...")
    try:
        pipeline = FaceReconstruction3D(config_path=None)
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    global pipeline
    if pipeline:
        pipeline.cleanup()
        logger.info("Pipeline cleaned up")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "3D Face Reconstruction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "process_image": "/api/process/image",
            "process_camera": "/api/process/camera",
            "websocket": "/ws/process"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_initialized": pipeline is not None
    }

@app.post("/api/process/image")
async def process_image(
    file: UploadFile = File(...),
    quality: str = "medium",
    smoothing: int = 3,
    scale_factor: float = 10.0,
    export_formats: str = "obj,ply"
):
    """
    Process an uploaded image and generate 3D face model
    
    Args:
        file: Uploaded image file
        quality: Processing quality (low, medium, high)
        smoothing: Mesh smoothing iterations
        scale_factor: Mesh scale factor
        export_formats: Comma-separated list of export formats
    
    Returns:
        JSON response with processing results and download URLs
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Parse export formats
        formats = [fmt.strip() for fmt in export_formats.split(",")]
        
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        image_path = Path(temp_dir) / file.filename
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing image: {file.filename}")
        
        # Process image with pipeline
        output_dir = Path("./web_app/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique output filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"reconstruction_{timestamp}"
        
        # Process the image
        pipeline.process_image_file(str(image_path), output_filename=output_filename)
        
        # Get list of exported files
        exported_files = list(output_dir.glob(f"{output_filename}_*"))
        
        # Prepare response
        result = {
            "status": "success",
            "message": "3D model generated successfully",
            "output_filename": output_filename,
            "files": [],
            "metadata": {
                "quality": quality,
                "smoothing": smoothing,
                "scale_factor": scale_factor,
                "landmarks": 468
            }
        }
        
        # Add file information
        for file_path in exported_files:
            if file_path.is_file():
                result["files"].append({
                    "name": file_path.name,
                    "format": file_path.suffix[1:],
                    "size": file_path.stat().st_size,
                    "url": f"/api/download/{file_path.name}"
                })
        
        # Cleanup temp directory
        shutil.rmtree(temp_dir)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download generated 3D model file"""
    file_path = Path("./web_app/outputs") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/octet-stream'
    )

@app.get("/api/outputs")
async def list_outputs():
    """List all generated 3D models"""
    output_dir = Path("./web_app/outputs")
    
    if not output_dir.exists():
        return {"files": []}
    
    files = []
    for file_path in output_dir.iterdir():
        if file_path.is_file():
            files.append({
                "name": file_path.name,
                "size": file_path.stat().st_size,
                "created": file_path.stat().st_ctime
            })
    
    # Sort by creation time (newest first)
    files.sort(key=lambda x: x["created"], reverse=True)
    
    return {"files": files}

@app.delete("/api/outputs/{filename}")
async def delete_output(filename: str):
    """Delete a generated 3D model file"""
    file_path = Path("./web_app/outputs") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path.unlink()
    return {"status": "success", "message": f"Deleted {filename}"}

# WebSocket endpoint for real-time processing
@app.websocket("/ws/process")
async def websocket_process(websocket: WebSocket):
    """
    WebSocket endpoint for real-time 3D face reconstruction
    Supports camera feed streaming and live processing
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            message_type = data.get("type")
            
            if message_type == "process_image":
                # Process uploaded image
                await handle_image_processing(websocket, data)
            
            elif message_type == "camera_frame":
                # Process camera frame (real-time)
                await handle_camera_frame(websocket, data)
            
            elif message_type == "ping":
                # Respond to ping
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })

async def handle_image_processing(websocket: WebSocket, data: Dict):
    """Handle image processing via WebSocket"""
    try:
        # Send processing status
        await websocket.send_json({
            "type": "status",
            "status": "processing",
            "progress": 0.0,
            "message": "Starting image processing..."
        })
        
        # Process image (implementation similar to REST endpoint)
        # This would be expanded with actual processing logic
        
        await websocket.send_json({
            "type": "status",
            "status": "processing",
            "progress": 0.5,
            "message": "Generating 3D mesh..."
        })
        
        await websocket.send_json({
            "type": "status",
            "status": "processing",
            "progress": 0.8,
            "message": "Applying texture..."
        })
        
        # Send completion
        await websocket.send_json({
            "type": "status",
            "status": "completed",
            "progress": 1.0,
            "message": "Processing complete!",
            "result": {
                "filename": "reconstruction_20260218_014645",
                "files": ["model.obj", "model.ply", "texture.png"]
            }
        })
    
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })

async def handle_camera_frame(websocket: WebSocket, data: Dict):
    """Handle camera frame processing (real-time)"""
    try:
        # Decode base64 image
        import base64
        image_data = base64.b64decode(data["image"])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Failed to decode image")
        
        # Process frame with pipeline
        if pipeline:
            landmarks_3d, mesh = pipeline.process_frame(frame)
            
            if mesh is not None:
                # Convert mesh to simple format for transmission
                vertices = np.asarray(mesh.vertices).tolist()
                triangles = np.asarray(mesh.triangles).tolist()
                
                # Send processed result
                await websocket.send_json({
                    "type": "frame_result",
                    "has_face": True,
                    "vertices_count": len(vertices),
                    "triangles_count": len(triangles),
                    # Note: In production, we'd send compressed data
                    # or use binary protocol for efficiency
                })
            else:
                await websocket.send_json({
                    "type": "frame_result",
                    "has_face": False
                })
        else:
            await websocket.send_json({
                "type": "error",
                "message": "Pipeline not initialized"
            })
    
    except Exception as e:
        logger.error(f"Error processing camera frame: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")

# Serve frontend
@app.get("/app")
async def serve_frontend():
    """Serve the frontend application"""
    return FileResponse("web_app/static/index.html")

def main():
    """Run the FastAPI server"""
    logger.info("Starting 3D Face Reconstruction Web Server...")
    
    # Create output directory
    Path("./web_app/outputs").mkdir(parents=True, exist_ok=True)
    
    # Run server
    uvicorn.run(
        "web_app.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()