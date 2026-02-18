/**
 * 3D Face Reconstruction Web Application
 * Frontend JavaScript for Three.js viewer and API communication
 */

// Global variables
let scene, camera, renderer, controls;
let currentMesh = null;
let currentTexture = null;
let isWireframe = false;
let isTextureEnabled = true;
let uploadedFile = null;
let cameraStream = null;
let ws = null;

// API base URL
const API_BASE = '';

// Initialize Three.js scene
function initThreeJS() {
    const container = document.getElementById('viewer-container');
    const canvas = document.getElementById('three-canvas');

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    // Camera
    camera = new THREE.PerspectiveCamera(
        75,
        container.clientWidth / container.clientHeight,
        0.1,
        1000
    );
    camera.position.z = 5;

    // Renderer
    renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: true,
        alpha: true
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enableZoom = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.0;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);

    const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
    backLight.position.set(-5, -5, -5);
    scene.add(backLight);

    // Grid helper
    const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
    gridHelper.position.y = -2;
    scene.add(gridHelper);

    // Handle window resize
    window.addEventListener('resize', onWindowResize);

    // Start animation loop
    animate();
}

function onWindowResize() {
    const container = document.getElementById('viewer-container');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Load 3D model
function loadModel(format, url) {
    const loadingOverlay = document.getElementById('loadingOverlay');
    loadingOverlay.style.display = 'flex';

    document.getElementById('noModelMessage').style.display = 'none';

    let loader;

    switch (format.toLowerCase()) {
        case 'obj':
            loader = new THREE.OBJLoader();
            break;
        case 'ply':
            loader = new THREE.PLYLoader();
            break;
        case 'gltf':
        case 'glb':
            loader = new THREE.GLTFLoader();
            break;
        default:
            console.error('Unsupported format:', format);
            loadingOverlay.style.display = 'none';
            return;
    }

    loader.load(
        url,
        function (object) {
            // Remove previous mesh
            if (currentMesh) {
                scene.remove(currentMesh);
            }

            // Process loaded object
            if (format.toLowerCase() === 'gltf' || format.toLowerCase() === 'glb') {
                currentMesh = object.scene;
            } else {
                currentMesh = object;
            }

            // Apply materials
            currentMesh.traverse(function (child) {
                if (child.isMesh) {
                    child.material = new THREE.MeshPhongMaterial({
                        color: 0x4e73df,
                        shininess: 100,
                        wireframe: isWireframe
                    });
                }
            });

            // Center and scale mesh
            const box = new THREE.Box3().setFromObject(currentMesh);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());

            currentMesh.position.sub(center);
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 3 / maxDim;
            currentMesh.scale.setScalar(scale);

            scene.add(currentMesh);

            // Update model info
            document.getElementById('vertexCount').textContent = countVertices(currentMesh);
            document.getElementById('triangleCount').textContent = countTriangles(currentMesh);
            document.getElementById('modelInfoCard').style.display = 'block';

            loadingOverlay.style.display = 'none';
        },
        function (xhr) {
            if (xhr.lengthComputable) {
                const percentComplete = xhr.loaded / xhr.total * 100;
                console.log('Model ' + percentComplete + '% loaded');
            }
        },
        function (error) {
            console.error('Error loading model:', error);
            loadingOverlay.style.display = 'none';
            alert('Error loading 3D model');
        }
    );
}

function countVertices(object) {
    let count = 0;
    object.traverse(function (child) {
        if (child.isMesh && child.geometry) {
            count += child.geometry.attributes.position.count;
        }
    });
    return count;
}

function countTriangles(object) {
    let count = 0;
    object.traverse(function (child) {
        if (child.isMesh && child.geometry) {
            if (child.geometry.index) {
                count += child.geometry.index.count / 3;
            } else {
                count += child.geometry.attributes.position.count / 3;
            }
        }
    });
    return count;
}

// Viewer controls
function toggleWireframe() {
    isWireframe = !isWireframe;

    if (currentMesh) {
        currentMesh.traverse(function (child) {
            if (child.isMesh) {
                child.material.wireframe = isWireframe;
            }
        });
    }
}

function toggleTexture() {
    isTextureEnabled = !isTextureEnabled;
    // Texture toggle implementation would go here
    console.log('Texture toggled:', isTextureEnabled);
}

function resetCamera() {
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    controls.reset();
}

function takeScreenshot() {
    renderer.render(scene, camera);
    const dataURL = renderer.domElement.toDataURL('image/png');

    const link = document.createElement('a');
    link.download = '3d_face_model_screenshot.png';
    link.href = dataURL;
    link.click();
}

// File upload handling
function setupFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // Click to browse
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }

    uploadedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewImage = document.getElementById('previewImage');
        previewImage.src = e.target.result;
        document.getElementById('previewContainer').style.display = 'block';
    };
    reader.readAsDataURL(file);

    // Enable process button
    document.getElementById('processBtn').disabled = false;
}

// Camera handling
function startCamera() {
    const videoElement = document.createElement('video');
    videoElement.autoplay = true;

    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            cameraStream = stream;
            videoElement.srcObject = stream;

            // Create modal for camera
            const modal = document.createElement('div');
            modal.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.8);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 9999;
            `;

            modal.innerHTML = `
                <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3>Camera Feed</h3>
                    <video id="cameraVideo" autoplay style="max-width: 100%; max-height: 400px;"></video>
                    <br><br>
                    <button id="captureBtn" class="btn btn-primary">Capture</button>
                    <button id="closeCameraBtn" class="btn btn-secondary">Close</button>
                </div>
            `;

            document.body.appendChild(modal);

            const cameraVideo = document.getElementById('cameraVideo');
            cameraVideo.srcObject = stream;

            document.getElementById('captureBtn').addEventListener('click', () => {
                const canvas = document.createElement('canvas');
                canvas.width = cameraVideo.videoWidth;
                canvas.height = cameraVideo.videoHeight;
                canvas.getContext('2d').drawImage(cameraVideo, 0, 0);

                canvas.toBlob((blob) => {
                    const file = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
                    handleFileSelect(file);
                    document.body.removeChild(modal);
                    stopCamera();
                });
            });

            document.getElementById('closeCameraBtn').addEventListener('click', () => {
                document.body.removeChild(modal);
                stopCamera();
            });
        })
        .catch((error) => {
            console.error('Error accessing camera:', error);
            alert('Could not access camera. Please ensure camera permissions are granted.');
        });
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
}

// Process image
async function processImage() {
    if (!uploadedFile) {
        alert('Please select an image first');
        return;
    }

    const quality = document.getElementById('qualitySelect').value;
    const smoothing = parseInt(document.getElementById('smoothingRange').value);
    const scaleFactor = parseFloat(document.getElementById('scaleRange').value);

    const formats = [];
    if (document.getElementById('formatObj').checked) formats.push('obj');
    if (document.getElementById('formatPly').checked) formats.push('ply');
    if (document.getElementById('formatStl').checked) formats.push('stl');
    if (document.getElementById('formatGltf').checked) formats.push('gltf');

    if (formats.length === 0) {
        alert('Please select at least one export format');
        return;
    }

    // Show progress
    document.getElementById('progressCard').style.display = 'block';
    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('progressText').textContent = 'Uploading image...';

    // Create form data
    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('quality', quality);
    formData.append('smoothing', smoothing);
    formData.append('scale_factor', scaleFactor);
    formData.append('export_formats', formats.join(','));

    try {
        document.getElementById('progressBar').style.width = '30%';
        document.getElementById('progressText').textContent = 'Processing image...';

        const response = await fetch(`${API_BASE}/api/process/image`, {
            method: 'POST',
            body: formData
        });

        document.getElementById('progressBar').style.width = '70%';
        document.getElementById('progressText').textContent = 'Generating 3D model...';

        if (!response.ok) {
            throw new Error('Processing failed');
        }

        const result = await response.json();

        document.getElementById('progressBar').style.width = '100%';
        document.getElementById('progressText').textContent = 'Complete!';

        // Update file list
        updateFileList(result.files);

        // Load first 3D model into viewer
        if (result.files.length > 0) {
            const objFile = result.files.find(f => f.format === 'obj') || result.files[0];
            loadModel(objFile.format, objFile.url);
        }

        // Hide progress after delay
        setTimeout(() => {
            document.getElementById('progressCard').style.display = 'none';
        }, 2000);

    } catch (error) {
        console.error('Error processing image:', error);
        document.getElementById('progressText').textContent = 'Error: ' + error.message;
        alert('Error processing image: ' + error.message);
    }
}

function updateFileList(files) {
    const fileList = document.getElementById('fileList');
    fileList.innerHTML = '';

    files.forEach(file => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div>
                <div class="file-name">${file.name}</div>
                <div class="file-size">${formatFileSize(file.size)}</div>
            </div>
            <a href="${file.url}" download="${file.name}" class="btn btn-sm btn-primary">
                <i class="fas fa-download"></i>
            </a>
        `;
        fileList.appendChild(fileItem);
    });
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Control panel event listeners
function setupControls() {
    const smoothingRange = document.getElementById('smoothingRange');
    smoothingRange.addEventListener('input', (e) => {
        document.getElementById('smoothingValue').textContent = e.target.value;
    });

    const scaleRange = document.getElementById('scaleRange');
    scaleRange.addEventListener('input', (e) => {
        document.getElementById('scaleValue').textContent = parseFloat(e.target.value).toFixed(1);
    });
}

// WebSocket connection
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/process`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        switch (data.type) {
            case 'status':
                handleStatusUpdate(data);
                break;
            case 'frame_result':
                handleFrameResult(data);
                break;
            case 'error':
                console.error('WebSocket error:', data.message);
                break;
        }
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

function handleStatusUpdate(data) {
    document.getElementById('progressCard').style.display = 'block';
    document.getElementById('progressBar').style.width = (data.progress * 100) + '%';
    document.getElementById('progressText').textContent = data.message;

    if (data.status === 'completed') {
        setTimeout(() => {
            document.getElementById('progressCard').style.display = 'none';
        }, 2000);
    }
}

function handleFrameResult(data) {
    console.log('Frame result:', data);
    // Handle real-time camera frame processing results
}

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initThreeJS();
    setupFileUpload();
    setupControls();
    connectWebSocket();

    // Load example files
    loadOutputFiles();
});

function loadOutputFiles() {
    fetch(`${API_BASE}/api/outputs`)
        .then(response => response.json())
        .then(data => {
            if (data.files.length > 0) {
                updateFileList(data.files);
            }
        })
        .catch(error => {
            console.error('Error loading output files:', error);
        });
}

// Example and help functions
function loadExample() {
    alert('Example functionality would load a sample image and process it');
}

function showHelp() {
    alert(`
3D Face Reconstruction System Help

1. Upload an image or use your camera
2. Adjust quality and smoothing settings
3. Select export formats (OBJ, PLY, STL, GLTF)
4. Click "Generate 3D Model"
5. View and interact with the 3D model
6. Download the generated files

Viewer Controls:
- Left click + drag: Rotate
- Right click + drag: Pan
- Scroll: Zoom
- Use toolbar buttons for additional options
    `);
}