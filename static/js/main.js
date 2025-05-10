// Global variables
let scene, camera, renderer, controls, pointCloud;
let isProcessing = false;
let logPollingInterval = null;
let siftPollingInterval = null;
let gifPollingInterval = null;

// DOM Elements
const modelContainer = document.getElementById('model-container');
const generateBtn = document.getElementById('generate-btn');
const cancelBtn = document.getElementById('cancel-btn');
const objectSelect = document.getElementById('object-select');
const optimizationRadios = document.querySelectorAll('input[name="optimization"]');
const showFeaturesCheckbox = document.getElementById('show-features');
const logsContainer = document.getElementById('logs-container');
const resetViewBtn = document.getElementById('reset-view-btn');
const pointSizeSlider = document.getElementById('point-size');

// Initialize the 3D scene
function initScene() {
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    // Create camera
    const width = modelContainer.clientWidth;
    const height = modelContainer.clientHeight;
    camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = 5;

    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    modelContainer.appendChild(renderer.domElement);

    // Add orbit controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    controls.screenSpacePanning = false;
    controls.maxPolarAngle = Math.PI;
    controls.update();

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    // Add coordinate axes
    const axesHelper = new THREE.AxesHelper(1);
    scene.add(axesHelper);

    // Handle window resize
    window.addEventListener('resize', onWindowResize);

    // Start animation loop
    animate();
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Handle window resize
function onWindowResize() {
    const width = modelContainer.clientWidth;
    const height = modelContainer.clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

// Load PCD model
function loadModel(objectName, optimization) {
    // Clear existing model
    if (pointCloud) {
        scene.remove(pointCloud);
        pointCloud = null;
    }

    // Show loading indicator
    showLoading(true);

    // Check if SIFT feature matching is enabled
    const showSIFT = document.getElementById('show-features').checked;

    // Step 1: If SIFT is enabled, load SIFT feature matches first
    if (showSIFT) {
        console.log("SIFT feature matching is enabled. Loading SIFT matches first...");

        // Show loading indicator in SIFT container
        const siftContainer = document.getElementById('sift-container') || createSiftContainer();
        siftContainer.style.display = 'block';
        siftContainer.innerHTML = '<h3>SIFT Feature Matches</h3><div class="loading-overlay"><div class="spinner"></div><p>Generating SIFT feature matches...</p></div>';

        // Fetch SIFT images
        fetch(`/api/sift-images?object=${objectName}&generate=true`)
            .then(response => response.json())
            .then(data => {
                if (data.count > 0) {
                    // Display SIFT images
                    displaySiftImages(data.images);
                    console.log("SIFT feature matches displayed. Now loading 3D model...");
                } else {
                    console.log("No SIFT images found. Continuing with model loading...");
                    siftContainer.innerHTML = '<h3>SIFT Feature Matches</h3><p class="placeholder-text">No SIFT feature matches available</p>';
                }

                // Step 2: Now proceed with loading the 3D model
                loadModelFile();
            })
            .catch(error => {
                console.error('Error loading SIFT images:', error);
                const siftContainer = document.getElementById('sift-container');
                if (siftContainer) {
                    siftContainer.innerHTML = '<h3>SIFT Feature Matches</h3><p class="placeholder-text">Error loading SIFT images</p>';
                }

                // Continue with model loading even if SIFT fails
                loadModelFile();
            });
    } else {
        // If SIFT is not enabled, just load the model directly
        loadModelFile();
    }

    // Helper function to load the actual model file
    function loadModelFile() {
        // Fetch the model from our API endpoint
        const apiUrl = `/api/model/${objectName}/${optimization}`;

        // Create PCD loader
        const loader = new THREE.PCDLoader();

        console.log(`Loading 3D model for ${objectName} with ${optimization} optimization...`);

        // First check if the file exists
        fetch(apiUrl)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Model not found (${response.status})`);
                }
                return response.blob();
            })
            .then(blob => {
                // Create a URL for the blob
                const url = URL.createObjectURL(blob);

                // Load the PCD file from the blob URL
                loader.load(url, (points) => {
                    pointCloud = points;

                    // Rotate the model to correct orientation (flip it right-side up)
                    pointCloud.rotation.x = Math.PI; // Rotate 180 degrees around X-axis

                    scene.add(pointCloud);

                    // Center the model
                    const box = new THREE.Box3().setFromObject(pointCloud);
                    const center = box.getCenter(new THREE.Vector3());
                    pointCloud.position.sub(center);

                    // Reset camera position
                    resetView();

                    // Hide loading indicator
                    showLoading(false);

                    // Clean up the blob URL
                    URL.revokeObjectURL(url);

                    console.log("3D model loaded. Now loading GIFs...");

                    // Step 3: After model is loaded, load GIFs
                    setTimeout(() => {
                        loadGifForObject(objectName);
                    }, 500); // Small delay for better visual sequence
                },
                // Progress callback
                (xhr) => {
                    console.log((xhr.loaded / xhr.total * 100) + '% loaded');
                },
                // Error callback
                (error) => {
                    console.error('Error loading model:', error);
                    addLogEntry('Error loading model: ' + error.message);
                    showLoading(false);
                });
            })
            .catch(error => {
                console.error('Error fetching model:', error);
                addLogEntry('Error fetching model: ' + error.message);
                showLoading(false);
            });
    }
}

// Load GIF for the selected object
function loadGifForObject(objectName) {
    // Show loading indicator in GIF container
    const gifContainer = document.getElementById('gif-container');
    gifContainer.innerHTML = '<div class="loading-overlay"><div class="spinner"></div></div>';

    console.log(`Loading GIF for object: ${objectName}`);

    // Clear any previous GIF cache to ensure we get a fresh one
    const timestamp = new Date().getTime();

    // Fetch GIFs for this object with cache-busting
    fetch(`/api/gifs?object=${objectName}&t=${timestamp}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log(`Received ${data.count} GIFs for ${data.object}`);

            // Clear the container
            gifContainer.innerHTML = '';

            if (data.count > 0) {
                // Display the first GIF
                const gifData = data.gifs[0];

                // Create a container for the GIF and caption
                const gifWrapper = document.createElement('div');
                gifWrapper.className = 'gif-wrapper';

                // Create the image element
                const imgElement = document.createElement('img');
                imgElement.src = `data:image/gif;base64,${gifData.data}`;
                imgElement.alt = `Animation for ${objectName}`;
                imgElement.className = 'gif-image';
                imgElement.title = `${gifData.name} (${objectName})`;

                // Add data attributes for debugging
                imgElement.dataset.object = objectName;
                imgElement.dataset.filename = gifData.name;

                // Add the image to the wrapper
                gifWrapper.appendChild(imgElement);

                // Add a caption with the object name and filename
                const caption = document.createElement('p');
                caption.className = 'gif-caption';

                // Get the display name for the object
                const displayName = getObjectDisplayName(objectName);

                // Create a span for the object name with special styling
                const objectSpan = document.createElement('span');
                objectSpan.className = 'object-name';
                objectSpan.textContent = displayName;

                // Set the caption text
                caption.textContent = 'Animation for ';
                caption.appendChild(objectSpan);

                // Add the filename in smaller text
                const filenameSpan = document.createElement('span');
                filenameSpan.className = 'filename';
                filenameSpan.textContent = ` (${gifData.name})`;
                caption.appendChild(filenameSpan);

                gifWrapper.appendChild(caption);

                // Add the wrapper to the container
                gifContainer.appendChild(gifWrapper);

                console.log(`Displayed GIF: ${gifData.name} for ${objectName}`);
            } else {
                // No GIFs found
                gifContainer.innerHTML = `<p class="placeholder-text">No animations available for ${getObjectDisplayName(objectName)}</p>`;
                console.log(`No GIFs found for ${objectName}`);
            }
        })
        .catch(error => {
            console.error('Error loading GIF:', error);
            gifContainer.innerHTML = `<p class="placeholder-text">Error loading animation: ${error.message}</p>`;
        });
}

// Helper function to get display name for an object
function getObjectDisplayName(objectValue) {
    const objectMap = {
        'castard': 'Castard',
        'spyderman': 'Spyderman',
        'new_box2': 'New Box 2'
    };

    return objectMap[objectValue] || objectValue;
}

// Reset camera view
function resetView() {
    if (!pointCloud) return;

    // Get bounding box of the model
    const box = new THREE.Box3().setFromObject(pointCloud);
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    // Position camera to see the entire model
    const fov = camera.fov * (Math.PI / 180);
    const cameraZ = Math.abs(maxDim / Math.sin(fov / 2));

    camera.position.set(0, 0, cameraZ * 1.5);
    camera.lookAt(0, 0, 0);

    // Update controls
    controls.target.set(0, 0, 0);
    controls.update();
}

// Show/hide loading indicator
function showLoading(show) {
    // Remove existing loading overlay if any
    const existingOverlay = document.querySelector('.loading-overlay');
    if (existingOverlay) {
        existingOverlay.remove();
    }

    if (show) {
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';

        const spinner = document.createElement('div');
        spinner.className = 'spinner';

        overlay.appendChild(spinner);
        modelContainer.appendChild(overlay);
    }
}

// Generate 3D model
function generateModel() {
    if (isProcessing) return;

    // Get selected options
    const objectName = objectSelect.value;
    const optimization = document.querySelector('input[name="optimization"]:checked').value;

    // Update UI
    isProcessing = true;
    generateBtn.disabled = true;
    cancelBtn.disabled = false;

    // Clear logs
    logsContainer.innerHTML = '';
    addLogEntry(`Starting 3D model generation for ${objectName} using ${optimization} method...`);

    // Make API request
    fetch('/api/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            object: objectName,
            optimization: optimization
        })
    })
    .then(response => response.json())
    .then(data => {
        addLogEntry(data.message);

        // Start polling for logs
        startLogPolling();

        // If pre-optimized model is available, load it immediately
        if (data.status === 'ready') {
            loadModel(objectName, optimization);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        addLogEntry('Error: ' + error.message);
        isProcessing = false;
        generateBtn.disabled = false;
        cancelBtn.disabled = true;
    });
}

// Cancel the current process
function cancelProcess() {
    fetch('/api/cancel', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        addLogEntry('Process cancelled.');
        isProcessing = false;
        generateBtn.disabled = false;
        cancelBtn.disabled = true;
        stopLogPolling();
    })
    .catch(error => {
        console.error('Error:', error);
        addLogEntry('Error cancelling process: ' + error.message);
    });
}

// Start polling for logs, SIFT images, and GIFs
function startLogPolling() {
    if (logPollingInterval) {
        clearInterval(logPollingInterval);
    }

    if (siftPollingInterval) {
        clearInterval(siftPollingInterval);
    }

    if (gifPollingInterval) {
        clearInterval(gifPollingInterval);
    }

    // Poll for logs
    logPollingInterval = setInterval(() => {
        fetch('/api/logs')
            .then(response => response.json())
            .then(data => {
                // Add new logs
                data.logs.forEach(log => {
                    addLogEntry(log);
                });

                // Check if process is still running
                if (!data.running && isProcessing) {
                    // Process completed
                    isProcessing = false;
                    generateBtn.disabled = false;
                    cancelBtn.disabled = true;
                    stopPolling();

                    // Load the generated model
                    const objectName = objectSelect.value;
                    const optimization = document.querySelector('input[name="optimization"]:checked').value;
                    loadModel(objectName, optimization);
                }
            })
            .catch(error => {
                console.error('Error polling logs:', error);
            });
    }, 1000);

    // Poll for SIFT images
    siftPollingInterval = setInterval(() => {
        fetch('/api/sift-images')
            .then(response => response.json())
            .then(data => {
                if (data.count > 0) {
                    // Display SIFT images
                    displaySiftImages(data.images);
                }
            })
            .catch(error => {
                console.error('Error polling SIFT images:', error);
            });
    }, 2000);

    // Poll for GIFs
    const objectName = objectSelect.value;
    fetchAndDisplayGifs(objectName);
}

// Stop all polling
function stopPolling() {
    if (logPollingInterval) {
        clearInterval(logPollingInterval);
        logPollingInterval = null;
    }

    if (siftPollingInterval) {
        clearInterval(siftPollingInterval);
        siftPollingInterval = null;
    }

    if (gifPollingInterval) {
        clearInterval(gifPollingInterval);
        gifPollingInterval = null;
    }
}

// Fetch GIFs for the selected object (but don't display them yet)
function fetchAndDisplayGifs(objectName) {
    // We'll just store the object name for later use
    // The actual GIF loading happens after model generation
    console.log(`GIFs available for ${objectName}`);

    // Reset the GIF container to show placeholder
    const gifContainer = document.getElementById('gif-container');
    if (gifContainer) {
        gifContainer.innerHTML = '<p class="placeholder-text">GIF will appear here after model generation</p>';
    }
}

// Display SIFT feature matching images
function displaySiftImages(images) {
    // Get the SIFT images container
    let siftContainer = document.getElementById('sift-container');
    if (!siftContainer) {
        siftContainer = createSiftContainer();
    }

    // Clear existing content
    siftContainer.innerHTML = '';

    // Add title
    const title = document.createElement('h3');
    title.textContent = 'SIFT Feature Matches';
    siftContainer.appendChild(title);

    // Add description
    const description = document.createElement('p');
    description.className = 'sift-description';
    description.textContent = 'Showing pairwise SIFT feature matches used in the 3D reconstruction process';
    siftContainer.appendChild(description);

    // Create image container
    const imageContainer = document.createElement('div');
    imageContainer.className = 'sift-images-container';
    siftContainer.appendChild(imageContainer);

    // Add images
    if (images && images.length > 0) {
        // Define captions based on index
        const captions = [
            'Pairwise Registration: Frames 1 → 2',
            'Pairwise Registration: Frames 2 → 3',
            'Loop Closure: Last Frame → First Frame'
        ];

        images.forEach((imageData, index) => {
            const imgWrapper = document.createElement('div');
            imgWrapper.className = 'sift-image-wrapper';

            // Add caption above the image
            const captionTop = document.createElement('p');
            captionTop.className = 'sift-caption-top';
            captionTop.textContent = captions[index] || `Feature Match ${index + 1}`;
            imgWrapper.appendChild(captionTop);

            // Create image element
            const imgElement = document.createElement('img');
            imgElement.src = `data:image/jpeg;base64,${imageData}`;
            imgElement.alt = `SIFT Match ${index + 1}`;
            imgElement.className = 'sift-image';

            // Create image container with zoom capability
            const imgContainer = document.createElement('div');
            imgContainer.className = 'sift-image-container';
            imgContainer.appendChild(imgElement);
            imgWrapper.appendChild(imgContainer);

            // Add explanation caption
            const caption = document.createElement('p');
            caption.className = 'sift-caption';

            // Different explanations based on the match type
            if (index === 2) {
                caption.innerHTML = 'Loop closure match connects the last and first frames to ensure global consistency';
            } else {
                caption.innerHTML = 'Green lines show matched SIFT features between consecutive frames';
            }

            imgWrapper.appendChild(caption);

            // Add to the main container
            imageContainer.appendChild(imgWrapper);
        });
    } else {
        // No images found
        const placeholder = document.createElement('p');
        placeholder.className = 'placeholder-text';
        placeholder.textContent = 'No SIFT feature matches available';
        imageContainer.appendChild(placeholder);
    }
}

// Add log entry to the logs container
function addLogEntry(text) {
    const logEntry = document.createElement('p');
    logEntry.className = 'log-entry';
    logEntry.textContent = text;
    logsContainer.appendChild(logEntry);

    // Auto-scroll to bottom
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

// Update point size
function updatePointSize() {
    if (pointCloud && pointCloud.material) {
        pointCloud.material.size = parseFloat(pointSizeSlider.value);
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initialize 3D scene
    initScene();

    // Fetch available objects
    fetch('/api/objects')
        .then(response => response.json())
        .then(objects => {
            // Clear any existing options
            objectSelect.innerHTML = '';

            // Populate object select with the new format
            objects.forEach(object => {
                const option = document.createElement('option');
                option.value = object.value;
                option.textContent = object.label;
                objectSelect.appendChild(option);
            });

            // Fetch GIFs for the default selected object
            const defaultObject = objectSelect.value;
            if (defaultObject) {
                fetchAndDisplayGifs(defaultObject);
            }
        })
        .catch(error => {
            console.error('Error fetching objects:', error);
        });

    // Add event listener for object selection change
    objectSelect.addEventListener('change', function() {
        const selectedObject = this.value;
        fetchAndDisplayGifs(selectedObject);

        // Hide SIFT container when changing objects
        const siftContainer = document.getElementById('sift-container');
        if (siftContainer) {
            siftContainer.style.display = 'none';
        }

        // Uncheck the SIFT checkbox
        document.getElementById('show-features').checked = false;
    });

    // Add event listener for SIFT feature checkbox
    document.getElementById('show-features').addEventListener('change', function() {
        const siftContainer = document.getElementById('sift-container');

        if (this.checked) {
            // Just create the container if it doesn't exist, but don't load anything yet
            if (!siftContainer) {
                createSiftContainer();
            } else {
                siftContainer.style.display = 'block';
                siftContainer.innerHTML = '<h3>SIFT Feature Matches</h3><p class="placeholder-text">SIFT feature matches will be shown when you click "Generate 3D Model"</p>';
            }

            console.log("SIFT feature matching enabled. Will show matches when Generate button is clicked.");
        } else {
            // Hide SIFT container
            if (siftContainer) {
                siftContainer.style.display = 'none';
            }
        }
    });

    // Add event listeners
    generateBtn.addEventListener('click', generateModel);
    cancelBtn.addEventListener('click', cancelProcess);
    resetViewBtn.addEventListener('click', resetView);
    pointSizeSlider.addEventListener('input', updatePointSize);
});

// Create SIFT container
function createSiftContainer() {
    const siftContainer = document.createElement('div');
    siftContainer.id = 'sift-container';
    siftContainer.className = 'sift-container';
    siftContainer.style.display = 'block';

    // Add title
    const title = document.createElement('h3');
    title.textContent = 'SIFT Feature Matches';
    siftContainer.appendChild(title);

    // Add loading indicator
    const loadingOverlay = document.createElement('div');
    loadingOverlay.className = 'loading-overlay';
    loadingOverlay.innerHTML = '<div class="spinner"></div>';
    siftContainer.appendChild(loadingOverlay);

    // Add to the page after the visualization windows
    document.querySelector('.visualization-windows').after(siftContainer);

    return siftContainer;
}
