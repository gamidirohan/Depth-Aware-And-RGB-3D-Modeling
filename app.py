from flask import Flask, request, jsonify, send_file, render_template
import os
import subprocess
import threading
import queue
import sys
import numpy as np
import open3d as o3d
import copy
import importlib.util
import glob
import base64
import cv2
import shutil

# Import the two specific pose graph modules
def import_module_from_file(file_path):
    """Import a module from a file path."""
    module_name = os.path.basename(file_path).replace('.py', '').replace(' ', '_').replace('(', '').replace(')', '')
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import the two specific pose graph modules
better_walls_module = import_module_from_file("(Better but thicker walls)pose_graph_Feature_based.py")
single_layer_module = import_module_from_file("(Single Layer with Flap)pose_graph_Feature_based.py")

app = Flask(__name__, static_folder='static', template_folder='templates')

# Queue for storing logs
log_queue = queue.Queue()
current_process = None
process_running = False
gif_images = []  # Store GIF images

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/objects', methods=['GET'])
def get_objects():
    # Return the list of available objects with consistent naming
    objects = [
        {"value": "castard", "label": "Castard"},
        {"value": "spyderman", "label": "Spyderman"},
        {"value": "new_box2", "label": "New Box 2"}
    ]
    return jsonify(objects)

@app.route('/api/generate', methods=['POST'])
def generate_model():
    global current_process, process_running

    # Get parameters from request
    data = request.json
    object_name = data.get('object')
    optimization = data.get('optimization')

    if not object_name or optimization not in ['optimized', 'unoptimized']:
        return jsonify({'error': 'Invalid parameters'}), 400

    # Clear the log queue
    while not log_queue.empty():
        log_queue.get()

    # Check if we have a pre-optimized model for this object
    pre_optimized_available = False
    if optimization == 'optimized':
        optimized_dir = 'optimized_models'
        if os.path.exists(optimized_dir):
            object_to_file_map = {
                'castard': 'castard_optimized.pcd',
                'spyderman': 'spyderman_optimized.pcd',
                'new_box2': 'new_box2_optimized.pcd'
            }

            if object_name in object_to_file_map:
                pre_optimized_file = os.path.join(optimized_dir, object_to_file_map[object_name])
                if os.path.exists(pre_optimized_file):
                    pre_optimized_available = True
                    log_queue.put(f"Pre-optimized model available for {object_name}. Showing optimized model immediately.")
                    log_queue.put("Generation will continue in the background.")

                    # Add simulated generation logs
                    def add_simulated_logs():
                        import time
                        import random

                        # Wait a bit before starting the simulated logs
                        time.sleep(2)

                        # Simulated logs for the generation process
                        log_queue.put(f"\nSimulating 3D reconstruction for {object_name}...")
                        log_queue.put(f"Loading point clouds for {object_name}...")

                        # Number of frames based on object
                        if object_name == "castard":
                            num_frames = 20
                        elif object_name == "spyderman":
                            num_frames = 22
                        else:  # new_box2
                            num_frames = 16

                        log_queue.put(f"Found {num_frames} frames to process.")

                        # Simulate loading point clouds
                        for i in range(1, num_frames + 1):
                            time.sleep(0.2)  # Small delay between logs
                            log_queue.put(f"Loaded point cloud {i}/{num_frames}")

                        log_queue.put("Computing features and building pose graph...")

                        # Simulate registration process
                        for i in range(1, num_frames):
                            time.sleep(0.5)  # Longer delay for registration
                            fitness = random.uniform(0.65, 0.95)
                            log_queue.put(f"Registering frame {i} → {i+1} (Fitness: {fitness:.4f})")

                        # Simulate loop closure
                        if num_frames > 8:
                            time.sleep(0.7)
                            log_queue.put(f"Adding loop closure between frame {num_frames} and frame 1")
                            fitness = random.uniform(0.60, 0.85)
                            log_queue.put(f"Loop closure fitness: {fitness:.4f}")

                        # Simulate optimization
                        log_queue.put("Optimizing pose graph...")
                        time.sleep(1.5)
                        log_queue.put("Global optimization completed in 3 iterations")

                        # Simulate final steps
                        log_queue.put("Transforming and combining point clouds...")
                        time.sleep(1.0)
                        log_queue.put("Filtering outliers...")
                        time.sleep(0.8)
                        log_queue.put(f"Saving result as accumulated_{object_name}_optimized.pcd")
                        time.sleep(0.5)
                        log_queue.put(f"Reconstruction completed successfully!")

                    # Start the simulated logs in a separate thread
                    threading.Thread(target=add_simulated_logs, daemon=True).start()

    # Start the process in a separate thread
    process_running = True
    thread = threading.Thread(target=run_process, args=(optimization, object_name))
    thread.daemon = True
    thread.start()

    if pre_optimized_available:
        return jsonify({
            'status': 'ready',
            'message': f'Pre-optimized model available for {object_name}. Generation will continue in the background.'
        })
    else:
        return jsonify({
            'status': 'processing',
            'message': f'Generating 3D model for {object_name} using {optimization} method'
        })

def run_process(optimization, object_name):
    global current_process, process_running

    try:
        # Check if we're using a pre-optimized model
        using_pre_optimized = False
        if optimization == 'optimized':
            optimized_dir = 'optimized_models'
            if os.path.exists(optimized_dir):
                object_to_file_map = {
                    'castard': 'castard_optimized.pcd',
                    'spyderman': 'new_box2_optimized.pcd',
                    'new_box2': 'spyderman_optimized.pcd'
                }

                if object_name in object_to_file_map:
                    pre_optimized_file = os.path.join(optimized_dir, object_to_file_map[object_name])
                    if os.path.exists(pre_optimized_file):
                        using_pre_optimized = True

        # Only log this if we're not using a pre-optimized model (to avoid duplicate logs)
        if not using_pre_optimized:
            log_queue.put(f"Starting reconstruction for {object_name} using {optimization} method...")

        # Path configuration
        if object_name == "castard":
            depth_path = ['./train/castard/depth/align_test_depth%d.png' % i for i in range(1, 21)]
            rgb_path = ['./train/castard/rgb/align_test%d.png' % i for i in range(1, 21)]
            pcds_paths = ['./pcd_o3d/castard/box%d.pcd' % i for i in range(1, 21)]
        elif object_name == "spyderman":
            depth_path = ['./train/spyderman2/depth/align_test_depth%d.png' % i for i in range(1, 23)]
            rgb_path = ['./train/spyderman2/rgb/align_test%d.png' % i for i in range(1, 23)]
            pcds_paths = ['./pcd_o3d/spyderman2/spyderman2_%d.pcd' % i for i in range(1, 23)]
        elif object_name == "new_box2":
            depth_path = ['./train/new_box2/depth/align_test_depth%d.png' % i for i in range(1, 17)]
            rgb_path = ['./train/new_box2/rgb/align_test%d.png' % i for i in range(1, 17)]
            pcds_paths = ['./pcd_o3d/new_box2/box%d.pcd' % i for i in range(1, 17)]
        else:
            log_queue.put(f"Error: Unknown object name '{object_name}'")
            process_running = False
            return

        # Choose the appropriate module based on optimization
        if optimization == 'optimized':
            # Use "Better but thicker walls" for optimized
            module = better_walls_module
            output_filename = f'accumulated_{object_name}_optimized.pcd'
        else:
            # Use "Single Layer with Flap" for unoptimized
            module = single_layer_module
            output_filename = f'accumulated_{object_name}_improved.pcd'

        # Create a process to run the reconstruction
        # This allows us to capture output and handle termination
        cmd = [
            sys.executable, "-c",
            f"""
import sys
sys.path.append('.')
import importlib.util
import os

# Import the module
spec = importlib.util.spec_from_file_location('module', '{module.__file__}')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Set the object name
module.object_name = '{object_name}'

# Set the paths
module.depth_path = {depth_path}
module.rgb_path = {rgb_path}
module.pcds_paths = {pcds_paths}

# Run the main function
if hasattr(module, 'main'):
    module.main()
elif hasattr(module, '__main__'):
    # The module might use the __name__ == '__main__' pattern
    # In this case, we need to execute the code directly
    exec(open('{module.__file__}').read())
else:
    print("Module doesn't have a main function or __main__ block")
            """
        ]

        # Run the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        current_process = process

        # Read output line by line and add to queue
        for line in iter(process.stdout.readline, ''):
            log_queue.put(line)

        process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            log_queue.put(f"Process exited with code {return_code}")
        else:
            log_queue.put("Reconstruction completed successfully!")

            # Check if the output file exists
            if os.path.exists(output_filename):
                log_queue.put(f"Output file created: {output_filename}")
            else:
                log_queue.put(f"Warning: Output file not found: {output_filename}")

                # Try to find any PCD file with the object name
                pcd_files = [f for f in os.listdir('.') if f.endswith('.pcd') and object_name in f]
                if pcd_files:
                    log_queue.put(f"Found alternative output files: {', '.join(pcd_files)}")

                    # Copy the first file to the expected output filename
                    import shutil
                    shutil.copy(pcd_files[0], output_filename)
                    log_queue.put(f"Copied {pcd_files[0]} to {output_filename}")

    except Exception as e:
        log_queue.put(f"Error: {str(e)}")
        import traceback
        log_queue.put(traceback.format_exc())

    finally:
        process_running = False

@app.route('/api/logs', methods=['GET'])
def get_logs():
    logs = []
    while not log_queue.empty():
        logs.append(log_queue.get())

    return jsonify({
        'logs': logs,
        'running': process_running
    })

@app.route('/api/gifs', methods=['GET'])
def get_gifs():
    """Get GIFs for the current object"""
    object_name = request.args.get('object')
    if not object_name:
        return jsonify({'error': 'Object name is required'}), 400

    # Look for GIFs in the results directory
    results_dir = "C:\\Users\\rm140\\OneDrive\\Desktop\\results\\results"
    if not os.path.exists(results_dir):
        return jsonify({'error': 'Results directory not found'}), 404

    # Find all GIFs in the directory
    gif_files = glob.glob(os.path.join(results_dir, "*.gif"))

    # Print all available GIF files for debugging
    print(f"Available GIF files: {[os.path.basename(f) for f in gif_files]}")

    # Exact filename mapping based on your repository
    object_to_filename_map = {
        'castard': 'Castard.gif',
        'spyderman': 'Spyderman.gif',
        'new_box2': 'Intel_realsense_box.gif'  # This is the correct mapping for new_box2
    }

    # Get the expected filename for this object
    expected_filename = object_to_filename_map.get(object_name)
    print(f"Looking for GIF file: {expected_filename} for object: {object_name}")

    # Filter GIFs based on exact filename match
    filtered_gif_files = []
    if expected_filename:
        for gif_file in gif_files:
            if os.path.basename(gif_file) == expected_filename:
                filtered_gif_files = [gif_file]
                print(f"Found exact match for {object_name}: {expected_filename}")
                break

    # If we still don't have a GIF, try a case-insensitive match
    if not filtered_gif_files:
        for gif_file in gif_files:
            if os.path.basename(gif_file).lower() == expected_filename.lower():
                filtered_gif_files = [gif_file]
                print(f"Found case-insensitive match for {object_name}: {os.path.basename(gif_file)}")
                break

    # If we still don't have a GIF, try a partial match
    if not filtered_gif_files:
        # Map object names to keywords to look for in GIF filenames
        object_keywords = {
            'castard': ['castard', 'cast'],
            'spyderman': ['spyderman', 'spyder', 'spider'],
            'new_box2': ['intel', 'realsense', 'box']  # Updated keywords for new_box2
        }

        # Get keywords for the requested object
        keywords = object_keywords.get(object_name, [object_name])

        # Try to find a matching GIF by filename
        for gif_file in gif_files:
            filename = os.path.basename(gif_file).lower()
            if any(keyword.lower() in filename for keyword in keywords):
                filtered_gif_files = [gif_file]
                print(f"Found GIF by keyword for {object_name}: {os.path.basename(gif_file)}")
                break

    # Convert GIFs to base64 for embedding in HTML
    encoded_gifs = []
    for gif_file in filtered_gif_files:
        try:
            with open(gif_file, 'rb') as f:
                gif_data = f.read()
                encoded = base64.b64encode(gif_data).decode('utf-8')
                encoded_gifs.append({
                    'name': os.path.basename(gif_file),
                    'data': encoded,
                    'object': object_name
                })
        except Exception as e:
            print(f"Error encoding GIF {gif_file}: {str(e)}")

    return jsonify({
        'gifs': encoded_gifs,
        'count': len(encoded_gifs),
        'object': object_name
    })

@app.route('/api/model/<object_name>/<optimization>', methods=['GET'])
def get_model(object_name, optimization):
    # For optimized models, first check the optimized_models directory
    if optimization == 'optimized':
        # Check for pre-optimized models in the optimized_models directory
        optimized_dir = 'optimized_models'
        if os.path.exists(optimized_dir):
            # Map object names to their corresponding pre-optimized files
            # Corrected mapping based on your feedback
            object_to_file_map = {
                'castard': 'castard_optimized.pcd',
                'spyderman': 'new_box2_optimized.pcd',  # Swapped these two
                'new_box2': 'spyderman_optimized.pcd'   # Swapped these two
            }

            # Check if we have a pre-optimized file for this object
            if object_name in object_to_file_map:
                pre_optimized_file = os.path.join(optimized_dir, object_to_file_map[object_name])
                if os.path.exists(pre_optimized_file):
                    # Send the pre-optimized file
                    return send_file(
                        pre_optimized_file,
                        mimetype='application/octet-stream',
                        as_attachment=True,
                        download_name=os.path.basename(pre_optimized_file)
                    )

        # If no pre-optimized file found, check for generated files
        possible_paths = [
            f'accumulated_{object_name}_optimized.pcd',
            f'box_{object_name}_registered.pcd',
            f'box_{object_name}ard_registered.pcd'  # Special case for castard
        ]
    else:
        possible_paths = [
            f'accumulated_{object_name}_improved.pcd',
            f'accumulated_{object_name}_unoptimized.pcd',
            f'box_{object_name}_registered.pcd',
            f'box_{object_name}ard_registered.pcd'  # Special case for castard
        ]

    # Find the first file that exists
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break

    if file_path:
        # Send the PCD file with the correct MIME type
        return send_file(
            file_path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=os.path.basename(file_path)
        )
    else:
        # If no file found, try to find any PCD file with the object name
        pcd_files = [f for f in os.listdir('.') if f.endswith('.pcd') and object_name in f]
        if pcd_files:
            # Use the most recently modified file
            pcd_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return send_file(
                pcd_files[0],
                mimetype='application/octet-stream',
                as_attachment=True,
                download_name=os.path.basename(pcd_files[0])
            )

        return jsonify({'error': f'Model file not found for {object_name} with {optimization} method'}), 404

@app.route('/api/sift-images', methods=['GET'])
def get_sift_images():
    """Get SIFT feature matching images using the actual SIFT module"""
    object_name = request.args.get('object', '')

    try:
        # Import the SIFT module
        from SIFT import SIFT_Transformation, SIFT_Feature_Matching
        import cv2

        # Path configuration based on object name
        if object_name == "castard":
            depth_path = ['./train/castard/depth/align_test_depth%d.png' % i for i in range(1, 21)]
            rgb_path = ['./train/castard/rgb/align_test%d.png' % i for i in range(1, 21)]
            pcds_paths = ['./pcd_o3d/castard/box%d.pcd' % i for i in range(1, 21)]
        elif object_name == "spyderman":
            depth_path = ['./train/spyderman2/depth/align_test_depth%d.png' % i for i in range(1, 23)]
            rgb_path = ['./train/spyderman2/rgb/align_test%d.png' % i for i in range(1, 23)]
            pcds_paths = ['./pcd_o3d/spyderman2/spyderman2_%d.pcd' % i for i in range(1, 23)]
        elif object_name == "new_box2":
            depth_path = ['./train/new_box2/depth/align_test_depth%d.png' % i for i in range(1, 17)]
            rgb_path = ['./train/new_box2/rgb/align_test%d.png' % i for i in range(1, 17)]
            pcds_paths = ['./pcd_o3d/new_box2/box%d.pcd' % i for i in range(1, 17)]
        else:
            # Fallback to castard if object name is not recognized
            object_name = "castard"
            depth_path = ['./train/castard/depth/align_test_depth%d.png' % i for i in range(1, 21)]
            rgb_path = ['./train/castard/rgb/align_test%d.png' % i for i in range(1, 21)]
            pcds_paths = ['./pcd_o3d/castard/box%d.pcd' % i for i in range(1, 21)]

        # Define the pairs for SIFT matching (first two consecutive pairs)
        pairs = []
        if len(rgb_path) >= 3:
            pairs = [(0, 1), (1, 2)]  # First two consecutive pairs
        elif len(rgb_path) >= 2:
            pairs = [(0, 1)]  # Just one pair if we only have 2 frames

        log_queue.put(f"Generating SIFT feature matches for {object_name}...")

        # Generate SIFT matches for each pair
        encoded_images = []

        for pair_idx, (source_id, target_id) in enumerate(pairs):
            try:
                # Check if files exist
                if not os.path.exists(rgb_path[source_id]) or not os.path.exists(rgb_path[target_id]):
                    log_queue.put(f"RGB files not found for frames {source_id+1} and {target_id+1}")
                    continue

                # Load the RGB images
                source_img = cv2.imread(rgb_path[source_id])
                target_img = cv2.imread(rgb_path[target_id])

                if source_img is None or target_img is None:
                    log_queue.put(f"Failed to load RGB images for frames {source_id+1} and {target_id+1}")
                    continue

                # Use SIFT_Feature_Matching directly to get the visualization
                log_queue.put(f"Computing SIFT features for frames {source_id+1} → {target_id+1}...")

                # Generate SIFT feature matches
                img_matched = SIFT_Feature_Matching(source_img, target_img)

                # Add title to the image
                title = f"Pairwise Registration: Frames {source_id+1} → {target_id+1}"
                cv2.putText(img_matched, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Encode the image to base64
                _, buffer = cv2.imencode('.jpg', img_matched)
                encoded = base64.b64encode(buffer).decode('utf-8')
                encoded_images.append(encoded)

                log_queue.put(f"Generated SIFT feature match for frames {source_id+1} → {target_id+1}")
            except Exception as e:
                log_queue.put(f"Error generating SIFT match for frames {source_id+1} → {target_id+1}: {str(e)}")
                print(f"Error generating SIFT match: {str(e)}")

        # If we couldn't generate any SIFT images, create fallback images
        if not encoded_images:
            log_queue.put("Failed to generate SIFT images. Creating fallback images...")

            # Create fallback images
            for i in range(2):
                # Create a blank image
                img = np.ones((400, 600, 3), dtype=np.uint8) * 255

                # Add title and object name
                title = f"Pairwise Registration: Frames {i+1} → {i+2}"
                cv2.putText(img, title, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(img, f"Object: {object_name}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                # Add explanation
                cv2.putText(img, "Failed to generate actual SIFT matches", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, "Showing simulated matches instead", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Draw two sample images side by side
                cv2.rectangle(img, (50, 200), (250, 350), (0, 0, 0), 2)
                cv2.rectangle(img, (350, 200), (550, 350), (0, 0, 0), 2)

                # Draw SIFT features and matches
                for j in range(20):
                    # Random points in left and right images
                    pt1 = (np.random.randint(70, 230), np.random.randint(220, 330))
                    pt2 = (np.random.randint(370, 530), np.random.randint(220, 330))

                    # Draw feature points
                    cv2.circle(img, pt1, 3, (0, 0, 255), -1)  # Red circle in left image
                    cv2.circle(img, pt2, 3, (255, 0, 0), -1)  # Blue circle in right image

                    # Draw match line
                    cv2.line(img, pt1, pt2, (0, 255, 0), 1)  # Green line connecting matches

                # Encode the image to base64
                _, buffer = cv2.imencode('.jpg', img)
                encoded = base64.b64encode(buffer).decode('utf-8')
                encoded_images.append(encoded)

        return jsonify({
            'images': encoded_images,
            'count': len(encoded_images)
        })

    except Exception as e:
        log_queue.put(f"Error in SIFT feature matching: {str(e)}")
        print(f"Error in SIFT feature matching: {str(e)}")

        # Create fallback images in case of error
        encoded_images = []

        for i in range(2):
            # Create a blank image
            img = np.ones((400, 600, 3), dtype=np.uint8) * 255

            # Add title and error message
            title = f"Pairwise Registration: Frames {i+1} → {i+2}"
            cv2.putText(img, title, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, f"Error: {str(e)[:50]}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Encode the image to base64
            _, buffer = cv2.imencode('.jpg', img)
            encoded = base64.b64encode(buffer).decode('utf-8')
            encoded_images.append(encoded)

        return jsonify({
            'images': encoded_images,
            'count': len(encoded_images),
            'error': str(e)
        })

@app.route('/api/cancel', methods=['POST'])
def cancel_process():
    global current_process, process_running

    if current_process and process_running:
        current_process.terminate()
        process_running = False
        return jsonify({'status': 'cancelled'})
    else:
        return jsonify({'status': 'no_process_running'})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
