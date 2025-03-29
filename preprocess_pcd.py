import cv2
import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
from termcolor import colored
import traceback

def rgbd_to_pcd(count, total_files, debug=True):
    print(colored(f"[Processing: Apple {count}/{total_files}]", "cyan"))
    print("-" * 50)
    
    # Read RGB and depth images
    rgb_path = os.path.join('train', 'apple_2', 'rgb', f'align_test_{count}.png')
    depth_path = os.path.join('train', 'apple_2', 'depth', f'align_test_depth_{count}.png')

    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print(colored(f"[ERROR] Missing files: {rgb_path} or {depth_path}", "red"))
        return

    # Debug: Print file information
    if debug:
        print(colored(f"[DEBUG] Loading RGB: {rgb_path}", "magenta"))
        print(colored(f"[DEBUG] Loading Depth: {depth_path}", "magenta"))

    source_color = o3d.io.read_image(rgb_path)
    source_depth = o3d.io.read_image(depth_path)

    # Debug: Check image dimensions
    if debug:
        print(colored(f"[DEBUG] RGB image dimensions: {np.asarray(source_color).shape}", "magenta"))
        print(colored(f"[DEBUG] Depth image dimensions: {np.asarray(source_depth).shape}", "magenta"))

    if source_color is None or source_depth is None:
        print(colored(f"[ERROR] Failed to read images: {rgb_path} or {depth_path}", "red"))
        return

    K = np.array(
         [[597.522, 0.0, 312.885],
         [0.0, 597.522, 239.870],
         [0.0, 0.0, 1.0]], dtype=np.float64)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.intrinsic_matrix = K

    # Debug: Print camera parameters
    if debug:
        print(colored(f"[DEBUG] Camera intrinsic matrix:\n{K}", "magenta"))

    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        source_color, source_depth, depth_scale=1000, convert_rgb_to_intensity=False, depth_trunc=1)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, intrinsic)
    
    # Debug: Initial point cloud info
    if debug:
        print(colored(f"[DEBUG] Initial point cloud: {len(pcd.points)} points", "magenta"))
        print(colored(f"[DEBUG] Point cloud bounds: Min {np.min(pcd.points, axis=0)}, Max {np.max(pcd.points, axis=0)}", "magenta"))
    
    # Check if initial point cloud is empty
    if len(pcd.points) == 0:
        print(colored("[ERROR] Initial point cloud is empty", "red"))
        return
    
    print(colored("🟢 Step 1: Precomputing Neighbors", "green"))
    # Debug: Begin plane segmentation
    if debug:
        print(colored("[DEBUG] Starting plane segmentation with distance_threshold=0.02", "magenta"))
    
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    
    # Debug: Plane segmentation results
    if debug:
        print(colored(f"[DEBUG] Plane model: {plane_model}", "magenta"))
        print(colored(f"[DEBUG] Inliers: {len(inliers)} points", "magenta"))
        print(colored(f"[DEBUG] Outlier cloud: {len(outlier_cloud.points)} points", "magenta"))
    
    # Check if there are points after plane segmentation
    if len(outlier_cloud.points) == 0:
        print(colored("[ERROR] No points after plane segmentation, using original cloud", "red"))
        outlier_cloud = pcd  # Fallback to original point cloud
    
    # DBSCAN clustering
    # Debug: DBSCAN parameters
    if debug:
        print(colored("[DEBUG] Starting DBSCAN with eps=0.02, min_points=500", "magenta"))
    
    labels = np.array(outlier_cloud.cluster_dbscan(eps=0.02, min_points=500, print_progress=True))
    
    # Debug: DBSCAN results
    if debug:
        unique_labels = np.unique(labels)
        print(colored(f"[DEBUG] DBSCAN found {len(unique_labels)} unique labels: {unique_labels}", "magenta"))
        for label in unique_labels:
            if label >= 0:  # Ignore noise (-1)
                print(colored(f"[DEBUG] Cluster {label} has {np.sum(labels == label)} points", "magenta"))
    
    # Check if any clusters were found
    if len(np.unique(labels)) <= 1 and -1 in np.unique(labels):
        print(colored("[WARNING] No clusters found with DBSCAN, trying with smaller min_points", "yellow"))
        # Try with smaller min_points value
        if debug:
            print(colored("[DEBUG] Retrying DBSCAN with eps=0.02, min_points=100", "magenta"))
        
        labels = np.array(outlier_cloud.cluster_dbscan(eps=0.02, min_points=100, print_progress=True))
        
        # Debug: Updated DBSCAN results
        if debug:
            unique_labels = np.unique(labels)
            print(colored(f"[DEBUG] Retry DBSCAN found {len(unique_labels)} unique labels: {unique_labels}", "magenta"))
            for label in unique_labels:
                if label >= 0:  # Ignore noise (-1)
                    print(colored(f"[DEBUG] Cluster {label} has {np.sum(labels == label)} points", "magenta"))
    
    # If still no clusters, use all points except plane
    if len(np.unique(labels)) <= 1 and -1 in np.unique(labels):
        print(colored("[WARNING] Still no clusters found, using all non-plane points", "yellow"))
        interest_pcd = outlier_cloud
    else:
        # Get the largest cluster (label 0)
        indexes = np.where(labels == 0)
        
        # Debug: Largest cluster info
        if debug:
            print(colored(f"[DEBUG] Largest cluster has {len(indexes[0])} points", "magenta"))
        
        # Extract Interest point clouds
        interest_pcd = o3d.geometry.PointCloud()
        interest_pcd.points = o3d.utility.Vector3dVector(np.asarray(outlier_cloud.points, np.float32)[indexes])
        interest_pcd.colors = o3d.utility.Vector3dVector(np.asarray(outlier_cloud.colors, np.float32)[indexes])
    
    # Check if point cloud is empty after clustering
    if len(interest_pcd.points) == 0:
        print(colored("[ERROR] No points after clustering, using outlier cloud", "red"))
        interest_pcd = outlier_cloud
    
    # Debug: Interest point cloud after clustering
    if debug:
        print(colored(f"[DEBUG] Interest point cloud after clustering: {len(interest_pcd.points)} points", "magenta"))
    
    # Load mask image
    mask_file = os.path.join('train', 'apple_2', 'mask', f'align_test_mask_{count}.png')
    
    # Debug: Mask file info
    if debug:
        print(colored(f"[DEBUG] Loading mask file: {mask_file}", "magenta"))
    
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(colored(f"[ERROR] Mask file missing: {mask_file}", "red"))
        return
    
    # Debug: Mask dimensions
    if debug:
        print(colored(f"[DEBUG] Mask dimensions: {mask.shape}", "magenta"))
        print(colored(f"[DEBUG] Mask values - Min: {mask.min()}, Max: {mask.max()}, Positive pixels: {np.sum(mask > 0)}", "magenta"))
    
    mask = mask > 0  # Convert to binary mask
    
    print(colored("🟢 Step 2: Applying Mask Filtering", "green"))
    points = np.asarray(interest_pcd.points)
    colors = np.asarray(interest_pcd.colors)
    height, width = mask.shape

    # Debug: Project 3D points to 2D
    if debug:
        print(colored("[DEBUG] Projecting 3D points to 2D for mask filtering", "magenta"))
        print(colored(f"[DEBUG] Points shape: {points.shape}", "magenta"))
    
    # Calculate 2D projection coordinates and apply mask
    u = np.clip((points[:, 0] * K[0, 0] / points[:, 2] + K[0, 2]).astype(int), 0, width - 1)
    v = np.clip((points[:, 1] * K[1, 1] / points[:, 2] + K[1, 2]).astype(int), 0, height - 1)
    mask_indices = mask[v, u]
    
    # Debug: Mask application results
    if debug:
        print(colored(f"[DEBUG] Projected points in mask: {np.sum(mask_indices)} of {len(mask_indices)}", "magenta"))
    
    filtered_points = points[mask_indices]
    filtered_colors = colors[mask_indices]

    # Check if points remain after mask filtering
    if len(filtered_points) == 0:
        print(colored("[ERROR] No points after mask filtering, using pre-mask point cloud", "red"))
        filtered_points = points
        filtered_colors = colors

    interest_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    interest_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    # Debug: Point cloud after mask filtering
    if debug:
        print(colored(f"[DEBUG] Point cloud after mask filtering: {len(interest_pcd.points)} points", "magenta"))
    
    print(colored("🟡 Step 3: Applying Height-based Filtering", "yellow"))
    y = np.asarray(interest_pcd.points)[:, 1]
    
    # Debug: Y-coordinate statistics
    if debug:
        y_min, y_max = np.min(y), np.max(y)
        print(colored(f"[DEBUG] Y-coordinate range: Min={y_min:.4f}, Max={y_max:.4f}, Mean={np.mean(y):.4f}", "magenta"))
        print(colored(f"[DEBUG] Applying height threshold of 0.18", "magenta"))
    
    idx = np.where(y < 0.18)[0]
    
    # Debug: Height filtering results
    if debug:
        print(colored(f"[DEBUG] Points below height threshold: {len(idx)} of {len(y)}", "magenta"))
    
    # Check if any points meet the height criteria
    if len(idx) == 0:
        print(colored("[WARNING] No points after height filtering, using higher threshold", "yellow"))
        # Try with a higher threshold
        if debug:
            print(colored("[DEBUG] Retrying with height threshold of 0.25", "magenta"))
        
        idx = np.where(y < 0.25)[0]
        
        # Debug: Updated height filtering results
        if debug:
            print(colored(f"[DEBUG] Points below adjusted height threshold: {len(idx)} of {len(y)}", "magenta"))
        
        # If still no points, skip height filtering
        if len(idx) == 0:
            print(colored("[ERROR] Still no points after adjusted height filtering, skipping height filter", "red"))
            idx = np.arange(len(y))
    
    interest_pcd = interest_pcd.select_by_index(list(idx))
    
    # Debug: Point cloud after height filtering
    if debug:
        print(colored(f"[DEBUG] Point cloud after height filtering: {len(interest_pcd.points)} points", "magenta"))
    
    # Check if point cloud is empty after height filtering
    if len(interest_pcd.points) == 0:
        print(colored("[ERROR] Empty point cloud after height filtering", "red"))
        # We'll skip saving this file
        return
    
    print(colored("🔵 Step 4: Outlier Removal", "blue"))
    try:
        # First statistical outlier removal
        nb_neighbors = min(1000, len(interest_pcd.points) - 1)
        
        # Debug: Statistical outlier removal parameters
        if debug:
            print(colored(f"[DEBUG] Statistical outlier removal with nb_neighbors={nb_neighbors}, std_ratio=0.8", "magenta"))
        
        cl, ind = interest_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=0.8)
        
        # Debug: Statistical outlier removal results
        if debug:
            print(colored(f"[DEBUG] After statistical outlier removal: {len(cl.points)} points (removed {len(interest_pcd.points) - len(cl.points)})", "magenta"))
        
        # Check if point cloud is empty after statistical outlier removal
        if len(cl.points) == 0:
            print(colored("[WARNING] Empty cloud after statistical outlier removal, using original", "yellow"))
            cl = interest_pcd
        
        # Second radius outlier removal
        nb_points = min(100, len(cl.points) - 1)
        
        # Debug: Radius outlier removal parameters
        if debug:
            print(colored(f"[DEBUG] Radius outlier removal with nb_points={nb_points}, radius=0.01", "magenta"))
        
        final_cl, ind = cl.remove_radius_outlier(nb_points=nb_points, radius=0.01)
        
        # Debug: Radius outlier removal results
        if debug:
            print(colored(f"[DEBUG] After radius outlier removal: {len(final_cl.points)} points (removed {len(cl.points) - len(final_cl.points)})", "magenta"))
        
        # Check if point cloud is empty after radius outlier removal
        if len(final_cl.points) == 0:
            print(colored("[WARNING] Empty cloud after radius outlier removal, using previous", "yellow"))
            final_cl = cl
        
    except Exception as e:
        # Debug: Detailed exception information
        if debug:
            print(colored(f"[DEBUG] Exception details:\n{traceback.format_exc()}", "magenta"))
        
        print(colored(f"[ERROR] Exception during outlier removal: {str(e)}", "red"))
        # Use the point cloud before outlier removal
        final_cl = interest_pcd
    
    # Final check before saving
    if len(final_cl.points) == 0:
        print(colored("[ERROR] Final point cloud is empty, cannot save", "red"))
        return
    
    # Debug: Final point cloud statistics
    if debug:
        points_array = np.asarray(final_cl.points)
        print(colored(f"[DEBUG] Final point cloud: {len(final_cl.points)} points", "magenta"))
        print(colored(f"[DEBUG] Final point cloud bounds: Min {np.min(points_array, axis=0)}, Max {np.max(points_array, axis=0)}", "magenta"))
    
    # Ensure output directory exists
    output_dir = os.path.join('pcd_o3d', 'apple_2')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'apple_{count}.pcd')
    
    # Debug: Saving information
    if debug:
        print(colored(f"[DEBUG] Saving point cloud to: {output_path}", "magenta"))
    
    success = o3d.io.write_point_cloud(output_path, final_cl)
    
    if success:
        print(colored(f"✅ Successfully saved: {output_path}", "green"))
    else:
        print(colored("❌ ERROR: Write PCD failed", "red"))
        # Debug: Attempt to diagnose write failure
        if debug:
            print(colored(f"[DEBUG] Write failed - checking file path existence: {os.path.exists(os.path.dirname(output_path))}", "magenta"))
            print(colored(f"[DEBUG] Write failed - point cloud validity: Points={len(final_cl.points)}, Has normals={final_cl.has_normals()}", "magenta"))
    
    print("-" * 50)

if __name__ == '__main__':
    # Add debug flag - set to False to reduce output
    DEBUG_MODE = True
    
    rgb_dir = os.path.join('train', 'apple_2', 'rgb')
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    file_numbers = sorted([int(f.split('_')[-1].split('.')[0]) for f in rgb_files])

    total_files = len(file_numbers)
    
    # Debug: Initial processing information
    if DEBUG_MODE:
        print(colored(f"[DEBUG] Processing {total_files} files from {rgb_dir}", "magenta"))
        print(colored(f"[DEBUG] File numbers range: {min(file_numbers)} to {max(file_numbers)}", "magenta"))

    for count in tqdm(file_numbers, desc=f"Processing Apples ({total_files})", unit="apple"):
        try:
            rgbd_to_pcd(count, total_files, debug=DEBUG_MODE)
        except Exception as e:
            print(colored(f"[CRITICAL ERROR] Processing failed for apple {count}: {str(e)}", "red"))
            # Debug: Full exception traceback
            if DEBUG_MODE:
                print(colored(f"[DEBUG] Exception traceback:\n{traceback.format_exc()}", "magenta"))
            continue