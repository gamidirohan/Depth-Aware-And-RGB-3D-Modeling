import cv2
import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
from termcolor import colored

def rgbd_to_pcd(count, total_files):
    print(colored(f"[Processing: Apple {count}/{total_files}]", "cyan"))
    print("-" * 50)
    
    # Read RGB and depth images
    rgb_path = os.path.join('train', 'apple_2', 'rgb', f'align_test_{count}.png')
    depth_path = os.path.join('train', 'apple_2', 'depth', f'align_test_depth_{count}.png')

    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print(colored(f"[ERROR] Missing files: {rgb_path} or {depth_path}", "red"))
        return

    source_color = o3d.io.read_image(rgb_path)
    source_depth = o3d.io.read_image(depth_path)

    if source_color is None or source_depth is None:
        print(colored(f"[ERROR] Failed to read images: {rgb_path} or {depth_path}", "red"))
        return

    K = np.array(
         [[597.522, 0.0, 312.885],
         [0.0, 597.522, 239.870],
         [0.0, 0.0, 1.0]], dtype=np.float64)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.intrinsic_matrix = K

    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        source_color, source_depth, depth_scale=1000, convert_rgb_to_intensity=False, depth_trunc=1)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, intrinsic)
    
    print(colored("🟢 Step 1: Precomputing Neighbors", "green"))
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    
    labels = np.array(outlier_cloud.cluster_dbscan(eps=0.02, min_points=500, print_progress=True))
    indexes = np.where(labels == 0)
    
    # Extract Interest point clouds
    interest_pcd = o3d.geometry.PointCloud()
    interest_pcd.points = o3d.utility.Vector3dVector(np.asarray(outlier_cloud.points, np.float32)[indexes])
    interest_pcd.colors = o3d.utility.Vector3dVector(np.asarray(outlier_cloud.colors, np.float32)[indexes])
    
    # Load mask image
    mask_file = os.path.join('train', 'apple_2', 'mask', f'align_test_mask_{count}.png')
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) > 0  # Convert to binary mask

    if mask is None:
        print(colored(f"[ERROR] Mask file missing: {mask_file}", "red"))
        return
    
    print(colored("🟢 Step 2: Applying Mask Filtering", "green"))
    points = np.asarray(interest_pcd.points)
    colors = np.asarray(interest_pcd.colors)
    height, width = mask.shape

    u = np.clip((points[:, 0] * K[0, 0] / points[:, 2] + K[0, 2]).astype(int), 0, width - 1)
    v = np.clip((points[:, 1] * K[1, 1] / points[:, 2] + K[1, 2]).astype(int), 0, height - 1)
    mask_indices = mask[v, u]
    filtered_points = points[mask_indices]
    filtered_colors = colors[mask_indices]

    interest_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    interest_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    print(colored("🟡 Step 3: Applying Height-based Filtering", "yellow"))
    y = np.asarray(interest_pcd.points)[:, 1]
    idx = np.where(y < 0.18)[0]
    interest_pcd = interest_pcd.select_by_index(list(idx))
    
    print(colored("🔵 Step 4: Outlier Removal", "blue"))
    cl, ind = interest_pcd.remove_statistical_outlier(nb_neighbors=1000, std_ratio=0.8)
    cl, ind = cl.remove_radius_outlier(nb_points=100, radius=0.01)
    
    output_path = os.path.join('pcd_o3d', 'apple_2', f'apple_{count}.pcd')
    success = o3d.io.write_point_cloud(output_path, cl)
    
    if success:
        print(colored(f"✅ Successfully saved: {output_path}", "green"))
    else:
        print(colored("❌ ERROR: Write PCD failed", "red"))
    print("-" * 50)

if __name__ == '__main__':
    rgb_dir = os.path.join('train', 'apple_2', 'rgb')
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    file_numbers = sorted([int(f.split('_')[-1].split('.')[0]) for f in rgb_files])

    total_files = len(file_numbers)

    for count in tqdm(file_numbers, desc=f"Processing Apples ({total_files})", unit="apple"):
        rgbd_to_pcd(count, total_files)