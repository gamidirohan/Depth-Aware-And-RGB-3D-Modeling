import open3d as o3d
import numpy as np
import argparse
import os
from termcolor import colored

def visualize_pcd(pcd_path):
    """
    Load and visualize a point cloud from a PCD file
    """
    if not os.path.exists(pcd_path):
        print(colored(f"[ERROR] PCD file not found: {pcd_path}", "red"))
        return False
    
    print(colored(f"[INFO] Loading PCD file: {pcd_path}", "cyan"))
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    if pcd is None or len(pcd.points) == 0:
        print(colored(f"[ERROR] Failed to load PCD or empty point cloud", "red"))
        return False
    
    # Print information about the point cloud
    print(colored("Point Cloud Information:", "green"))
    print(f"  - Number of points: {len(pcd.points)}")
    print(f"  - Has colors: {pcd.has_colors()}")
    print(f"  - Has normals: {pcd.has_normals()}")
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud Viewer - {os.path.basename(pcd_path)}", width=1024, height=768)
    vis.add_geometry(pcd)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    
    # Set view control
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    print(colored("\nVisualization Controls:", "yellow"))
    print("  - Left mouse: Rotate")
    print("  - Ctrl + Left mouse: Pan/Move")
    print("  - Wheel: Zoom in/out")
    print("  - Shift + Left mouse: Roll")
    print("  - 'h': Show/Hide help menu")
    print("  - 'r': Reset view")
    print("  - '-/+': Decrease/Increase point size")
    print(colored("Press 'q' or 'ESC' to close the window", "yellow"))
    
    vis.run()
    vis.destroy_window()
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PCD files")
    parser.add_argument("-p", "--path", type=str, help="Path to the PCD file")
    args = parser.parse_args()
    
    if args.path:
        pcd_path = args.path
    else:
        pcd_path = input("Enter the path to the PCD file: ")
    
    visualize_pcd(pcd_path)
