import numpy as np
import open3d as o3d
from SIFT import SIFT_Transformation
import matplotlib.pyplot as plt

def load_point_clouds(voxel_size=0.0, pcds_paths=None):
    """Load and downsample point clouds."""
    pcds = []
    print(f'Loading {len(pcds_paths)} point clouds...')
    for path in pcds_paths:
        pcd = o3d.io.read_point_cloud(path)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
    return pcds

def preprocess_point_cloud(pcd, voxel_size):
    """Preprocess point cloud by downsampling and computing normals."""
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh

def pairwise_registration(source, target, voxel_size, init_trans=np.identity(4)):
    """Perform pairwise registration with RANSAC followed by ICP refinement."""
    print(":: Apply point-to-plane ICP")

    # Ensure normals are estimated
    if not source.has_normals():
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    if not target.has_normals():
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # Set correspondence distances
    distance_coarse = voxel_size * 15
    distance_fine = voxel_size * 1.5

    # ICP coarse alignment
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, distance_coarse, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # ICP fine alignment
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, distance_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, distance_fine, icp_fine.transformation)

    # Check fitness score - higher is better (max 1.0)
    fitness = icp_fine.fitness
    print(f"Fitness score: {fitness}")

    return transformation_icp, information_icp, fitness

def build_pose_graph(pcds, rgb_path, depth_path, voxel_size):
    """Build pose graph for global registration."""
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            # Process only consecutive frames for odometry
            if target_id == source_id + 1:
                print(f"Registering {source_id} -> {target_id}")

                # Use SIFT to get initial transformation
                try:
                    init_trans, _, _, _, _, _, _ = SIFT_Transformation(
                        rgb_path[source_id], rgb_path[target_id],
                        depth_path[source_id], depth_path[target_id],
                        pcds[source_id], pcds[target_id],
                        distance_ratio=0.8)
                except:
                    print(f"SIFT failed for {source_id}->{target_id}, using identity")
                    init_trans = np.identity(4)

                # Refine with ICP
                transformation_icp, information_icp, fitness = pairwise_registration(
                    pcds[source_id], pcds[target_id], voxel_size, init_trans)

                # If fitness score is too low, try with different parameters
                if fitness < 0.3:
                    print(f"Low fitness score ({fitness}). Trying with different parameters.")
                    init_trans = np.identity(4)
                    transformation_icp, information_icp, fitness = pairwise_registration(
                        pcds[source_id], pcds[target_id], voxel_size * 2, init_trans)

                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))

            # Add non-sequential connections for every k frames (for loop closure)
            elif target_id % 5 == 0 and source_id % 5 == 0:
                print(f"Adding non-sequential connection {source_id} -> {target_id}")
                transformation_icp, information_icp, _ = pairwise_registration(
                    pcds[source_id], pcds[target_id], voxel_size, np.identity(4))

                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))

    # Add loop closure between last and first frame if we have enough frames
    if n_pcds >= 8:
        source_id = n_pcds - 1
        target_id = 0
        print(f"Adding loop closure {source_id} -> {target_id}")

        try:
            init_trans, _, _, _, _, _, _ = SIFT_Transformation(
                rgb_path[source_id], rgb_path[target_id],
                depth_path[source_id], depth_path[target_id],
                pcds[source_id], pcds[target_id],
                distance_ratio=0.8)
        except:
            print(f"SIFT failed for loop closure, using identity")
            init_trans = np.identity(4)

        transformation_icp, information_icp, _ = pairwise_registration(
            pcds[source_id], pcds[target_id], voxel_size, init_trans)

        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                     target_id,
                                                     transformation_icp,
                                                     information_icp,
                                                     uncertain=True))

    return pose_graph

def optimize_pose_graph(pose_graph, voxel_size):
    """Optimize pose graph for global consistency."""
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=voxel_size * 1.5,
        edge_prune_threshold=0.25,
        preference_loop_closure=5.0,  # Increased weight for loop closure
        reference_node=0)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

    return pose_graph

def filter_outliers(pcd, nb_points=20, std_ratio=2.0):
    """Filter outliers from point cloud."""
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_points, std_ratio=std_ratio)
    return cl

def visualize_registration_result(source, target, transformation):
    """Visualize registration result."""
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

if __name__ == "__main__":
    import copy

    object_name = "new_box2"  # "castard", "spyderman", or "new_box2"

    # Path configuration
    if object_name == "castard":
        depth_path = ['./train/castard/depth/align_test_depth%d.png' % i for i in range(1, 6)]
        rgb_path = ['./train/castard/rgb/align_test%d.png' % i for i in range(1, 6)]
        pcds_paths = ['./pcd_o3d/castard/box%d.pcd' % i for i in range(1, 21)]
    elif object_name == "spyderman":
        depth_path = ['./train/spyderman2/depth/align_test_depth%d.png' % i for i in range(1, 17)]
        rgb_path = ['./train/spyderman2/rgb/align_test%d.png' % i for i in range(1, 17)]
        pcds_paths = ['./pcd_o3d/spyderman2/spyderman2%d.pcd' % i for i in range(1, 17)]
    elif object_name == "new_box2":
        depth_path = ['./train/new_box2/depth/align_test_depth%d.png' % i for i in range(1, 17)]
        rgb_path = ['./train/new_box2/rgb/align_test%d.png' % i for i in range(1, 17)]
        pcds_paths = ['./pcd_o3d/new_box2/box%d.pcd' % i for i in range(1, 17)]

    # Parameters
    voxel_size = 0.002  # Adjusted voxel size to better handle the scale of your data

    # Load point clouds
    print("Loading point clouds...")
    pcds = load_point_clouds(voxel_size, pcds_paths)

    # Visualize input point clouds with their original RGB colors
    print("Visualizing input point clouds with RGB colors...")
    # No need to paint uniform colors - use the original RGB colors from the point clouds
    o3d.visualization.draw_geometries(pcds)

    # Preprocess point clouds - ensure normals are computed
    print("Preprocessing point clouds...")
    for pcd in pcds:
        if not pcd.has_normals():
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # Build pose graph
    print("Building pose graph...")
    pose_graph = build_pose_graph(pcds, rgb_path, depth_path, voxel_size)

    # Optimize pose graph
    print("Optimizing pose graph...")
    optimized_pose_graph = optimize_pose_graph(pose_graph, voxel_size)

    # Transform points and combine
    print("Transforming and combining point clouds...")
    accumulated_pcd = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        print(f"Transformation matrix for point cloud {point_id}:")
        print(optimized_pose_graph.nodes[point_id].pose)
        transformed_pcd = copy.deepcopy(pcds[point_id])
        transformed_pcd.transform(optimized_pose_graph.nodes[point_id].pose)
        accumulated_pcd += transformed_pcd

    # Filter outliers for cleaner result
    print("Filtering outliers...")
    cleaned_pcd = filter_outliers(accumulated_pcd, nb_points=30, std_ratio=2.0)

    # Visualize final result with RGB colors
    print("Visualizing final result with RGB colors...")
    vis = o3d.visualization.Visualizer()
    vis.create_window('Final RGB Point Cloud')
    vis.add_geometry(cleaned_pcd)

    # Set rendering options to ensure RGB colors are displayed
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    opt.point_color_option = o3d.visualization.PointColorOption.Color  # Ensure RGB colors are used

    vis.run()
    vis.destroy_window()

    # Save the result
    output_filename = f'accumulated_{object_name}_improved.pcd'
    o3d.io.write_point_cloud(output_filename, cleaned_pcd)
    print(f"Result saved as {output_filename}")

    # Optional: Create a visualization with all transformed point clouds and RGB colors
    print("Creating additional visualization with coordinate frame...")
    vis = o3d.visualization.Visualizer()
    vis.create_window('3D Reconstructed Object with RGB')

    # Add coordinate frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(axis)

    # Add the combined point cloud
    vis.add_geometry(cleaned_pcd)

    # Set rendering options to ensure RGB colors are displayed
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    opt.point_color_option = o3d.visualization.PointColorOption.Color  # Ensure RGB colors are used

    vis.run()
    vis.destroy_window()