import sys
import os
import numpy as np
import open3d as o3d
import copy

def run_reconstruction(object_name, optimization):
    """
    Run the 3D reconstruction directly without creating temporary scripts

    Args:
        object_name: Name of the object to reconstruct (castard, spyderman, new_box2)
        optimization: Whether to use optimized or unoptimized method (optimized, unoptimized)
    """
    print(f"Running reconstruction for {object_name} using {optimization} method...")

    if optimization == "optimized":
        # Import the optimized module
        import pose_graph_Feature_based_optimized_fixed as optimized_module

        # Save the original object_name
        original_object_name = optimized_module.object_name if hasattr(optimized_module, 'object_name') else None

        # Set the object_name
        optimized_module.object_name = object_name

        # Run the main function
        try:
            print(f"Running optimized reconstruction for {object_name}...")

            # Path configuration
            if object_name == "castard":
                depth_path = ['./train/castard/depth/align_test_depth%d.png' % i for i in range(1, 21)]
                rgb_path = ['./train/castard/rgb/align_test%d.png' % i for i in range(1, 21)]
                pcds_paths = ['./pcd_o3d/castard/box%d.pcd' % i for i in range(1, 21)]
            elif object_name == "spyderman":
                # Check both naming conventions for spyderman files
                depth_path = ['./train/spyderman2/depth/align_test_depth%d.png' % i for i in range(1, 17)]
                rgb_path = ['./train/spyderman2/rgb/align_test%d.png' % i for i in range(1, 17)]

                # Try both naming patterns for spyderman
                pcds_paths = []
                for i in range(1, 17):
                    # Try first naming pattern
                    path1 = f'./pcd_o3d/spyderman2/spyderman2{i}.pcd'
                    # Try second naming pattern
                    path2 = f'./pcd_o3d/spyderman2/spyderman2_{i}.pcd'

                    if os.path.exists(path1):
                        pcds_paths.append(path1)
                    elif os.path.exists(path2):
                        pcds_paths.append(path2)
                    else:
                        print(f"Warning: No file found for spyderman frame {i}")
            elif object_name == "new_box2":
                depth_path = ['./train/new_box2/depth/align_test_depth%d.png' % i for i in range(1, 17)]
                rgb_path = ['./train/new_box2/rgb/align_test%d.png' % i for i in range(1, 17)]
                pcds_paths = ['./pcd_o3d/new_box2/box%d.pcd' % i for i in range(1, 17)]

            # Parameters
            voxel_size = 0.002  # Adjusted voxel size to better handle the scale of your data

            # Load point clouds
            print("Loading point clouds...")
            pcds = []
            print(f'Loading {len(pcds_paths)} point clouds...')
            for path in pcds_paths:
                try:
                    pcd = o3d.io.read_point_cloud(path)
                    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
                    pcds.append(pcd_down)
                    print(f"Successfully loaded {path}")
                except Exception as e:
                    print(f"Error loading {path}: {str(e)}")

            if len(pcds) == 0:
                print("Error: No point clouds were loaded. Check file paths.")
                return

            # Visualize input point clouds with their original RGB colors
            print("Visualizing input point clouds with RGB colors...")
            o3d.visualization.draw_geometries(pcds)

            # Preprocess point clouds - ensure normals are computed
            print("Preprocessing point clouds...")
            for pcd in pcds:
                if not pcd.has_normals():
                    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

            # Build pose graph
            print("Building pose graph...")
            pose_graph = optimized_module.build_pose_graph(pcds, rgb_path, depth_path, voxel_size)

            # Optimize pose graph
            print("Optimizing pose graph...")
            optimized_pose_graph = optimized_module.optimize_pose_graph(pose_graph, voxel_size)

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
            cleaned_pcd = optimized_module.filter_outliers(accumulated_pcd, nb_points=30, std_ratio=2.0)

            # Save the result with multiple naming conventions for compatibility
            output_filenames = [
                f'accumulated_{object_name}_optimized.pcd'
            ]

            # Save with all naming conventions
            for output_filename in output_filenames:
                o3d.io.write_point_cloud(output_filename, cleaned_pcd)
                print(f"Result saved as {output_filename}")

            # Visualize final result
            o3d.visualization.draw_geometries([cleaned_pcd])

            print("Optimized reconstruction completed successfully!")
        except Exception as e:
            print(f"Error during optimized reconstruction: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore the original object_name
            if original_object_name is not None:
                optimized_module.object_name = original_object_name

    else:  # unoptimized
        # Import the unoptimized module
        import pose_graph_Feature_based_NO as unoptimized_module

        # Save the original object_name
        original_object_name = unoptimized_module.object_name if hasattr(unoptimized_module, 'object_name') else None

        # Set the object_name
        unoptimized_module.object_name = object_name

        # Run the main function
        try:
            print(f"Running unoptimized reconstruction for {object_name}...")

            # Get the main code
            import importlib
            importlib.reload(unoptimized_module)

            # Save with additional naming conventions for compatibility
            output_filenames = [
                f'accumulated_{object_name}_unoptimized.pcd',
                f'accumulated_{object_name}_improved.pcd'
            ]

            # Special case for castard
            if object_name == 'castard':
                output_filenames.append(f'box_castard_registered.pcd')

            # Find the original output file
            original_file = None
            possible_files = [
                f'accumulated_{object_name}_improved.pcd',
                f'box_{object_name}_registered.pcd',
                f'box_{object_name}ard_registered.pcd'
            ]

            for file in possible_files:
                if os.path.exists(file):
                    original_file = file
                    break

            # If we found the original file, copy it to all naming conventions
            if original_file:
                pcd = o3d.io.read_point_cloud(original_file)
                for output_filename in output_filenames:
                    if output_filename != original_file:
                        o3d.io.write_point_cloud(output_filename, pcd)
                        print(f"Result saved as {output_filename}")

            print("Unoptimized reconstruction completed successfully!")
        except Exception as e:
            print(f"Error during unoptimized reconstruction: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore the original object_name
            if original_object_name is not None:
                unoptimized_module.object_name = original_object_name

if __name__ == "__main__":
    # Check if we have the right number of arguments
    if len(sys.argv) != 3:
        print("Usage: python direct_reconstruction.py <object_name> <optimization>")
        print("  object_name: castard, spyderman, or new_box2")
        print("  optimization: optimized or unoptimized")
        sys.exit(1)

    object_name = sys.argv[1]
    optimization = sys.argv[2]

    # Validate arguments
    if object_name not in ["castard", "spyderman", "new_box2"]:
        print(f"Error: Invalid object name '{object_name}'")
        print("Valid object names: castard, spyderman, new_box2")
        sys.exit(1)

    if optimization not in ["optimized", "unoptimized"]:
        print(f"Error: Invalid optimization '{optimization}'")
        print("Valid optimizations: optimized, unoptimized")
        sys.exit(1)

    # Run the reconstruction
    run_reconstruction(object_name, optimization)
