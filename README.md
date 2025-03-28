# Depth-Aware-And-RGB-3D-Modeling


With RGB-D cameras, we can get multiple RGB and Depth images and convert them to point clouds easily. Leveraging this, we can reconstruct single object with multi-view RGB and Depth images. To acheive this, point clouds from multi-view need to be registered. This task is also known as **point registration"**, whose goal is to find transfomration matrix between source and target point clouds. The alignment consists of two sub-alignment, Initial alignment and Alignment refinement. 

## Dataset Formatting

This project uses the [RGB-D dataset](http://www.cs.washington.edu/rgbd-dataset) from the University of Washington. Once downloaded, the dataset is formatted using the provided `format_repo.py` script to organize the files into the required structure.

### How to Use

1. **Download the dataset:**
   Download the RGB-D dataset from [here](http://www.cs.washington.edu/rgbd-dataset).

2. **Format the dataset:**
   Use the `format_repo.py` script to format the dataset as required for this repository. Update the `source_directory` and `target_directory` variables in the script to point to your dataset location.
   Easier to do with each object placed in the train directory, updating the directory variables in the code, and run.

   ```python
   if __name__ == "__main__":
       source_directory = r"train\apple_1"  # Update this path
       target_directory = source_directory  # Update this path
       reorganize_dataset(source_directory, target_directory)
   ```

3. **Run the script:**
   Execute the script to format the dataset.

   ```bash
   python format_repo.py
   ```

## RGB-D Camera Spec
- Model: Intel realsense D415
- Intrinsic parameters in 640x480 RGB, Depth image.<br> 
```
K = [[597.522, 0.0, 312.885],
     [0.0, 597.522, 239.870],
     [0.0, 0.0, 1.0]]
```
## Requirements

# myenv Virtual Environment File

Below is a sample requirements file for setting up your virtual environment with the necessary dependencies:

Create the virtual environment:
  
  python -m venv myenv
  # Activate the environment:
  # On Windows:
  ./myenv/Scripts/activate
  # On Unix or MacOS:
  source myenv/bin/activate

Then, create a file named requirements.txt with the following content:

pyrealsense2 only works with python 3.9

--------------------------------------------------
open3d
pyrealsense2
opencv-python
numpy
kornia
--------------------------------------------------

Finally, install the dependencies using:

  pip install -r requirements.txt

## Align RGB and Depth Image & Depth Filtering
Due to the different position of RGB and Depth lens, aligning them should be done to get exact point clouds. This project used alignment function offered by pyrealsense2 package. Raw depth data are so noisy that depth filtering should be needed. Pyrealsense2 library, developed by Intel, offers filtering methods of depth images. In this project, spatial-filter was used that smoothes noise and preserves edge components in depth images. <br>

```capture_aligned_images.py``` : **Capture RGB and Depth Image, align them** Set image path for saving RGB and Depth images. Press 'S' to capture scene.  

## Pre-Process Point Clouds
Single object might be a part of the scanned data. In order to get points of interested objects, pre-processing should be implemented. Plane-removal, outlier-removal, DBSCAN clustering were executed to extract object. Open3D offers useful functions to filter points.

```preprocess_pcd.py``` : **RGB-D Images to Object's Point clouds.** Plane-removal, points outlier-filtering, DBSCAN clustering were applied.

## Feature based Registration (Local Registration)
Initial alignment can be acheived through finding transformation matrix between feature points. The position of 3D points can be estimated with back-projection rays and depth values from depth image. Transformation matrix can be estimated with 3D corresponding feature points from souce and target point clouds, with RANSAC procedure. In order to find robust correspondeces in object area, extracted object 3D points were reprojected and the bounding area were obtained to filter outliers.<br><br>
```SIFT.py```: Find 3d transformation matrix with SIFT feature points<br>
```ORB.py```: Find 3d transformation matrix with ORB feature points<br>
```LoFTR.py```: Find 3d transformation matrix with LoFTR feature points<br>

> **Reprjection Constraints**
- Before <br>
![image](https://user-images.githubusercontent.com/50229148/207070426-1ba44e95-91ce-4f27-a5aa-790c82466651.png)
- After <br>
![image](https://user-images.githubusercontent.com/50229148/207070623-72690f60-20a7-4a4f-8e43-15d8a960783d.png)


## ICP based Registration (Global Registration) 
With ICP algorithm implemented in Open3D, refine initial transformation matrix. In this project, Point-to-Plane ICP method was used.

## Pose Graph Optimization
Pose graph optimization is a non-linear optimization of poses, frequently used in SLAM. Node represents poses of camera, edge represents relative pose between two cameras(nodes). In loop closure, the error occurs between predicted and measurement in loop nodes due to inaccurate camera poses. The goal of pose graph optimization is to minimize error between loop nodes(X_i, X_j), shown below. Levenberg-marquardt(LM) method was used for non-linear optimization<br><br> <img src="https://user-images.githubusercontent.com/50229148/207068648-7660b2f5-3d09-4fe5-92ae-1582f424e82d.png" width="500" height="300"> <br><br>

#### Effect of Pose graph optimzation
<img src="https://user-images.githubusercontent.com/50229148/207066364-70a2d1f5-0659-44fb-9fe3-2e934a765d22.gif" width="400" height="240"><img src="https://user-images.githubusercontent.com/50229148/207066377-2f863df7-4c54-4f0e-98ed-e22d7198e507.gif" width="400" height="240"> <br>
You can find the difference of registration quality between unoptimized(left) and optimized(right) in reconstruction result. In unoptimized result, error exists due to accumulated pose error. <br>

## Run Registration <br>
```Pose_graph_ICP.py```: Run registration with ICP method<br>
```pose_graph_Feature_based.py```: Run registration with Feature based(SIFT) method<br>

## Results <br>
The object was reconstructed with multiple different view of RGB-D Images. <br>

The reconstructed point clouds is below. <br>
#### Object 3D Reconstruction (Feature based)
<img src="https://user-images.githubusercontent.com/50229148/207057731-dd2b725b-4f1b-498a-8264-c0a37118fb1b.gif" width="400" height="240"><img src="https://user-images.githubusercontent.com/50229148/207055985-3f8fd7f2-305d-4b92-bff0-7f62675179ea.gif" width="400" height="240">

#### Object 3D Reconstruction (ICP based)
<img src="https://user-images.githubusercontent.com/50229148/207057841-18e17230-c2f8-4d73-834f-309f4a788ba8.gif" width="400" height="240"><img src="https://user-images.githubusercontent.com/50229148/207056463-6033a29a-d25f-4100-8e3d-c5c1883d9eb4.gif" width="400" height="240">


#### Human shape 3D Reconstruction 
<img src="https://user-images.githubusercontent.com/50229148/207057339-28d455ff-27b5-4209-9afe-6133ea94e4b2.gif" width="400" height="240"><img src="https://user-images.githubusercontent.com/50229148/207056403-2bf1fd3e-0f4b-418f-a3e1-9286207f2d34.gif" width="400" height="240">

## Pushing to GitHub

To push your changes to a new or existing branch on GitHub, follow these steps:

1. **Initialize Git (if not already initialized):**

   ```bash
   git init
   ```

2. **Add the remote repository:**

   ```bash
   git remote add origin https://github.com/gamidirohan/Depth-Aware-And-RGB-3D-Modeling.git
   ```

   If the remote already exists, update it:

   ```bash
   git remote set-url origin https://github.com/gamidirohan/Depth-Aware-And-RGB-3D-Modeling.git
   ```

3. **Add all files to the staging area:**

   ```bash
   git add .
   ```

4. **Commit the changes with a message:**

   ```bash
   git commit -m "Initial commit"
   ```

5. **Push the changes to the remote repository:**

   ```bash
   git push -u origin main
   ```

   If you are pushing to a different branch, replace `main` with your branch name:

   ```bash
   git push -u origin your-branch-name
   ```

Feel free to ask if you have any questions or need further assistance!