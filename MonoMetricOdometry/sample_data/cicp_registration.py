__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""
# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""ICP variant that uses both geometry and color for registration"""

import open3d as o3d
import numpy as np
import copy


def draw_registration_result(source, target, transformation, fp_output=None):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw([source_temp, target])
    if fp_output is not None:
        o3d.io.write_point_cloud(fp_output, source_temp)

print("Load two point clouds and show initial pose ...")
# ply_data = o3d.data.DemoColoredICPPointClouds()
# source = o3d.io.read_point_cloud(ply_data.paths[0])
# target = o3d.io.read_point_cloud(ply_data.paths[1])

source = o3d.io.read_point_cloud(r"D:\mono-metric-depth\MonoMetricOdometry\sample_data_2\3d_result\pcd\MonoMetricOdometry\frame-002985_1701313572850_fx_2038_fy_2038.ply")
target = o3d.io.read_point_cloud(r"D:\mono-metric-depth\MonoMetricOdometry\sample_data_2\3d_result\pcd\MonoMetricOdometry\frame-002987_1701313572950_fx_2038_fy_2038.ply")

# Filter points in source
source_points = np.asarray(source.points)
source_colors = np.asarray(source.colors)
dist_xz = np.linalg.norm(source_points[:, [0, 2]], axis=1)  # Compute norm2 distance for x-z
mask_source = (dist_xz > 50) | (source_points[:, 1] > 5)  # Create a mask for the conditions
source.points = o3d.utility.Vector3dVector(source_points[~mask_source])  # Apply the mask
source.colors = o3d.utility.Vector3dVector(source_colors[~mask_source])  # Apply the mask to colors

# Filter points in target
target_points = np.asarray(target.points)
target_colors = np.asarray(target.colors)
dist_xz = np.linalg.norm(target_points[:, [0, 2]], axis=1)  # Compute norm2 distance for x-z
mask_target = (dist_xz > 50) | (target_points[:, 1] > 5) #| (dist_xz < 25) # Create a mask for the conditions
target.points = o3d.utility.Vector3dVector(target_points[~mask_target])  # Apply the mask
target.colors = o3d.utility.Vector3dVector(target_colors[~mask_target])  # Apply the mask to colors

# Visualize the point clouds
o3d.visualization.draw_geometries([source])
o3d.visualization.draw_geometries([target])
o3d.visualization.draw_geometries([source, target])

if __name__ == "__main__":
    # Draw initial alignment.
    current_transformation = np.identity(4)
    # draw_registration_result(source, target, current_transformation)
    print(current_transformation)

    # Colored pointcloud registration.
    # This is implementation of following paper:
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017.
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("Colored point cloud registration ...\n")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("2. Estimate normal")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp, "\n")
    source.paint_uniform_color([1, 0, 0])  # Red
    target.paint_uniform_color([0, 0, 1])  # Blue
    draw_registration_result(
        source, target, result_icp.transformation,
        fp_output=r"D:\mono-metric-depth\MonoMetricOdometry\sample_data_2/video_0_warped.ply"
    )
    print(current_transformation)

    # Extract rotation (3x3 upper left) and translation (3x1 upper right)
    rotation = current_transformation[:3, :3].copy()
    translation = current_transformation[:3, 3].copy()
    from scipy.spatial.transform import Rotation
    # Convert rotation matrix to Euler angles
    r = Rotation.from_matrix(rotation)
    euler_angles = r.as_euler('xyz', degrees=True)

    print("Euler angles (in degrees): ", euler_angles)
    print("Translation (in meters): ", translation)