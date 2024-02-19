__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""
"""
the sample data is from: "E:\psd_data\data_calib_20231129\work_dir_epsilon_0.1\length_0060_0070_num_16\segment_0000"
sample data are selected out accoring to time stamp:
video_0-frame-002981_1701313572650.jpg
video_1-frame-002988_1701313572650.jpg
"""
from PIL import Image
import numpy as np
import open3d as o3d
from tqdm import tqdm


class MonoDepthMapping:

    def __init__(self, fps_img, K=None):
        """
            given a list of image paths, compute the depth map and register together. often 3 to 5 images shall be
            enough, and they shall be sequential in time in a straight line trajectory, produced by SfM-Engine.
        """
        self.fps_img = fps_img

        # Load Images
        self.images = []
        for fp_img in self.fps_img:
            self.images.append(Image.open(fp_img))

        self.img_shape = self.images[0].size

        # Load Model and Compute Depth
        depth_engine = GetDepthAnythingDisp()

        if K is None:
            self.K = np.array([[3100, 0, self.img_shape[1] / 2], [0, 3100, self.img_shape[0] / 2], [0, 0, 1]])
        else:
            self.K = K

        fx, fy, cx, cy, width, height = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2], self.img_shape[0], \
        self.img_shape[1]
        # Create PinholeCameraIntrinsic object
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        self.disps = []
        for i, image in enumerate(tqdm(self.images)):
            disp = depth_engine(image)
            self.disps.append(disp)
            img_rgb = o3d.io.read_image(self.fps_img[i].as_posix())
            depth = 1000 / (disp + 100)
            # depth[depth > 300] = np.nan
            img_depth = o3d.geometry.Image(depth)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_rgb, img_depth)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
            o3d.visualization.draw_geometries([pcd])

        self._compute_sequential_transformation()

    def _compute_sequential_transformation(self, K=None, Bf=1000):
        """
            compute the transformation between sequential images
        """
        # Convert disps to depth by assuming a Bf value
        depths = []
        for disp in self.disps:
            d_temp = Bf / disp
            d_temp[np.abs(d_temp) > 100] = 0
            d_temp[np.isnan(d_temp)] = 0
            depths.append(d_temp)

        # Compute transformation between sequential images
        if K is None:
            K = np.array([[3100, 0, self.img_shape[1] / 2], [0, 3100, self.img_shape[0] / 2], [0, 0, 1]])

        # Convert image and depth to point cloud
        point_clouds = []
        for image, depth in zip(self.images, depths):
            point_clouds.append(self.image_depth_to_point_cloud(image, depth, K))

    @staticmethod
    def image_depth_to_point_cloud(image, depth, K, visualize=False):
        """
            convert image and depth to point cloud in open3d capatible format
        """
        # Get image size
        width, height = image.size
        source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            np.array(image), depth)
        target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, K)
        # Get pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Get intrinsic matrix
        inverse_K = np.linalg.inv(K)
        selected_mask = depth > 0
        selected_u, selected_v = u[selected_mask], v[selected_mask]
        selected_z = depth[depth > 0].reshape([-1, 1])

        pcd = reconstruct_pcd(depth, fx=K[0, 0], fy=K[1, 1], u0=K[0, 2], v0=K[1, 2], pcd_base=None, mask=selected_mask)

        coords_2d_homo = np.vstack([selected_u, selected_v, np.ones(selected_z.shape)])
        ray_coordinates = coords_2d_homo.T @ inverse_K.T

        assert np.allclose(ray_coordinates[:, -1], 1)

        # Get point cloud in camera frame
        coords_3d = ray_coordinates * selected_z

        # Convert image to numpy array and normalize color to [0, 1]
        image_np = np.array(image) / 255.0

        # Get color for each point
        colors = image_np[selected_u, selected_v]

        # Convert point cloud to open3d format
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coords_3d))

        # Set color for each point
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Visualize point cloud
        if visualize:
            o3d.visualization.draw_geometries([point_cloud])

        # Write the point cloud out
        fp_out = Path("output_point_cloud.pcd").resolve()
        o3d.io.write_point_cloud(fp_out.as_posix(), point_cloud)

        return point_cloud


class ImageLoaderWithCalib:
    def __init__(self, fps_img, K):
        self.fps_img = fps_img
        self.K = K

    def __getitem__(self, idx):
        img = Image.open(self.fps_img[idx])
        return img, self.K

    def __len__(self):
        return len(self.fps_img)


if __name__ == "__main__":
    from pathlib import Path

    fps_img = [Path(r"D:\mono-metric-depth\data\wild_road\frame-002953_1701313571250.jpg"),
               Path(r"D:\mono-metric-depth\data\wild_road\frame-002954_1701313571300.jpg"),
               Path(r"D:\mono-metric-depth\data\wild_road\frame-002955_1701313571350.jpg")]
    mono_depth_mapping = MonoDepthMapping(fps_img)
