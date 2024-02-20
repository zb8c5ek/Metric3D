__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""
import cv2
import numpy as np
from ImageLoaderWithCalib.CalibLoader import CalibLoader
from tqdm import tqdm

class ImageLoader:
    def __init__(self, image_dir, calib_file):
        self.image_dir = image_dir
        self.calib = CalibLoader(calib_file)
        self.image_info = {}  # dictionary to store image names, paths and undistorted images

    def load_images(self, path=None):
        if path is None:
            path = self.image_dir
        if path.is_dir():
            img_files = list(path.glob('*.jpg'))  # adjust the pattern ('*.jpg') according to your images
        elif path.is_file():
            img_files = [path]
        else:
            raise ValueError(f"Invalid path: {path}")

        for img_file in tqdm(img_files, total=len(img_files)):
            img = cv2.imread(str(img_file))
            h, w = img.shape[:2]
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.calib.intrinsic, self.calib.distorted[:4], np.eye(3), self.calib.intrinsic, (w, h), cv2.CV_16SC2)

            # undistort
            dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            # store image name, path and undistorted image in the dictionary
            self.image_info[img_file.name] = (img_file, dst)

        return self.image_info

def show_cv_img_in_pil(img):
    from PIL import Image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.show()


if __name__ == "__main__":
    from pathlib import Path

    fp_img = Path(
        r"E:\psd_data\data_calib_20231129\work_dir_20240219-0.050\traj_analyzer0.050\length_0060_0070_num_10\segment_0008\cam_0_120_degree").resolve()
    fp_calib = Path(
        r"D:\mono-metric-depth\MonoMetricOdometry\sample_data\SU_U_M692H_20240202\cam1_params.yml").resolve()

    image_loader = ImageLoader(fp_img, fp_calib)
    image_info = image_loader.load_images()

    dp_output = fp_img.parent / "undistorted"
    dp_output.mkdir(exist_ok=True)
    for img_name, (img_path, img) in image_info.items():
        new_name = f"{img_path.stem}_distorted.jpg"
        cv2.imwrite(str(dp_output / new_name), img)

