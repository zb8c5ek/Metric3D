__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""
import yaml
import numpy as np

class CalibLoader:
    def __init__(self, file_path):
        with open(file_path, 'r') as stream:
            lines = stream.readlines()

        # Remove the first two lines
        lines = lines[2:]

        # Join the remaining lines back into a string
        data_string = ''.join(lines)
        # Now you can load the YAML data from the string
        data_loaded = yaml.safe_load(data_string)
        camera_data = data_loaded['camera']

        self.intrinsic_full = np.array(camera_data['intrinsic']).reshape(4, 4)
        self.intrinsic = self.intrinsic_full[0:3, 0:3]
        self.distorted = np.array(camera_data['distorted'])
        self.cam2ego_R = camera_data['cam2ego_R']
        self.cam2ego_t = camera_data['cam2ego_t']
        self.ego2gnd_t = camera_data['ego2gnd_t']
        self.image_height = camera_data['image_height'][0]
        self.image_width = camera_data['image_width'][0]
        self.nearest_pt_threshold = camera_data['nearest_pt_threshold'][0]
        self.view_range = camera_data['view_range'][0]
        self.linear_range = camera_data['linear_range'][0]


if __name__ == "__main__":
    from pathlib import Path

    fp_calib = Path(
        r"D:\mono-metric-depth\MonoMetricOdometry\sample_data\SU_U_M692H_20240202\cam1_params.yml").resolve()
    calibration = CalibLoader(fp_calib.as_posix())
    print(calibration.intrinsic)
    print(calibration.distorted)
    # and so on for the other attributes
