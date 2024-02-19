__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""
import os
import os.path as osp
import cv2
import time
import sys
from pathlib import Path
CODE_SPACE = Path(r"D:\mono-metric-depth").resolve().as_posix()
sys.path.append(CODE_SPACE)
import argparse
import mmcv
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
from datetime import timedelta
import random
import numpy as np
from mono.utils.logger import setup_logger
import glob
from mono.utils.comm import init_env
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import do_scalecano_test_with_custom_data
from mono.utils.mldb import load_data_info, reset_ckpt_path
from mono.utils.custom_data import load_from_annos, load_data
# mono/configs/HourglassDecoder/convlarge.0.3_150.py
# --load-from
# ./weight/convlarge_hourglass_0.3_150_step750k_v1.1.pth
# --test_data_path
# D:\mono-metric-depth\MonoMetricOdometry\sample_data\images
# --show-dir
# D:\mono-metric-depth\MonoMetricOdometry\sample_data\3d_result
# --launcher
# None

def bringup(show_dir: str = r"D:\mono-metric-depth\MonoMetricOdometry\sample_data\3d_result"):
    os.chdir(CODE_SPACE)
    fn_config = r"D:\mono-metric-depth\mono/configs/HourglassDecoder/convlarge.0.3_150.py"
    cfg = Config.fromfile(fn_config)
    cfg.show_dir = show_dir
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    cfg.load_from = r"D:\mono-metric-depth/weight/convlarge_hourglass_0.3_150_step750k_v1.1.pth"

    # load data info
    data_info = {}
    load_data_info('data_info', data_info=data_info)
    cfg.mldb_info = data_info
    # update check point info
    reset_ckpt_path(cfg.model, data_info)

    # create show dir
    os.makedirs(osp.abspath(cfg.show_dir), exist_ok=True)

    # init the logger before other steps
    cfg.log_file = osp.join(cfg.show_dir, f'{timestamp}.log')
    logger = setup_logger(cfg.log_file)

    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.distributed = False
    logger.info(f'Distributed training: {cfg.distributed}')

    # dump config
    cfg.dump(osp.join(cfg.show_dir, osp.basename(fn_config)))
    return cfg

def main_script(local_rank: int, cfg: dict, launcher: str, test_data: list):

    test_data_path = args.test_data_path
    if not os.path.isabs(test_data_path):
        test_data_path = osp.join(CODE_SPACE, test_data_path)

    if 'json' in test_data_path:
        test_data = load_from_annos(test_data_path)
    else:
        test_data = load_data(args.test_data_path)

    if not cfg.distributed:
        main_worker(0, cfg, args.launcher, test_data)
    else:
        # distributed training
        if args.launcher == 'ror':
            local_rank = cfg.dist_params.local_rank
            main_worker(local_rank, cfg, args.launcher, test_data)
        else:
            mp.spawn(main_worker, nprocs=cfg.dist_params.num_gpus_per_node, args=(cfg, args.launcher, test_data))

    if cfg.distributed:
        cfg.dist_params.global_rank = cfg.dist_params.node_rank * cfg.dist_params.num_gpus_per_node + local_rank
        cfg.dist_params.local_rank = local_rank


        torch.cuda.set_device(local_rank)
        default_timeout = timedelta(minutes=30)
        dist.init_process_group(
            backend=cfg.dist_params.backend,
            init_method=cfg.dist_params.dist_url,
            world_size=cfg.dist_params.world_size,
            rank=cfg.dist_params.global_rank,
            timeout=default_timeout)

    logger = setup_logger(cfg.log_file)
    # build model
    model = get_configured_monodepth_model(cfg, )

    # config distributed training
    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
                                                          device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # load ckpt
    model, _, _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
    model.eval()
    # TODO: use this function to load the data and hence test
    do_scalecano_test_with_custom_data(
        model,
        cfg,
        test_data,
        logger,
        cfg.distributed,
        local_rank
    )


class ComputeDepth:
    def __init__(self, fp_img, calib):
        pass


if __name__ == '__main__':
    cfg = bringup()
    # main_worker(0, cfg, args.launcher, test_data)