# -*- coding: utf-8 -*-
from tqdm.auto import tqdm
import os
import random
import sys
import time
import numpy as np
from PIL import Image
import torch
import open3d as o3d
import trimesh
import argparse
import logging
from pytorch3d.io import load_ply
from pytorch3d.loss import chamfer_distance as CD
from pytorch3d.ops import iterative_closest_point as ICP
# 新增依赖库
import pandas as pd  # <--- 添加pandas支持
import ipdb
import warnings

warnings.filterwarnings("ignore")


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Process and generate point cloud data."
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        help="PointCloud directory",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        help="GroundTruth directory",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for inference"
    )
    args = parser.parse_args()
    return args


def find_ply_files(directory):
    ply_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".ply"):
                ply_files.append(os.path.join(root, file))
    ply_files.sort()
    return ply_files


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_logger(args):
    script_name = os.path.basename(__file__)
    script_name_without_extension = os.path.splitext(script_name)[0]
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                "./logs/{}_seed{}_evaluation_{}.log".format(
                    script_name_without_extension,
                    args.seed,
                    time.strftime("%Y-%m-%d--%H-%M-%S"),
                )
            ),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


def main(args):
    os.makedirs("./logs", exist_ok=True)

    # 新增结果目录创建
    os.makedirs("./results", exist_ok=True)  # <--- 创建结果目录
    logger = get_logger(args)
    set_seed(args.seed)
    pred_pcd_list = find_ply_files(args.pred_dir)
    logger.info("Evaluating on {} pointclouds".format(len(pred_pcd_list)))
    # 修改数据结构：使用字典记录结果
    results = {  # <--- 替换原来的cd_list和error_list
        "Filename": [],
        "CD (e-3)": [],
        "Status": []
    }

    for pred_pcd_path in tqdm(pred_pcd_list):
        file_name = os.path.basename(pred_pcd_path)  # <--- 更安全的文件名获取方式
        logger.debug("Processing {}".format(file_name))

        # 初始化默认值
        cd_value = np.nan
        status = "Error"

        try:
            gt_pcd_path = os.path.join(args.gt_dir, file_name)
            if not os.path.exists(gt_pcd_path):
                raise FileNotFoundError(f"GT file not found: {gt_pcd_path}")
            # 加载点云数据
            gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)
            pred_pcd = o3d.io.read_point_cloud(pred_pcd_path)

            # 空文件检查
            if len(gt_pcd.points) == 0 or len(pred_pcd.points) == 0:  # <--- 新增空数据检查
                raise ValueError("Empty point cloud detected")

            # 转换为Tensor并中心化
            gt_pcd_tensor = torch.tensor(np.array(gt_pcd.points),
                                         dtype=torch.float32,  # <--- 明确指定数据类型
                                         device=args.device).unsqueeze(0)
            gt_pcd_tensor -= gt_pcd_tensor.mean(dim=1, keepdim=True)

            pred_pcd_tensor = torch.tensor(np.array(pred_pcd.points),
                                           dtype=torch.float32,
                                           device=args.device).unsqueeze(0)
            pred_pcd_tensor -= pred_pcd_tensor.mean(dim=1, keepdim=True)

            # 计算CD
            chamfer_distance = CD(pred_pcd_tensor, gt_pcd_tensor)[0].item()

            if np.isnan(chamfer_distance):
                raise ValueError("CD calculation resulted in NaN")

            # 记录成功结果
            cd_value = chamfer_distance * 1000  # 转换为e-3单位
            status = "Success"

        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")
        finally:
            # 记录所有结果
            results["Filename"].append(file_name)
            results["CD (e-3)"].append(cd_value)
            results["Status"].append(status)  # <--- 新增状态列
    # 生成结果表格
    df = pd.DataFrame(results)  # <--- 使用pandas创建DataFrame

    # 保存Excel文件（带时间戳）
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    parts = gt_pcd_path.split('/')
    # 提取 train_airplane_0.1_sample_ge
    result = parts[-6]  # 倒数第五个元素

    excel_path = os.path.join("./results", f"{result}.xlsx")  # <--- 输出路径
    df.to_excel(excel_path, index=False)

    # 输出统计信息
    valid_cd = df[df["Status"] == "Success"]["CD (e-3)"]
    logger.info(f"Valid files: {len(valid_cd)}/{len(pred_pcd_list)}")
    logger.info("Average CD: {:.4f} e-3".format(valid_cd.mean()))
    logger.info(f"Results saved to {excel_path}")


if __name__ == "__main__":
    args = arg_parser()
    main(args)