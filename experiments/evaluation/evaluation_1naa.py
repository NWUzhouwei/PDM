import torch
import numpy as np
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from pprint import pprint
from metrics.PyTorchEMD.emd import earth_mover_distance as EMD
from metrics.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from metrics.ChamferDistancePytorch.fscore import fscore
from tqdm import tqdm
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

import ipdb

import warnings
warnings.filterwarnings("ignore")

cham3D = chamfer_3DDist()

# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


def EMD_CD(sample_pcs, ref_pcs, batch_size,  reduced=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    fs_lst = []
    iterator = range(0, N_sample, batch_size)

    for b_start in iterator:
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        dl, dr, _, _ = cham3D(sample_batch.cuda(), ref_batch.cuda())
        fs = fscore(dl, dr)[0].cpu()
        fs_lst.append(fs)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

        emd_batch = EMD(sample_batch.cuda(), ref_batch.cuda(), transpose=False)
        emd_lst.append(emd_batch)

    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)
    fs_lst = torch.cat(fs_lst).mean()
    results = {
        'MMD-CD': cd,
        'MMD-EMD': emd,
        'fscore': fs_lst
    }
    return results

def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, accelerated_cd=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(N_sample)
    for sample_b_start in tqdm(iterator):
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            dl, dr, _, _ = cham3D(sample_batch_exp.cuda().float(), ref_batch.cuda().float())
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1).detach().cpu())

            emd_batch = EMD(sample_batch_exp.cuda().float(), ref_batch.cuda().float(), transpose=False)
            emd_lst.append(emd_batch.view(1, -1).detach().cpu())

        cd_lst = torch.cat(cd_lst, dim=1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd


# Adapted from https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }


def compute_all_metrics(sample_pcs, ref_pcs, batch_size):
    results = {}

    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref_pcs, sample_pcs, batch_size)

    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({
        "%s-CD" % k: v for k, v in res_cd.items()
    })

    res_emd = lgan_mmd_cov(M_rs_emd.t())
    results.update({
        "%s-EMD" % k: v for k, v in res_emd.items()
    })

    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(ref_pcs, ref_pcs, batch_size)
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(sample_pcs, sample_pcs, batch_size)

    # 1-NN results
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update({
        "1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if 'acc' in k
    })
    one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
    results.update({
        "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
    })

    return results


#######################################################
# JSD : from https://github.com/optas/latent_3d_points
#######################################################
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds, as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False, verbose=False):
    """Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    """another way of computing JSD"""

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))




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

    logger = get_logger(args)
    set_seed(args.seed)

    pred_pcd_list = find_ply_files(args.pred_dir)
    logger.info("Evaluating on {} pointclouds".format(len(pred_pcd_list)))

    error_list = []
    cd_list = []
    ref=[]
    sample=[]
    results_list = []
    for pred_pcd_path in tqdm(pred_pcd_list):
        logger.debug("Processing {}".format(pred_pcd_path))

        file_name = pred_pcd_path.split("/")[-1]
        gt_pcd_path = os.path.join(args.gt_dir, file_name)
        if not os.path.exists(gt_pcd_path):
            logger.debug("GT file not found: {}".format(gt_pcd_path))
            error_list.append(pred_pcd_path)
            continue

        gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)
        gt_pcd_tensor = (
            torch.tensor(np.array(gt_pcd.points)).unsqueeze(0).to(args.device)
        )
        # gt_pcd_tensor = gt_pcd_tensor - gt_pcd_tensor.mean(dim=1, keepdim=True)
        ref.append(gt_pcd_tensor)
        pred_pcd = o3d.io.read_point_cloud(pred_pcd_path)
        pred_pcd_tensor = (
            torch.tensor(np.array(pred_pcd.points)
                         ).unsqueeze(0).to(args.device)
        )
        sample.append(pred_pcd_tensor)
    ref_pcs = torch.cat(ref, dim=0).contiguous()
    sample_pcs= torch.cat(sample, dim=0).contiguous()
    results = compute_all_metrics(sample_pcs, ref_pcs, 100)
    results = {k: (v.cpu().detach().item()
                   if not isinstance(v, float) else v) for k, v in results.items()}
    pprint(results)
        # pred_pcd_tensor = pred_pcd_tensor - \
        #     pred_pcd_tensor.mean(dim=1, keepdim=True)
        # metrics_results = compute_all_metrics(pred_pcd_tensor, gt_pcd_tensor, batch_size=1)
        # results = {k: (v.cpu().detach().item()
        #                if not isinstance(v, float) else v) for k, v in metrics_results.items()}
        # pprint(results)
        # results_list.append(metrics_results)

        # chamfer_distance_numpy = CD(pred_pcd_tensor, gt_pcd_tensor)[
        #     0].cpu().numpy()
        # if np.isnan(chamfer_distance_numpy):
        #     error_list.append(pred_pcd_path)
        #     continue
        # else:
        #     chamfer_distance = float(chamfer_distance_numpy) * 1000
        #     cd_list.append(chamfer_distance)
        # logger.debug(
        #     "CD: {} e-3, Mean CD: {} e-3".format(
        #         chamfer_distance, np.mean(cd_list))
        # )


    pprint(results)
    # logger.info("Mean list: {} e-3".format(np.mean(results_list)))
    # logger.info("Mean CD: {} e-3".format(np.mean(cd_list)))
    # logger.info("Error list: {}".format(error_list))

if __name__ == "__main__":
    args = arg_parser()
    main(args)
