import datetime
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, List, Optional
from PIL import Image
import time
import logging
from pathlib import Path
import random
import numpy as np
from pvd import Model, generate_pvd_xyz, prepare_pvd_model
import trimesh
import warnings
from tqdm.auto import tqdm

import hydra
import torch
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import functional as TVF
from pytorch3d.structures import Pointclouds
from pytorch3d.io import IO
from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
from model import ConditionalPointCloudDiffusionModel_ge, get_model_ge
import training_utils
import diffusion_utils
from dataset import get_dataset
from model import get_model, ConditionalPointCloudDiffusionModel
from config.structured import ProjectConfig

torch.multiprocessing.set_sharing_strategy("file_system")


def prepare_prior_model(cfg: ProjectConfig):
    cfg.model.point_cloud_model = "mamba3d_ge"
    model = ConditionalPointCloudDiffusionModel_ge(**cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        print("ge model resume path:%s" % cfg.aux_run.prior_ckpt)
        resumed_param = torch.load(cfg.aux_run.prior_ckpt)
        try:
            model.load_state_dict(resumed_param["model"])
        except:
            model.load_state_dict(resumed_param["model_state"])
    return model


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: ProjectConfig):
    # Accelerator
    accelerator = Accelerator(
        mixed_precision=cfg.run.mixed_precision,
        cpu=cfg.run.cpu,
        gradient_accumulation_steps=cfg.optimizer.gradient_accumulation_steps,
    )

    # Logging
    training_utils.setup_distributed_print(accelerator.is_main_process)
    if cfg.logging.wandb and accelerator.is_main_process:
        wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.run.name,
            job_type=cfg.run.job,
            config=OmegaConf.to_container(cfg),
        )
        wandb.run.log_code(
            root=hydra.utils.get_original_cwd(),
            include_fn=lambda p: any(
                p.endswith(ext)
                for ext in (".py", ".json", ".yaml", ".md", ".txt.", ".gin")
            ),
            exclude_fn=lambda p: any(
                s in p for s in ("output", "tmp", "wandb", ".git", ".vscode")
            ),
        )
        cfg: ProjectConfig = DictConfig(
            wandb.config.as_dict()
        )  # get the config back from wandb for hyperparameter sweeps

    # Configuration
    print(OmegaConf.to_yaml(cfg))
    print(f"Current working directory: {os.getcwd()}")

    # Set random seed
    training_utils.set_seed(cfg.run.seed)
    # Model
    model = get_model(cfg)
    print(f"Parameters (total): {sum(p.numel() for p in model.parameters()):_d}")
    print(
        f"Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}"
    )

    # Exponential moving average of model parameters
    if cfg.ema.use_ema:
        from torch_ema import ExponentialMovingAverage

        model_ema = ExponentialMovingAverage(model.parameters(), decay=cfg.ema.decay)
        model_ema.to(accelerator.device)
        print("Initialized model EMA")
    else:
        model_ema = None
        print("Not using model EMA")

    # Optimizer and scheduler
    optimizer = training_utils.get_optimizer(cfg, model, accelerator)
    scheduler = training_utils.get_scheduler(cfg, optimizer)

    # Resume from checkpoint and create the initial training state
    train_state: training_utils.TrainState = training_utils.resume_from_checkpoint(
        cfg, model, optimizer, scheduler, model_ema
    )

    # Datasets
    dataloader_train, dataloader_val, dataloader_vis = get_dataset(cfg)

    # Compute total training batch size
    total_batch_size = (
        cfg.dataloader.batch_size
        * accelerator.num_processes
        * accelerator.gradient_accumulation_steps
    )

    # Setup
    (
        model,
        optimizer,
        scheduler,
        dataloader_train,
        dataloader_val,
        dataloader_vis,
    ) = accelerator.prepare(
        model, optimizer, scheduler, dataloader_train, dataloader_val, dataloader_vis
    )

    if cfg.run.manual_seed:
        generator = torch.Generator().manual_seed(cfg.run.manual_seed)
    else:
        generator = None

    print("Building prior model...")
    # ge_path = cfg.aux_run.prior_ckpt
    # opt = {
    #     "model": ge_path,
    #     "nc": 3,
    #     "embed_dim": 64,
    #     "attention": True,
    #     "dropout": 0.1,
    # }
    prior_model = prepare_prior_model(cfg)
    print("Built prior model!")

    # Type hints
    model: ConditionalPointCloudDiffusionModel
    optimizer: torch.optim.Optimizer

    if cfg.run.job == "sample_bdm_blending":
        # Whether or not to use EMA parameters for sampling
        if cfg.run.sample_from_ema:
            assert model_ema is not None
            model_ema.to(accelerator.device)
            sample_context = model_ema.average_parameters
        else:
            sample_context = nullcontext
        # Sample
        with sample_context():
            sample_bdm_blending(
                cfg=cfg,
                model=model,
                prior_model=prior_model,
                dataloader=dataloader_val,
                accelerator=accelerator,
                generator=generator,
            )
        if cfg.logging.wandb and accelerator.is_main_process:
            wandb.finish()
        time.sleep(5)
        return

    else:
        raise ValueError(f"Invalid job: {cfg.run.job}")


@torch.no_grad()
def bdm_blending(
    accelerator: Accelerator,
    batch: FrameData or dict,
    cfg: ProjectConfig,
    model: ConditionalPointCloudDiffusionModel,
    prior_model: ConditionalPointCloudDiffusionModel_ge,
    generator: torch.Generator = None,
):
    """
    we fulfill our need to interact two diffusion model in this function
    the meaning of some variables should be clearified here

    1) status: this variable is set to be either 1 or 0, which means it is on the branch of prior or recon
    specially, 1 represents prior and 0 represents recon
    2) timelist: obviously
    3) timephase: a phase that specify the start and the end of interaction
    4) pred_pc: we recommend to use only one variable to transmit tensor of point cloud between two branches.
    """

    img = batch.image_rgb
    mask = batch.fg_probability
    camera = batch.camera

    roll_step = cfg.aux_run.roll_step  # 1
    milestones = cfg.aux_run.milestones  # [64,62,60,56,8,4,2,0]
    times = len(milestones) - 1

    if cfg.run.diffusion_scheduler == "ddim":
        prior_roll_step = int(roll_step * 16)  # 16
        prior_milestones = [
            int(i / 64 * 1000) for i in milestones
        ]  # [1000,968,936,872,128,64,32,0]
    else:
        assert cfg.run.diffusion_scheduler == "ddpm"
        prior_roll_step = roll_step
        prior_milestones = milestones

    B = img.shape[0]
    num_points = cfg.dataset.max_points
    device = model.point_cloud_model.device

    pred_pc = torch.randn(B, num_points, 3).to(device)
    pred_pc -= pred_pc.mean(dim=1, keepdim=True)

    # start
    for i in range(times):  # (0, 1, ..., 19)
        if i == 0:  # only sample recon
            print(
                "Go through recon model from {} to {}".format(
                    milestones[i], milestones[i + 1] - roll_step
                )
            )
            pred_pc = model.interaction_sample(
                pred_pc,
                camera,
                img,
                mask,
                return_sample_every_n_steps=10,
                scheduler=cfg.run.diffusion_scheduler,
                num_inference_steps=cfg.run.num_inference_steps,
                disable_tqdm=(not accelerator.is_main_process),
                start_time=milestones[i],  # 64
                end_time=milestones[i + 1] - roll_step,  # 61
            )  # (B, N, 3)
        elif i == times - 1:  # only sample recon
            print(
                "Go through recon model from {} to {}".format(
                    milestones[i] - roll_step, milestones[i + 1]
                )
            )
            pred_pc = model.interaction_sample(
                pred_pc,
                camera,
                img,
                mask,
                return_sample_every_n_steps=10,
                scheduler=cfg.run.diffusion_scheduler,
                num_inference_steps=cfg.run.num_inference_steps,
                disable_tqdm=(not accelerator.is_main_process),
                start_time=milestones[i] - roll_step,  # 1
                end_time=milestones[i + 1],  # 0
            )  # (B, N, 3)
        else:
            print(
                "Go through recon model from {} to {}".format(
                    milestones[i] - roll_step, milestones[i + 1]
                )
            )
            pred_pc = model.interaction_sample(
                pred_pc,
                camera,
                img,
                mask,
                return_sample_every_n_steps=10,
                scheduler=cfg.run.diffusion_scheduler,
                num_inference_steps=cfg.run.num_inference_steps,
                disable_tqdm=(not accelerator.is_main_process),
                start_time=milestones[i] - roll_step,
                end_time=milestones[i + 1],
            )  # (B, N, 3)

            print("Begin to combine outputs of recon model and prior model")
            print(
                "Go through Branch1: recon model from {} to {}".format(
                    milestones[i + 1], milestones[i + 1] - roll_step
                )
            )
            pred_pc_recon = pred_pc.clone()
            pred_out_recon = model.interaction_sample(
                pred_pc_recon,
                camera,
                img,
                mask,
                return_sample_every_n_steps=10,
                scheduler=cfg.run.diffusion_scheduler,
                num_inference_steps=cfg.run.num_inference_steps,
                disable_tqdm=(not accelerator.is_main_process),
                start_time=milestones[i + 1],
                end_time=milestones[i + 1] - roll_step,
            )  # (B, N, 3)
            # pred_out_recon = pred_out_recon.unsqueeze(1)  # (B, 1, N, 3)

            print(
                "Go through Branch2: prior model from {} to {}".format(
                    prior_milestones[i + 1], prior_milestones[i + 1] - prior_roll_step
                )
            )
            pred_pc_prior = pred_pc.clone()
            # pred_out_prior = prior(
            #     prior_model,
            #     pred_pc_prior,
            #     start_time=prior_milestones[i + 1],
            #     end_time=prior_milestones[i + 1] - prior_roll_step,
            # )  # (B, N, 3)
            pred_out_prior = prior_model.interaction_sample(
                pred_pc_prior,
                camera,
                img,
                mask,
                return_sample_every_n_steps=10,
                scheduler=cfg.run.diffusion_scheduler,
                num_inference_steps=cfg.run.num_inference_steps,
                disable_tqdm=(not accelerator.is_main_process),
                start_time=prior_milestones[i + 1],
                end_time=prior_milestones[i + 1] - prior_roll_step,
            )  # (B, N, 3)
            # pred_out_prior = pred_out_prior.unsqueeze(1)  # (B, 1, N, 3)

            print("Applying Dynamic Weighted Sampling Strategy")
            x_r = pred_out_recon
            x_g = pred_out_prior

            diff = x_g - x_r
            dist_sq = torch.sum(diff**2, dim=-1, keepdim=True)

            delta = cfg.fusion.fusion_delta

            weights = 1.0 / (1.0 + torch.exp(-delta * dist_sq))

            pred_pc = weights * x_g + (1.0 - weights) * x_r

            # # random choose which point cloud to go
            # print("Random choose branch to go")
            # pred_out = torch.cat(
            #     [pred_out_recon, pred_out_prior], dim=1
            # )  # (B, 2, N, 3)
            # pred_out = pred_out.permute(0, 2, 1, 3)  # (B, N, 2, 3)
            # indices = torch.randint(
            #     0,
            #     2,
            #     (
            #         pred_out.shape[0],
            #         pred_out.shape[1],
            #     ),
            #     generator=generator,
            # ).long()  # (B, N)
            # pred_pc = pred_out[
            #     torch.arange(pred_out.shape[0]).unsqueeze(1),
            #     torch.arange(pred_out.shape[1]).unsqueeze(0),
            #     indices,
            #     :,
            # ]  # (B, N, 3)

    output = Pointclouds(pred_pc)
    return output


@torch.no_grad()
def sample_bdm_blending(
    *,
    cfg: ProjectConfig,
    model: torch.nn.Module,
    prior_model: torch.nn.Module,
    dataloader: Iterable,
    accelerator: Accelerator,
    output_dir: str = "sample_bdm_blending",
    generator: torch.Generator = None,
):
    # Eval mode
    model.eval()
    progress_bar: Iterable[FrameData] = tqdm(
        dataloader, disable=(not accelerator.is_main_process)
    )

    # Output dir
    output_dir: Path = Path(output_dir)

    # Setup logs directory and logger
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True, parents=True)

    # Configure logging
    log_file = logs_dir / f"sampling_bdm_blending_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting BDM blending sampling process")
    logger.info(f"Output directory: {output_dir.absolute()}")
    if cfg.run.num_sample_batches is not None:
        logger.info(f"Number of sample batches: {cfg.run.num_sample_batches}")
    else:
        logger.info(f"Number of sample batches: all ({len(dataloader)} batches)")
    logger.info(f"Number of generated samples per input: {cfg.run.num_samples}")

    # PyTorch3D IO
    io = IO()

    # Initialize timing and memory tracking
    total_start_time = time.time()
    batch_times = []

    # Get initial GPU memory usage
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB

    total_input_samples_num = 0

    # Visualize
    for batch_idx, batch in enumerate(progress_bar):
        if isinstance(batch, dict):
            batch = FrameData(**batch)

        batch_start_time = time.time()
        batch_size = len(batch.image_rgb)
        total_input_samples_num += batch_size

        progress_bar.set_description(
            f"Processing batch {batch_idx:4d} / {len(dataloader):4d}"
        )

        if (
            cfg.run.num_sample_batches is not None
            and batch_idx >= cfg.run.num_sample_batches
        ):
            break

        batch_sample_times = []

        # Optionally produce multiple samples for each point cloud
        for sample_idx in range(cfg.run.num_samples):
            sample_start_time = time.time()

            # Filestring
            filename = (
                f"{{name}}-{sample_idx}.{{ext}}"
                if cfg.run.num_samples > 1
                else "{name}.{ext}"
            )
            filestr = str(output_dir / "{dir}" / "{category}" / filename)

            # BDM blending
            output = bdm_blending(
                accelerator=accelerator,
                batch=batch,
                cfg=cfg,
                model=model,
                prior_model=prior_model,
                generator=generator,
            )
            assert isinstance(output, Pointclouds)

            # Save individual samples
            for i in range(len(output)):
                frame_number = batch.frame_number[i]
                sequence_name = batch.sequence_name[i]
                sequence_category = batch.sequence_category[i]
                (output_dir / "gt" / sequence_category).mkdir(
                    exist_ok=True, parents=True
                )
                (output_dir / "pred" / sequence_category).mkdir(
                    exist_ok=True, parents=True
                )
                (output_dir / "images" / sequence_category).mkdir(
                    exist_ok=True, parents=True
                )

                pcd = batch.sequence_point_cloud[i]
                if not isinstance(pcd, Pointclouds):
                    pcd = Pointclouds(pcd[None])
                # Save ground truth
                io.save_pointcloud(
                    data=pcd,
                    path=filestr.format(
                        dir="gt",
                        category=sequence_category,
                        name=sequence_name,
                        ext="ply",
                    ),
                )

                # Save generation
                io.save_pointcloud(
                    data=output[i],
                    path=filestr.format(
                        dir="pred",
                        category=sequence_category,
                        name=sequence_name,
                        ext="ply",
                    ),
                )

                # Save input images
                filename = filestr.format(
                    dir="images",
                    category=sequence_category,
                    name=sequence_name,
                    ext="png",
                )
                TVF.to_pil_image(batch.image_rgb[i]).save(filename)

            # Log sample time
            sample_time = time.time() - sample_start_time
            batch_sample_times.append(sample_time)
            logger.info(f"Sample {sample_idx}: sample_time={sample_time:.2f}s")

        total_batch_sample_time = sum(batch_sample_times)
        avg_batch_sample_time = (
            total_batch_sample_time / len(batch_sample_times)
            if batch_sample_times
            else 0
        )

        # Calculate batch time and memory usage
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)

        # Log batch summary
        logger.info("------- Batch Time Summary -------")
        logger.info(f"Average total time per sample: {avg_batch_sample_time:.2f}s")
        logger.info(
            f"Batch {batch_idx} (size {batch_size}) completed in {batch_time:.2f}s"
        )

    # Calculate total statistics
    total_time = time.time() - total_start_time
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        max_memory_used = peak_memory - initial_memory
        logger.info(f"Initial GPU memory usage: {initial_memory:.2f} GB")
        logger.info(f"Peak GPU memory usage: {peak_memory:.2f} GB")
        logger.info(
            f"Maximum GPU memory used during sampling: {max_memory_used:.2f} GB"
        )

    # Log summary
    logger.info("=" * 50)
    logger.info("BDM BLENDING SAMPLING SUMMARY")
    logger.info("=" * 50)
    logger.info(
        f"Total samples generated: {total_input_samples_num * cfg.run.num_samples}"
    )
    logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f}min)")
    logger.info(f"Average batch time: {avg_batch_time:.2f}s")
    logger.info(
        f"Average sample time: {total_time/total_input_samples_num*cfg.run.num_samples:.2f}s"
    )
    logger.info(f"Log file saved to: {log_file.absolute()}")

    print("Saved samples to: ")
    print(output_dir.absolute())
    print(f"Logs saved to: {log_file.absolute()}")


if __name__ == "__main__":
    main()
