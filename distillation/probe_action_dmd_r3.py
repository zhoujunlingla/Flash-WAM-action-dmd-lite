"""Offline probes for action-DMD-lite.

This script answers the minimal question before expensive training:
does the fake score produce a direction different from plain teacher
distillation, and does its cheap video conditioning matter?
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wan_va"))

from patches import install_flash_attn_stub
install_flash_attn_stub()

from action_dmd import masked_huber, normalize_dmd_gradient, pooled_video_stats
from distributed.util import init_distributed
from trainer import FlashWAMDistiller
from utils import init_logger, logger


def _dist_mean(value, device):
    t = torch.tensor([float(value), 1.0], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / t[1].clamp(min=1)).item()


def _scheduler_sigma_for_timesteps(scheduler, timesteps, device, dtype):
    sched_t = scheduler.timesteps.to(device)
    sched_s = scheduler.sigmas.to(device)
    flat = timesteps.reshape(-1).to(sched_t.dtype)
    idx = torch.argmin((sched_t[:, None] - flat[None]).abs(), dim=0)
    return sched_s[idx].reshape(timesteps.shape).to(device=device, dtype=dtype)


@torch.no_grad()
def _student_endpoints(trainer, batch):
    batch = trainer.convert_input_format(batch)
    bsz = batch["latents"].shape[0]
    ref_shape = batch["latents"].shape
    num_frames = ref_shape[2]
    input_dict = trainer._prepare_input_dict(batch)

    video_ts = input_dict["latent_dict"]["timesteps"]
    sigma_video = _scheduler_sigma_for_timesteps(
        trainer.train_scheduler_latent, video_ts, trainer.device, torch.float32
    )[0]

    action_ts = input_dict["action_dict"]["timesteps"]
    sigma_action = _scheduler_sigma_for_timesteps(
        trainer.train_scheduler_action, action_ts, trainer.device, torch.float32
    )[0]

    video_v_seq, action_v_seq = trainer.student(input_dict, train_mode=True)
    video_v = trainer._extract_video_v(video_v_seq, ref_shape, bsz)
    video_x0 = trainer._consistency_function(
        video_v,
        input_dict["latent_dict"]["noisy_latents"],
        sigma_video.to(trainer.device),
    )

    action_v = trainer._extract_action_v(action_v_seq, num_frames)
    sigma_a_5d = sigma_action[None, None, :, None, None].to(action_v)
    action_x0 = input_dict["action_dict"]["noisy_latents"] - sigma_a_5d * action_v
    return input_dict, video_x0, action_x0, batch["actions_mask"].float()


@torch.no_grad()
def _teacher_endpoint(trainer, base_input_dict, video_x0, action_sigma, timesteps_dmd, sigma_dmd, num_frames):
    input_dict_dmd = {
        "latent_dict": {
            **base_input_dict["latent_dict"],
            "noisy_latents": video_x0.detach(),
            "timesteps": torch.zeros_like(base_input_dict["latent_dict"]["timesteps"]),
        },
        "action_dict": {
            **base_input_dict["action_dict"],
            "noisy_latents": action_sigma.detach(),
            "timesteps": timesteps_dmd,
        },
        "chunk_size": base_input_dict["chunk_size"],
        "window_size": base_input_dict["window_size"],
    }
    _, teacher_action_v_seq = trainer.teacher(input_dict_dmd, train_mode=True)
    teacher_action_v = trainer._extract_action_v(teacher_action_v_seq, num_frames)
    return trainer._action_x0_from_v(action_sigma, teacher_action_v, sigma_dmd)


def _coord_mean_update(state, action_x0, mask):
    if state["sum"] is None:
        state["sum"] = torch.zeros_like(action_x0[:1].float())
        state["count"] = torch.zeros_like(mask[:1].float())
    if tuple(state["sum"].shape[1:]) != tuple(action_x0.shape[1:]):
        return
    state["sum"] += (action_x0.float() * mask.float()).sum(dim=0, keepdim=True)
    state["count"] += mask.float().sum(dim=0, keepdim=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--fake-train-batches", type=int, default=int(os.environ.get("PROBE_FAKE_TRAIN_BATCHES", "32")))
    parser.add_argument("--heldout-batches", type=int, default=int(os.environ.get("PROBE_HELDOUT_BATCHES", "8")))
    parser.add_argument("--resume-from-path", default=os.environ.get("PROBE_RESUME_PATH"))
    parser.add_argument("--resume-from-step", type=int, default=None)
    parser.add_argument("--teacher-model-path", default=os.environ.get("TEACHER_PATH"))
    parser.add_argument("--dataset-path", default=os.environ.get("DATASET_PATH"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "/tmp/action_dmd_probe"))
    args = parser.parse_args()

    config_mod = os.environ.get("CONFIG_FILE", "config_robotwin_1v2a")
    config = importlib.import_module(config_mod).cfg

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    config.enable_wandb = False
    config.enable_action_dmd = True
    config.action_dmd_weight = 0.0
    config.action_endpoint_weight = 0.0
    config.fake_action_updates = int(os.environ.get("FAKE_ACTION_UPDATES", "1"))
    config.output_dir = args.output_dir
    if args.teacher_model_path:
        config.teacher_model_path = args.teacher_model_path
    if args.dataset_path:
        config.dataset_path = args.dataset_path
        config.empty_emb_path = os.path.join(args.dataset_path, "empty_emb.pt")
    if args.resume_from_path:
        config.resume_from_path = args.resume_from_path
    if args.resume_from_step is not None:
        config.resume_from_step = args.resume_from_step

    trainer = FlashWAMDistiller(config)
    trainer.student.eval()
    trainer.teacher.eval()
    trainer.fake_action_score.train()

    coord_state = {"sum": None, "count": None}
    stats_bank = []
    fake_train_losses = []

    for _ in range(args.fake_train_batches):
        batch = trainer._get_next_batch()
        _, video_x0, action_x0, mask = _student_endpoints(trainer, batch)
        video_stats = pooled_video_stats(video_x0.detach())
        _coord_mean_update(coord_state, action_x0.detach(), mask.detach())
        stats_bank.append(video_stats.detach().float().cpu())
        fake_loss = trainer._update_fake_action_score(action_x0.detach(), video_stats.detach(), mask.detach())
        fake_train_losses.append(float(fake_loss.detach().float().item()))

    if coord_state["sum"] is not None:
        coord_mean = coord_state["sum"] / coord_state["count"].clamp(min=1)
    else:
        coord_mean = None

    metrics = {
        "fake_loss": [],
        "const_loss": [],
        "copy_loss": [],
        "zero_stats_loss": [],
        "shuffle_stats_loss": [],
        "r3_cos_fake_vs_distill": [],
        "r3_cos_normed_vs_distill": [],
        "fake_train_loss": fake_train_losses,
    }

    trainer.fake_action_score.eval()
    for _ in range(args.heldout_batches):
        batch = trainer._get_next_batch()
        input_dict, video_x0, action_x0, mask = _student_endpoints(trainer, batch)
        bsz, _, num_frames, _, _ = action_x0.shape
        video_stats = pooled_video_stats(video_x0.detach())
        sigma_dmd, timesteps_dmd = trainer._sample_action_dmd_timesteps(bsz, num_frames)
        action_sigma = trainer._add_action_dmd_noise(action_x0, timesteps_dmd)

        teacher_x0 = _teacher_endpoint(
            trainer, input_dict, video_x0, action_sigma, timesteps_dmd, sigma_dmd, num_frames
        )
        fake_x0 = trainer.fake_action_score(action_sigma, sigma_dmd, video_stats)
        zero_x0 = trainer.fake_action_score(action_sigma, sigma_dmd, torch.zeros_like(video_stats))
        if stats_bank:
            bank = torch.cat(stats_bank, dim=0).to(device=trainer.device, dtype=torch.float32)
            pick = torch.randint(0, bank.shape[0], (video_stats.shape[0],), device=trainer.device)
            shuffle_stats = bank[pick]
        else:
            shuffle_stats = torch.zeros_like(video_stats)
        shuffle_x0 = trainer.fake_action_score(action_sigma, sigma_dmd, shuffle_stats)

        if coord_mean is not None and tuple(coord_mean.shape[1:]) == tuple(action_x0.shape[1:]):
            const = coord_mean.to(device=trainer.device, dtype=action_x0.dtype).expand_as(action_x0)
        else:
            const = torch.zeros_like(action_x0)

        fake_loss = masked_huber(fake_x0, action_x0, mask, config.huber_c)
        const_loss = masked_huber(const, action_x0, mask, config.huber_c)
        copy_loss = masked_huber(action_sigma, action_x0, mask, config.huber_c)
        zero_loss = masked_huber(zero_x0, action_x0, mask, config.huber_c)
        shuffle_loss = masked_huber(shuffle_x0, action_x0, mask, config.huber_c)

        d_fake = ((fake_x0.float() - teacher_x0.float()) * mask.float()).flatten(1)
        d_distill = ((action_x0.float() - teacher_x0.float()) * mask.float()).flatten(1)
        normed = normalize_dmd_gradient(
            fake_x0.float() - teacher_x0.float(), mask.float(), sigma_dmd,
            min_scale=config.action_dmd_grad_min_scale,
        )
        d_normed = (normed.float() * mask.float()).flatten(1)

        metrics["fake_loss"].append(float(fake_loss.item()))
        metrics["const_loss"].append(float(const_loss.item()))
        metrics["copy_loss"].append(float(copy_loss.item()))
        metrics["zero_stats_loss"].append(float(zero_loss.item()))
        metrics["shuffle_stats_loss"].append(float(shuffle_loss.item()))
        metrics["r3_cos_fake_vs_distill"].append(float(F.cosine_similarity(d_fake, d_distill, dim=1, eps=1e-8).mean().item()))
        metrics["r3_cos_normed_vs_distill"].append(float(F.cosine_similarity(d_normed, d_distill, dim=1, eps=1e-8).mean().item()))

    summary = {}
    for key, values in metrics.items():
        if not values:
            continue
        local_mean = sum(values) / len(values)
        summary[key + "_mean"] = _dist_mean(local_mean, trainer.device)

    fake = summary.get("fake_loss_mean")
    if fake is not None and fake > 0:
        summary["R1_fake_over_const"] = summary.get("fake_loss_mean", 0.0) / max(summary.get("const_loss_mean", 1e-12), 1e-12)
        summary["R1_fake_over_copy"] = summary.get("fake_loss_mean", 0.0) / max(summary.get("copy_loss_mean", 1e-12), 1e-12)
        summary["R2_zero_over_real"] = summary.get("zero_stats_loss_mean", 0.0) / max(fake, 1e-12)
        summary["R2_shuffle_over_real"] = summary.get("shuffle_stats_loss_mean", 0.0) / max(fake, 1e-12)

    if rank == 0:
        out = {
            "config": {
                "resume_from_path": args.resume_from_path,
                "fake_train_batches": args.fake_train_batches,
                "heldout_batches": args.heldout_batches,
                "action_dmd_sigma_min": config.action_dmd_sigma_min,
                "action_dmd_sigma_max": config.action_dmd_sigma_max,
            },
            "summary": summary,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        logger.info(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    init_logger()
    main()
