from copy import deepcopy
import os

from config import cfg as base_cfg


cfg = deepcopy(base_cfg)

cfg.__name__ = "Flash-WAM RoboTwin: video 1 step, action 2 steps"

cfg.distill_mode = os.environ.get("DISTILL_MODE", "flashwam")
cfg.distill_video = True
cfg.distill_action = True
cfg.action_aware = True

cfg.num_ddim_timesteps = 1
cfg.num_ddim_timesteps_action = 2

cfg.action_distill_mode = "x0"
cfg.action_loss_weight = 1.0
cfg.action_aware_weight = 0.01

cfg.snr_shift = 5.0
cfg.action_snr_shift = 1.0
cfg.num_train_timesteps = 1000

cfg.ema_decay = 0.995
cfg.loss_type = "huber"
cfg.huber_c = 0.001
cfg.sigma_data = 0.5
cfg.cfg_min = 2.0
cfg.cfg_max = 10.0

cfg.learning_rate = 5e-6
cfg.beta1 = 0.9
cfg.beta2 = 0.999
cfg.weight_decay = 0.0
cfg.max_grad_norm = 2.0
cfg.warmup_steps = 100
cfg.max_train_steps = int(os.environ.get("MAX_TRAIN_STEPS", "10000"))

cfg.batch_size = 1
cfg.gradient_accumulation_steps = 8
cfg.load_worker = 0
cfg.noisy_cond_prob = 0.0
cfg.cfg_prob = 0.0

cfg.save_interval = int(os.environ.get("SAVE_INTERVAL", "500"))
cfg.gc_interval = 50
cfg.enable_wandb = os.environ.get("ENABLE_WANDB", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
cfg.wandb_entity = None
cfg.seed = 42
