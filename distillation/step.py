"""Consistency / loss training step (StepMixin)."""
import math

import torch
import torch.nn.functional as F
from einops import rearrange

from utils import data_seq_to_patch, logger
from consistency import scalings_for_boundary_conditions


class StepMixin:
    # ==================================================================
    # Extract video v-prediction from model output -> [B, C, F, H, W]
    # ==================================================================
    def _extract_video_v(self, video_pred, ref_shape, batch_size):
        return data_seq_to_patch(
            self.patch_size, video_pred,
            ref_shape[-3], ref_shape[-2], ref_shape[-1],
            batch_size=batch_size,
        )

    # ==================================================================
    # Extract action v-prediction from model output -> [B, C, F, N, 1]
    # ==================================================================
    def _extract_action_v(self, action_pred, num_frames):
        return rearrange(action_pred, 'b (f n) c -> b c f n 1', f=num_frames)

    # ==================================================================
    # Consistency function: f(x_t, t) = c_skip * x_t + c_out * pred_x0
    # ==================================================================
    def _consistency_function(self, v_pred, noisy_latent, sigma, sigma_data=None):
        """
        pred_x0 = x_t - sigma * v    (FlowMatch inversion)
        f(x_t, t) = c_skip * x_t + c_out * pred_x0
        """
        if sigma_data is None:
            sigma_data = self.config.sigma_data
        sigma_5d = sigma[None, None, :, None, None].to(v_pred.dtype).to(v_pred.device)
        c_skip, c_out = scalings_for_boundary_conditions(
            sigma_5d, sigma_data=sigma_data)
        pred_x0 = noisy_latent - sigma_5d * v_pred
        return c_skip * noisy_latent + c_out * pred_x0

    def _sigma_5d(self, sigma, like):
        return sigma[None, None, :, None, None].to(dtype=like.dtype, device=like.device)

    def _ids_to_sigma_timestep(self, scheduler, ids):
        ids = ids.to(self.device)
        sigmas = scheduler.sigmas.to(self.device)[ids]
        timesteps = scheduler.timesteps.to(self.device)[ids]
        return sigmas, timesteps

    def _end_ids_from_ratio(self, start_ids, base_k, ratio):
        stride = max(1, int(round(float(base_k) * float(ratio))))
        return (start_ids.to(self.device) + stride).clamp(
            max=self.config.num_train_timesteps - 1)

    def _action_flowmap_ratios_and_weights(self):
        ratios = list(getattr(self.config, "action_flowmap_stride_ratios", [0.5, 1.0]))
        weights = list(getattr(self.config, "action_flowmap_loss_weights", [0.5, 1.0]))
        if len(ratios) != len(weights):
            raise ValueError(
                "action_flowmap_stride_ratios and action_flowmap_loss_weights "
                "must have the same length")
        if not ratios:
            return [1.0], [1.0]
        total = sum(float(w) for w in weights)
        if total <= 0:
            weights = [1.0 for _ in ratios]
            total = float(len(weights))
        return ratios, [float(w) / total for w in weights]

    def _action_teacher_substeps(self, ratio, min_ratio):
        min_steps = max(1, int(getattr(self.config, "action_flowmap_teacher_min_substeps", 1)))
        max_steps = max(min_steps, int(getattr(self.config, "action_flowmap_teacher_max_substeps", 4)))
        if min_ratio <= 0:
            return min_steps
        scaled = int(math.ceil(float(ratio) / float(min_ratio)))
        return max(min_steps, min(max_steps, scaled))

    def _max_action_flowmap_stride(self):
        ratios, _ = self._action_flowmap_ratios_and_weights()
        return max(1, max(int(round(float(self.k_action) * float(r))) for r in ratios))

    def _resample_action_flowmap_source(self, input_dict):
        action_dict = input_dict['action_dict']
        clean_action = action_dict['latent']
        max_stride = self._max_action_flowmap_stride()
        max_start = int(self.config.num_train_timesteps) - 1 - max_stride
        if max_start < 0:
            raise ValueError(
                "action flow-map stride is longer than the training schedule; "
                "reduce ACTION_FLOWMAP_STRIDE_RATIOS or increase num_ddim_timesteps_action")

        B = clean_action.shape[0]
        num_frames = clean_action.shape[2]
        source_ids = torch.randint(
            0, max_start + 1, (num_frames,), device=self.device)
        sigma_start, timesteps_start = self._ids_to_sigma_timestep(
            self.train_scheduler_action, source_ids)

        noise = torch.randn_like(clean_action)
        noisy_action = (1.0 - self._sigma_5d(sigma_start, clean_action)) * clean_action + \
            self._sigma_5d(sigma_start, clean_action) * noise
        targets = noise - clean_action

        action_mask = action_dict.get('actions_mask')
        if action_mask is not None:
            mask = action_mask.float()
            noisy_action = noisy_action * mask
            targets = targets * mask

        action_dict['noisy_latents'] = noisy_action
        action_dict['targets'] = targets
        action_dict['timesteps'] = timesteps_start[None].repeat(B, 1)
        action_dict['cond_timesteps'] = torch.zeros_like(action_dict['timesteps'])
        return source_ids, torch.tensor(0.0, device=self.device)

    def _huber_or_l2_video(self, pred, target):
        if self.config.loss_type == "huber":
            c = self.config.huber_c
            diff = pred.float() - target.detach().float()
            return torch.mean(torch.sqrt(diff ** 2 + c ** 2) - c)
        return F.mse_loss(pred.float(), target.detach().float())

    def _huber_or_l2_action(self, pred, target, mask):
        mask = mask.float()
        diff = (pred.float() * mask) - (target.detach().float() * mask)
        if self.config.loss_type == "huber":
            c = self.config.huber_c
            return (torch.sqrt(diff ** 2 + c ** 2) - c).sum() / \
                mask.sum().clamp(min=1)
        return (diff ** 2).sum() / mask.sum().clamp(min=1)

    def _action_flow_input(
        self,
        base_input_dict,
        video_cache,
        action_latents,
        action_timesteps,
        action_target_timesteps=None,
    ):
        latent_zero_ts = torch.zeros_like(base_input_dict['latent_dict']['timesteps'])
        action_dict = {
            **base_input_dict['action_dict'],
            'noisy_latents': action_latents,
            'timesteps': action_timesteps,
            'cond_timesteps': base_input_dict['action_dict']['cond_timesteps'],
        }
        if action_target_timesteps is not None:
            action_dict['action_target_timesteps'] = action_target_timesteps
        return {
            'latent_dict': {
                **base_input_dict['latent_dict'],
                'noisy_latents': video_cache,
                'latent': video_cache,
                'timesteps': latent_zero_ts,
                'cond_timesteps': latent_zero_ts,
            },
            'action_dict': action_dict,
            'chunk_size': base_input_dict['chunk_size'],
            'window_size': base_input_dict['window_size'],
        }

    @torch.no_grad()
    def _teacher_action_rollout(
        self,
        base_input_dict,
        video_cache,
        action_start,
        start_ids,
        end_ids,
        substeps,
    ):
        B = action_start.shape[0]
        num_frames = action_start.shape[2]
        current = action_start.detach()
        start_ids = start_ids.to(self.device)
        end_ids = end_ids.to(self.device)
        last_ids = start_ids

        for step_idx in range(substeps):
            alpha = float(step_idx + 1) / float(substeps)
            next_ids = torch.round(
                start_ids.float() + (end_ids.float() - start_ids.float()) * alpha
            ).long().clamp(max=self.config.num_train_timesteps - 1)
            next_ids = torch.maximum(next_ids, last_ids)

            sigma_cur, timestep_cur = self._ids_to_sigma_timestep(
                self.train_scheduler_action, last_ids)
            sigma_next, _ = self._ids_to_sigma_timestep(
                self.train_scheduler_action, next_ids)
            teacher_input = self._action_flow_input(
                base_input_dict=base_input_dict,
                video_cache=video_cache,
                action_latents=current,
                action_timesteps=timestep_cur[None].repeat(B, 1),
            )
            _, teacher_action_v_seq = self.teacher(teacher_input, train_mode=True)
            teacher_action_v = self._extract_action_v(teacher_action_v_seq, num_frames)
            current = current + teacher_action_v * (
                self._sigma_5d(sigma_next, teacher_action_v) -
                self._sigma_5d(sigma_cur, teacher_action_v)
            )
            last_ids = next_ids

        return current

    def _action_flowmap_losses(
        self,
        base_input_dict,
        student_video_cache,
        action_ts_ids,
        sigma_start_action,
        actions_mask,
    ):
        B = base_input_dict['action_dict']['noisy_latents'].shape[0]
        num_frames = base_input_dict['action_dict']['noisy_latents'].shape[2]
        action_start = base_input_dict['action_dict']['noisy_latents']
        action_timesteps = base_input_dict['action_dict']['timesteps']
        mask = actions_mask.float()

        ratios, weights = self._action_flowmap_ratios_and_weights()
        min_ratio = min(ratios)
        flowmap_loss = torch.tensor(0.0, device=self.device)
        endpoint_loss = torch.tensor(0.0, device=self.device)
        self_consistency_loss = torch.tensor(0.0, device=self.device)
        preds = {}

        for ratio, weight in zip(ratios, weights):
            end_ids = self._end_ids_from_ratio(action_ts_ids, self.k_action, ratio)
            sigma_end_action, timesteps_end_action = self._ids_to_sigma_timestep(
                self.train_scheduler_action, end_ids)
            substeps = self._action_teacher_substeps(ratio, min_ratio)
            target_action = self._teacher_action_rollout(
                base_input_dict=base_input_dict,
                video_cache=student_video_cache.detach(),
                action_start=action_start,
                start_ids=action_ts_ids,
                end_ids=end_ids,
                substeps=substeps,
            )

            student_input = self._action_flow_input(
                base_input_dict=base_input_dict,
                video_cache=student_video_cache.detach(),
                action_latents=action_start,
                action_timesteps=action_timesteps,
                action_target_timesteps=timesteps_end_action[None].repeat(B, 1),
            )
            _, student_action_v_seq = self.student(student_input, train_mode=True)
            student_action_v = self._extract_action_v(student_action_v_seq, num_frames)
            pred_action = action_start + student_action_v * (
                self._sigma_5d(sigma_end_action, student_action_v) -
                self._sigma_5d(sigma_start_action, student_action_v)
            )
            flowmap_loss = flowmap_loss + float(weight) * self._huber_or_l2_action(
                pred_action, target_action, mask)
            preds[float(ratio)] = {
                "pred": pred_action,
                "sigma": sigma_end_action,
                "timesteps": timesteps_end_action,
            }

        endpoint_weight = float(getattr(self.config, "action_flowmap_endpoint_weight", 0.0))
        if endpoint_weight > 0:
            zero_timesteps = torch.zeros_like(action_timesteps)
            endpoint_input = self._action_flow_input(
                base_input_dict=base_input_dict,
                video_cache=student_video_cache.detach(),
                action_latents=action_start,
                action_timesteps=action_timesteps,
                action_target_timesteps=zero_timesteps,
            )
            _, endpoint_action_v_seq = self.student(endpoint_input, train_mode=True)
            endpoint_action_v = self._extract_action_v(endpoint_action_v_seq, num_frames)
            endpoint_pred = action_start - self._sigma_5d(
                sigma_start_action, endpoint_action_v) * endpoint_action_v
            endpoint_loss = self._huber_or_l2_action(
                endpoint_pred,
                base_input_dict['action_dict']['latent'].detach(),
                mask,
            )

        sc_weight = float(getattr(self.config, "action_flowmap_self_consistency_weight", 0.0))
        if sc_weight > 0 and len(preds) >= 2:
            sorted_ratios = sorted(preds)
            mid_ratio = sorted_ratios[0]
            long_ratio = sorted_ratios[-1]
            mid = preds[mid_ratio]
            long = preds[long_ratio]
            mid_source = mid["pred"].detach()
            mid_input = self._action_flow_input(
                base_input_dict=base_input_dict,
                video_cache=student_video_cache.detach(),
                action_latents=mid_source,
                action_timesteps=mid["timesteps"][None].repeat(B, 1),
                action_target_timesteps=long["timesteps"][None].repeat(B, 1),
            )
            _, mid_action_v_seq = self.student(mid_input, train_mode=True)
            mid_action_v = self._extract_action_v(mid_action_v_seq, num_frames)
            via_mid = mid_source + mid_action_v * (
                self._sigma_5d(long["sigma"], mid_action_v) -
                self._sigma_5d(mid["sigma"], mid_action_v)
            )
            self_consistency_loss = self._huber_or_l2_action(
                via_mid, long["pred"].detach(), mask)

        return flowmap_loss, endpoint_loss, self_consistency_loss

    # ==================================================================
    # One training step
    # ==================================================================
    def _train_step(self, batch, batch_idx):
        batch = self.convert_input_format(batch)

        B = batch['latents'].shape[0]
        ref_shape = batch['latents'].shape     # [B, C, F, H, W]
        num_frames = ref_shape[2]
        actions_mask = batch.get('actions_mask')
        enable_action_flowmap = (
            self.distill_action and
            getattr(self.config, "enable_action_flowmap", False)
        )

        # ---- 1. Prepare input_dict (identical to native training) ----
        input_dict = self._prepare_input_dict(batch)
        action_flowmap_clip_rate = torch.tensor(0.0, device=self.device)
        if enable_action_flowmap:
            action_ts_ids, action_flowmap_clip_rate = \
                self._resample_action_flowmap_source(input_dict)

        # ---- 2. Compute sigma_start and sigma_end for LCM (video) ----
        video_timesteps = input_dict['latent_dict']['timesteps'][0]  # [F]
        sched_ts = self.train_scheduler_latent.timesteps  # [1000]
        video_ts_ids = torch.argmin(
            (sched_ts[:, None] - video_timesteps.cpu()).abs(), dim=0)  # [F]

        sigma_start = self.train_scheduler_latent.sigmas[video_ts_ids].to(self.device)
        end_ids = (video_ts_ids + self.k).clamp(max=self.config.num_train_timesteps - 1)
        sigma_end = self.train_scheduler_latent.sigmas[end_ids].to(self.device)
        timesteps_end = self.train_scheduler_latent.timesteps[end_ids].to(self.device)

        # ---- 2b. Compute sigma pairs for actions ----
        if self.distill_action:
            if not enable_action_flowmap:
                action_timesteps = input_dict['action_dict']['timesteps'][0]  # [F]
                sched_ts_a = self.train_scheduler_action.timesteps
                action_ts_ids = torch.argmin(
                    (sched_ts_a[:, None] - action_timesteps.cpu()).abs(), dim=0)

            sigma_start_action, _ = self._ids_to_sigma_timestep(
                self.train_scheduler_action, action_ts_ids)
            end_ids_action = (action_ts_ids + self.k_action).clamp(
                max=self.config.num_train_timesteps - 1)
            sigma_end_action, timesteps_end_action = self._ids_to_sigma_timestep(
                self.train_scheduler_action, end_ids_action)

        # ---- 3. Teacher CFG Euler step ----
        cfg_scale = self.config.cfg_min + torch.rand(1).item() * (
            self.config.cfg_max - self.config.cfg_min)

        with torch.no_grad():
            # Conditioned forward
            video_v_cond, action_v_cond = self.teacher(input_dict, train_mode=True)

            # Unconditioned forward (replace text_emb with empty)
            B_emb = input_dict['latent_dict']['text_emb'].shape[0]
            empty_emb = self.empty_emb.expand(B_emb, -1, -1)
            input_dict_uncond = {
                'latent_dict': {**input_dict['latent_dict'], 'text_emb': empty_emb},
                'action_dict': {**input_dict['action_dict'], 'text_emb': empty_emb},
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
            video_v_uncond, _ = self.teacher(input_dict_uncond, train_mode=True)

            # CFG combination (video only — action_guidance_scale=1, no CFG)
            video_v_cfg = video_v_uncond + cfg_scale * (video_v_cond - video_v_uncond)

            # Video Euler step -> x_prev
            video_v_cfg_5d = self._extract_video_v(video_v_cfg, ref_shape, B)
            sigma_s = sigma_start[None, None, :, None, None].to(video_v_cfg_5d)
            sigma_e = sigma_end[None, None, :, None, None].to(video_v_cfg_5d)
            x_prev = input_dict['latent_dict']['noisy_latents'] + \
                     video_v_cfg_5d * (sigma_e - sigma_s)

            # Action Euler step -> x_prev_action for the original Flash-WAM path.
            if self.distill_action and not enable_action_flowmap:
                action_v_5d = self._extract_action_v(action_v_cond, num_frames)
                sigma_s_a = sigma_start_action[None, None, :, None, None].to(action_v_5d)
                sigma_e_a = sigma_end_action[None, None, :, None, None].to(action_v_5d)
                x_prev_action = input_dict['action_dict']['noisy_latents'] + \
                                action_v_5d * (sigma_e_a - sigma_s_a)

        # ---- 4. Online student consistency prediction at sigma_start ----
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if not should_sync:
            self.student.set_requires_gradient_sync(False)
        else:
            self.student.set_requires_gradient_sync(True)

        student_video_v_seq, student_action_v_seq = self.student(input_dict, train_mode=True)

        # ---- 4a. Video consistency prediction ----
        if self.distill_video:
            student_video_v = self._extract_video_v(student_video_v_seq, ref_shape, B)
            student_video_pred = self._consistency_function(
                student_video_v,
                input_dict['latent_dict']['noisy_latents'],
                sigma_start,
            )

        # ---- 4b. Action prediction for the original Flash-WAM path ----
        if self.distill_action and not enable_action_flowmap:
            student_action_v = self._extract_action_v(student_action_v_seq, num_frames)
            if self.action_distill_mode == "x0":
                # Action consistency function: f = x_sigma - sigma * v
                sigma_s_a_5d = sigma_start_action[None, None, :, None, None].to(student_action_v)
                student_action_pred = input_dict['action_dict']['noisy_latents'] - \
                                      sigma_s_a_5d * student_action_v
            else:
                student_action_pred = self._consistency_function(
                    student_action_v,
                    input_dict['action_dict']['noisy_latents'],
                    sigma_start_action,
                )

        # ---- 5. Target student prediction at sigma_end on x_prev ----
        input_dict_end = {
            'latent_dict': {
                **input_dict['latent_dict'],
                'noisy_latents': x_prev.detach(),
                'timesteps': timesteps_end[None].repeat(B, 1),
            },
            'action_dict': {**input_dict['action_dict']},
            'chunk_size': input_dict['chunk_size'],
            'window_size': input_dict['window_size'],
        }
        if self.distill_action and not enable_action_flowmap:
            input_dict_end['action_dict'] = {
                **input_dict['action_dict'],
                'noisy_latents': x_prev_action.detach(),
                'timesteps': timesteps_end_action[None].repeat(B, 1),
            }

        with torch.no_grad():
            target_video_v_seq, target_action_v_seq = self.target_student(
                input_dict_end, train_mode=True)

            if self.distill_video:
                target_video_v = self._extract_video_v(target_video_v_seq, ref_shape, B)
                target_video_pred = self._consistency_function(
                    target_video_v, x_prev, sigma_end,
                )

            if self.distill_action and not enable_action_flowmap:
                target_action_v = self._extract_action_v(target_action_v_seq, num_frames)
                if self.action_distill_mode == "x0":
                    sigma_e_a_5d = sigma_end_action[None, None, :, None, None].to(target_action_v)
                    target_action_pred = x_prev_action - sigma_e_a_5d * target_action_v
                else:
                    target_action_pred = self._consistency_function(
                        target_action_v, x_prev_action, sigma_end_action,
                    )

        # ---- 6. Loss ----
        video_loss = torch.tensor(0.0, device=self.device)
        if self.distill_video:
            video_loss = self._huber_or_l2_video(student_video_pred, target_video_pred)

        action_flowmap_loss = torch.tensor(0.0, device=self.device)
        action_endpoint_loss = torch.tensor(0.0, device=self.device)
        action_self_consistency_loss = torch.tensor(0.0, device=self.device)
        action_loss = torch.tensor(0.0, device=self.device)
        if self.distill_action:
            if enable_action_flowmap:
                student_video_cache = (
                    student_video_pred if self.distill_video
                    else input_dict['latent_dict']['latent']
                )
                action_flowmap_loss, action_endpoint_loss, action_self_consistency_loss = \
                    self._action_flowmap_losses(
                        input_dict,
                        student_video_cache,
                        action_ts_ids,
                        sigma_start_action,
                        actions_mask,
                    )
                action_loss = action_flowmap_loss
            else:
                action_loss = self._huber_or_l2_action(
                    student_action_pred,
                    target_action_pred,
                    actions_mask,
                )

        # ---- 6b. Action-aware regularizer (native flow matching MSE) ----
        action_aware_loss = torch.tensor(0.0, device=self.device)
        if self.action_aware:
            student_action_v = self._extract_action_v(student_action_v_seq, num_frames)
            action_targets = input_dict['action_dict']['targets']
            mask = actions_mask.float()
            aa_diff = (student_action_v.float() - action_targets.float().detach()) * mask
            action_aware_loss = (aa_diff ** 2).sum() / mask.sum().clamp(min=1)

        sc_weight = float(getattr(self.config, 'action_flowmap_self_consistency_weight', 0.0))
        sc_warmup_steps = int(getattr(
            self.config, 'action_flowmap_self_consistency_warmup_steps', 0))
        if sc_weight > 0.0 and sc_warmup_steps > 0:
            sc_weight *= min(1.0, float(self.step + 1) / float(sc_warmup_steps))

        loss = video_loss + self.config.action_loss_weight * action_loss \
               + getattr(self.config, 'action_aware_weight', 0.0) * action_aware_loss \
               + getattr(self.config, 'action_flowmap_endpoint_weight', 0.0) * action_endpoint_loss \
               + sc_weight * action_self_consistency_loss

        loss = loss / self.gradient_accumulation_steps

        if not torch.isfinite(loss):
            if self.config.rank == 0:
                logger.warning(f"[step {self.step}] NaN/Inf loss, skipping")
            loss = torch.zeros(1, device=self.device, requires_grad=True)

        loss.backward()

        return {
            "loss": loss.detach(),
            "video_loss": video_loss.detach(),
            "action_loss": action_loss.detach() if self.distill_action else action_loss,
            "action_aware_loss": action_aware_loss.detach() if self.action_aware else action_aware_loss,
            "action_flowmap_loss": action_flowmap_loss.detach(),
            "action_endpoint_loss": action_endpoint_loss.detach(),
            "action_self_consistency_loss": action_self_consistency_loss.detach(),
            "action_flowmap_clip_rate": action_flowmap_clip_rate.detach(),
            "should_sync": should_sync,
        }
