# Flash-WAM + Action-only DMD-lite

This branch starts from the official Flash-WAM distillation code and adds a
disabled-by-default action distribution refinement.  The goal is to improve
RoboTwin success without changing Flash-WAM's one-step inference path.

## Motivation

Flash-WAM uses modality-aware consistency distillation:

1. video stream: Karras/LCM-style consistency for high-dimensional video latents;
2. action stream: linear x0 consistency for low-noise, precision-sensitive actions;
3. small action flow-matching regularizer.

This is stable, but it is still mainly a paired teacher-trajectory objective.
The new branch adds a lightweight action-only distribution correction so that
the one-step student action endpoint distribution is nudged toward the frozen
teacher action distribution.

## What Is Added

The extension is enabled only with:

```bash
ACTION_DMD_ENABLE=1
```

New components:

1. `FakeActionScore`
   - A small action-only fake score / x0 predictor.
   - It estimates the current student action endpoint distribution.
   - It does not model video distribution.

2. Recent replay buffer
   - Stores detached student action endpoints, pooled video-cache statistics,
     and action masks.
   - Helps the fake score avoid lagging behind the moving student distribution.

3. Scheduler-consistent action DMD loss
   - Uses the native LingBot-VA / Flash-WAM action scheduler to add noise.
   - Uses the same action x0 conversion as Flash-WAM action consistency:
     `x0 = x_sigma - sigma * v`.
   - Detaches student video condition for the first version, preventing action
     DMD gradients from hacking the video cache.

## Student Loss

With action DMD enabled:

```text
L_student =
  L_video_CM
+ lambda_action * L_action_CM
+ lambda_fm * L_action_FM
+ lambda_endpoint * L_endpoint
+ warmup(step) * lambda_dmd * L_action_DMD
```

`L_video_CM` and `L_action_CM` are the original Flash-WAM losses.

`L_endpoint` anchors the executable action endpoint to the dataset action.  This
prevents the distribution term from making actions look plausible but imprecise.

`L_action_DMD` is a pseudo-target loss:

```text
A_sigma = action_scheduler.add_noise(A_student, epsilon, sigma)
A_teacher_x0 = teacher_action_x0(A_sigma, sigma | stopgrad(student_video), C)
A_fake_x0 = fake_action_score(A_sigma, sigma | stopgrad(student_video), C)
g = normalize_sigma(mask * (A_fake_x0 - A_teacher_x0))
A_pseudo = stopgrad(A_student - eta * g)
L_action_DMD = || mask * (A_student - A_pseudo) ||^2
```

## Fake Score Loss

The fake score is trained on detached student action endpoints:

```text
A_sigma = action_scheduler.add_noise(stopgrad(A_student), epsilon, sigma)
A_fake_x0 = fake_action_score(A_sigma, sigma | stopgrad(student_video), C)
L_fake = Huber(mask * A_fake_x0, mask * stopgrad(A_student))
```

The fake score is updated multiple times per student step:

```bash
FAKE_ACTION_UPDATES=2
```

## Suggested First Run

Start from an already working Flash-WAM checkpoint:

```bash
ACTION_DMD_ENABLE=1 \
ACTION_DMD_WEIGHT=5e-4 \
ACTION_DMD_WARMUP_STEPS=1000 \
ACTION_DMD_SIGMA_MIN=0.05 \
ACTION_DMD_SIGMA_MAX=0.30 \
ACTION_ENDPOINT_WEIGHT=0.05 \
FAKE_ACTION_UPDATES=2 \
python distillation/train.py ...
```

Recommended ablations:

1. official Flash-WAM baseline;
2. `+ L_endpoint` only;
3. `+ fake score` with `ACTION_DMD_WEIGHT=0`;
4. `+ normalized action DMD`;
5. compare `ACTION_DMD_SIGMA_MIN=0.05` vs `0.02`;
6. compare `FAKE_ACTION_UPDATES=1/2/4`.

The first success criterion is not action MSE alone.  Check RoboTwin task
success, contact/gripper phases, DMD gradient norm by sigma, and fake-score
loss stability.
