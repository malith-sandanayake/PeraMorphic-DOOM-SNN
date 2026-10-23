# TODO: DOOM SNN Implementation Plan

## Phase 1: Make SNN Training Functional (P0)
- [x] 1.1 Fix `compute_ppo_loss` — replace `torch.log(policy_rates)` with proper softmax log-probs
- [x] 1.2 Fix `_forward_values` — thread `values_pred` from `train_epoch` through to `compute_ppo_loss`
- [x] 1.3 Implement SNN experience collection loop (replace `pass` with actual rollout)

## Phase 2: Make ANN Training Use DoomNet (P0)
- [x] 2.1 Wire DoomNet into SB3 `DoomPolicy` (override forward to use DoomNet)
- [x] 2.2 Apply YAML configs in ANN training (pass loaded config to `train()`)
- [x] 2.3 Fix EvaluationCallback save path (use `model.env.metadata` instead of `self.locals`)

## Phase 3: Fix SNN Evaluation (P1)
- [x] 3.1 Use correct T value in SNN evaluation (expand frame to `(1, T, C, H, W)`)

## Phase 4: Infrastructure (P2)
- [x] 4.1 Create `tests/` directory with unit tests (ann, snn, reward, preprocess)
- [x] 4.2 Fix video recording (add cv2.VideoWriter)
- [x] 4.3 Add `--learning-rate` CLI argument
- [x] 4.4 Fix ANN→SNN conversion to support DoomNet-wrapped policies (`_extract_architecture`)

## Phase 5: FPGA Export (P3 — Future)
- [ ] Deferred until training pipeline is functional
