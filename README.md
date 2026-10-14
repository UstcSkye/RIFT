# RIFT

RIFT is an efficient, privacy-preserving federated transfer framework for traffic forecasting with limited target observations. This repository contains the main **2-bit upload / 4-bit broadcast** configuration: predictive rotated source compression, target-conditioned aggregation, and affine model-increment broadcasts.

## Environment

Python 3.10+ and a CUDA-enabled PyTorch build are recommended. Install PyTorch for your GPU, then run:

```bash
pip install -r requirements.txt
```

The default device is `cuda:0`. CPU execution is available explicitly with `--device cpu`. The implementation runs the server and city clients in one process and exchanges serialized messages between them.

## Data

Data files are not included. Use the prepared speed datasets from [ST-GFSL](https://github.com/RobinLu1209/ST-GFSL#data) and the 2019 regional traffic data from [LargeST](https://github.com/liuxu77/LargeST). Arrange the processed arrays as follows:

```text
data/traffic/
  metr-la/   dataset.npy, matrix.npy
  pems-bay/  dataset.npy, matrix.npy
  chengdu/   dataset.npy, matrix.npy
  shenzhen/  dataset.npy, matrix.npy
  sd/        dataset.npy, matrix.npy
  oc/        dataset.npy, matrix.npy
  la/        dataset.npy, matrix.npy
  gba/       dataset.npy, matrix.npy
```

`dataset.npy` has shape `[time, nodes, features]` in chronological order, with unnormalized speed or flow in channel 0. Speed datasets also provide raw time-of-day phase in `[0,1)` in channel 1. For LargeST, phase is constructed from the sample index. `matrix.npy` is a nonnegative `[nodes, nodes]` adjacency matrix. Preserve the prepared sensor ordering and time alignment. The legacy LargeST folder name `orange` is accepted for `oc`. Raw LargeST HDF5 files must first be converted to these arrays; they cannot be renamed to `.npy`.

There are two independent four-city groups. A target uses the other three cities in its group as sources. Chengdu and Shenzhen have 144 observations per day; all other cities have 288. Inputs contain 12 historical steps and forecasts contain 6 steps.

**Evaluation uses the held-out test interval by default.** Speed datasets use `[floor(0.8T), T)`. LargeST uses `[84096, 105120)` of the prepared 2019 sequence. Windows are fully contained in the evaluation interval and advance by one step. MAE and RMSE are computed in original units at horizons 1, 3, and 6; their arithmetic means are also reported.

Target training uses the first three days, with normalization fitted only on this support interval. Speed source records come from complete days in the first 80% of each source sequence. LargeST source records use `[0, 73440)`, comprising 255 complete days. Evaluation observations are never used to fit target normalization.

## Run

Train and evaluate one target:

```bash
python run.py train --target chengdu --data-root data/traffic --output outputs/chengdu_seed0
python run.py train --target sd --data-root data/traffic --output outputs/sd_seed0
```

Change the target, seed, device, or output location as needed:

```bash
python run.py train --target pems-bay --seed 1 --device cuda:0 --data-root data/traffic --output outputs/pems_bay_seed1 --upload-bits 2 --download-bits 4
```

Supported targets are `chengdu`, `shenzhen`, `metr-la`, `pems-bay`, `sd`, `oc`, `la`, and `gba`. Run each seed in a new output directory. Existing output directories are not overwritten.

Defaults are 10 source-participating rounds followed by 10 target-only rounds, 4 private source optimizer steps per participating round, 2 target MSE epochs per round, and 50 final target MAE epochs. Target batch size is 32. Source Poisson sampling uses an expected batch of 128 node/day records, each containing 16 within-day windows. The joint clipping bound is 0.25 and the source noise multiplier is 0.8261. Final adaptation uses a fresh Adam optimizer with learning rates 0.002 for the decoder and graph residual and 0.001 for the temporal base.

Use `--sigma 0` for the noise-off control. It retains projection, joint clipping, the optimizer pathway, and private source-normalization releases, but adds no Gaussian noise to source gradients. Its `results.json` marks `source_privacy.mode` as `noise_off`, sets `training_epsilon` to `null`, and reports `no_finite_training_dp_bound`; no finite privacy guarantee is claimed for the noise-off training process. Negative and nonfinite noise multipliers are rejected.

## Checkpoint reuse

Evaluate a saved target directly, without retraining:

```bash
python run.py evaluate --checkpoint outputs/chengdu_seed0/target.pt --data-root data/traffic
```

Restart final adaptation from the saved pre-adaptation model:

```bash
python run.py adapt --checkpoint outputs/chengdu_seed0/before_adaptation.pt --data-root data/traffic --output outputs/chengdu_readapted
```

The adaptation command starts a fresh optimizer and shuffle stream; it is not an interrupted-federation resume. Checkpoints use this release's `rift_target_v1` format and include target model state, normalization statistics, and configuration. Historical internal experiment bundles require conversion before use.

Each training run writes `config.json`, `before_adaptation.pt`, `target.pt`, and `results.json`. Results include per-horizon metrics, evaluation boundaries, and serialized communication bytes. Target checkpoints are local artifacts and are not federated payloads.

## Method details

- **Source privacy:** source training uses the temporal core and local calendar embeddings. Shared gradients are projected onto public subspaces and jointly clipped with calendar gradients before Gaussian perturbation. Source graph residuals are inactive. Source optimizer and calendar states remain local.
- **Target prediction:** the target trains the temporal core and its private sparse graph residual. The graph branch has hidden dimension 8 and at most 4 neighbors per node. Target updates do not use source privacy perturbation.
- **Predictive source uploads:** each sender and the server maintain separate decoded histories in the full shared space. History is projected into the current public basis, subtracted from the source update, and the residual is encoded using a signed block-Hadamard transform, a four-level codebook, packed 2-bit indices, and one FP32 scale per tensor. Both endpoints advance from decoded messages.
- **Aggregation:** cosine similarity between decoded source innovations and the decoded target update determines softmax weights. Exponential smoothing uses a previous-weight fraction of 0.25. The target contributes 0.9 of the aggregate during source-participating rounds; it contributes the full update in target-only rounds.
- **Broadcasts:** the server compresses the aggregated model increment relative to the preceding common decoded state using affine 4-bit quantization. Each tensor carries packed indices and FP32 minimum/scale values. The server and receiving clients commit the same decoded model. Initialization, public bases, and the final handoff remain FP32. This is the affine-Q4 configuration, not the experimental rotated-Q4 broadcast.

Communication is counted from actual serialized application messages, once per recipient. Source uploads, target uploads, basis exchange, model broadcasts, initialization, and the final handoff are distinguished in `results.json`. Network-protocol overhead is excluded; MB denotes 1,000,000 bytes.

Source normalization uses cached noisy mean/std releases in `data/source_statistics/`, configurable with `--stats-dir`. A missing cache is generated from bounded source records with replacement zCDP budget 0.1 per city (0.05 for each statistic). Reuse the same cache across runs on the same prepared data. Do not reuse a cache for another dataset merely because its dimensions match. Model seeds control initialization and public dropout; private sampling and Gaussian noise use independent secret randomness. Fresh runs therefore need not reproduce historical seed-specific values bit for bit. The source guarantee must account for preprocessing and all repeated training releases under a consistent adjacency definition; this code does not assign a fixed end-to-end epsilon from the noise multiplier alone. Targets are outside the source privacy guarantee.

## Code layout

```text
run.py                     training, adaptation, and evaluation entry point
rift/
  model.py                 target predictor and private graph residual
  temporal_model.py        shared temporal backbone
  data.py                  city records, windows, and source statistics
  privacy.py               private source optimizer
  record_gradients.py      node/day gradients
  protocol.py              public bases, joint clipping, and message framing
  rotation.py              signed block-Hadamard Q2 codec
  source_wire.py           packed source residual messages
  compression.py           predictive histories and affine Q4 broadcasts
  aggregation.py           target-conditioned source weighting
  engine.py                federated simulation and checkpoint handling
```

The temporal backbone follows our FedDEFT implementation. RIFT adds source privacy processing, predictive compression, innovation-based aggregation, and compressed model synchronization. The repository excludes datasets, model checkpoints, manuscript assets, and internal experiment logs.
