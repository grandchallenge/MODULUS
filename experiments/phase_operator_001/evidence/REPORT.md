# PHASE-OPERATOR-001 — L4 experimental report

**Disposition:** REVISE / CONDITIONAL CONTINUE, no protected merge or scaled performance claim.
**Platform:** NVIDIA L4, PyTorch 2.14.1+cu130, CUDA 13.0.

## Work completed

- Synthetic teacher: 30 training runs (3 seeds × 2 phase regimes × 5 variants, 1,000 steps), 30,000 optimizer steps.
- TinyStories: one 500-step pretrained backbone plus 15 adapter/full-finetuning runs (3 seeds × 5 methods × 200 steps).
- WikiText-2 transfer: 18 runs (3 seeds × 6 methods × 600 steps) from the same pretrained checkpoint.
- Centered phase versus bounded LoRA: 6 runs (3 × 2 × 600).
- LoRA norm frontier: 12 runs (3 × 4 × 600).
- In total 82 completed training runs and 55,100 optimizer steps, all on the L4, with no CPU fallback. Synthetic trials are NOT language-model training.
- Randomized CUDA verification: 120 spectral checks, 120 phase-smoothness checks, 120 factorization equivalences, 10 finite-difference comparisons, 60 causal prefix tests, 60 batch-isolation tests, 5 state round-trips.

## Synthetic held-out normalized MSE

Teacher family explicitly generates phase-modulated operators. Zero-output baseline = 1.0; smaller is better. 3 seeds each.

| Teacher scale | Student | Params | In-domain mean | Shifted-context mean |
|---|---|---:|---:|---:|
| 0.30 | joint | 1,801 | 0.511959 | 0.598874 |
| 0.30 | svd | 1,801 | 0.489854 | 0.528448 |
| 0.30 | oracle | 265 | 0.000000 | 0.000000 |
| 0.30 | real | 1,933 | 0.251240 | 0.402636 |
| 0.30 | lora | 1,536 | 0.086293 | 0.314286 |
| 1.50 | joint | 1,801 | 0.625535 | 0.639624 |
| 1.50 | svd | 1,801 | 0.744752 | 0.732411 |
| 1.50 | oracle | 265 | 0.000009 | 0.000009 |
| 1.50 | real | 1,933 | 0.648793 | 1.182266 |
| 1.50 | lora | 1,536 | 0.881717 | 1.163207 |

**Boundary:** oracle and snapshot-SVD methods use privileged teacher information. Strong-phase context shift reveals conditional family utility, not downstream generalization.

## TinyStories, held-out next-token validation

Official train revision: f54c09fd23315a6f9c86f9dc80f725de7d8f9c64; tokenizer revision 607a30d783dfa663caf39e06633721c8d4cfcd7e.
Train token SHA256: 6c69c8919752dfc3f0f2d42a5c7dc1bdd86d11baa7d11ff93e55b15aec83847c; validation token SHA256: d403b9e7419556fde8d4572aaf4bf90f05f621c6c555afe6a8f149ab9255647e.
Finite pools: 450,000 train and 50,000 validation tokens; evaluator uses 16,384 positions. One base seed, three adapter seeds.

| Method | Trainable scalars | Validation NLL | Mean perplexity |
|---|---:|---:|---:|
| pretrain | 5,053,920 | 4.68933 | 108.78 |
| lora | 3,072 | 4.68196 | 107.98 |
| real | 6,938 | 4.68370 | 108.17 |
| phase | 6,162 | 4.68520 | 108.33 |
| static | 4,626 | 4.68570 | 108.39 |
| full | 5,053,920 | 4.34355 | 76.99 |

## WikiText-2 domain transfer and TinyStories retention

WikiText revision: b08601e04326c79dfdd32d625aee71d232d685c3; training SHA256 4f8096b222abe0871eb97954c86eea5ae43ea82e2093a1d3af932ec00d33ef8e; validation SHA256 41031ef01333bba898e542366cd3d04e351e3dace84d032905ac0c538b33b9bc.
Backbone checkpoint SHA256: 30797abcceab2a01c3b7303454867acdec71433095cb90e78c6bd7f6828189c8.
Starting WikiText NLL 10.14547, starting TinyStories NLL 4.68933. Transfer uses 600 steps and 16,384 held-out token positions per domain.

| Method | Trainable scalars | WikiText NLL (mean) | WikiText perplexity (mean) | Original TinyStories NLL |
|---|---:|---:|---:|---:|
| lora | 3,072 | 9.01585 | 8232.82 | 5.34075 |
| lora16 | 6,144 | 8.99968 | 8100.58 | 5.42419 |
| real | 6,938 | 9.18944 | 9807.56 | 4.91417 |
| phase | 6,162 | 9.25271 | 10444.13 | 4.80380 |
| static | 4,626 | 9.29472 | 10889.09 | 4.86068 |
| full | 5,053,920 | 6.71095 | 821.38 | 6.86778 |

## Zero-centering and constrained LoRA

The centered phase update starts with zero adaptation and an immediately nonzero controller gradient. Bounded LoRA uses a differentiable spectral-norm clip inside its forward computation.

| Method | WikiText NLL | Original TinyStories NLL |
|---|---:|---:|
| centered | 9.18665 | 4.82006 |
| bounded_lora16 | 9.10787 | 5.00997 |
| LoRA16 spectral cap 0.10 | 9.30218 | 4.81611 |
| LoRA16 spectral cap 0.15 | 9.21461 | 4.89648 |
| LoRA16 spectral cap 0.20 | 9.16385 | 4.92983 |
| LoRA16 spectral cap 0.25 | 9.13045 | 4.98231 |

## Inference

- On the synthetic phase teacher, oracle framing is learnable; ordinary paired-basis identification remains the principal bottleneck.
- LoRA outruns and generally outperforms ordinary phase conditioning on real-language target loss.
- Phase/centered phase shows lower original-domain deterioration under transfer. A 0.15-capped LoRA control loses more original-domain performance than centered phase at marginally worse target loss, but the tested cap grid, initialization and optimizer settings are not a complete matched Pareto frontier.
- This is a constrained adaptation/retention research lead, **not a breakthrough in holographic compression**.

## Limitations and review gates

- One TinyStories base checkpoint and three adapter initialization/sampling seeds; no independent backbone-seed replication.
- Dataset pools are finite and windows repeatedly sampled; results are not full-corpus perplexities.
- Narrow width 96, depth 2, context 64; phase budgets, LoRA ranks and actual kernel costs differ.
- Spectral-cap LoRA uses full spectral-norm computation in its forward pass; throughput is not representative of an optimized implementation.
- The first GPU stress run failed due to precision/tolerance mismatch. Full-precision rerun passed with maximal dense error 1.19e-7; see the saved stress log and source.
- No trained-model checkpoint is admitted to this provider PR; exact replay requires pinned HF data and independent CUDA execution.
- Next gate: independent internal review of source, >=3 backbone seeds, stronger baseline tuning, exact norm-matched Pareto curves, long-context token evaluation, independent GPU replay, then CONTINUE/RETIRE.
