# PHASE-OPERATOR-001 — GPU evidence package

**Status:** GPU experimental evidence from GCL Colab NVIDIA L4, with protected repository review pending. No independent certification and no main-branch admission. See MODULUS issue #36 and draft PR #37.

## Exact hypothesis

A bounded, phase-conditioned operator may adapt a fixed causal Transformer with less original-domain forgetting than an equally budgeted simple adapter. Its matrix-free update is A(z) = g [Uc diag(cos(Cz+b)) + Us diag(sin(Cz+b))] V^T. With [Uc,Us] and V orthonormal, the fixed-context operator norm is at most |g|. This does not bound the full nonlinear Transformer Jacobian.

## Reproducible source

- substantial_synthetic.py: a phase-generated teacher and five student families, 3 seeds × 2 phase strengths × 5 variants × 1,000 steps = 30 GPU runs. Oracle-fixed bases and SVD over teacher operator snapshots receive privileged information and cannot count as ordinary deployment baselines.
- substantial_tinystories.py: pretrained 96-wide, 2-layer, 4-head causal GPT with context 64 and tied vocabulary embedding. Real HF TinyStories and GPT-2 tokenizer pinned by commit revisions. Base 500 steps; 5 adapter/full-finetuning variants × 3 seeds × 200 steps = 16 runs including the base.
- wikitext_transfer.py: one frozen pretrained base adapted to WikiText-2, six variants × three seeds × 600 steps = 18 runs. Measures target-domain WikiText validation AND original-domain TinyStories retention. Includes matched LoRA rank 16.
- followup_adapters.py: zero-centered phase initialization versus spectrally capped LoRA rank 16, 3 seeds each × 600 steps.
- frontier_sweep.py: 4 spectral-norm caps (0.10, 0.15, 0.20, 0.25) × 3 seeds × 600 steps; tests whether phase sits on the transfer–retention frontier.
- stress_tests.py: randomized geometry, numerical gradients, causal prefix, batch isolation, and state round-trip assertions on the L4.

The runners are standalone Python but use a shared Colab /content workspace. Their data and source revisions and token-pool SHA256 fingerprints are preserved in the evidence JSONs. Actual train/validation token pools are not committed. The base checkpoint is also not committed and must be recreated from the pinned source/data manifest.

## Budget and limits

TinyStories token pool: 450,000 train and 50,000 validation; WikiText-2: 650,000 train and 90,000 validation. Evaluation uses 16,384 deterministic token positions in each domain, rather than full-dataset perplexity. Train windows are sampled with replacement. Adapter seeds 42–44 share ONE pretrained base seed. Baselines are not matched for FLOPs or update norm except where explicitly stated. Training throughput is model/batch-specific and includes periodic validation.

The initial strict numerical-stress run failed with about 6e-5 absolute difference under accelerator matmul precision. Disabling reduced-precision FP32 matmul and rerunning with explicit mixed tolerances passed, with maximum observed difference about 1.2e-7. Preserve both facts in independent review.

## Claim boundary and decision

No full-network compression, production scaling, transformer-generalization gain, novel Fourier transform, or protected admission is claimed. Real-language target-domain results favor ordinary LoRA; bounded phase conditioning reduces original-domain degradation at the expense of transfer quality. Whether this tradeoff survives a norm-matched LoRA frontier remains the decisive question.

Repository governance: MODULUS is a provider, not a mathematical certification authority. No protected branch writes or claims of deployment are implied by this PR.
