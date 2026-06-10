# Project Context: TW-PLD with LightGCL

This document explains the complete conceptual background of the model. Read this before reading the equations or writing code. The goal is for anyone joining the project to understand not just *what* the model does, but *why* each design choice was made.

---

## The Three Problems We're Solving

Modern recommendation systems on platforms like Spotify, YouTube, and Amazon rely on **implicit feedback** — clicks, plays, views — rather than explicit ratings. This is cheap to collect but introduces three serious problems that current methods handle poorly:

### Problem 1: Noise in implicit feedback

Not every click means "I like this." Users tap items by accident, click out of curiosity, or are exposed to items in high-visibility positions. A naive model treats every interaction as a real preference signal, which causes it to recommend more of the wrong things.

### Problem 2: Cold-start sparsity

New users have very little interaction history. Collaborative filtering methods that learn from user behaviour have almost nothing to work with. Graph-based methods like LightGCN improve this somewhat by drawing on global structure, but they still assume that whatever interactions a user does have are clean signals.

### Problem 3: Preference drift over time

User preferences change. A user who bought hobbyist photography gear two years ago might now buy professional equipment. A model that treats every recorded interaction as equally valid will continue recommending based on outdated patterns.

---

## The Three Tools We're Combining

Each problem has an existing solution in the literature. None of them, however, address all three together.

### Tool 1: LightGCL (the backbone)

LightGCL learns user and item embeddings by maintaining two views of the user-item graph:
- **Local view**: propagation through the actual interaction graph
- **Global view**: propagation through an SVD-augmented graph that captures broad collaborative patterns

A contrastive learning objective forces these two views to agree. This makes embeddings robust to sparsity because the global view fills in patterns the local view can't see.

**Strength**: handles cold-start well.
**Weakness**: doesn't filter noise — happily aligns local and global views even on accidental clicks.

### Tool 2: PLD (Personalized Loss Distribution)

PLD identifies and suppresses noisy interactions per user. For each user, it computes the mean and standard deviation of training losses across their interactions, and flags any interaction whose loss is outside a typical range.

**Strength**: tailored denoising per user rather than a global threshold.
**Weakness**: 
1. Per-user statistics are unstable when the user has few interactions (fails on cold-start).
2. Treats all interactions as equally informative — doesn't account for preference drift.

### Tool 3: Temporal Weighting (our contribution)

Each interaction is assigned a weight based on how long ago it occurred, using exponential decay:

```
w_ui = exp(-λ · Δt_ui)
```

These weights are used in PLD's mean and variance calculations, so recent behaviour shapes the denoising threshold more strongly than ancient behaviour.

This addresses preference drift directly. A user whose tastes evolved from hobbyist to professional photography will have their "typical loss" range defined by recent professional purchases, not stale hobbyist purchases.

---

## Why Combine All Three?

Each tool fixes one problem but creates or fails to address others. The combination is necessary because the problems interact:

- A new user's few interactions may include noise → need both cold-start handling AND denoising
- A long-term user's old interactions may have stopped being relevant → need both denoising AND temporal awareness
- A user whose tastes evolved has effectively become two different users → temporal weighting separates "old preferences" from "current preferences"

No single tool handles all three. The hybrid is the contribution.

---

## The Critical Design Choice: Integration over Composition

There are two ways to combine PLD with LightGCL:

### Naive approach (composition)
1. Run PLD as a preprocessing filter — remove the noisy interactions
2. Train LightGCL on the cleaned data

This is easier to implement but **incorrect**. PLD has no access to graph structure when filtering, so it might remove interactions that look weird locally but are consistent with the user's broader neighbourhood (= a genuine emerging interest).

### Our approach (integration)
1. PLD operates *inside* the LightGCL training loop
2. PLD uses the current model's losses to decide what looks noisy
3. The denoising decisions then shape how embeddings update
4. Updated embeddings change what looks like noise in the next step
5. Feedback loop between graph structure and noise filtering

This is harder to implement but produces a model that uses graph-level and loss-level signals together. **This integration is the core methodological contribution of the project.**

---

## Key Design Decisions and Why

### Decision: Hard threshold (not soft threshold)

We use a hard threshold for the denoising decision:

```
n_ui = 1   if loss is within typical range
n_ui = β   otherwise (β = 0.1)
```

**Reasoning**: PLD uses a hard threshold. To make a clean comparison and demonstrate that improvements come specifically from temporal weighting, we keep everything else identical to PLD and change only one thing. A soft threshold would be a defensible engineering improvement but would confound the experimental comparison — we wouldn't know whether gains came from temporal weighting or from the threshold change.

A soft threshold is noted in the report's Recommendations section as future work.

### Decision: Per-user reference time t_now(u)

The temporal weight uses each user's own most recent interaction as the reference point, not a global timestamp:

```
t_now(u) = max(t_ui for i ∈ O_u)
Δt_ui = t_now(u) - t_ui
```

**Reasoning**: A global reference time punishes inactive users for being inactive. If a user was active in June 2023 but the dataset extends to January 2026, all their interactions would be discounted by 2.5 years — even though *from their perspective*, June 2023 was their "now." Per-user reference captures decay relative to each user's active period, which is what we actually want to measure.

Implementation cost is trivial: one max operation per user.

### Decision: Pre-compute temporal weights once

Temporal weights w_ui don't change during training (timestamps don't change), so they are computed once before training begins and stored.

**Reasoning**: Efficiency. Avoids recomputing exp() millions of times per epoch.

### Decision: Synthetic noise injection, not logged production noise

Evaluation injects noise at 5%, 10%, 15%, and 20% by randomly replacing a fraction of positive interactions with unrelated items.

**Reasoning**: This is the standard protocol in the recommendation denoising literature (used by PLD, DenoiseRec, and others). Real production noise is correlated with position bias, time of day, and device — but we don't have access to that data at this scale. Synthetic noise provides a controlled, reproducible benchmark.

This is acknowledged as a limitation in the Discussion section.

---

## What's In Scope

- Implicit feedback recommendation (binary user-item interactions)
- Three benchmark datasets: Gowalla, Yelp2018, MIND
- Synthetic noise injection at 5%/10%/15%/20%
- Comparison against LightGCN, LightGCN+PLD, LightGCL, LightGCL+PLD (no temporal), LightGCL+TW-PLD (full proposed model)
- Recall@K and NDCG@K evaluation, K ∈ {20, 50}
- Cold-start subgroup analysis (users with <5 or <10 interactions)
- Ablation on the temporal weighting component

---

## What's Out of Scope

- Explicit feedback (ratings)
- Social signals, demographics, contextual features (location, device, time of day)
- Fairness analysis across demographic groups
- Interpretability methods (attention visualization, saliency)
- Real-world deployment, A/B testing, production logging
- Cross-domain transfer
- Transformer-based or sequence-based variants
- Multi-task learning

These were considered and explicitly excluded to keep the project tractable within the semester timeline.

---

## How the Pieces Fit Together (high-level flow)

```
1. Load interaction graph with timestamps
2. Pre-compute SVD-augmented graph (global view)
3. Pre-compute temporal weights w_ui for every interaction
4. For each training epoch:
     a. Propagate embeddings through local view → e_u^(L)
     b. Propagate embeddings through global view → e_u^(G)
     c. Compute contrastive loss between two views
     d. For each batch:
        - Compute BPR loss per interaction
        - Compute weighted per-user statistics (μ_u, σ_u) using w_ui
        - Flag interactions whose loss exceeds μ_u + k·σ_u as noise
        - Compute final loss = denoised BPR + α·contrastive + γ·L2
     e. Backpropagate and update embeddings
5. Evaluate on held-out test set at multiple noise levels
```

The details are in `docs/pseudocode.md` and the formal math is in `docs/equations.md`.

---

## How to Validate the Implementation

When the implementation is complete, these are the checks that confirm it's working:

1. **Baseline reproduction**: LightGCL with no PLD, no noise, no temporal weighting should reproduce the published LightGCL numbers on Gowalla, Yelp2018, MIND (within seed variance).

2. **PLD reproduction**: Adding PLD with all w_ui = 1 should reproduce published PLD numbers.

3. **Sanity check on temporal weights**: w_ui for the user's most recent interaction should equal 1.0; older interactions should monotonically decrease.

4. **Sanity check on weighted stats**: If all w_ui = 1, weighted mean and variance should equal ordinary mean and variance.

5. **Sanity check on denoising**: At 0% noise, n_ui should be 1 for almost all interactions (only natural outliers flagged). At 20% noise, the model should flag roughly that fraction.

6. **Performance check**: On clean data, the proposed model should match LightGCL. As noise increases, it should degrade more slowly than baselines.

If any of these fail, there's a bug to find before running full experiments.

---

## Author Notes

This document was assembled from a design discussion before implementation began. Update it as decisions evolve during implementation, but mark changes with the date and reason. The conceptual stability of this document is more important than its completeness — when reasoning gets re-done, paste it here so future readers don't have to re-derive it.
