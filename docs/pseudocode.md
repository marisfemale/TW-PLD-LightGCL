# Training Loop Pseudocode — TW-PLD with LightGCL

This document describes the complete training procedure in plain language. Each step corresponds to one or more equations in `equations.md` and one or more lines of code in the implementation.

---

## Algorithm 1: TW-PLD with LightGCL — Training procedure

```
INPUT:  User-item graph R with timestamps t_ui
        Hyperparameters: λ, α, k, β, K, d, q, τ, γ, η, batch_size, epochs

OUTPUT: Trained embeddings e_u for all u ∈ U, e_i for all i ∈ I

═══════════════════════════════════════════════════════════════════════
PRE-TRAINING SETUP (run once before training begins)
═══════════════════════════════════════════════════════════════════════

1.  Initialize:
       Random embeddings e_u^(0), e_i^(0) for all users and items
       Compute SVD-augmented graph view R̃ from R         (Eq. 2)
       Compute t_now(u) = max(t_ui) for every user        (Eq. 7)
       Compute Δt_ui = t_now(u) - t_ui for all interactions
       Compute temporal weights w_ui = exp(-λ · Δt_ui)    (Eq. 7)
       
       Note: w_ui values do not change during training,
       so they are stored once and reused every epoch.

═══════════════════════════════════════════════════════════════════════
MAIN TRAINING LOOP
═══════════════════════════════════════════════════════════════════════

2.  FOR each epoch = 1 to total_epochs:

3.      FOR each mini-batch B sampled from R:

        ─────────────────────────────────────────────────────────────
        STAGE 1 — LightGCL encoder (Equations 1, 3, 4)
        ─────────────────────────────────────────────────────────────
        
4.      Propagate K layers on local view  → E^(L)         (Eq. 1)
        
5.      Propagate K layers on global view → E^(G)         (Eq. 3)
        
6.      Compute contrastive loss L_CL between E^(L) and E^(G)
            using temperature τ                            (Eq. 11)

        ─────────────────────────────────────────────────────────────
        STAGE 2 — Per-interaction BPR loss (Equations 5, 6)
        ─────────────────────────────────────────────────────────────
        
7.      FOR each interaction (u, i) in B:
        
8.          Compute prediction score s(u, i) = e_u · e_i  (Eq. 5)
            
9.          Sample a negative item j ∉ O_u
            
10.         Compute BPR loss
                ℓ_ui = -log σ(s(u,i) - s(u,j))             (Eq. 6)

        ─────────────────────────────────────────────────────────────
        STAGE 3 — TW-PLD denoising (Equations 8, 9, 10)
        ─────────────────────────────────────────────────────────────
        
11.     FOR each user u present in B:
        
12.         Gather (ℓ_ui, w_ui) for all of u's interactions in B
            
13.         Compute weighted mean
                μ_u = Σ(w_ui · ℓ_ui) / Σ(w_ui)             (Eq. 8)
            
14.         Compute weighted std-dev σ_u from same weights (Eq. 9)
            
15.         FOR each interaction (u, i):
        
16.             IF ℓ_ui > μ_u + k · σ_u:
17.                 Set n_ui = β            ← flagged as noisy
18.             ELSE:
19.                 Set n_ui = 1            ← treated as clean

        ─────────────────────────────────────────────────────────────
        STAGE 4 — Joint loss and parameter update (Equation 12)
        ─────────────────────────────────────────────────────────────
        
20.     Compute weighted BPR loss:
            L_BPR = Σ_(u,i) n_ui · ℓ_ui
            
21.     Compute joint loss:
            L_total = L_BPR + α · L_CL + γ · ||Θ||²₂        (Eq. 12)
            
22.     Backpropagate L_total
        
23.     Update embeddings: e ← e - η · ∇L_total

        END FOR (mini-batch loop)

24.     Evaluate on validation set (Recall@K, NDCG@K)
        
25.     IF early-stopping criterion met:
            BREAK

        END FOR (epoch loop)

26. RETURN trained embeddings e_u, e_i
```

---

## Notes on Each Stage

### Pre-training setup (line 1)

The pre-training setup is the most subtle part of the algorithm. Three things happen here:

1. **Random initialization** — every user and item gets a random `d`-dimensional vector. These are the parameters the model will learn.

2. **SVD decomposition** — done once, on CPU, before training. The resulting matrices `U_q`, `Σ_q`, `V_q` are stored. Computing SVD on a large sparse matrix can be slow; use `scipy.sparse.linalg.svds` for efficiency.

3. **Temporal weight pre-computation** — every interaction's `w_ui` is computed once and stored. Since timestamps don't change, recomputing every epoch would be wasteful. Store these as a tensor aligned with the interaction list.

### Stage 1: Encoder (lines 4–6)

In each mini-batch, both views propagate forward simultaneously. This is the most compute-intensive part of training. Pay attention to:
- **Sparse matrix operations** for local-view propagation (use PyTorch's sparse tensor support)
- **Dense matrix operations** for global-view propagation (since R̃ is dense)
- **Memory efficiency** — for large graphs, propagation needs to be batched

### Stage 2: BPR loss (lines 7–10)

This is straightforward but requires negative sampling. The standard approach:
- For each positive (u, i), sample one negative item j uniformly at random
- Check that j ∉ O_u (rejection sampling)
- Use the sampled negative in the BPR loss

For efficiency, batch the negative sampling: pre-sample many negatives, then check membership in a vectorized way.

### Stage 3: TW-PLD denoising (lines 11–19)

This is where the novelty lives. Critical details:

- **Per-user statistics** are computed using only interactions present in the current batch.
- For users with very few interactions in the batch, the statistics will be noisy. Consider maintaining a running estimate of `μ_u` and `σ_u` across batches using exponential moving averages.
- **Temporal weights `w_ui` are looked up**, not recomputed (they were pre-computed in step 1).
- The denoising weight `n_ui` is a tensor of the same shape as the loss tensor, with values in {1, β}.

### Stage 4: Joint loss (lines 20–23)

Standard PyTorch training step:
- Compute scalar `L_total`
- Call `L_total.backward()`
- Step the optimizer (Adam is typical)
- Zero gradients before the next iteration

---

## Implementation Considerations

### Memory and speed

- LightGCL's two views double the memory cost compared to LightGCN. Use mixed-precision training (`torch.cuda.amp`) where possible.
- Use sparse tensor operations for the local view propagation; dense operations for the global view.
- Gowalla is the smallest dataset and should be implemented first to debug. Yelp2018 and MIND require more memory.

### Reproducibility

- Set seeds in PyTorch, NumPy, and Python's `random` module at the start of every script.
- Use `torch.backends.cudnn.deterministic = True` for full reproducibility (at a small speed cost).
- Save the config used for each run alongside the results.

### Early stopping

- Validate every N epochs (e.g., N = 5)
- Track Recall@20 on the validation set
- Stop if no improvement for, say, 10 consecutive validation checks
- Save the best model checkpoint

### Edge cases to handle

- **User with one interaction in a batch**: weighted mean is the loss itself, weighted std is 0. The denoising threshold is then `loss + 0 = loss`, so the interaction is never flagged. This is acceptable behaviour — sparse users get no denoising, which is what we'd want.
- **User with no interactions in a batch**: skip the user's denoising computation.
- **All weights summing to 0**: shouldn't happen in practice (weights are positive), but guard with a small epsilon in the denominator: `Σ w + 1e-10`.
- **Numerical overflow in exp()**: clip Δt values or use `log1p` carefully.

---

## Cross-References

| Stage | Equations | Lines in pseudocode |
|---|---|---|
| Pre-training setup | Eq. 2, 7 | 1 |
| Stage 1: Encoder | Eq. 1, 3, 4, 11 | 4–6 |
| Stage 2: BPR loss | Eq. 5, 6 | 7–10 |
| Stage 3: Denoising | Eq. 8, 9, 10 | 11–19 |
| Stage 4: Joint loss | Eq. 12 | 20–23 |
| Evaluation | (separate) | 24 |

For the full equations with symbol definitions, see `equations.md`.  
For symbol-to-variable naming, see `notation_table.md`.
