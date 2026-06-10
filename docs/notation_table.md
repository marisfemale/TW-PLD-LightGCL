# Notation Table — TW-PLD with LightGCL

This document maps mathematical symbols used in the equations to suggested Python variable names. Following this consistently across the codebase will make it much easier to verify that the implementation matches the math.

---

## Sets and Indices

| Math symbol | Python name | Meaning |
|---|---|---|
| `U` | `users` or `n_users` | Set of all users (or its size) |
| `I` | `items` or `n_items` | Set of all items (or its size) |
| `u` | `u` or `user_idx` | A user index |
| `i` | `i` or `item_idx` | An item index |
| `j` | `j` or `neg_item_idx` | A sampled negative item |
| `v` | `v` | Another user (used in contrastive denominator) |
| `O_u` | `user_items[u]` or `O_u` | Set of items user *u* interacted with |
| `O_i` | `item_users[i]` or `O_i` | Set of users who interacted with item *i* |
| `B` | `batch` | Current mini-batch of interactions |
| `k` (layer) | `layer` or `k_layer` | Current propagation layer index |

---

## Matrices and Tensors

| Math symbol | Python name | Shape | Meaning |
|---|---|---|---|
| `R` | `R` or `interaction_matrix` | (n_users, n_items) | Original interaction matrix (sparse) |
| `R̃` | `R_tilde` or `global_view` | (n_users, n_items) | SVD-reconstructed dense matrix |
| `U_q` | `U_q` | (n_users, q) | Left singular vectors |
| `Σ_q` | `Sigma_q` | (q, q) | Diagonal singular values |
| `V_q` | `V_q` | (n_items, q) | Right singular vectors |
| `e_u^(k)` | `e_u_layer[k]` or `e_u_local[k]` | (n_users, d) | Local-view user embeddings at layer k |
| `e_i^(k)` | `e_i_layer[k]` or `e_i_local[k]` | (n_items, d) | Local-view item embeddings at layer k |
| `ẽ_u^(k)` | `e_u_global[k]` | (n_users, d) | Global-view user embeddings at layer k |
| `ẽ_i^(k)` | `e_i_global[k]` | (n_items, d) | Global-view item embeddings at layer k |
| `e_u^(L)` | `e_u_local_final` | (n_users, d) | Final local-view embedding |
| `e_u^(G)` | `e_u_global_final` | (n_users, d) | Final global-view embedding |
| `e_i^(L)` | `e_i_local_final` | (n_items, d) | Final local-view item embedding |
| `e_i^(G)` | `e_i_global_final` | (n_items, d) | Final global-view item embedding |
| `Θ` | `model.parameters()` | varies | All trainable parameters (initial embeddings) |

---

## Scalars and Per-Interaction Values

| Math symbol | Python name | Meaning |
|---|---|---|
| `R_ui` | `R[u, i]` | Binary interaction flag (1 or 0) |
| `t_ui` | `t_ui` or `timestamp` | Timestamp of interaction (u, i) |
| `t_now(u)` | `t_now_u` or `user_t_now[u]` | User u's most recent interaction timestamp |
| `Δt_ui` | `delta_t_ui` | Time elapsed since interaction (u, i) |
| `w_ui` | `w_ui` or `temporal_weight` | Temporal weight of interaction (u, i) |
| `s(u, i)` | `score_ui` | Predicted preference score for (u, i) |
| `ℓ_ui` | `loss_ui` or `bpr_loss` | BPR loss for interaction (u, i) |
| `μ_u` | `mu_u` | Weighted mean of user u's losses |
| `σ_u` | `sigma_u` | Weighted std-dev of user u's losses |
| `n_ui` | `n_ui` or `denoise_weight` | Denoising weight for interaction (u, i) |

---

## Losses

| Math symbol | Python name | Meaning |
|---|---|---|
| `L_BPR^(TW-PLD)` | `l_bpr` or `loss_bpr_denoised` | Temporally-weighted, denoised BPR loss |
| `L_CL` | `l_cl` or `loss_contrastive` | LightGCL contrastive loss |
| `L_total` | `l_total` or `loss` | Final loss being minimised |
| `‖Θ‖₂²` | `l2_reg` or `weight_decay_term` | L2 regularisation term |

---

## Hyperparameters

| Math symbol | Python name | Role |
|---|---|---|
| `d` | `embedding_dim` | Embedding dimension |
| `K` | `n_layers` or `K` | Number of graph convolution layers |
| `q` | `svd_rank` or `q` | SVD truncation rank |
| `λ` | `lambda_decay` | Temporal decay rate (avoid the Python keyword `lambda`) |
| `k` | `k_threshold` | Noise threshold multiplier (rename to avoid clash with layer `k`) |
| `β` | `beta_suppress` | Noise suppression factor |
| `τ` | `tau` | Contrastive temperature |
| `α` | `alpha` | Contrastive loss weight |
| `γ` | `gamma` or `weight_decay` | L2 regularisation weight |
| `η` | `learning_rate` or `lr` | Optimiser learning rate |

---

## Important Naming Conventions

### Avoid Python keyword conflicts

- `λ` (lambda) → use `lambda_decay`, **not** `lambda` (Python reserved word)
- `k` (threshold) → use `k_threshold`, since `k` is also used for the layer index
- `K` (layers) → use `K` or `n_layers`

### Suffix conventions

- `_local` for local-view quantities (e.g., `e_u_local`)
- `_global` for global-view quantities (e.g., `e_u_global`)
- `_final` for post-layer-combination embeddings (e.g., `e_u_local_final`)
- `_u`, `_i` for per-user, per-item quantities
- `_ui` for per-interaction quantities

### Tensor shape comments

In code, add shape annotations on tensor variables:

```python
e_u = self.user_embeddings  # shape: (n_users, embedding_dim)
w_ui = temporal_weights[u_idx, i_idx]  # shape: (batch_size,)
mu_u = compute_weighted_mean(losses, weights)  # shape: (n_users_in_batch,)
```

This catches shape bugs early and serves as documentation.

---

## Example: A Variable Naming Cheat Sheet

When reading the equations and translating to code, this mapping should be quick reference:

```
Math:  ℓ_ui = -log σ(s(u,i) - s(u,j))
Code:  loss_ui = -torch.log(torch.sigmoid(score_pos - score_neg))

Math:  w_ui = exp(-λ · Δt_ui)
Code:  w_ui = torch.exp(-lambda_decay * delta_t_ui)

Math:  μ_u = Σ(w_ui · ℓ_ui) / Σ(w_ui)
Code:  mu_u = (w_ui * loss_ui).sum() / w_ui.sum()

Math:  n_ui = 1 if ℓ_ui ≤ μ_u + k·σ_u else β
Code:  n_ui = torch.where(loss_ui <= mu_u + k_threshold * sigma_u, 1.0, beta_suppress)

Math:  L_total = Σ n_ui·ℓ_ui + α·L_CL + γ·||Θ||²
Code:  l_total = (n_ui * loss_ui).sum() + alpha * l_cl + gamma * l2_reg
```

Use these patterns as a reference when writing the model class.

---

## Cross-References

For the full equations using these symbols, see `equations.md`.  
For the procedural flow, see `pseudocode.md`.
