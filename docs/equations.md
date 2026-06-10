# Formal Equations — TW-PLD with LightGCL

This document contains the complete mathematical specification of the model. Every equation includes inline definitions of its symbols so no cross-referencing is needed.

The equations are presented in the order they execute during training. Each equation corresponds to specific lines in `pseudocode.md`.

---

## Stage 1: LightGCL Encoder

### Equation 1 — Local view propagation

For each graph convolution layer, embeddings are updated by aggregating neighbour embeddings from the previous layer:

$$\mathbf{e}_u^{(k)} = \sum_{i \in \mathcal{O}_u} \frac{1}{\sqrt{|\mathcal{O}_u| \cdot |\mathcal{O}_i|}} \, \mathbf{e}_i^{(k-1)}$$

$$\mathbf{e}_i^{(k)} = \sum_{u \in \mathcal{O}_i} \frac{1}{\sqrt{|\mathcal{O}_u| \cdot |\mathcal{O}_i|}} \, \mathbf{e}_u^{(k-1)}$$

**Where:**
- `e_u^(k)` — embedding vector of user *u* at propagation layer *k*
- `e_i^(k)` — embedding vector of item *i* at propagation layer *k*
- `k` — current propagation layer, k ∈ {1, 2, ..., K}
- `K` — total number of graph convolution layers (a hyperparameter)
- `O_u` — set of items that user *u* interacted with
- `O_i` — set of users who interacted with item *i*
- `|O_u|`, `|O_i|` — number of interactions for user *u* and item *i* respectively (the node degrees)
- The square-root normalisation prevents popular users/items from dominating propagation

**In plain terms:** each user's new embedding is a weighted average of the items they interacted with (and vice versa). The normalisation prevents popular users or items from dominating the propagation.

---

### Equation 2 — SVD decomposition of the interaction matrix

The user-item interaction matrix is decomposed using truncated singular value decomposition with rank *q*:

$$\mathbf{R} \approx \mathbf{U}_q \mathbf{\Sigma}_q \mathbf{V}_q^\top$$

The reconstructed dense matrix forms the **global view** of the interaction graph:

$$\tilde{\mathbf{R}} = \mathbf{U}_q \mathbf{\Sigma}_q \mathbf{V}_q^\top$$

**Where:**
- `R` — the original user-item interaction matrix of shape |U| × |I|, where R_ui = 1 if user *u* interacted with item *i*, else 0
- `|U|` — total number of users
- `|I|` — total number of items
- `q` — truncation rank, controlling how many latent patterns to retain (a hyperparameter, typically q = 5)
- `U_q` — left singular vectors, shape |U| × q (user-side patterns)
- `Σ_q` — diagonal matrix of singular values, shape q × q (pattern strengths)
- `V_q` — right singular vectors, shape |I| × q (item-side patterns)
- `V_q^T` — the transpose of V_q
- `R̃` — the reconstructed dense matrix (the global view), shape |U| × |I|

**In plain terms:** SVD finds the top q strongest patterns in the interaction data and rebuilds a smoothed version of the graph that has soft connections everywhere, not just on real interactions. This view is computed once before training.

---

### Equation 3 — Global view propagation

Propagation on the SVD-reconstructed graph follows the same form as Equation 1, but uses the dense reconstruction as edge weights:

$$\tilde{\mathbf{e}}_u^{(k)} = \sum_{i \in \mathcal{I}} \tilde{\mathbf{R}}_{ui} \cdot \mathbf{e}_i^{(k-1)}$$

$$\tilde{\mathbf{e}}_i^{(k)} = \sum_{u \in \mathcal{U}} \tilde{\mathbf{R}}_{ui} \cdot \mathbf{e}_u^{(k-1)}$$

**Where:**
- `ẽ_u^(k)` — global-view embedding of user *u* at layer *k* (tilde distinguishes it from the local-view embedding)
- `ẽ_i^(k)` — global-view embedding of item *i* at layer *k*
- `I` — the full set of all items
- `U` — the full set of all users
- `R̃_ui` — the entry at row *u*, column *i* in the SVD-reconstructed matrix (the soft connection strength between user *u* and item *i*)
- Unlike Equation 1, the sum runs over **all** items (and users), not just observed interactions

**In plain terms:** in the global view, every user is softly connected to every item with strength given by the SVD reconstruction, so even users with few real interactions get embeddings shaped by broad patterns.

---

### Equation 4 — Final embedding (layer combination)

The final embedding for each user is the average across all propagation layers, computed separately for each view:

$$\mathbf{e}_u^{(L)} = \frac{1}{K+1} \sum_{k=0}^{K} \mathbf{e}_u^{(k)}, \qquad \mathbf{e}_u^{(G)} = \frac{1}{K+1} \sum_{k=0}^{K} \tilde{\mathbf{e}}_u^{(k)}$$

**Where:**
- `e_u^(L)` — final local-view embedding of user *u* (the superscript (L) denotes "Local")
- `e_u^(G)` — final global-view embedding of user *u* (the superscript (G) denotes "Global")
- `e_u^(0)` — the initial (randomly initialised) embedding at layer 0
- `e_u^(k)` — local-view embedding at layer *k*, from Equation 1
- `ẽ_u^(k)` — global-view embedding at layer *k*, from Equation 3
- `K` — total number of layers
- `K + 1` — the divisor, since we average over layers 0 through K inclusive
- The same operation is applied analogously to item embeddings e_i^(L) and e_i^(G)

**In plain terms:** rather than using only the final layer, we average across all layers (0 through K) so the embedding captures information at every neighbourhood scale.

---

## Stage 2: BPR Loss and Temporal Weighting

### Equation 5 — Predicted preference score

For a user-item pair, the predicted score is the inner product of their final local-view embeddings:

$$s(u, i) = \mathbf{e}_u^{(L)} \cdot \mathbf{e}_i^{(L)}$$

**Where:**
- `s(u, i)` — the model's predicted preference score for user *u* on item *i* (a single scalar)
- `e_u^(L)` — final local-view embedding of user *u* (from Equation 4)
- `e_i^(L)` — final local-view embedding of item *i* (from Equation 4)
- The dot product `·` here is the standard vector inner product, summed over all *d* embedding dimensions

**In plain terms:** higher dot product means the user and item point in similar directions in embedding space, which the model interprets as higher preference.

---

### Equation 6 — BPR loss per interaction

For each positive interaction and a sampled negative item, the Bayesian Personalized Ranking loss is:

$$\ell_{ui} = -\log \sigma\big(s(u, i) - s(u, j)\big)$$

**Where:**
- `ℓ_ui` — BPR loss for the interaction between user *u* and item *i*
- `s(u, i)` — predicted score for the positive (observed) interaction, from Equation 5
- `s(u, j)` — predicted score for a negative item *j* (sampled uniformly from items the user has *not* interacted with: j ∉ O_u)
- `σ(·)` — the sigmoid function, defined as σ(x) = 1 / (1 + exp(−x)), which maps any real number into the range (0, 1)
- `log` — natural logarithm
- A small loss means the model correctly ranks the positive above the negative; a large loss means it doesn't

**In plain terms:** the loss is low when the model scores the real interaction higher than a random non-interaction, and high when it gets that ordering wrong.

---

### Equation 7 — Temporal weight

For each interaction, the temporal weight uses exponential decay with a per-user reference time:

$$w_{ui} = \exp(-\lambda \cdot \Delta t_{ui})$$

$$\Delta t_{ui} = t_{\text{now}}(u) - t_{ui}$$

$$t_{\text{now}}(u) = \max_{i \in \mathcal{O}_u} t_{ui}$$

**Where:**
- `w_ui` — temporal weight assigned to the interaction between user *u* and item *i*; falls in the range (0, 1]
- `λ` (lambda) — temporal decay rate, a positive hyperparameter (larger λ = faster decay)
- `Δt_ui` — elapsed time between the interaction and the user's reference time (units must match λ, e.g., days)
- `t_ui` — timestamp of the interaction between user *u* and item *i*
- `t_now(u)` — reference time for user *u*, defined as the timestamp of *u*'s most recent interaction
- `max(...)` — the maximum function, returning the largest timestamp across all of *u*'s interactions
- `exp(x)` — the exponential function, e^x
- When Δt_ui = 0 (most recent interaction), w_ui = exp(0) = 1
- As Δt_ui grows, w_ui shrinks toward 0

**In plain terms:** each interaction gets a weight between 0 and 1 based on how long ago it happened, relative to the user's own most recent interaction. Recent interactions get weights near 1; older ones decay toward 0.

---

## Stage 3: TW-PLD Denoising

### Equation 8 — Weighted mean of per-user losses

For each user, the weighted mean of their interaction losses is:

$$\mu_u = \frac{\sum_{i \in \mathcal{O}_u} w_{ui} \cdot \ell_{ui}}{\sum_{i \in \mathcal{O}_u} w_{ui}}$$

**Where:**
- `μ_u` (mu) — weighted mean of losses for user *u*; represents the "typical" loss level for this user, with recent interactions counting more
- `O_u` — set of items user *u* has interacted with (the same set used in Equation 1)
- `w_ui` — temporal weight from Equation 7
- `ℓ_ui` — BPR loss from Equation 6
- Numerator: total weighted loss across all of *u*'s interactions
- Denominator: total weight (acts as the "weighted sample size")
- If all w_ui = 1, this reduces to the ordinary mean

**In plain terms:** instead of an ordinary average, recent losses count more in computing the "typical" loss for this user.

---

### Equation 9 — Weighted standard deviation

The weighted standard deviation of losses for user *u*:

$$\sigma_u = \sqrt{\frac{\sum_{i \in \mathcal{O}_u} w_{ui} \cdot (\ell_{ui} - \mu_u)^2}{\sum_{i \in \mathcal{O}_u} w_{ui}}}$$

**Where:**
- `σ_u` (sigma) — weighted standard deviation of losses for user *u*; represents how much losses typically deviate from μ_u
- `(ℓ_ui − μ_u)²` — squared deviation of each loss from the weighted mean
- `w_ui` — temporal weight from Equation 7 (recent deviations matter more)
- `μ_u` — weighted mean from Equation 8
- `√(...)` — square root, which converts the weighted variance into a standard deviation (same units as the loss)
- If all w_ui = 1, this reduces to the ordinary standard deviation

**In plain terms:** how much losses typically deviate from the user's recent norm. This defines what counts as "outside the normal range" for this user.

---

### Equation 10 — Denoising weight (hard threshold)

Each interaction receives a denoising weight based on whether its loss falls within the user's typical range:

$$n_{ui} = \begin{cases} 1 & \text{if } \ell_{ui} \leq \mu_u + k \cdot \sigma_u \\ \beta & \text{otherwise} \end{cases}$$

**Where:**
- `n_ui` — denoising weight for the interaction between user *u* and item *i*
- `ℓ_ui` — BPR loss for this interaction (from Equation 6)
- `μ_u` — user *u*'s weighted mean loss (from Equation 8)
- `σ_u` — user *u*'s weighted standard deviation of losses (from Equation 9)
- `k` — threshold multiplier, a positive hyperparameter (typically k ∈ [1.0, 2.0]; higher *k* = stricter, fewer interactions flagged)
- `β` (beta) — suppression factor for flagged noisy interactions (typically β = 0.1; setting β = 0 removes them entirely)
- `μ_u + k · σ_u` — the upper threshold above which losses are considered abnormally high

**In plain terms:** interactions whose loss is within the user's typical range get full weight; those whose loss is unusually high are flagged as likely noise and suppressed to a small fraction of their weight.

---

## Stage 4: Joint Loss and Optimisation

### Equation 11 — LightGCL contrastive loss

The contrastive loss aligns local and global view embeddings using the InfoNCE formulation:

$$\mathcal{L}_{CL} = \sum_{u \in \mathcal{U}} -\log \frac{\exp\big(\cos(\mathbf{e}_u^{(L)}, \, \mathbf{e}_u^{(G)}) / \tau\big)}{\sum_{v \in \mathcal{U}} \exp\big(\cos(\mathbf{e}_u^{(L)}, \, \mathbf{e}_v^{(G)}) / \tau\big)}$$

**Where:**
- `L_CL` — total contrastive loss summed over all users (CL = Contrastive Loss)
- `U` — full set of users in the batch
- `u` — the "anchor" user (the one whose two views we want to align)
- `v` — any user in the batch, including *u* itself (used in the denominator over all candidates)
- `e_u^(L)` — local-view embedding of user *u* (from Equation 4)
- `e_u^(G)` — global-view embedding of user *u* (from Equation 4)
- `e_v^(G)` — global-view embedding of user *v*
- `cos(a, b)` — cosine similarity between vectors *a* and *b*, defined as (a · b) / (‖a‖ · ‖b‖), ranging from −1 to 1
- `τ` (tau) — temperature hyperparameter, a positive scalar (typically τ ∈ [0.1, 1.0]); smaller τ = sharper distinctions between positive and negative pairs
- `exp(·)` — exponential function
- Numerator: similarity between user *u*'s own two views (this is what we want to maximise)
- Denominator: sum of similarities between *u*'s local view and every user's global view (the "competition")
- An analogous loss is also computed on the item side and added; both terms together form the full L_CL

**In plain terms:** this loss pulls each user's local and global embeddings toward each other, while pushing them away from every other user's global embedding. It forces the two views to agree on who each user is.

---

### Equation 12 — Joint training objective

The total loss combines the temporally-weighted, PLD-denoised BPR loss with the contrastive loss and a regularisation term:

$$\mathcal{L}_{\text{total}} = \underbrace{\sum_{(u,i) \in \mathcal{B}} n_{ui} \cdot \ell_{ui}}_{\mathcal{L}_{BPR}^{\text{TW-PLD}}} + \alpha \cdot \mathcal{L}_{CL} + \gamma \cdot \|\Theta\|_2^2$$

**Where:**
- `L_total` — final loss being minimised at each training step
- `L_BPR^(TW-PLD)` — the temporally-weighted, denoised BPR loss (the underbraced first term)
- `B` — the current mini-batch of interactions being processed
- `(u, i) ∈ B` — each positive interaction in the batch
- `n_ui` — denoising weight from Equation 10 (1 for clean interactions, β for noisy ones)
- `ℓ_ui` — BPR loss from Equation 6
- `α` (alpha) — contrastive loss coefficient, a positive hyperparameter balancing recommendation accuracy and view consistency
- `L_CL` — contrastive loss from Equation 11
- `γ` (gamma) — L2 regularisation coefficient, a small positive hyperparameter (typically γ = 10⁻⁴)
- `Θ` (capital theta) — the set of all trainable parameters in the model (i.e., all initial embeddings e_u^(0) and e_i^(0))
- `‖Θ‖₂²` — squared L2 norm of the parameters, equal to the sum of squares of every parameter value (this penalises large weights to prevent overfitting)

**In plain terms:** the model learns to (a) rank real interactions above non-interactions while ignoring flagged noise, (b) keep its two views consistent, and (c) avoid overfitting. The three terms balance these goals.

---

## Hyperparameter Summary

For convenience, here are all hyperparameters that appear in the equations. Concrete values will be tuned during implementation and reported in Section 4.4 of the final report.

| Symbol | Role | Equation | Typical range |
|---|---|---|---|
| `d` | Embedding dimension | (implicit) | 32–128 |
| `K` | Number of propagation layers | 1, 3, 4 | 2–3 |
| `q` | SVD truncation rank | 2 | 5–10 |
| `λ` | Temporal decay rate | 7 | 10⁻⁴ to 10⁻² per day |
| `k` | Noise threshold multiplier | 10 | 1.0–2.0 |
| `β` | Noise suppression factor | 10 | 0.0–0.2 |
| `τ` | Contrastive temperature | 11 | 0.1–1.0 |
| `α` | Contrastive loss weight | 12 | 0.1–1.0 |
| `γ` | L2 regularisation weight | 12 | 10⁻⁵ to 10⁻³ |

---

## Equation-to-Pseudocode Mapping

| Pseudocode line | Equation |
|---|---|
| 1 (pre-compute SVD) | Eq. 2 |
| 1 (pre-compute weights) | Eq. 7 |
| 4 (local view propagation) | Eq. 1 |
| 5 (global view propagation) | Eq. 3 |
| (end of stage 1, final embedding) | Eq. 4 |
| 6 (contrastive loss) | Eq. 11 |
| 8 (prediction score) | Eq. 5 |
| 9–10 (BPR loss) | Eq. 6 |
| 13 (weighted mean) | Eq. 8 |
| 14 (weighted std) | Eq. 9 |
| 15–19 (noise flagging) | Eq. 10 |
| 20–21 (joint loss) | Eq. 12 |
