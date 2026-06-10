# TW-PLD with LightGCL

A robust recommender system that combines graph contrastive learning, personalized denoising, and temporal weighting to address three problems simultaneously: noisy implicit feedback, cold-start sparsity, and preference drift over time.

## Project Context

This is the implementation codebase for a Master's capstone project (PRT840) at Charles Darwin University. The full design rationale, conceptual background, and scope are documented in `CONTEXT.md` — read that first.

## What This Model Does

The model integrates three existing components into a single end-to-end training procedure:

1. **LightGCL** — graph contrastive learning backbone for stable embeddings under sparse data
2. **PLD (Personalized Loss Distribution)** — per-user denoising of noisy implicit feedback
3. **Temporal Weighting** — exponential decay so recent interactions count more than older ones (the novel contribution)

The novelty is in the **integration**: PLD operates inside the LightGCL training loop (not as preprocessing), and PLD's statistics use temporally-weighted means and variances.

## Research Goal

Demonstrate that TW-PLD with LightGCL outperforms both LightGCN and PLD-only baselines on three benchmark datasets (Gowalla, Yelp2018, MIND) under controlled noise injection at rates between 5% and 20%, measured by Recall@K and NDCG@K.

## Current Status

- ✅ Methodology finalized
- ✅ Twelve formal equations specified (see `docs/equations.md`)
- ✅ Training loop pseudocode complete (see `docs/pseudocode.md`)
- ✅ System architecture documented (see `docs/system_diagram.png`)
- 🔄 Implementation phase — starting now

## Where to Start

New contributors should read these documents in order before touching the code:

1. `CONTEXT.md` — full conceptual background and rationale
2. `docs/equations.md` — the formal model specification
3. `docs/pseudocode.md` — the training loop in plain language
4. `docs/notation_table.md` — symbol definitions used in the code

## Folder Structure

```
TW-PLD-LightGCL/
├── README.md                    This file
├── CONTEXT.md                   Full conceptual background
├── docs/
│   ├── system_diagram.png       Architecture diagram
│   ├── notation_table.md        Symbol definitions
│   ├── pseudocode.md            Training loop in plain language
│   └── equations.md             All 12 formal equations
├── references/
│   ├── citations.md             Required reference papers
│   └── (PDF files when available)
├── data/                        Datasets (Gowalla, Yelp2018, MIND)
├── notebooks/                   Colab notebooks for GPU training
├── src/                         Implementation code
└── outputs/                     Per-run training artifacts
```

## Implementation Stack

- **Language**: Python 3.10+
- **Framework**: PyTorch
- **Compute**: Google Colab Pro (GPU acceleration)
- **Starting codebase**: Official LightGCL PyTorch implementation
- **Datasets**: Gowalla, Yelp2018, MIND (standard recommendation benchmarks)

## Authors

PRT840 Capstone Project, Semester 1, 2026
Charles Darwin University, Faculty of Science and Technology

- Ngoc Loi Vong (S392159)
- Md Abir Hossain (S389056)
- Luong Thuy Dieu Nguyen (S379862)
- Rakibul Hasan (S383811)

Supervisor: Dr. Yan Zhang
