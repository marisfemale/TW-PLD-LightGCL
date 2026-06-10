# Source folder

This is where the implementation lives. The recommended file structure:

```
src/
├── config.py            Hyperparameter configurations
├── data.py              Dataset loading, preprocessing, noise injection
├── model.py             TW-PLD-LightGCL model class
├── train.py             Training loop
├── evaluate.py          Recall@K and NDCG@K evaluation
├── utils.py             Helper functions
└── run_experiment.py    Top-level experiment runner
```
