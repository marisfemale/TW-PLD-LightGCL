# Reference Papers

These are the papers that the model is built on. If PDF files are available, place them in this folder alongside this document. Filenames should be: `lightgcl.pdf`, `pld.pdf`, `lightgcn.pdf`, etc.

---

## Primary References (Essential)

### LightGCL (the backbone)

Cai, X., Huang, C., Xia, L., and Ren, X. (2023). **LightGCL: Simple yet effective graph contrastive learning for recommendation.** *International Conference on Learning Representations (ICLR)*.

- Official code: https://github.com/HKUDS/LightGCL
- arXiv: https://arxiv.org/abs/2302.08191
- Why it matters: this is the encoder backbone. The official PyTorch implementation should be the starting point for the codebase.

### PLD (Personalized Loss Distribution)

Zhang, K., Cao, Q., Wu, Y., Sun, F., Shen, H., and Cheng, X. (2025). **Personalized denoising implicit feedback for robust recommender system.** *Proceedings of the ACM Web Conference (WWW 2025)*.

- Why it matters: this is the denoising method we are extending. Our temporal weighting is added on top of PLD's per-user loss distribution framework.

### LightGCN (the simpler baseline)

He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., and Wang, M. (2020). **LightGCN: Simplifying and powering graph convolution network for recommendation.** *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval*, pp. 639–648.

- Why it matters: comparison baseline. LightGCL is built on top of LightGCN's propagation rule.

---

## Secondary References (Supporting)

### Temporal dynamics in collaborative filtering

Koren, Y. (2009). **Collaborative filtering with temporal dynamics.** *Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 447–456.

- Foundational paper for temporal modelling in recommendation. Our temporal weighting builds on this tradition.

### Implicit feedback noise — foundational

Hu, Y., Koren, Y., and Volinsky, C. (2008). **Collaborative filtering for implicit feedback datasets.** *Proceedings of the 8th IEEE International Conference on Data Mining (ICDM)*, pp. 263–272.

- Establishes the implicit feedback problem.

### Implicit feedback noise — denoising methods

Wang, W., Feng, F., He, X., Nie, L., and Chua, T.-S. (2021). **Learning robust recommender from noisy implicit feedback.** *arXiv preprint arXiv:2112.01160*.

Sun, J., Guo, W., Zhang, H., Zhang, J., Liu, Z., and He, X. (2020). **Framework for analyzing and mitigating noise in implicit feedback.** *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

### Graph neural networks for recommendation

Wang, X., He, X., Wang, M., Feng, F., and Chua, T.-S. (2019). **Neural graph collaborative filtering.** *Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval*, pp. 165–174.

Hamilton, W. L., Ying, R., and Leskovec, J. (2017). **Inductive representation learning on large graphs.** *Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS)*, pp. 1025–1035.

### Matrix factorization

Koren, Y., Bell, R., and Volinsky, C. (2009). **Matrix factorization techniques for recommender systems.** *Computer*, 42(8), pp. 30–37.

### Datasets

Wu, F., Qiao, Y., Chen, J. H., Wu, C., Qi, T., Lian, J., Liu, D., Xing, X., Gao, G., Xie, X., and Zhou, M. (2020). **MIND: A large-scale dataset for news recommendation.** *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pp. 3597–3606.

### Survey

Zhang, K., Cao, Q., Sun, F., Wu, Y., Tao, S., Shen, H., and Cheng, X. (2025). **Robust recommender system: A survey and future directions.** *ACM Computing Surveys*.

---

## Useful Resources

- LightGCL official repository: https://github.com/HKUDS/LightGCL
- Gowalla dataset: https://snap.stanford.edu/data/loc-gowalla.html
- Yelp dataset: https://www.yelp.com/dataset
- MIND dataset: https://msnews.github.io/

If running into implementation issues, the LightGCL repo's `Issues` tab on GitHub is a good first place to check.
