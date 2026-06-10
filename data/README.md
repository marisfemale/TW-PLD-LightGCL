# Data folder

Place dataset files here. The model expects:

```
data/
├── gowalla/
│   ├── train.txt
│   ├── test.txt
│   └── (timestamps if available)
├── yelp2018/
│   ├── train.txt
│   ├── test.txt
│   └── (timestamps if available)
└── mind/
    ├── train.txt
    ├── test.txt
    └── (timestamps if available)
```

The exact file format follows LightGCL's official repository conventions. See `references/citations.md` for dataset download links.

Note: timestamp information is required for the temporal weighting component. If a dataset doesn't include timestamps natively, you'll need to source them separately or use a synthetic ordering (and document this in the report).
