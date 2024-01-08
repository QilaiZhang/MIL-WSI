# MIL-WSI
MIL-WSI provides APIs to easily train and test state-of-the-art MIL models





### Feature Extracting

Run with one GPU or CPU:

```python
python scripts/extract_features.py --data_path [DATA_PATH] --feature_path [FEATURE_PATH]
```

Run with multiple GPUs:

```
torchrun --nproc-per-node [GPU_NUM] scripts/extract_features.py --launcher pytorch --data_path [DATA_PATH] --feature_path [FEATURE_PATH]
```