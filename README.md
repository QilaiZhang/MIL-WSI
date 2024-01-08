# MIL-WSI
MIL-WSI provides APIs to easily train and test state-of-the-art MIL models



```shell
conda env create -f enviroment.yaml
```




### Feature Extracting

Run with one GPU or CPU:

```shell
python scripts/extract_features.py --data_path [DATA_PATH] --feature_path [FEATURE_PATH]
```

Run with multiple GPUs:

```shell
torchrun --nproc-per-node [GPU_NUM] scripts/extract_features.py --launcher pytorch --data_path [DATA_PATH] --feature_path [FEATURE_PATH]
```