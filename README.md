# TED4STL: Trend-Error Decomposition For Self-Supervised Time Series Learning for Multivariate Forecasting Task

<p align="center">
<img src=".\images\1_pipeline_step_1.png" width = "600" alt="" align=center />
</p>

<p align="center">
<img src=".\images\2_pipeline_step_1.png" width = "600" alt="" align=center />
</p>

## Datasets

The datasets are available at the following links:

| Dataset | Link |
|-|-|
| ETTh1 & ETTm1 | [https://github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| Electricity | [https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) |
| Weather | [https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR) |
| Traffic | [http://pems.dot.ca.gov/](http://pems.dot.ca.gov/) |


## Statistics

To extract the statitics of the models, move to their root directory and run the following command:

```sh
python3 exctract_csv.py --directory forecasting/B<batch_size>_E<repr_dim>/ [--type raw]
```
