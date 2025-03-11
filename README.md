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
| ETT* | [https://github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| Exchange | [https://github.com/laiguokun/multivariate-time-series-data]( https://github.com/laiguokun/multivariate-time-series-data) |
| WTH | [https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR) |
| Electricity | [https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) |
| Traffic | [http://pems.dot.ca.gov/](http://pems.dot.ca.gov/) |
| Weather | [https://www.bgc-jena.mpg.de/wetter/+(https://www.bgc-jena.mpg.de/wetter/) |
| Ili | [https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html) |

## Run models

To run the models move to the root directory of each of them and follow the instructions reported in the ```README.md``` available.

## Statistics

To extract the statistics of the models, move to their root directory and run the following command:

```sh
python3 exctract_csv.py --directory forecasting/B<batch_size>_E<repr_dim>/ [--type raw]
```
