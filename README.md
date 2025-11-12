# TED4STL: Trend-Error Decomposition For Self-Supervised Time Series Learning in Multivariate Forecasting Task

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
| Weather | [https://www.bgc-jena.mpg.de/wetter/](https://www.bgc-jena.mpg.de/wetter/) |
| Ili | [https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html) |

All datasets are provided in .csv format, except for the **electricity** dataset, which requires a preprocessing step. To preprocess it, please navigate to the datasets directory of each model and run the following command:
```sh
python3 preprocess_electricity.py
```

## Run models

All the models require Python 3.11 and each has its requirements reported in the `requirements.txt` file in the model root directory. 
Please create a `.venv` environment for each model and install the requirements before running. 
To run the models move to the root directory of each of them and follow the instructions reported in the ```README.md``` file.

## Statistics

To extract the statistics of the models, move to their root directory and run the following command:

```sh
python3 exctract_csv.py --directory forecasting/B<batch_size>_E<repr_dim>/ [--type raw]
```
