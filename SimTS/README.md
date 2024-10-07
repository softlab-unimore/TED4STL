# SimTS: Implementation with 1 - step pipeline

<p align="center">
    <img src="/images/simts.png" alt="simts" width=600>
</p>

## Dependencies
A list of dependencies is provided in ```requirements.txt```. After creating a virtual environment, we recommend installing dependencies via ```pip```:

```shell
pip install -r /path/to/requirements.txt
```
## Run the experiments

To run the experiments on forecasting task use the following command:

```shell
sh forecasting.sh
```
After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/forecasting/B{batch_size}_E{output_repr_dim}/<mode>/DatasetName__RunName_Date_Time/`.

To run the experiments on classification task use the following command:

```shell
sh classification.sh
```
After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/classfication/B{batch_size}_E{output_repr_dim}/<mode>/DatasetName__RunName_Date_Time/`.
