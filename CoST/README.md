# CoST: Implemenation with 2 - step pipeline

<p align="center">
<img src=".\images\cost.png" width = "600" alt="" align=center />
</p>

  
## Requirements
Required dependencies can be installed by:
```sh
pip install -r requirements.txt
```


## Run the experiments

To run the experiments on forecasting task use the following command:

```shell
sh forecasting.sh
```
After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/forecasting/B{batch_size}_E{output_repr_dim}/<mode>/DatasetName__RunName_Date_Time/`.

