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

To run the experiments on the forecasting task use the following command:

```sh
sh forecasting.sh
```

To run the short-term forecasting please add the `--short_term` parameter in each command reported inside the `forecasting.sh` file.

After training and evaluation, the trained encoder, output, and evaluation metrics can be found in `training/forecasting/B{batch_size}_E{output_repr_dim}/<mode>/DatasetName__RunName_Date_Time/`.

