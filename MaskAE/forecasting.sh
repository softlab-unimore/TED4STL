# To run the experiments on MAE with the pipeline integration on forecasting task
python3 script_forecasting_dlinear.py --dataset ETTh1 --mode dlinear
python3 finetune_script_forecasting_dlinear.py --dataset ETTh1 --mode dlinear

python3 script_forecasting_dlinear.py --dataset ETTh2 --mode dlinear
python3 finetune_script_forecasting_dlinear.py --dataset ETTh2 --mode dlinear

python3 script_forecasting_dlinear.py --dataset ETTm1 --mode dlinear
python3 finetune_script_forecasting_dlinear.py --dataset ETTm1 --mode dlinear

python3 script_forecasting_dlinear.py --dataset ETTm2 --mode dlinear
python3 finetune_script_forecasting_dlinear.py --dataset ETTm2 --mode dlinear

python3 script_forecasting_dlinear.py --dataset exchange_rate --mode dlinear
python3 finetune_script_forecasting_dlinear.py --dataset exchange_rate --mode dlinear

python3 script_forecasting_dlinear.py --dataset  WTH --mode dlinear
python3 finetune_script_forecasting_dlinear.py --dataset WTH --mode dlinear

python3 script_forecasting_dlinear.py --dataset electricity --mode dlinear
python3 finetune_script_forecasting_dlinear.py --dataset electricity --mode dlinear

python3 script_forecasting_dlinear.py --dataset weather --mode dlinear
python3 finetune_script_forecasting_dlinear.py --dataset weather --mode dlinear

python3 script_forecasting_dlinear.py --dataset traffic --mode dlinear
python3 finetune_script_forecasting_dlinear.py --dataset traffic --mode dlinear

python3 script_forecasting_dlinear.py --dataset  national_illness --mode dlinear --n_length 96 --batch_size 32
python3 finetune_script_forecasting_dlinear.py --dataset national_illness --mode dlinear --n_length 96 --finetune_batch_size 32 --batch_size 32

# To run the experiments on MAE without the pipeline integration on forecasting task
python3 script_forecasting_normal.py --dataset ETTh1 --mode MAE
python3 finetune_script_forecasting_normal.py --dataset ETTh1 --mode MAE

python3 script_forecasting_normal.py --dataset ETTh2 --mode MAE
python3 finetune_script_forecasting_normal.py --dataset ETTh2 --mode MAE

python3 script_forecasting_normal.py --dataset ETTm1 --mode MAE
python3 finetune_script_forecasting_normal.py --dataset ETTm1 --mode MAE

python3 script_forecasting_normal.py --dataset ETTm2 --mode MAE
python3 finetune_script_forecasting_normal.py --dataset ETTm2 --mode MAE

python3 script_forecasting_normal.py --dataset exchange_rate --mode MAE
python3 finetune_script_forecasting_normal.py --dataset exchange_rate --mode MAE

python3 script_forecasting_normal.py --dataset  WTH --mode MAE
python3 finetune_script_forecasting_normal.py --dataset WTH --mode MAE

python3 script_forecasting_normal.py --dataset electricity --mode MAE
python3 finetune_script_forecasting_normal.py --dataset electricity --mode MAE

python3 script_forecasting_normal.py --dataset weather --mode MAE
python3 finetune_script_forecasting_normal.py --dataset weather --mode MAE

python3 script_forecasting_normal.py --dataset traffic --mode MAE
python3 finetune_script_forecasting_normal.py --dataset traffic --mode MAE

python3 script_forecasting_normal.py --dataset  national_illness --mode MAE --n_length 96 --batch_size 32
python3 finetune_script_forecasting_normal.py --dataset national_illness --mode MAE --n_length 96 --finetune_batch_size 32 --batch_size 32
