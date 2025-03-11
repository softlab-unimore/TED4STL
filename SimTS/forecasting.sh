
# To run SimTS model with pipeline integration on forecastin task
python3 train_pipeline.py --dataset ETTh1 --eval
python3 train_pipeline.py --dataset ETTh2 --eval
python3 train_pipeline.py --dataset ETTm1 --eval
python3 train_pipeline.py --dataset ETTm2 --eval
python3 train_pipeline.py --dataset exchange_rate --eval
python3 train_pipeline.py --dataset WTH --eval
python3 train_pipeline.py --dataset electricity --eval
python3 train_pipeline.py --dataset weather --eval
python3 train_pipeline.py --dataset traffic --eval
python3 train_pipeline.py --dataset national_illness --eval

# To run original SimTS model on forecasting task
python3 train_simts.py --dataset ETTh1 --eval
python3 train_simts.py --dataset ETTh2 --eval
python3 train_simts.py --dataset ETTm1 --eval
python3 train_simts.py --dataset ETTm2 --eval
python3 train_simts.py --dataset exchange_rate --eval
python3 train_simts.py --dataset WTH --eval
python3 train_simts.py --dataset electricity --eval
python3 train_simts.py --dataset weather --eval
python3 train_simts.py --dataset traffic --eval
python3 train_simts.py --dataset national_illness --eval

