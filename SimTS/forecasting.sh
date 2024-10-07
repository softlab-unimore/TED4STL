
# To run SimTS model with pipeline integration on forecastin task
python3 script.py --dataset ETTh1 --eval
python3 script.py --dataset ETTm1 --eval
python3 script.py --dataset WTH --eval
python3 script.py --dataset electricity --eval
python3 script.py --dataset traffic --eval
python3 script.py --dataset national_illness --eval

# To run original SimTS model on forecasting task
python3 train_simts.py --dataset ETTh1 --eval
python3 train_simts.py --dataset ETTm1 --eval
python3 train_simts.py --dataset WTH --eval
python3 train_simts.py --dataset electricity --eval
python3 train_simts.py --dataset traffic --eval
python3 train_simts.py --dataset national_illness --eval

