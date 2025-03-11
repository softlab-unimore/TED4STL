
# To execute the Ts2Vec model with the pipeline integration on forecasting tasks
python3 script_forecasting.py --mode ts2vec-one-loss --dataset ETTh1 --eval
python3 script_forecasting.py --mode ts2vec-one-loss --dataset ETTh2 --eval
python3 script_forecasting.py --mode ts2vec-one-loss --dataset ETTm1 --eval
python3 script_forecasting.py --mode ts2vec-one-loss --dataset ETTm2 --eval
python3 script_forecasting.py --mode ts2vec-one-loss --dataset exchange_rate --eval
pyhton3 script_forecasting.py --mode ts2vec-one-loss --dataset WTH --eval
pyhton3 script_forecasting.py --mode ts2vec-one-loss --dataset electricity --eval
pyhton3 script_forecasting.py --mode ts2vec-one-loss --dataset weather --eval
pyhton3 script_forecasting.py --mode ts2vec-one-loss --dataset traffic --eval
pyhton3 script_forecasting.py --mode ts2vec-one-loss --dataset national_illness --eval


# To execute the Ts2Vec original model
python3 script_forecasting.py --mode ts2vec --dataset ETTh1 --eval
python3 script_forecasting.py --mode ts2vec --dataset ETTh2 --eval
python3 script_forecasting.py --mode ts2vec --dataset ETTm1 --eval
python3 script_forecasting.py --mode ts2vec --dataset ETTm2 --eval
python3 script_forecasting.py --mode ts2vec --dataset exchange_rate --eval
pyhton3 script_forecasting.py --mode ts2vec --dataset WTH --eval
pyhton3 script_forecasting.py --mode ts2vec --dataset electricity --eval
pyhton3 script_forecasting.py --mode ts2vec --dataset weather --eval
pyhton3 script_forecasting.py --mode ts2vec --dataset traffic --eval
pyhton3 script_forecasting.py --mode ts2vec --dataset national_illness --eval





