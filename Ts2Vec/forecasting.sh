
# To execute the Ts2Vec model with the pipeline integration on forecasting tasks
python3 train_pipeline.py --mode ts2vec-one-loss --dataset ETTh1 --loader forecast_csv --eval
python3 train_pipeline.py --mode ts2vec-one-loss --dataset ETTh2 --loader forecast_csv --eval
python3 train_pipeline.py --mode ts2vec-one-loss --dataset ETTm1 --loader forecast_csv --eval
python3 train_pipeline.py --mode ts2vec-one-loss --dataset ETTm2 --loader forecast_csv --eval
python3 train_pipeline.py --mode ts2vec-one-loss --dataset exchange_rate --loader forecast_csv --eval
pyhton3 train_pipeline.py --mode ts2vec-one-loss --dataset WTH --loader forecast_csv --eval
pyhton3 train_pipeline.py --mode ts2vec-one-loss --dataset electricity --loader forecast_csv --eval
pyhton3 train_pipeline.py --mode ts2vec-one-loss --dataset weather --loader forecast_csv --eval
pyhton3 train_pipeline.py --mode ts2vec-one-loss --dataset traffic --loader forecast_csv --eval
pyhton3 train_pipeline.py --mode ts2vec-one-loss --dataset national_illness --loader forecast_csv --eval


# To execute the Ts2Vec original model
python3 train.py --mode ts2vec --dataset ETTh1 --loader forecast_csv --eval
python3 train.py --mode ts2vec --dataset ETTh2 --loader forecast_csv --eval
python3 train.py --mode ts2vec --dataset ETTm1 --loader forecast_csv --eval
python3 train.py --mode ts2vec --dataset ETTm2 --loader forecast_csv --eval
python3 train.py --mode ts2vec --dataset exchange_rate --loader forecast_csv --eval
pyhton3 train.py --mode ts2vec --dataset WTH --loader forecast_csv --loader forecast_csv --eval
pyhton3 train.py --mode ts2vec --dataset electricity --loader forecast_csv --eval
pyhton3 train.py --mode ts2vec --dataset weather --loader forecast_csv --eval
pyhton3 train.py --mode ts2vec --dataset traffic --loader forecast_csv --eval
pyhton3 train.py --mode ts2vec --dataset national_illness --loader forecast_csv --eval





