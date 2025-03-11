import argparse
import json

import torch.cuda

from ts2vec import TS2Vec
import datautils
import utils
import os
import time
import datetime
from tasks.forecasting import eval_forecasting
from ts2vec_ablation import TS2VecAblation
from ts2vec_dlinear import TS2VecDlinear


def create_model(type_of_train, dim, n_time_cols, current_device, configuration):
    if 'ts2vec-dlinear' in type_of_train.lower():
        return TS2VecDlinear(input_dims=dim, device=current_device, mode=type_of_train, n_time_cols=n_time_cols, **configuration)
    return TS2Vec(input_dims=dim, device=current_device, mode=type_of_train, n_time_cols=n_time_cols, **configuration)

if __name__ == "__main__":
    config = argparse.ArgumentParser()
    config.add_argument('--dataset', type=str, default='ETTh1', help='The dataset name')
    config.add_argument('--run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    config.add_argument('--mode', type=str, default='ts2vec-Dlinear-one-loss')
    config.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    config.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    config.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    config.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    config.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    config.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    config.add_argument('--iters', type=int, default=None, help='The number of iterations')
    config.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    config.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    config.add_argument('--seed', type=int, default=None, help='The random seed')
    config.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    config.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    config.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    config.add_argument('--ci', action='store_true', default=False)
    config.add_argument('--short-term', action='store_true', default=False)
    args = config.parse_args()

    # set GPU
    device = utils.init_dl_program(0, seed=42, max_threads=8)

    print("-------------- LOAD DATASET: PREPROCESSING ------------------------")

    if args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_time_cols = datautils.load_forecast_csv(args.dataset, short_term=args.short_term)
        train_data = data[:, train_slice]
    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_time_cols = datautils.load_forecast_csv(args.dataset, short_term=args.short_term, univar=True)
        train_data = data[:, train_slice]
    else:
        raise ValueError(f"Unknown loader {args.loader}.")

    print("Data after StandarScaler on n_coviariate_cols and original features")
    print(data.shape)
    print("train:", train_slice)
    print("valid: ", valid_slice)
    print("test:", test_slice)
    print("pred_lens: ", pred_lens)
    print("n_time_cols:", n_time_cols)
    print('------------')
    print(scaler)

    train_data = data[:, train_slice]
    valid_data = data[:, valid_slice]
    test_data = data[:, test_slice]
    print(train_data.shape)

    #Creation of dirs to store results
    run_dir = f'{args.path}/training/forecasting/{args.mode}/{args.dataset}__ {utils.name_with_datetime("forecast_multivar")}'
    os.makedirs(run_dir, exist_ok=True)


    print("\n------------------- TRAINING ENCODER -------------------\n")

    if not args.ci:
        input_dim = train_data.shape[-1]
        # input_dim = train_data.shape[-1] - n_time_cols
        if args.mode == 'feature':
            input_dim = train_data.shape[-1] + train_data.shape[-1] - n_time_cols

        config = dict(
            batch_size=args.batch_size,
            lr=args.lr,
            output_dims=args.repr_dims,
            max_train_length=args.max_train_length,
            ci=args.ci
        )

        # Train a TS2Vec model
        model = create_model(args.mode, input_dim, n_time_cols, device, config)
        # model = TS2VecAblation(input_dims=input_dim, device=device, mode='ts2vec-ablation-err', n_time_cols=n_time_cols, **config)
    else:
        config = dict(
            batch_size=4,
            lr=args.lr,
            output_dims=32,
            max_train_length=args.max_train_length,
            ci=args.ci
        )

        # Train a TS2Vec model
        model = create_model(args.mode, 1, n_time_cols, device, config)

    t = time.time()

    loss_log = model.fit(
        train_data,
        n_epochs=None,
        n_iters=None,
        verbose=True
    )
    model.save(f'{run_dir}/model_avg.pkl', f'{run_dir}/model_err.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        print("\n----------------- EVAL FORECASTING -------------------\n")

        out, eval_res = eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_time_cols)

        print("\n----------------- FINAL RESULTS --------------------\n")

        utils.pkl_save(f'{run_dir}/out.pkl', out)
        utils.pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        with open(f'{run_dir}/eval_res.json', 'w') as json_file:
            json.dump(eval_res, json_file, indent=4)

        print('Evaluation result:', eval_res)


    torch.cuda.empty_cache()

    print("Finished")
