import json

import torch
import random
import numpy as np
import argparse
import os
import time
import datetime
import tasks
import datautils
import utils
from simts_dlinear import SimTSDlinear
from utils import init_dl_program, pkl_save


def save_checkpoint_callback(
        save_every=1,
        unit='epoch'
):
    assert unit in ('epoch', 'iter')

    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')

    return callback


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='electricity', help='The dataset name')
    parser.add_argument('--kernels', type=int, nargs='+',
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 1, 2, 4, 8, 16, 32, 64, 128, 256],
                        help='The kernel sizes used in the mixture of AR expert layers')
    parser.add_argument('--mode', type=str, default='dlinear', help='The mode used for training')
    parser.add_argument('--mask_dir', default=None, help='directory to dynamask dataset')
    parser.add_argument('--dir', default='training/forecasting',
                        help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, default='forecast_csv',
                        help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0,
                        help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000,
                        help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--K', type=int, default=201, help='K in the paper')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None,
                        help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=42, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None,
                        help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--experiment', type=str, default='base',
                        help='Type of experiment to perform (defaluts to base)')
    parser.add_argument('--area', type=int, default=None, help='Type of experiment to perform (defaluts to base)')
    parser.add_argument('--mix', action="store_true", help='Whether to perform mix')
    parser.add_argument('--short_term', action="store_true", help='Whether to perform short term forecasting')
    parser.add_argument('--kernel_size', default=25, type=int, help='The kernel size used in the moving average')
    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    else:
        device = 'cpu'

    print('Loading data... ', end='')

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

    print('done', train_data.shape)

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        kernel_list=args.kernels,
        output_dims=args.repr_dims,
        raw_length=train_data.shape[1],
        K=args.K,
        n_time_cols=n_time_cols if task_type == 'forecasting' else 0,
        max_train_length=args.max_train_length,
        experiment=args.experiment,
        experiment_args={'sigma': 0.5},
        mix=args.mix,
        task_type=task_type,
        kernel_size=args.kernel_size
    )

    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    if task_type == 'forecasting':
        run_dir = './' + args.dir + f'/B{args.batch_size}_E{args.repr_dims}/' + args.mode + '/' + args.dataset + '__' + utils.name_with_datetime('forecast_multivar')
    else:
        assert False
    os.makedirs(run_dir, exist_ok=True)

    t = time.time()
    print("train_data size:", train_data.shape)

    model = SimTSDlinear(
        input_dims=train_data.shape[-1],
        device=device,
        **config
    )

    loss_log, best_net_avg, best_net_err = model.fit(
        train_data,
        # pert_data=None,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )

    # model.net = best_net
    model.save(f'{run_dir}/model_avg.pkl', f'{run_dir}/model_err.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        if task_type == 'forecasting':
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_time_cols, run_dir)
        else:
            assert False
        pkl_save(f'{run_dir}/out.pkl', out)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        print('Evaluation result:', eval_res)

        with open(f'{run_dir}/eval_res.json', 'w') as json_file:
            json.dump(eval_res, json_file, indent=4)

    print("Finished.")
