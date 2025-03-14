import argparse
import json
import math
import time
import datetime

from tqdm import tqdm
from models.model import *
from datautils import load_forecast_csv
from save_model import save_model
from moving_avg_tensor_dataset import TimeSeriesDatasetWithMovingAvg
from models.model_dlinear import MAE_ViT_Dlinear
import random

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def custom_collate_fn(batch, n_time_cols=7):
    # Stack della lista di tensori in un unico tensore
    data = torch.stack([item[0] for item in batch], dim=0)
    total_covariate = (data.shape[2] - n_time_cols)//2

    result_data_avg = torch.cat([data[:, :, :n_time_cols], data[:, :, n_time_cols:n_time_cols+total_covariate]], dim=2)
    result_data_err = torch.cat([data[:, :, :n_time_cols], data[:, :, n_time_cols + total_covariate:]], dim=2)
    return result_data_avg, result_data_err

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="MAE")

    parser.add_argument('--dataset', default='ETTh1', type=str)
    parser.add_argument('--mode', default='dlinear', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_device_batch_size', default=64, type=int)
    parser.add_argument('--base_learning_rate', default=1.5e-4, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--total_epoch', default=200, type=int)
    parser.add_argument('--warmup_epoch', default=5, type=int)
    parser.add_argument('--emb_dim', default=64, type=int)
    parser.add_argument('--model_path', default='training', type=str)
    parser.add_argument('--n_length', default=336, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--labelled_ratio', default=0.1, type=float)
    parser.add_argument('--finetune_batch_size', default=64, type=int)
    parser.add_argument('--finetune_base_learning_rate', default=0.001, type=float)
    parser.add_argument('--finetune_epochs', default=31, type=int)
    parser.add_argument('--finetune_seed', default=42, type=int)
    parser.add_argument('--pretrain', default=True, type=bool)
    parser.add_argument('--short_term', action='store_true', default=False)
    parser.add_argument('--kernel_size', default=25, type=int)

    args = parser.parse_args()
    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'mps'
    print("Device:", device)

    # Load dataset
    print("Dataset:", args.dataset)

    print("-------------- LOAD DATASET: PREPROCESSING ------------------------")

    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_time_cols = load_forecast_csv(args.dataset, args.short_term)
    dataset_name = args.dataset

    train_data = data[:, train_slice]
    vali_data = data[:, valid_slice]
    test_data = data[:, test_slice]

    for pred_len in pred_lens:
        print('----------------------------------------')
        print('Start training for pred_len:', pred_len)
        print('----------------------------------------')

        train_dataset = TimeSeriesDatasetWithMovingAvg(torch.from_numpy(train_data).to(torch.float), n_time_cols=n_time_cols, seq_len=args.n_length, pred_len=pred_len, kernel_size=args.kernel_size)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=load_batch_size,
            shuffle=True,
            # collate_fn=custom_collate_fn
        )

        model = MAE_ViT_Dlinear(
            sample_shape=[train_data.shape[0]*train_data.shape[2], args.n_length],
            patch_size=(train_data.shape[0]*train_data.shape[2], args.patch_size),
            mask_ratio=args.mask_ratio
        ).to(device)

        optim = torch.optim.AdamW(
            model.parameters(),
            lr=args.base_learning_rate * args.batch_size / 256,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay
        )

        lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

        t = time.time()

        step_count = 0
        optim.zero_grad()
        min_loss = 100
        for e in range(args.total_epoch):
            model.train()
            print('===== start training ======')
            losses = []
            # for sample, label in tqdm(iter(train_loader)):
            for sample_avg, sample_err, _ in tqdm(iter(train_loader)):
                step_count += 1

                sample_avg = sample_avg.swapaxes(1, 2)
                sample_avg = np.expand_dims(sample_avg, axis=1)

                sample_err = sample_err.swapaxes(1, 2)
                sample_err = np.expand_dims(sample_err, axis=1)

                sample_avg = torch.tensor(sample_avg, dtype=torch.float32).to(device)
                sample_err = torch.tensor(sample_err, dtype=torch.float32).to(device)


                predicted_sample, mask = model(sample_avg, sample_err)
                loss = torch.mean((predicted_sample - (sample_avg + sample_err)) ** 2 * mask) / args.mask_ratio
                loss.backward()
                if step_count % steps_per_update == 0:
                    optim.step()
                    optim.zero_grad()
                losses.append(loss.item())
            lr_scheduler.step()
            avg_loss = sum(losses) / len(losses)
            print(f'In epoch {e}, average traning loss is {avg_loss}.')

            ''' save pre-trained model '''
            if avg_loss < min_loss:
                min_loss = avg_loss
                save_model(args, model, args.mode, pred_len, optim)
                print("Model update with loss {}.".format(min_loss))

        t = time.time() - t

        print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
        with open(
                f'./{args.model_path}/forecasting/B{args.batch_size}_E{args.emb_dim}/{args.mode}/Pretrained_{args.dataset}_{args.emb_dim}_{pred_len}/exec_time.json',
                'w') as json_file:
            json.dump({'exec_time': t, 'exec_time_formatted': f'{datetime.timedelta(seconds=t)}'}, json_file, indent=4)

    print("Finished")