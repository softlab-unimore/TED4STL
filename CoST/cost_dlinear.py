import sys, math, random, copy
from typing import Union, Callable, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torch.utils.data import TensorDataset, DataLoader, Dataset

import numpy as np
from einops import rearrange, repeat, reduce

from models.encoder_dlinear import CoSTEncoderDlinear
from moving_avg_tensor_dataset import TimeSeriesDatasetWithMovingAvg
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan


def custom_collate_fn(batch, n_time_cols=7):
    # Stack della lista di tensori in un unico tensore
    data_1 = torch.stack([item[0] for item in batch], dim=0)
    data_2 = torch.stack([item[1] for item in batch], dim=0)
    total_covariate = (data_1.shape[2] - n_time_cols)//2

    result_data1_avg = torch.cat([data_1[:, :, :n_time_cols], data_1[:, :, n_time_cols:n_time_cols+total_covariate]], dim=2)
    result_data1_err = torch.cat([data_1[:, :, :n_time_cols], data_1[:, :, n_time_cols + total_covariate:]], dim=2)

    result_data2_avg = torch.cat([data_2[:, :, :n_time_cols], data_2[:, :, n_time_cols:n_time_cols + total_covariate]], dim=2)
    result_data2_err = torch.cat([data_2[:, :, :n_time_cols], data_2[:, :, n_time_cols + total_covariate:]], dim=2)

    return result_data1_avg, result_data1_err, result_data2_avg, result_data2_err

def custom_collate_fn_eval(batch, n_time_cols=7):
    # Stack della lista di tensori in un unico tensore
    data_1 = torch.stack([item[0] for item in batch], dim=0)
    total_covariate = (data_1.shape[2] - n_time_cols)//2

    result_data1_avg = torch.cat([data_1[:, :, :n_time_cols], data_1[:, :, n_time_cols:n_time_cols+total_covariate]], dim=2)
    result_data1_err = torch.cat([data_1[:, :, :n_time_cols], data_1[:, :, n_time_cols + total_covariate:]], dim=2)

    return result_data1_avg, result_data1_err

def create_custom_dataLoader(dataset, batch_size, n_time_cols=7, eval=False):
    def collate_fn(batch):
        return custom_collate_fn(batch, n_time_cols=n_time_cols)

    def collate_fn_eval(batch):
        return custom_collate_fn_eval(batch, n_time_cols=n_time_cols)

    if eval:
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_eval)

    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, drop_last=True, collate_fn=collate_fn)


class PretrainDataset(TensorDataset):

    def __init__(self,
                 data,
                 sigma,
                 p=0.5,
                 multiplier=10,
                 n_time_cols=7):
        self.data = data
        self.p = p
        self.sigma = sigma
        self.multiplier = multiplier
        self.N, self.T, self.D = data.shape # num_ts, time, dim
        self.n_time_cols = n_time_cols
        self.data1 = self.transform(self.data)
        self.data2 = self.transform(self.data)
        self.dataset1 = TimeSeriesDatasetWithMovingAvg(self.data1, n_time_cols=self.n_time_cols)
        self.dataset2 = TimeSeriesDatasetWithMovingAvg(self.data2, n_time_cols=self.n_time_cols)
        super().__init__(self.dataset1.tensors[0], self.dataset2.tensors[0])

    def __getitem__(self, item):
        ts1 = self.tensors[0][item % self.N]
        ts2 = self.tensors[1][item % self.N]
    #     return self.transform(ts), self.transform(ts)
        return ts1, ts2

    def __len__(self):
        return self.data.size(0) * self.multiplier

    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x):
        if random.random() > self.p:
            return x
        return x + (torch.randn(x.shape) * self.sigma)

    def scale(self, x):
        if random.random() > self.p:
            return x
        return x * (torch.randn(x.size(-1)) * self.sigma + 1)

    def shift(self, x):
        if random.random() > self.p:
            return x
        return x + (torch.randn(x.size(-1)) * self.sigma)


class CoSTModel(nn.Module):
    def __init__(self,
                 encoder_q: nn.Module, encoder_k: nn.Module,
                 kernels: List[int],
                 device: Optional[str] = 'cuda',
                 dim: Optional[int] = 128,
                 alpha: Optional[float] = 0.05,
                 K: Optional[int] = 65536,
                 m: Optional[float] = 0.999,
                 T: Optional[float] = 0.07):
        super().__init__()

        self.K = K
        self.m = m
        self.T = T
        self.device = device

        self.kernels = kernels

        self.alpha = alpha

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        # create the encoders
        self.head_q = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.head_k = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer('queue', F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))


    def compute_loss(self, q, k, k_negs):
        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, k_negs])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators - first dim of each batch
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = F.cross_entropy(logits, labels)

        return loss

    def convert_coeff(self, x, eps=1e-6):
        amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
        phase = torch.atan2(x.imag, x.real + eps)
        return amp, phase

    def instance_contrastive_loss(self, z1, z2):
        B, T = z1.size(0), z1.size(1)
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

    def forward(self, xq_avg, xq_err, xk_avg, xk_err):
        # compute query features
        rand_idx = np.random.randint(0, xq_avg.shape[1])

        q_t, q_s = self.encoder_q(xq_avg, xq_err)
        if q_t is not None:
            q_t = F.normalize(self.head_q(q_t[:, rand_idx]), dim=-1)

        # compute key features
        with torch.no_grad():  # no gradient for keys
            self._momentum_update_key_encoder()  # update key encoder
            k_t, k_s = self.encoder_k(xk_avg, xk_err)
            if k_t is not None:
                k_t = F.normalize(self.head_k(k_t[:, rand_idx]), dim=-1)

        loss = 0

        loss += self.compute_loss(q_t, k_t, self.queue.clone().detach())
        self._dequeue_and_enqueue(k_t)

        q_s = F.normalize(q_s, dim=-1)
        _, k_s = self.encoder_q(xk_avg, xk_err)
        k_s = F.normalize(k_s, dim=-1)

        q_s_freq = fft.rfft(q_s, dim=1)
        k_s_freq = fft.rfft(k_s, dim=1)
        q_s_amp, q_s_phase = self.convert_coeff(q_s_freq)
        k_s_amp, k_s_phase = self.convert_coeff(k_s_freq)

        seasonal_loss = self.instance_contrastive_loss(q_s_amp, k_s_amp) + \
                        self.instance_contrastive_loss(q_s_phase,k_s_phase)
        loss += (self.alpha * (seasonal_loss/2))

        return loss

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update for key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        # replace keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr


class CoSTDlinear:
    def __init__(self,
                 input_dims: int,
                 kernels: List[int],
                 alpha: bool,
                 max_train_length: int,
                 output_dims: int = 320,
                 hidden_dims: int = 64,
                 depth: int = 10,
                 device: 'str' ='cuda',
                 lr: float = 0.001,
                 batch_size: int = 16,
                 after_iter_callback: Union[Callable, None] = None,
                 after_epoch_callback: Union[Callable, None] = None,
                 n_time_cols=7):

        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.n_time_cols = n_time_cols

        if kernels is None:
            kernels = []

        self.net = CoSTEncoderDlinear(
            input_dims=input_dims, output_dims=output_dims,
            kernels=kernels,
            length=max_train_length,
            hidden_dims=hidden_dims, depth=depth,
        ).to(self.device)

        self.cost = CoSTModel(
            self.net,
            copy.deepcopy(self.net),
            kernels=kernels,
            dim=self.net.component_dims,
            alpha=alpha,
            K=256,
            device=self.device,
        ).to(self.device)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0

    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        assert train_data.ndim == 3

        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
                
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        multiplier = 1 if train_data.shape[0] >= self.batch_size else math.ceil(self.batch_size / train_data.shape[0])
        train_dataset = PretrainDataset(torch.from_numpy(train_data).to(torch.float), sigma=0.5, multiplier=multiplier, n_time_cols=self.n_time_cols)
        # train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        train_loader = create_custom_dataLoader(train_dataset, self.batch_size, n_time_cols=self.n_time_cols)

        optimizer = torch.optim.SGD([p for p in self.cost.parameters() if p.requires_grad],
                                    lr=self.lr,
                                    momentum=0.9,
                                    weight_decay=1e-4)
        
        loss_log = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            # for batch in train_loader
            for xq_avg, xq_err, xk_avg, xk_err in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                # x_q, x_k = map(lambda x: x.to(self.device), batch)
                if self.max_train_length is not None and xq_avg.size(1) > self.max_train_length and xk_avg.size(1) > self.max_train_length \
                        and xq_err.size(1) > self.max_train_length and xk_err.size(1) > self.max_train_length:
                    window_offset = np.random.randint(xq_avg.size(1) - self.max_train_length + 1)
                    xq_avg = xq_avg[:, window_offset : window_offset + self.max_train_length]
                    xq_err = xq_err[:, window_offset : window_offset + self.max_train_length]
                    xk_avg = xk_avg[:, window_offset : window_offset + self.max_train_length]
                    xk_err = xk_err[:, window_offset : window_offset + self.max_train_length]

                optimizer.zero_grad()

                xq_avg = xq_avg.to(self.device)
                xq_err = xq_err.to(self.device)
                xk_avg = xk_avg.to(self.device)
                xk_err = xk_err.to(self.device)

                loss = self.cost(xq_avg, xq_err, xk_avg, xk_err)

                loss.backward()
                optimizer.step()

                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

                if n_iters is not None:
                    adjust_learning_rate(optimizer, self.lr, self.n_iters, n_iters)

                # break # only one iteration
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

            if n_epochs is not None:
                adjust_learning_rate(optimizer, self.lr, self.n_epochs, n_epochs)

            # break # only one epoch
            
        return loss_log
    
    def _eval_with_pooling(self, x_avg, x_err, mask=None, slicing=None, encoding_window=None):
        out_t, out_s = self.net(x_avg.to(self.device, non_blocking=True), x_err.to(self.device, non_blocking=True))  # l b t d
        out = torch.cat([out_t[:, -1], out_s[:, -1]], dim=-1)
        return rearrange(out.cpu(), 'b d -> b () d')
    
    def encode(self, data, mode, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
        if mode == 'forecasting':
            encoding_window = None
            slicing = None
        else:
            raise NotImplementedError(f"mode {mode} has not been implemented")

        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        dataset = TimeSeriesDatasetWithMovingAvg(torch.from_numpy(data).to(torch.float), n_time_cols=self.n_time_cols)
        # loader = DataLoader(dataset, batch_size=batch_size)
        loader = create_custom_dataLoader(dataset, batch_size, n_time_cols=self.n_time_cols, eval=True)

        with torch.no_grad():
            output = []
            for x, y in loader:
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer1 = []
                        calc_buffer2 = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        y_sliding = torch_pad_nan(
                            x[:, max(l, 0): min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer1, dim=0),
                                    torch.cat(calc_buffer2, dim=0),
                                    mask,
                                    slicing=slicing,
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer1 = []
                                calc_buffer2 = []
                                calc_buffer_l = 0
                            calc_buffer1.append(x_sliding)
                            calc_buffer2.append(y_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                y_sliding,
                                mask,
                                slicing=slicing,
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer1, dim=0),
                                torch.cat(calc_buffer2, dim=0),
                                mask,
                                slicing=slicing,
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer1 = []
                            calc_buffer2 = []
                            calc_buffer_l = 0
                    
                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, y, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)


def adjust_learning_rate(optimizer, lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
