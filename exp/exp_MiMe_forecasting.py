from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
from utils.dtw_metric import accelerated_dtw
import random

warnings.filterwarnings('ignore')

class exp_MiMe_forcasting(Exp_Basic):
    def __init__(self, args):
        super(exp_MiMe_forcasting, self).__init__(args)
        self.model_num = len(self.args.model_gins)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss(reduction='none')
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        f_dim = -1 if self.args.features == 'MS' else 0

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                all_loss = 0
                for j in range(self.model_num):

                    # forward
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, j)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, j)

                    outputs = outputs[:, -self.args.pred_len:, f_dim:]

                    loss_matrix = criterion(outputs, batch_y_target)
                    dim_loss = loss_matrix.mean().detach()
                    all_loss += dim_loss.detach()

                total_loss.append(all_loss.detach().cpu().numpy())

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        print("shuffle_together")

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        criterion = self._select_criterion()
        model_opts = self._select_optimizer()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler

        f_dim = -1 if self.args.features == 'MS' else 0

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()

            epoch_time = time.time()

            for j in range(self.model_num):
                print(f"the {j}th shuffle")
                fix_seed = 2021 + j
                random.seed(fix_seed)
                torch.manual_seed(fix_seed)
                np.random.seed(fix_seed)

                train_data, train_loader = self._get_data(flag='train')

                total_loss = 0

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    if i % self.model_num == 0:
                        self.model.zero_grad()
                        total_loss = 0

                    iter_count += 1
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x.clone(), batch_x_mark.clone(), dec_inp.clone(),
                                                 batch_y_mark.clone(), i % self.model_num)
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            loss = criterion(outputs, batch_y.clone())
                            loss = loss.mean()
                            total_loss = total_loss + loss
                    else:
                        outputs = self.model(batch_x.clone(), batch_x_mark.clone(), dec_inp.clone(),
                                             batch_y_mark.clone(), i % self.model_num)
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y.clone())
                        loss = loss.mean()
                        total_loss = total_loss + loss

                    if (i + 1) % self.model_num == 0:
                        if self.args.use_amp:
                            scaler.scale(total_loss).backward()
                            scaler.step(model_opts)
                            scaler.update()
                        else:
                            total_loss.backward()
                            model_opts.step()
                        train_loss.append(total_loss.item())

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_opts, epoch + 1, self.args)

        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))
        #
        # return self.model

        return

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='online_test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(
                os.path.join('./ckpt/checkpoint', f'{self.args.data_path.split("/")[0]}_{self.args.pred_len}',
                             'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        criterion = self._select_criterion()
        model_opts = self._select_optimizer()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler

        assembly = OGD(next(iter(test_loader))[1].shape[-1], self.model_num, self.device, self.args.eta_factor,
                       self.args.lamdba_factor)

        f_dim = -1 if self.args.features == 'MS' else 0
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            self.model.zero_grad()
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            all_outputs = []
            self.model.eval()
            with torch.no_grad():
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                for j in range(self.model_num):
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x.clone(), batch_x_mark.clone(), dec_inp.clone(),
                                                 batch_y_mark.clone(), j)
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    else:
                        outputs = self.model(batch_x.clone(), batch_x_mark.clone(), dec_inp.clone(),
                                             batch_y_mark.clone(), j)
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]

                    outputs = outputs[:, -self.args.pred_len:, :]
                    outputs = outputs.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = batch_y.shape
                        if outputs.shape[-1] != batch_y.shape[-1]:
                            outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                        outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(
                            shape)

                    outputs = outputs[:, :, f_dim:]
                    all_outputs.append(outputs)
                    torch.cuda.empty_cache()
                assembly.add_true(batch_y)

            self.model.train()
            for j in range(self.model_num):
                self.model.zero_grad()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x.clone(), batch_x_mark.clone(), dec_inp.clone(),
                                             batch_y_mark.clone(), j)
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y.clone()).mean()
                else:
                    outputs = self.model(batch_x.clone(), batch_x_mark.clone(), dec_inp.clone(),
                                         batch_y_mark.clone(), j)
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y.clone()).mean()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_opts)
                    scaler.update()
                else:
                    loss.backward()
                    model_opts.step()

            if test_data.scale and self.args.inverse:
                shape = batch_y.shape
                batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
            batch_y = batch_y.detach().cpu().numpy()
            batch_y = batch_y[:, :, f_dim:]

            pred = assembly.get_assembly_output(all_outputs)
            true = batch_y

            preds.append(pred)
            trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('mse:{}, mae:{}, dtw:{}\n'.format(mse, mae, dtw))
        self.append_mse_mae_to_csv(self.args, mae, mse, "result_mse_mae.csv")

        f.write('mse:{}, mae:{}, dtw:{}\n'.format(mse.mean(), mae.mean(), dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return

    def append_mse_mae_to_csv(self, args, mae, mse, csv_file='experiments_metrics.csv'):

        new_rows = [[args.model, args.data_path.split('\\')[0], args.pred_len, mse, mae]]
        columns = ['model', 'data', 'pred_len', 'mse', 'mae']
        new_df = pd.DataFrame(new_rows, columns=columns)

        if os.path.exists(csv_file):
            existing_df = pd.read_csv(csv_file)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        combined_df.to_csv(csv_file, index=False)
        return


class assembly_base:
    def __init__(self):
        pass

    def select_result(self, all_outputs, select_model):
        # [num_models, B, pred_len, num_features]
        all_outputs = np.stack(all_outputs, axis=0)
        B, pred_len, num_features = all_outputs.shape[1:]  # batch_size, pred_len, num_features
        selected_outputs = np.zeros((B, pred_len, num_features), dtype=all_outputs.dtype)
        for j in range(num_features):
            m_idx = select_model[j]
            selected_outputs[:, :, j] = all_outputs[m_idx, :, :, j]
        return selected_outputs

    def weight_result(self, all_outputs, weight):
        all_outputs = torch.stack([torch.tensor(o, device=self.device) for o in all_outputs], dim=0)
        selected_outputs = (weight.unsqueeze(1).unsqueeze(1) * all_outputs).sum(dim=0).detach().cpu().numpy()
        return selected_outputs


class OGD(assembly_base):
    def __init__(self, num_features, model_num, device, eta_factor, lamdba_factor):
        """
        x_init: torch.Tensor, initial point x_1 ∈ K
        step_sizes: list or callable, η_t
        projection_fn: function implementing Π_K
        """
        self.x = torch.distributions.Dirichlet(torch.ones(model_num, device=device)).sample(
            (num_features,))  # (num_features, model_num)
        self.projection_fn = self._projection
        self.device = device
        self.eta_factor = eta_factor
        self.lamdba_factor = lamdba_factor
        self.model_num = model_num
        self.t = 1

    def _projection(self, y: torch.Tensor, eps: float = 1e-12):
        """
        Euclidean projection onto the probability simplex.

        Args:
            y: Tensor of shape (..., d)
            eps: numerical stability

        Returns:
            x: projected tensor, same shape as y
        """
        # sort in descending order
        sorted_y, _ = torch.sort(y, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_y, dim=-1)

        # create index 1,2,...,d
        dim = y.size(-1)
        rho = torch.arange(1, dim + 1, device=y.device, dtype=y.dtype)

        # determine support
        support = sorted_y - (cumsum - 1) / rho > 0
        support_size = support.sum(dim=-1, keepdim=True)

        # compute tau
        tau = (cumsum.gather(
            -1, support_size - 1
        ) - 1) / support_size

        # projection
        x = torch.clamp(y - tau, min=0.0)
        return x

    def add_true(self, true):
        self.true = true.detach()

    def _update_weight(self, output):
        M = output.squeeze().permute(2, 1, 0)  # (model_num,B , L, num_features) -> (num_features, L, model_num)

        x = self.x  # (num_features, model_num)
        y = self.true.squeeze().permute(1, 0)  # (num_features, L)

        x_ = x.unsqueeze(-1)  # (num_features, model_num, 1)
        y_ = y.unsqueeze(-1)  # (num_features, L, 1)

        # lambda_ = 1e-6
        lamdba_ = self.lamdba_factor
        # grad = 2 * torch.bmm(M.transpose(1, 2), M @ x_ - y_) + lambda_ * x_ # (num_features, model_num,1)

        num_features, L, model_num = M.shape
        I_block = torch.eye(model_num, device=M.device, dtype=M.dtype)  # (model_num, model_num)
        I_block = I_block.unsqueeze(0).repeat(num_features, 1, 1) * lamdba_  # (num_features, model_num, model_num)
        M = torch.cat([M, I_block], dim=1)
        zero_block = torch.zeros(num_features, model_num, 1, device=y_.device, dtype=y_.dtype)
        y_ = torch.cat([y_, zero_block], dim=1)  # (num_features, L+model_num, 1)
        grad = 2 * torch.bmm(M.transpose(1, 2), M @ x_ - y_)  # (num_features, model_num,1)

        p = grad * self.eta_factor / self.t
        with torch.no_grad():
            self.x = self.projection_fn((x_ - p).squeeze())
        self.t = self.t + 1

    def get_assembly_output(self, all_outputs):
        """
        Perform one OGD update:
        x_t = Π_K(x_{t-1} - η_t ∇f_{t-1}(x_{t-1}))

        loss_fn: f_{t-1}(x), callable
        """
        all_outputs = torch.stack([torch.tensor(o, device=self.device) for o in all_outputs],
                                  dim=0)  # (model_num, B, L, num_features)
        selected_outputs = (self.x.T.unsqueeze(1).unsqueeze(1) * all_outputs).sum(dim=0).detach().cpu().numpy()
        save_usage_matrix_split(self.x.detach().cpu(), "weight/OGD", "usage")

        self._update_weight(all_outputs.detach())
        return selected_outputs


def save_usage_matrix_split(
        usage_matrix,
        dir_path="./usage_logs",
        prefix="train_usage"
):
    usage_matrix = np.asarray(usage_matrix)
    assert usage_matrix.ndim == 2, "usage_matrix must has two dimenstions (m, n)"

    m, n = usage_matrix.shape

    os.makedirs(dir_path, exist_ok=True)

    columns = [f"var_{i}" for i in range(n)]

    for i in range(m):
        row = usage_matrix[i:i + 1]  # shape (1, n)
        df = pd.DataFrame(row, columns=columns)

        path = os.path.join(dir_path, f"{prefix}_{i}.csv")

        if not os.path.exists(path):
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, mode="a", header=False, index=False)

    mean_row = usage_matrix.mean(axis=0, keepdims=True)  # shape (1, n)
    df_mean = pd.DataFrame(mean_row, columns=columns)

    mean_path = os.path.join(dir_path, f"{prefix}_mean.csv")

    if not os.path.exists(mean_path):
        df_mean.to_csv(mean_path, index=False)
    else:
        df_mean.to_csv(mean_path, mode="a", header=False, index=False)
