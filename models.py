import torch
import random
import matplotlib.pyplot as plt
from darts.models.forecasting.dlinear import DLinearModel
from darts.models.forecasting.auto_arima import AutoARIMA
from statsmodels.tsa.ar_model import AutoReg
from darts import TimeSeries
from sklearn.linear_model import LinearRegression
import numpy as np 
import time
import pandas as pd
from tqdm import tqdm
# from DLinear.models.DLinear import Model as DLinearRaw
from dataclasses import dataclass
from typing import List
import copy

class TSForecaster:
    def __init__(self, input_length, output_length) -> None:
        self.input_length = input_length
        self.output_length = output_length

    def fit_loader(self, train_loader, val_loader, lr, max_epochs):
        raise NotImplementedError()

    def fit_raw(self, scaled_train_df, scaled_val_df):
        raise NotImplementedError()

    def predict(self, input) -> torch.Tensor:
        raise NotImplementedError()
    
    def test_loss_acc(self, test_ds, skip_prop = 0.0, skip_seed = 43):
        mses = list()
        maes = list()
        random.seed(skip_seed)
        for input, truth in tqdm(test_ds):
            if random.random() < skip_prop:
                mses.append(np.nan)
                maes.append(np.nan)
                continue
            preds = self.predict(input)
            mses.append(((preds - truth)**2).mean())
            maes.append(torch.abs(preds - truth).mean())

        return mses, maes
    
    def test_loss_acc_loader(self, test_loader, device, skip_prop = 0.0, skip_seed = 43, progress_bar=True):
        mses = list()
        maes = list()
        random.seed(skip_seed)
        with torch.no_grad():
            for input, truth in (tqdm(test_loader) if progress_bar else test_loader):
                if random.random() < skip_prop:
                    mses.extend([np.nan] * input.shape[0])
                    maes.extend([np.nan] * input.shape[0])
                    continue
                truth = truth.to(device).view(truth.shape[0], -1)
                preds = self.predict(input.to(device))
                preds = preds.view(-1, self.output_length)
                mses.extend(((preds - truth)**2).mean(dim=-1).tolist())
                maes.extend(torch.abs(preds - truth).mean(dim=-1).tolist())

        return mses, maes
    
    def plot_predictions(self, ts_input, truth):
        with torch.no_grad():
            preds = self.predict(ts_input).cpu()
            ts_input = ts_input.cpu()
            truth = truth.cpu()
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(ts_input.flatten(), label="input")
            ax.plot(range(self.input_length + 1, self.input_length + self.output_length + 1), truth.flatten(), label="truth", alpha=0.5)
            ax.plot(range(self.input_length + 1, self.input_length + self.output_length + 1), preds.flatten(), label="forecast")
            ax.legend()

class GlobalMean(TSForecaster):

    def fit_raw(self, scaled_train_df, scaled_val_df):
        all_df = pd.concat([scaled_train_df, scaled_val_df])
        self.mean = all_df.to_numpy().mean()

    def predict(self, input):
        return torch.tensor([self.mean] * self.output_length)



class MovingAverage(TSForecaster):
    def fit_raw(self, scaled_train_df, scaled_val_df):
        all_df = pd.concat([scaled_train_df, scaled_val_df])
        pred_df = all_df.copy()
        self.mses = list()
        for window_length in tqdm(range(1, self.input_length + 1)):
            total_mse = 0
            for output_step in range(1, self.output_length + 1):
                if output_step == 1:
                    pred_df = all_df.rolling(window=window_length).mean().shift(output_step)
                else:
                    pred_df = (((self.input_length + 1) * pred_df - all_df.shift(self.input_length))/self.input_length).shift(1)

                mse = np.nanmean(((
                    all_df.iloc[self.input_length + output_step:-(self.output_length - output_step) or None] - 
                    pred_df[self.input_length + output_step:-(self.output_length - output_step) or None]
                )**2).to_numpy())
                total_mse += mse
            total_mse /= self.output_length
            self.mses.append(total_mse)

        self.window_size = np.argmin(self.mses) + 1
        print(f"Selected Window Size: {self.window_size}")

    def predict(self, input) -> torch.Tensor:
        preds = list()
        for output_step in range(1, self.output_length + 1):
            window_sum = sum(preds[-self.window_size:]) + input.flatten()[-(self.window_size - output_step + 1):0 if output_step > self.window_size else None].sum()
            preds.append(window_sum/self.window_size)
        return torch.tensor(preds)

# Deprecated
class AR(TSForecaster):
    def __init__(self, input_length, output_length) -> None:
        super().__init__(input_length, output_length)
        raise NotImplementedError("This class is deprecated")
    def fit_raw(self, scaled_train_df, scaled_val_df, val_ds):
        train_series = np.transpose(scaled_train_df.to_numpy()).flatten()
        # val_inputs = np.array([x for x, y in val_ds]).reshape(-1, self.input_length)
        # val_truths = np.array([y for x, y in val_ds]).reshape(-1, self.output_length)
        self.mses = list()
        for lags in tqdm(range(1, self.input_length + 1)):
            self.model = AutoReg(
                endog=train_series,
                lags=lags,
                trend='c'
            ).fit()

            sse = 0

            for series in range(scaled_val_df.shape[-1]):
                model_newdata = self.model.apply(scaled_val_df.iloc[:, series].to_numpy(), refit=False)
                preds = model_newdata.predict()
                sse += np.nanmean((preds - scaled_val_df.iloc[:, series].to_numpy())**2)

            # self.lags = lags
            # mse, mae = self.test_loss_acc(None, input_array=val_inputs, truth_array=val_truths)
            mse = sse/scaled_val_df.shape[-1]

            self.mses.append(mse)

        self.fit_preset(pd.concat((scaled_train_df, scaled_val_df)), np.argmin(self.mses) + 1)

    def fit_preset(self, scaled_train_df, lags):
        self.lags = lags
        train_series = np.transpose(scaled_train_df.to_numpy()).flatten()
        self.model = AutoReg(
                endog=train_series,
                lags=self.lags,
                trend='c'
        ).fit()
        print(f"Selected Lags: {self.lags}")

    def test_loss_acc(self, test_ds, skip_prop=0, skip_seed=43, input_array=None, truth_array=None):
        test_inputs = np.array([x for x, y in test_ds]).reshape(-1, self.input_length) if input_array is None else input_array
        test_truths = np.array([y for x, y in test_ds]).reshape(-1, self.output_length) if truth_array is None else truth_array

        params = np.flip(self.model.params)
        preds = None
        for i in tqdm(range(self.output_length)):
            concatted = test_inputs if preds is None else np.concatenate([test_inputs, preds], axis=-1)
            sliced = np.pad(concatted[:, -self.lags:], ((0, 0), (0, 1)), constant_values=1)
            next_pred = (sliced * params).sum(axis=-1)
            if preds is None:
                preds = next_pred.reshape(-1, 1)
            else:
                preds = np.concatenate([preds, next_pred.reshape(-1, 1)], axis=-1)
    
        return ((test_truths - preds)**2).mean(axis=-1), np.abs(test_truths - preds).mean(axis=-1)
    
    def test_loss_acc_stack(self, stack):
        params = np.flip(self.model.params)
        constant = params[0]
        params = torch.tensor(params[1:])

        concat = stack[:, :, -self.lags:]
        for i in tqdm(range(self.output_length)):
            preds = (stack[:, :, -3:] * params).sum(dim=-1)[:, :, None]

            
            concat = torch.concat([concat[:, :, 1:], preds], dim=-1)
    
    def test_loss_acc_loader(self, test_loader, device, skip_prop=0, skip_seed=43, progress_bar=True):
        with torch.no_grad():
            params = torch.tensor(np.flip(self.model.params).copy(), device=device)
            constant = params[0]
            params = params[1:]
            mses = list()
            for x, y in tqdm(test_loader):
                preds = None
                for i in range(self.output_length):
                    x = x.view(x.shape[0], x.shape[-1])
                    concatted = x.to(device) if preds is None else torch.concat([x.to(device), preds], dim=-1)
                    sliced = concatted[:, -self.lags:]
                    next_pred = (concatted * params).sum(dim=-1) + constant
                    if preds is None:
                        preds = next_pred.view(-1, 1)
                    else:
                        preds = torch.concat([preds, next_pred.view(-1, 1)], axis=-1)
                y = y.view(y.shape[0], y.shape[-1]).to(device)
                assert y.shape == preds.shape
                mses.extend(((y - preds)**2).mean(dim=-1).tolist())
                break
        
        return mses

    def predict(self, input) -> torch.Tensor:
        preds = self.model.append(input.flatten(), refit=False).forecast(steps=self.output_length)
        return torch.tensor(preds)
    
@dataclass 
class DummyAR:
    params: List[float]
    ar_lags: List[float]

def _unfold_df(df, device, input_length, output_length):
    data = torch.tensor(np.transpose(df.to_numpy()), dtype=torch.float32, device=device)
    return data.unfold(1, input_length + output_length, 1)
    
class ARv2(TSForecaster):
    def fit_raw(self, scaled_train_df, scaled_val_df, device, time_limit_hours, do_first_diff: bool, use_ols=False, first_lag=0, last_lag=None, search_space=None):
        print(f"{time_limit_hours} hours limit")

        self.do_first_diff = do_first_diff

        train_series = np.transpose(scaled_train_df.to_numpy()).flatten()
        if self.do_first_diff:
            train_series = np.diff(train_series, 1, axis=0)
        val_series_rw = _unfold_df(scaled_val_df, device, self.input_length, self.output_length)
        self.mses = list()
        start_time = time.perf_counter()
        max_lags = self.input_length if not do_first_diff else self.input_length - 1
        if last_lag is not None:
            max_lags = last_lag
        self.time_is_up = False
        space = range(first_lag, max_lags + 1, 1) if search_space is None else np.flip(search_space["window_len"],0).tolist()
        for lags in tqdm(space):
            if use_ols:
                if do_first_diff:
                    betas = self._fit_ols(scaled_train_df.diff(1).iloc[1:, :], device, lags)
                else:
                    betas = self._fit_ols(scaled_train_df, device, lags)
                model = DummyAR(
                    params=betas,
                    ar_lags=None if lags == 0 else betas[1:]
                )
            else:
                model = AutoReg(
                    endog=train_series,
                    lags=lags,
                    trend='c'
                ).fit()

            val_mse, _ = self._test_loss_acc_stack(model, val_series_rw, device)
            self.mses.append(val_mse)
            if (time.perf_counter() - start_time)/3600 > time_limit_hours:
                print(f"{time_limit_hours} hours have passed, time's up")
                self.time_is_up = True
                break
        print(f"Best Lags: {space[np.argmin(self.mses)]}")
        self.best_lags = space[np.argmin(self.mses)]
        return np.min(self.mses)

    def _fit_ols(self, scaled_train_df, device, input_length):
        torch.cuda.empty_cache()
        data_tensor = _unfold_df(scaled_train_df, torch.device("cpu"), input_length=input_length, output_length=1)
        y = copy.deepcopy(data_tensor[:, :, input_length].reshape(-1))
        y.to(device)
        if input_length == 0:
            del data_tensor
            betas = y.mean()[None]
        else:
            x = torch.nn.functional.pad(copy.deepcopy(data_tensor[:, :, :input_length]).reshape(-1, input_length), (0, 1), value=1)
            del data_tensor
            x.to(device)
            betas = torch.inverse(torch.matmul(torch.transpose(x, 1, 0), x))
            betas = torch.matmul(betas, torch.transpose(x, 1, 0))
            betas = torch.matmul(betas, y)
            del x, y
        return torch.flip(betas, [0])

    # def _unfold_df(self, df, device, input_length=None, output_length=None):
    #     data = torch.tensor(np.transpose(df.to_numpy()), dtype=torch.float32, device=device)
    #     if input_length is None:
    #         input_length = self.input_length
    #     if output_length is None:
    #         output_length = self.output_length
    #     return data.unfold(1, input_length + output_length, 1)

    def fit_preset(self, scaled_train_df, lags, do_first_diff: bool, device: torch.device, use_ols=False):
        self.do_first_diff = do_first_diff
        self.lags = lags
        if use_ols:
            if do_first_diff:
                betas = self._fit_ols(scaled_train_df.diff(1).iloc[1:, :], device, lags)
            else:
                betas = self._fit_ols(scaled_train_df, device, lags)
            self.model = DummyAR(
                params=betas,
                ar_lags=None if lags == 0 else betas[1:]
            )
        else:
            train_series = np.transpose(scaled_train_df.to_numpy()).flatten()
            if self.do_first_diff:
                train_series = np.diff(train_series, 1, axis=0)
            self.model = AutoReg(
                    endog=train_series,
                    lags=self.lags,
                    trend='c'
            ).fit()
        print(f"Selected Lags: {self.lags}")
    
    def _predict_stack(self, model, val_series_rw, device: torch.device) -> torch.Tensor:
        constant = model.params[0]

        if not self.do_first_diff:
            if model.ar_lags is None:
                return constant
        
            lags = len(model.ar_lags)
            lag_params = torch.flip(torch.tensor(model.params[1:], dtype=torch.float32, device=device), [0])
            assert len(lag_params) == lags

            val_preds = torch.zeros(val_series_rw.shape[0], val_series_rw.shape[1], self.output_length, dtype=torch.float32, device=device)
            for t in range(self.output_length):
                if t == 0:
                    val_preds[:, :, t] = (val_series_rw[:, :, -lags - self.output_length:-self.output_length] * lag_params).sum(dim=-1) 
                elif t < lags:
                    val_preds[:, :, t] = (val_series_rw[:, :, -lags + t - self.output_length:-self.output_length] * lag_params[:-t]).sum(dim=-1) + (val_preds[:, :, :t] * lag_params[-t:]).sum(dim=-1)
                else:
                    val_preds[:, :, t] = (val_preds[:, :, t-lags:t] * lag_params).sum(dim=-1)
                val_preds[:, :, t] = val_preds[:, :, t] + constant 
            return val_preds
        else:
            if model.ar_lags is None:
                deltas = torch.arange(1, self.output_length + 1, 1, device=device) * constant
                val_preds = val_series_rw[:, :, self.input_length - 1][:, :, None] + deltas
                return val_preds
            
            lags = len(model.ar_lags)
            lag_params = torch.flip(torch.tensor(model.params[1:], dtype=torch.float32, device=device), [0])
            assert len(lag_params) == lags

            val_diffs = torch.diff(val_series_rw, 1, dim=-1)
            val_diff_preds = torch.zeros(val_series_rw.shape[0], val_series_rw.shape[1], self.output_length, dtype=torch.float32, device=device)
            for t in range(self.output_length):
                if t == 0:
                    val_diff_preds[:, :, t] = (val_diffs[:, :, -lags - self.output_length:-self.output_length] * lag_params).sum(dim=-1)
                elif t < lags:
                    val_diff_preds[:, :, t] = (val_diffs[:, :, -lags + t - self.output_length:-self.output_length] * lag_params[:-t]).sum(dim=-1) + (val_diff_preds[:, :, :t] * lag_params[-t:]).sum(dim=-1)
                else:
                    val_diff_preds[:, :, t] = (val_diff_preds[:, :, t-lags:t] * lag_params).sum(dim=-1)
                val_diff_preds[:, :, t] = val_diff_preds[:, :, t] + constant 
            val_preds = val_series_rw[:, :, self.input_length - 1, None] + torch.cumsum(val_diff_preds, dim=-1)
            return val_preds

    def _test_loss_acc_stack(self, model, val_series_rw, device: torch.device, return_prediction=False):
        val_preds = self._predict_stack(model, val_series_rw, device)
        if return_prediction:
            return (
                ((val_preds - val_series_rw[:, :, -self.output_length:])**2).mean().item(), 
                torch.abs(val_preds - val_series_rw[:, :, -self.output_length:]).mean().item(),
                val_preds,
                val_series_rw[:, :, -self.output_length:]
            )
        return (
            ((val_preds - val_series_rw[:, :, -self.output_length:])**2).mean().item(), 
            torch.abs(val_preds - val_series_rw[:, :, -self.output_length:]).mean().item()
        )
       
    
    def test_loss_acc_df(self, scaled_test_df, device: torch.device, return_prediction=False):
        test_series_rw = _unfold_df(scaled_test_df, device, self.input_length, self.output_length)
        # print(test_series_rw.shape)
        return self._test_loss_acc_stack(self.model, test_series_rw, device, return_prediction)

    def predict(self, input) -> torch.Tensor:
        preds = self.model.append(input.flatten(), refit=False).forecast(steps=self.output_length)
        return torch.tensor(preds)
    
class VarOLS(TSForecaster):
    def __init__(self, input_length, output_length, device: torch.device) -> None:
        super().__init__(input_length, output_length) 
        self.device = device

    def fit_raw(self, scaled_train_df, scaled_val_df, device, time_limit_hours, do_first_diff: bool, first_lag=0, last_lag=None):
        print(f"{time_limit_hours} hours limit")

        self.do_first_diff = do_first_diff

        val_series_rw = _unfold_df(scaled_val_df, device, self.input_length, self.output_length)
        self.mses = list()
        start_time = time.perf_counter()
        max_lags = self.input_length if not do_first_diff else self.input_length - 1
        if last_lag is not None:
            max_lags = last_lag
        self.time_is_up = False
        for lags in tqdm(range(first_lag, max_lags + 1, 1)):
            if do_first_diff:
                betas = self._fit_ols(scaled_train_df.diff(1).iloc[1:, :], device, lags)
            else:
                betas = self._fit_ols(scaled_train_df, device, lags)
            model = DummyAR(
                params=betas,
                ar_lags=None if lags == 0 else betas[1:]
                )

            val_mse, _ = self._test_loss_acc_stack(model, val_series_rw, device)
            self.mses.append(val_mse)
            if (time.perf_counter() - start_time)/3600 > time_limit_hours:
                print(f"{time_limit_hours} hours have passed, time's up")
                self.time_is_up = True
                break
        print(f"Best Lags: {np.argmin(self.mses) + first_lag}")
        self.best_lags = np.argmin(self.mses) + first_lag
    
# class AutoAR(TSForecaster):
#     def fit_raw(self, scaled_train_df, scaled_val_df):
#         train_series = np.transpose(pd.concat([scaled_train_df, scaled_val_df]).to_numpy()).flatten()
#         train_darts = TimeSeries.from_times_and_values(pd.RangeIndex(len(train_series)), train_series)
#         self.model = AutoARIMA(
#             start_p=1,
#             d=1,
#             max_p=10
#         ).fit(train_darts)


#     def test_loss_acc(self, test_ds, n=96, skip_prop=0, skip_seed=43):
#         test_ts = list()
#         targets = list()
#         for x, y in tqdm(test_ds):
#             test_ts.append(TimeSeries.from_times_and_values(pd.RangeIndex(self.input_length), x[0].view(x[0].shape[0], -1).numpy()))
#             targets.append(y[0][:n].numpy())


#         preds = np.array([z.all_values().flatten() for z in self.model.predict(n=n, series=test_ts)])
#         targets = np.array(targets)

#         mse = ((preds - targets)**2).mean()
#         mae = np.abs(preds-targets).mean()
#         return mse, mae

#     def predict(self, input) -> Tensor:
#         preds = self.model.append(input.flatten(), refit=False).forecast(steps=self.output_length)
#         return torch.tensor(preds)

class DumbMLP(torch.nn.Module):
    def __init__(self, input_length, output_length) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(input_length, output_length)

    def forward(self, x):
        return self.layer(x)
    
    def init_optim(self, lr):
        self.optim = torch.optim.Adam(self.parameters(), lr)
    
class DumbMLP4TS(TSForecaster):
    def __init__(self, input_length, output_length, device, state_filename: str) -> None:
        super().__init__(input_length, output_length)
        self.mlp = DumbMLP(input_length, output_length).to(device)
        self.device = device
        self.state_filename = state_filename

    def load_last_state(self):
        self.mlp.load_state_dict(torch.load(self.state_filename))

    def fit_loader(self, train_loader, val_loader, lr, max_epochs, time_limit_hours, do_print = True, epoch_batches=None):
        print(f"{time_limit_hours} hours limit")
        self.mlp.init_optim(lr)
        self.mlp.train()
        best_val_loss = 99999
        best_val_epoch = 0
        start_time = time.perf_counter()
        self.time_is_up = False
        for epoch in tqdm(range(max_epochs)):
            for b, batch in enumerate(train_loader):
                x, y = batch
                self.mlp.optim.zero_grad()
                preds = self.predict(x.to(self.device))
                y = y.to(self.device).view(y.shape[0], -1)
                assert preds.shape == y.shape
                loss = torch.nn.functional.mse_loss(preds, y.to(self.device).view(y.shape[0], -1))
                loss.backward()
                self.mlp.optim.step()
                if (time.perf_counter() - start_time)/3600 > time_limit_hours:
                    print(f"{time_limit_hours} hours have passed, time's up")
                    self.time_is_up = True 
                    break
                if epoch_batches is not None and b >= epoch_batches:
                    break
            self.mlp.eval()
            with torch.no_grad():
                val_loss, val_acc = self.test_loss_acc_loader(val_loader, self.device, progress_bar=False)
                val_loss = np.mean(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    torch.save(self.mlp.state_dict(), self.state_filename)
                if epoch % 5 == 0:
                    train_loss, train_acc = self.test_loss_acc_loader(train_loader, self.device, progress_bar=False)
                    train_loss = np.mean(train_loss)
                    val_acc = np.mean(val_acc)
                    if do_print:
                        print(f"Train MSE: {train_loss:0.4f} | Val MSE: {val_loss:0.4f} | Val MAE: {val_acc:0.4f}")
            if self.time_is_up:
                break

        print(f"Best val epoch id {best_val_epoch}")
        self.load_last_state()
        return best_val_epoch
            

    def predict(self, input) -> torch.Tensor:
        x = input.to(self.device)
        return self.mlp(x.view(x.shape[0], -1))
    
class LinearOLS(TSForecaster):
    def __init__(self, input_length, output_length, device: torch.device) -> None:
        super().__init__(input_length, output_length)
        self.device = device

    def _fit_ols(self, scaled_train_df, device):
        torch.cuda.empty_cache()
        data_tensor = _unfold_df(scaled_train_df, torch.device("cpu"), self.input_length, self.output_length)
        y = data_tensor[:, :, self.input_length:self.input_length+self.output_length].reshape(-1, self.output_length)
        y.to(device)
        if self.input_length == 0:
            del data_tensor
            betas = y.mean(dim=0)[None]
        else:
            x = torch.nn.functional.pad(data_tensor[:, :, :self.input_length].reshape(-1, self.input_length), (0, 1), value=1)
            del data_tensor
            x.to(device)
            betas = torch.inverse(torch.matmul(torch.transpose(x, 1, 0), x))
            betas = torch.matmul(betas, torch.transpose(x, 1, 0))
            betas = torch.matmul(betas, y)
        return torch.flip(betas, [0])
    
    def fit(self, scaled_train_df, scaled_val_df, time_limit_hours, device=None):
        self.betas = self._fit_ols(pd.concat((scaled_train_df, scaled_val_df)), device or self.device)
    
    def _predict_stack(self, val_predictors_rw) -> torch.Tensor:
        constant = self.betas[-1]
        if self.betas.shape[0] == 1:
            return constant.expand(val_predictors_rw.shape[-2], -1)
        lag_params = self.betas[:-1]
    
        return torch.matmul(val_predictors_rw, lag_params) + constant
    
    def _test_loss_acc_stack(self, val_series_rw) -> float:
        val_preds = self._predict_stack(val_series_rw[:, :, :self.input_length])
        return (
            ((val_preds - val_series_rw[:, :, -self.output_length:])**2).mean().item(), 
            torch.abs(val_preds - val_series_rw[:, :, -self.output_length:]).mean().item()
        )
    
    def test_loss_acc_df(self, scaled_test_df, device: torch.device = None):
        test_series_rw = _unfold_df(scaled_test_df, device or self.device, self.input_length, self.output_length)
        return self._test_loss_acc_stack(test_series_rw)
    

    
class SimpleDLinear(TSForecaster):
    def fit_raw(self, scaled_train_df, scaled_val_df, max_epochs, lr, kernel_size=25):
        train_darts = [
            TimeSeries.from_times_and_values(scaled_train_df.index, scaled_train_df[col])
            for col in scaled_train_df.columns
        ]

        val_darts = [
            TimeSeries.from_times_and_values(scaled_val_df.index, scaled_val_df[col])
            for col in scaled_val_df.columns 
        ]

        self.model = DLinearModel(
            input_chunk_length=self.input_length,
            output_chunk_length=self.output_length,
            kernel_size=kernel_size,
            use_static_covariates=False,
            n_epochs=max_epochs,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": lr},
            save_checkpoints=True,
            batch_size=8
        ).fit(train_darts, val_series=val_darts)

    def load_state(self, path):
        self.model = DLinearModel.load_from_checkpoint(path, best=True)

    def predict(self, input) -> torch.Tensor:
        
        r = pd.RangeIndex(self.input_length)
        input_ts = TimeSeries.from_times_and_values(r, input.numpy().flatten())
        return torch.tensor(self.model.predict(n=96, series=input_ts).all_values()).flatten()
    
    def test_loss_acc(self, test_ds, n=96, skip_prop=0, skip_seed=43):
        test_ts = list()
        targets = list()
        for x, y in tqdm(test_ds):
            test_ts.append(TimeSeries.from_times_and_values(pd.RangeIndex(self.input_length), x[0][-self.input_length:].numpy()))
            targets.append(y[0][:n].numpy())


        preds = np.array([z.all_values().flatten() for z in self.model.predict(n=n, series=test_ts)])
        targets = np.array(targets)

        mse = ((preds - targets)**2).mean(axis=-1)
        mae = np.abs(preds-targets).mean(axis=-1)
        return mse, mae
    

@dataclass 
class DLinearRawConfig:
    seq_len: int
    pred_len: int
    enc_in: int
    individual: bool = False

class SimpleDLinear2(TSForecaster):
    def fit_loader(self, train_loader, val_loader, lr, max_epochs, time_limit_hours, do_print = True):
        self.model = DLinearRaw(
            DLinearRawConfig(
                seq_len=self.input_length,
                pred_len=self.output_length,
                enc_in=321
            )
        )

        self.device = torch.device("cuda:0")
        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr)

        self.model.train()
        for i in range(max_epochs):
            for x, y in tqdm(train_loader, desc=f"Epoch {i + 1}"):
                opt.zero_grad()
                x = x.permute(0, 2, 1)
                preds = self.model.forward(x.to(self.device))
                y = y[:, :, None].to(self.device)
                assert preds.shape == y.shape

                loss = torch.nn.functional.mse_loss(preds, y)
                loss.backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                val_mse, val_mae = self.test_loss_acc_loader(val_loader, self.device)
                print(f"Val MSE {val_mse}")



    # def load_state(self, path):
    #     self.model = DLinearModel.load_from_checkpoint(path, best=True)

    # def predict(self, input) -> torch.Tensor:
        
    #     r = pd.RangeIndex(self.input_length)
    #     input_ts = TimeSeries.from_times_and_values(r, input.numpy().flatten())
    #     return torch.tensor(self.model.predict(n=96, series=input_ts).all_values()).flatten()
    
    def test_loss_acc_loader(self, test_loader, device, skip_prop=0, skip_seed=43, progress_bar=True):
        total_mse = 0
        total_mae = 0
        n = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="Evaluating"):
                x = x.permute(0, 2, 1)
                preds = self.model.forward(x.to(device))
                y = y[:, :, None].to(device)
                assert preds.shape == y.shape

                total_mse += torch.nn.functional.mse_loss(preds, y, reduction='sum')
                total_mae += torch.abs(preds - y).sum()
                n += x.shape[0]
        
        return total_mse/(n * self.output_length), total_mae/(n * self.output_length)
    
# class ORCA(TSForecaster):
#     def __init__(self, input_length, output_length, config_file, device) -> None:
#         super().__init__(input_length, output_length)
#         use_determined = False
#         context = None
#         optimizer = None
#         scheduler = None 
#         n_train = None
#         id_best = None
#         root = "./datasets/"
#         with open(config_file, 'r') as stream:
#             args = AttrDict(yaml.safe_load(stream)['hyperparameters'])

#         dims, sample_shape, num_classes, loss, args = get_config(root, args)
#         self.model = wrapper1D(sample_shape, num_classes, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, target_seq_len=args.target_seq_len, drop_out=args.drop_out)
#         self.model, _, _, _, _, _ = load_state(use_determined, args, context, self.model, optimizer, scheduler, n_train, id_best, test=True)
#         self.model.output_raw = False
#         self.model.eval()
#         self.device = device
#         self.model.to(device)

#     def predict(self, input) -> Tensor:
#         x = input.view(input.shape[0], 1, input.shape[-1]).to(self.device)
#         return self.model(x)
        