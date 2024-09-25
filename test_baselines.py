import models
import pandas as pd
import torch
import numpy as np
from dataclasses import dataclass 
from typing import List
import dashed_dreams.DASH.src.data.ts_datasets as my

DEVICE = torch.device("cuda:0")
test_ar = models.ARv2(2, 2)
test_df = pd.DataFrame({
    0: [1, 2, 3, 4, 5, 6],
    1: [-1, -2, -3, -4, -5, -6],
})

@dataclass 
class DummyAR:
    params: List[float]
    ar_lags: List[float]

def test_unfold():
    expected = torch.tensor([
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6]
        ],
        [
            [-1, -2, -3, -4],
            [-2, -3, -4, -5],
            [-3, -4, -5, -6]
        ],
    ]).to(DEVICE)
    actual = test_ar._unfold_df(test_df, DEVICE)
    assert actual.shape == expected.shape
    assert (actual == expected).all()

def test_predict_d0():
    test_ar.do_first_diff = False
    test_ar.model = DummyAR(
        params=[-1, 1, 1],
        ar_lags=[1, 2]
    )

    expected = torch.tensor([
        [
            [2, 3],
            [4, 6],
            [6, 9]
        ],
        [
            [-4, -7],
            [-6, -10],
            [-8, -13]
        ],
    ]).to(DEVICE)

    actual = test_ar._predict_stack(
        test_ar.model, 
        test_ar._unfold_df(test_df, DEVICE),
        DEVICE
    )

    assert actual.shape == expected.shape
    assert (actual == expected).all()

def test_predict_d0_constant():
    test_ar.do_first_diff = False
    constant = 9
    test_ar.model = DummyAR(
        params=[constant],
        ar_lags=None
    )

    expected = torch.tensor([
        [
            [constant, constant],
            [constant, constant],
            [constant, constant]
        ],
        [
            [constant, constant],
            [constant, constant],
            [constant, constant]
        ],
    ]).to(DEVICE)

    actual = test_ar._predict_stack(
        test_ar.model, 
        test_ar._unfold_df(test_df, DEVICE),
        DEVICE
    )

    assert actual == constant
    assert (actual == expected).all()

def test_predict_d1():
    test_ar.do_first_diff = True
    test_ar.model = DummyAR(
        params=[1, -1],
        ar_lags=[1]
    )

    expected = torch.tensor([
        [
            [2, 3],
            [3, 4],
            [4, 5]
        ],
        [
            [0, -1],
            [-1, -2],
            [-2, -3]
        ],
    ]).to(DEVICE)

    actual = test_ar._predict_stack(
        test_ar.model, 
        test_ar._unfold_df(test_df, DEVICE),
        DEVICE
    )

    assert actual.shape == expected.shape
    assert (actual == expected).all()

def test_predict_d1_constant():
    test_ar.do_first_diff = True
    test_ar.model = DummyAR(
        params=[1],
        ar_lags=None
    )

    expected = torch.tensor([
        [
            [3, 4],
            [4, 5],
            [5, 6]
        ],
        [
            [-1, 0],
            [-2, -1],
            [-3, -2]
        ],
    ]).to(DEVICE)

    actual = test_ar._predict_stack(
        test_ar.model, 
        test_ar._unfold_df(test_df, DEVICE),
        DEVICE
    )

    assert actual.shape == expected.shape
    assert (actual == expected).all()

def test_mse():
    test_ar.do_first_diff = True
    test_ar.model = DummyAR(
        params=[1, -1],
        ar_lags=[1]
    )

    expected = torch.tensor([
        [
            [2, 3], # error = 1
            [3, 4],
            [4, 5]
        ],
        [
            [0, -1], # error = 3
            [-1, -2],
            [-2, -3]
        ],
    ]).to(DEVICE)

    actual = test_ar._predict_stack(
        test_ar.model, 
        test_ar._unfold_df(test_df, DEVICE),
        DEVICE
    )

    assert test_ar.test_loss_acc_df(test_df, DEVICE) == (5, 2)


def test_dataloaders():
    file_dict = {
        "ETTh1": "ETT-small/ETTh1.csv",
        "ETTh2": "ETT-small/ETTh2.csv",
        "ETTm1": "ETT-small/ETTm1.csv",
        "ETTm2": "ETT-small/ETTm2.csv",
        "ECL": "electricity/electricity.csv",
        "ER": "exchange_rate/exchange_rate.csv",
        "ILI": "illness/national_illness.csv",
        "Traffic": "traffic/traffic.csv",
        "Weather": "weather/weather.csv"
    }
    eps = 1e-5

    for file in file_dict.values():
        print(file)
        seq_len = 165
        horizon = 52
        batch_size = 28
        raw_illness_df = pd.read_csv(
            f"./datasets/all_six_datasets/{file}"
        )
        train_loader, val_loader, test_loader = my.get_timeseries_dataloaders(
            f"./datasets/all_six_datasets/{file}",
            batch_size, 
            seq_len=seq_len,
            forecast_horizon=horizon,
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2
        )
        train_data, val_data, test_data = train_loader.dataset, val_loader.dataset, test_loader.dataset
        train_df, val_df, test_df = pd.DataFrame(train_data.data), pd.DataFrame(val_data.data), pd.DataFrame(test_data.data)

        n = len(raw_illness_df)
        if 'ETTm' in file:
            n_train = 12 * 30 * 24 * 4
            n_val = 4 * 30 * 24 * 4
            n_test = 4 * 30 * 24 * 4
        elif 'ETTh' in file:
            n_train = 12 * 30 * 24
            n_val = 4 * 30 * 24
            n_test = 4 * 30 * 24
        else:
            n_train = int(0.7*n)
            n_test = int(0.2*n)
            n_val = n - n_train - n_test

        assert len(train_df) == n_train
        assert len(val_df) == n_val + seq_len
        assert len(test_df) == n_test + seq_len

        assert len(train_data) == (n_train - seq_len - horizon + 1) * (raw_illness_df.shape[1] - 1)
        assert len(val_data) == (n_val - horizon + 1) * (raw_illness_df.shape[1] - 1)
        assert len(test_data) == (n_test - horizon + 1) * (raw_illness_df.shape[1] - 1)

        mean = raw_illness_df.iloc[:n_train, 1:].mean()
        std = raw_illness_df.iloc[:n_train, 1:].std(ddof=0)

        scaled_illness_df = (raw_illness_df.iloc[:, 1:] - mean)/std

        assert not np.isnan(scaled_illness_df.to_numpy()).any()

        assert (train_df.to_numpy() == scaled_illness_df.iloc[:n_train, :].to_numpy()).all()
        assert (val_df.to_numpy() == scaled_illness_df.iloc[n_train - seq_len:n_train + n_val, :].to_numpy()).all()
        assert (test_df.to_numpy() == scaled_illness_df.iloc[n_train + n_val - seq_len:n_train + n_val + n_test, :].to_numpy()).all()

        x, y = train_data[0]
        assert x.shape == (1, seq_len)
        assert y.shape == (1, horizon)
        
        for i in range(scaled_illness_df.shape[1]):
            x, y = train_data[(n_train - seq_len - horizon + 1) * i]
            assert (np.abs(x[0].numpy() - scaled_illness_df.iloc[:seq_len, i].to_numpy()) < eps).all()
            assert (np.abs(y[0].numpy() - scaled_illness_df.iloc[seq_len:seq_len + horizon, i].to_numpy()) < eps).all()

            x, y = train_data[(n_train - seq_len - horizon + 1) * (i + 1) - 1]
            assert (np.abs(x[0].numpy() - scaled_illness_df.iloc[n_train - horizon - seq_len:n_train - horizon, i].to_numpy()) < eps).all()
            assert (np.abs(y[0].numpy() - scaled_illness_df.iloc[n_train - horizon:n_train, i].to_numpy()) < eps).all()

            x, y = val_data[(n_val - horizon + 1) * i]
            assert (np.abs(x[0].numpy() - scaled_illness_df.iloc[n_train - seq_len:n_train, i].to_numpy()) < eps).all()
            assert (np.abs(y[0].numpy() - scaled_illness_df.iloc[n_train:n_train + horizon, i].to_numpy()) < eps).all()

            x, y = val_data[(n_val - horizon + 1) * (i + 1) - 1]
            assert (np.abs(x[0].numpy() - scaled_illness_df.iloc[n_train + n_val - horizon - seq_len:n_train + n_val - horizon, i].to_numpy()) < eps).all()
            assert (np.abs(y[0].numpy() - scaled_illness_df.iloc[n_train + n_val - horizon:n_train + n_val, i].to_numpy()) < eps).all()

            x, y = test_data[(n_test - horizon + 1) * i]
            assert (np.abs(x[0].numpy() - scaled_illness_df.iloc[n_train + n_val - seq_len:n_train + n_val, i].to_numpy()) < eps).all()
            assert (np.abs(y[0].numpy() - scaled_illness_df.iloc[n_train + n_val:n_train + n_val + horizon, i].to_numpy()) < eps).all()

            x, y = test_data[(n_test - horizon + 1) * (i + 1) - 1]
            assert (np.abs(x[0].numpy() - scaled_illness_df.iloc[n_train + n_val + n_test - horizon - seq_len:n_train + n_val + n_test - horizon, i].to_numpy()) < eps).all()
            assert (np.abs(y[0].numpy() - scaled_illness_df.iloc[n_train + n_val + n_test - horizon:n_train + n_val + n_test, i].to_numpy()) < eps).all()

        for x, y in val_loader:
            assert x.shape == (batch_size, 1, seq_len)
            assert y.shape == (batch_size, 1, horizon)
            assert (np.abs(x[0][0].numpy() - scaled_illness_df.iloc[n_train - seq_len:n_train, 0].to_numpy()) < eps).all()
            assert (np.abs(y[0][0].numpy() - scaled_illness_df.iloc[n_train:n_train + horizon, 0].to_numpy()) < eps).all()
            assert (np.abs(x[-1][0].numpy() - scaled_illness_df.iloc[n_train - seq_len + batch_size - 1:n_train + batch_size - 1, 0].to_numpy()) < eps).all()
            assert (np.abs(y[-1][0].numpy() - scaled_illness_df.iloc[n_train + batch_size - 1:n_train + horizon + batch_size - 1, 0].to_numpy()) < eps).all()
            break



def test_dataloaders_multi_channel():
    file_dict = {
        "ETTh1": "ETT-small/ETTh1.csv",
        "ETTh2": "ETT-small/ETTh2.csv",
        "ETTm1": "ETT-small/ETTm1.csv",
        "ETTm2": "ETT-small/ETTm2.csv",
        "ECL": "electricity/electricity.csv",
        "ER": "exchange_rate/exchange_rate.csv",
        "ILI": "illness/national_illness.csv",
        "Traffic": "traffic/traffic.csv",
        "Weather": "weather/weather.csv"
    }
    eps = 1e-5

    for file in file_dict.values():
        print(file)
        seq_len = 165
        horizon = 52
        batch_size=28
        raw_illness_df = pd.read_csv(
            f"./datasets/all_six_datasets/{file}"
        )
        train_loader, val_loader, test_loader = my.get_timeseries_dataloaders(
            f"./datasets/all_six_datasets/{file}",
            batch_size, 
            seq_len=seq_len,
            forecast_horizon=horizon,
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
            multi_channel=True
        )
        train_data, val_data, test_data = train_loader.dataset, val_loader.dataset, test_loader.dataset
        train_df, val_df, test_df = pd.DataFrame(train_data.data), pd.DataFrame(val_data.data), pd.DataFrame(test_data.data)
        n_channels = raw_illness_df.shape[1] - 1

        n = len(raw_illness_df)
        if 'ETTm' in file:
            n_train = 12 * 30 * 24 * 4
            n_val = 4 * 30 * 24 * 4
            n_test = 4 * 30 * 24 * 4
        elif 'ETTh' in file:
            n_train = 12 * 30 * 24
            n_val = 4 * 30 * 24
            n_test = 4 * 30 * 24
        else:
            n_train = int(0.7*n)
            n_test = int(0.2*n)
            n_val = n - n_train - n_test

        assert len(train_df) == n_train
        assert len(val_df) == n_val + seq_len
        assert len(test_df) == n_test + seq_len

        assert len(train_data) == (n_train - seq_len - horizon + 1) 
        assert len(val_data) == (n_val - horizon + 1) 
        assert len(test_data) == (n_test - horizon + 1) 

        mean = raw_illness_df.iloc[:n_train, 1:].mean()
        std = raw_illness_df.iloc[:n_train, 1:].std(ddof=0)

        scaled_illness_df = (raw_illness_df.iloc[:, 1:] - mean)/std

        assert not np.isnan(scaled_illness_df.to_numpy()).any()

        assert (train_df.to_numpy() == scaled_illness_df.iloc[:n_train, :].to_numpy()).all()
        assert (val_df.to_numpy() == scaled_illness_df.iloc[n_train - seq_len:n_train + n_val, :].to_numpy()).all()
        assert (test_df.to_numpy() == scaled_illness_df.iloc[n_train + n_val - seq_len:n_train + n_val + n_test, :].to_numpy()).all()

        x, y = train_data[0]
        assert x.shape == (n_channels, seq_len)
        assert y.shape == (n_channels, horizon)
        

        x, y = train_data[0]
        assert (np.abs(x.numpy() - scaled_illness_df.iloc[:seq_len, :].T.to_numpy()) < eps).all()
        assert (np.abs(y.numpy() - scaled_illness_df.iloc[seq_len:seq_len + horizon, :].T.to_numpy()) < eps).all()

        x, y = train_data[n_train - seq_len - horizon]
        assert (np.abs(x.numpy() - scaled_illness_df.iloc[n_train - horizon - seq_len:n_train - horizon, :].T.to_numpy()) < eps).all()
        assert (np.abs(y.numpy() - scaled_illness_df.iloc[n_train - horizon:n_train, :].T.to_numpy()) < eps).all()

        x, y = val_data[0]
        assert (np.abs(x.numpy() - scaled_illness_df.iloc[n_train - seq_len:n_train, :].T.to_numpy()) < eps).all()
        assert (np.abs(y.numpy() - scaled_illness_df.iloc[n_train:n_train + horizon, :].T.to_numpy()) < eps).all()

        x, y = val_data[(n_val - horizon)]
        assert (np.abs(x.numpy() - scaled_illness_df.iloc[n_train + n_val - horizon - seq_len:n_train + n_val - horizon, :].T.to_numpy()) < eps).all()
        assert (np.abs(y.numpy() - scaled_illness_df.iloc[n_train + n_val - horizon:n_train + n_val, :].T.to_numpy()) < eps).all()

        x, y = test_data[0]
        assert (np.abs(x.numpy() - scaled_illness_df.iloc[n_train + n_val - seq_len:n_train + n_val, :].T.to_numpy()) < eps).all()
        assert (np.abs(y.numpy() - scaled_illness_df.iloc[n_train + n_val:n_train + n_val + horizon, :].T.to_numpy()) < eps).all()

        x, y = test_data[(n_test - horizon)]
        assert (np.abs(x.numpy() - scaled_illness_df.iloc[n_train + n_val + n_test - horizon - seq_len:n_train + n_val + n_test - horizon, :].T.to_numpy()) < eps).all()
        assert (np.abs(y.numpy() - scaled_illness_df.iloc[n_train + n_val + n_test - horizon:n_train + n_val + n_test, :].T.to_numpy()) < eps).all()

        for x, y in val_loader:
            assert x.shape == (batch_size, n_channels, seq_len)
            assert y.shape == (batch_size, n_channels, horizon)
            assert (np.abs(x[0].numpy() - scaled_illness_df.iloc[n_train - seq_len:n_train, :].T.to_numpy()) < eps).all()
            assert (np.abs(y[0].numpy() - scaled_illness_df.iloc[n_train:n_train + horizon, :].T.to_numpy()) < eps).all()
            assert (np.abs(x[-1].numpy() - scaled_illness_df.iloc[n_train - seq_len + batch_size - 1:n_train + batch_size - 1, :].T.to_numpy()) < eps).all()
            assert (np.abs(y[-1].numpy() - scaled_illness_df.iloc[n_train + batch_size - 1:n_train + horizon + batch_size - 1, :].T.to_numpy()) < eps).all()
            break