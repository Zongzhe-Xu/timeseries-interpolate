import argparse
import data.ts_datasets as my
import pandas as pd
from models import DumbMLP4TS, ARv2
import torch
import numpy as np
import pickle as pkl
import time 
import sys

def main():
    parser = argparse.ArgumentParser(description='Run AR search')
    parser.add_argument('--dataset', type=str, default='ETTh1_96', help='dataset to use')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--multi_channel', type=bool, default=False, help='use multi-channel data')
    parser.add_argument('--time_limit_hours', type=float, default=10, help='time limit in hours')
    parser.add_argument('--use_ols', type=bool, default=True, help='use OLS for AR')
    parser.add_argument('--search', action='store_true', help='do interploation search')
    args = parser.parse_args()

    print("Is interpolating?:", args.search)

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("Using GPU")
    else:
        DEVICE = torch.device("cpu")
        print("Using CPU")

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

    
    params = args.dataset.split("_")
    prefix = params[0]
    horizon = int(params[1])
    args.horizon = horizon
    assert prefix in file_dict, f"Invalid dataset {args.dataset} from possible {list(file_dict.keys())}"
    input_length = 96 if prefix == "ILI" else 512
    train_loader, val_loader, test_loader = my.get_timeseries_dataloaders(
        f"./data/ts_datasets/all_six_datasets/{file_dict[prefix]}", 
        batch_size=args.batch_size, 
        seq_len=input_length,
        forecast_horizon=horizon,
        multi_channel=args.multi_channel
    )
    train_data, val_data, test_data = train_loader.dataset, val_loader.dataset, test_loader.dataset
    train_df, val_df, test_df = pd.DataFrame(train_data.data), pd.DataFrame(val_data.data), pd.DataFrame(test_data.data)

    if not args.search:
        args.do_diff = True
        start_time = time.perf_counter()
        ar = ARv2(input_length, horizon)
        ar.fit_raw(train_df, val_df, DEVICE, args.time_limit_hours, args.do_diff, use_ols=args.use_ols, first_lag=400)
        print(f"AR search for {prefix} {args.horizon} finished in {time.perf_counter() - start_time} seconds)")
        ar.fit_preset(pd.concat((train_df, val_df)), ar.best_lags, args.do_diff, torch.device("cpu"), use_ols=args.use_ols)
        mse, mae = ar.test_loss_acc_df(test_df, torch.device("cpu"))
        print(f"Final AR metrics for {prefix} {args.horizon}: MSE {np.mean(mse)} | MAE {np.mean(mae)}")
        # other_bits = ((time.perf_counter() - start_time)/60, f"Reached Time Limit? {ar.time_is_up}", f"Best Lags {ar.lags}")
    else:
        search_space = {"window_len": 511-np.array([0]+[int(1.5**i) for i in range(16)][1:])}
        print("Search Space:", search_space)
        # exit()
        best_windows = []
        predictions = []
        for do_diff in [True, False]:
            args.do_diff = do_diff
            start_time = time.perf_counter()
            ar = ARv2(input_length, horizon)
            mse = ar.fit_raw(train_df, val_df, DEVICE, args.time_limit_hours, do_diff, use_ols=args.use_ols, first_lag=400, search_space=search_space)
            print(f"AR search for {prefix} {args.horizon} d= {do_diff} finished in {time.perf_counter() - start_time} seconds with validation MSE {mse}")
            best_windows.append(ar.best_lags)
            ar.fit_preset(train_df, ar.best_lags, do_diff, torch.device("cpu"), use_ols=args.use_ols)
            _, _, prediction, y = ar.test_loss_acc_df(val_df, torch.device("cpu"), return_prediction = True)
            # test_mse, test_mae = ar.test_loss_acc_df(test_df, torch.device("cpu"))
            # print(f"test metrics for {prefix} {args.horizon} d= {do_diff}: MSE {np.mean(test_mse)} | MAE {np.mean(test_mae)}")
            predictions.append(prediction)
        
        start_time = time.perf_counter()
        list1 = predictions[0].permute(1, 0, 2).flatten().cpu().numpy()
        list2 = predictions[1].permute(1, 0, 2).flatten().cpu().numpy()
        y = y.permute(1, 0, 2).flatten().cpu().numpy()
        # print("test:", np.mean((list1 - y)**2), np.mean((list2 - y)**2))
        # find alpha such that alpha * list1 + (1 - alpha) * list2 is closest to y
        alpha = np.sum(-2*(list2-y)*(list1-list2)) / np.sum(2*(list1-list2)**2)
        new_prediction = alpha * list1 + (1 - alpha) * list2
        mse = np.mean((new_prediction - y)**2)
        mae = np.mean(np.abs(new_prediction - y))
        print(f"combined validation metrics for {prefix} {args.horizon}: MSE {mse} | MAE {mae} | alpha {alpha}")

        ar1 = ARv2(input_length, horizon)
        ar1.fit_preset(pd.concat((train_df, val_df)), best_windows[0], True, torch.device("cpu"), use_ols=args.use_ols)
        _, _, prediction1, y1 = ar1.test_loss_acc_df(test_df, torch.device("cpu"), return_prediction = True)
        ar2 = ARv2(input_length, horizon)
        ar2.fit_preset(pd.concat((train_df, val_df)), best_windows[1], False, torch.device("cpu"), use_ols=args.use_ols)
        _, _, prediction2, y2 = ar2.test_loss_acc_df(test_df, torch.device("cpu"), return_prediction = True)
        list1 = prediction1.permute(1, 0, 2).flatten().cpu().numpy()
        list2 = prediction2.permute(1, 0, 2).flatten().cpu().numpy()
        assert torch.equal(y1, y2), "y1 and y2 are not equal"
        y = y1.permute(1, 0, 2).flatten().cpu().numpy()
        new_prediction = alpha * list1 + (1 - alpha) * list2
        mse_all = np.mean((new_prediction - y)**2)
        mae_all = np.mean(np.abs(new_prediction - y))
        mse1 = np.mean((list1 - y)**2)
        mae1 = np.mean(np.abs(list1 - y))
        mse2 = np.mean((list2 - y)**2)
        mae2 = np.mean(np.abs(list2 - y))
        print(f"test metrics for {prefix} {args.horizon} d= {True}: MSE {mse1} | MAE {mae1}")
        print(f"test metrics for {prefix} {args.horizon} d= {False}: MSE {mse2} | MAE {mae2}")
        print(f"combined test metrics for {prefix} {args.horizon}: MSE {mse_all} | MAE {mae_all} | alpha {alpha}")
        
            



    


if __name__ == "__main__":
    main()