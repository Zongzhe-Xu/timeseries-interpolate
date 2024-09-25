import argparse
import dashed_dreams.DASH.src.data.ts_datasets as my
import pandas as pd
from models import DumbMLP4TS, ARv2
import torch
import numpy as np
import pickle as pkl
import time 
import sys

def save_results(mse, mae, args, other_bits):
    try:
        with open(args.results_file, 'rb') as file:
            results_dict = pkl.load(file)
    except FileNotFoundError:
        results_dict = dict()

    if args.dataset not in results_dict:
        results_dict[args.dataset] = dict() 
    
    if args.do_mlp:
        model = "linear"
    elif args.do_diff:
        model = "ar_d1"
    else:
        model = "ar_d0"

    if model not in results_dict[args.dataset]:
        results_dict[args.dataset][model] = dict()

    results_dict[args.dataset][model][args.horizon] = (mse, mae, other_bits)

    with open(args.results_file, 'wb') as file:
        pkl.dump(results_dict, file)

def check_for_results(args):
    try:
        with open(args.results_file, 'rb') as file:
            results_dict = pkl.load(file)
    except FileNotFoundError:
        return False
    
    if args.dataset not in results_dict:
        return False

    if args.do_mlp:
        model = "linear"
    elif args.do_diff:
        model = "ar_d1"
    else:
        model = "ar_d0"

    if model not in results_dict[args.dataset]:
        return False
    
    if args.horizon not in results_dict[args.dataset][model]:
        return False
    
    return True
    

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="MLP AR experiments")

    # Add arguments
    parser.add_argument(
        '--results_file',
        type=str,
        required=True,
        help='Name of results filename'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Name of dataset'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        required=True,
        help='Forecast horizon'
    )
    parser.add_argument(
        '--time_limit_hours',
        type=float,
        required=True,
        help='Time limit in hours'
    )
    parser.add_argument(
        '--do_diff',
        action='store_true',
        help='Do AR with differencing instead of AR'
    )
    parser.add_argument(
        '--use_ols',
        action='store_true',
        help='Do AR with OLS instead of MLE'
    )
    parser.add_argument(
        '--do_mlp',
        action='store_true',
        help='Do MLP instead of AR'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite Existing Results'
    )

    # Parse the arguments
    args = parser.parse_args()

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

    if not args.overwrite and check_for_results(args):
        print("Already Exists")
        sys.exit(0)

    assert args.dataset in file_dict, f"Invalid dataset {args.dataset} from possible {list(file_dict.keys())}"
    input_length = 96 if args.dataset == "ILI" else 512
    train_loader, val_loader, test_loader = my.get_timeseries_dataloaders(
        f"./data/ts_datasets/all_six_datasets/{file_dict[args.dataset]}", 
        32, 
        seq_len=input_length,
        forecast_horizon=args.horizon
    )
    train_data, val_data, test_data = train_loader.dataset, val_loader.dataset, test_loader.dataset
    train_df, val_df, test_df = pd.DataFrame(train_data.data), pd.DataFrame(val_data.data), pd.DataFrame(test_data.data)
    DEVICE = torch.device("cuda:0")

    start_time = time.perf_counter()

    if args.do_mlp:
        torch.manual_seed(1234)
        dumb_mlp = DumbMLP4TS(input_length, args.horizon, DEVICE, f"./mlp_states/{args.dataset}_{args.horizon}.pt")
        best_val_epoch = dumb_mlp.fit_loader(train_loader, val_loader, 1e-5, 300, args.time_limit_hours, epoch_batches=2000)
        dumb_mlp.mlp.eval()
        mse, mae = dumb_mlp.test_loss_acc_loader(test_loader, dumb_mlp.device)
        print(f"Final MLP metrics for {args.dataset} {args.horizon}: MSE {np.mean(mse)} | MAE {np.mean(mae)}")
        mse = np.mean(mse)
        mae = np.mean(mae)
        other_bits = ((time.perf_counter() - start_time)/60, f"Reached Time Limit? {dumb_mlp.time_is_up}", f"Best Val Epoch {best_val_epoch}")
    else:
        ar = ARv2(input_length, args.horizon)
        ar.fit_raw(train_df, val_df, DEVICE, args.time_limit_hours, args.do_diff, use_ols=args.use_ols, first_lag=400)
        ar.fit_preset(pd.concat((train_df, val_df)), ar.best_lags, args.do_diff, torch.device("cpu"), use_ols=args.use_ols)
        mse, mae = ar.test_loss_acc_df(test_df, torch.device("cpu"))
        print(f"Final AR metrics for {args.dataset} {args.horizon}: MSE {np.mean(mse)} | MAE {np.mean(mae)}")
        other_bits = ((time.perf_counter() - start_time)/60, f"Reached Time Limit? {ar.time_is_up}", f"Best Lags {ar.lags}")

    save_results(
        mse,
        mae,
        args,
        other_bits
    )


if __name__ == "__main__":
    main()