# Setup
- DASH Setup here: https://docs.google.com/document/d/1_Gxt7_toKyBgZhMYS-BXBm12_xn3BHmgWgaEAr5cW0A/edit?usp=sharing 
- TimesFM setup here: https://github.com/google-research/timesfm 
- Other requirements in top level requirements.txt
- Raw data, download here: https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy (download the `all_six_dataset.zip` file)

My conda setup is as follows, but perhaps not necessary
- One environment for DASH (according to doc)
- One environment for TimesFM (according to TimesFM github instructions)
- One environment for other baselines (using `requirements.txt`)

# Data
All datasets used up to this point are forecasting datasets.

Raw data consists of csv files where each file corresponds to a single dataset, the columns are channels, and the rows are timesteps. The contents should include
- ETT-small datasets (4 of them): ETTh1, ETTh2, ETTm1, ETTm2
- Electricity (very big)
- Exchange Rate
- Illness (small)
- Traffic
- Weather (big)

In my setup, they were stored at the repo root level in a folder called `datasets`, such that the ETTh1 file was located in `[repo root]/datasets/all_six_datasets/ETT-small/ETTh1.csv`

None of the raw data is prescaled. Each dataset has fixed timesteps, but the length of the timestep varies between datasets. **The ETT-small datasets have pre-defined train/val/test splits (See https://huggingface.co/datasets/ETDataset/ett)**

# Problem Setup/Data Samples

Although each channel is a series of scalar values, the long-horizon forecasting problem is as follows: given input samples (x<sub>t-L-1</sub>, ..., x<sub>t</sub>), predict (x<sub>t+1</sub>, ..., x<sub>t+h</sub>) where:
- t is the timestemp index (assuming single channel)
- L is the input window 
- h is the forecast horizon

Thus, (x<sub>t-L-1</sub>, ..., x<sub>t</sub>) are the training features and (x<sub>t+1</sub>, ..., x<sub>t+h</sub>) are the labels for each t. As t varies, some values in the label will eventually appear in the input.

# Dataloaders 

Dataloaders are adopted from Mono's code, adopted from this file specifically: https://github.com/moment-timeseries-foundation-model/moment-research/blob/main/moment/data/forecasting_datasets.py 

They are located in `external_ts/my_extract.py`. The code datasets (not raw datasets) have been modified to only output single channels via index wrap-around. For example, if the raw csv has 5 channels and 100 rows, then the code dataset will "see" 500 items, with items 0-99 corresponding to channel 0, 100-199 corresponding to channel 1, etc. However, the code dataset will not mix data from different channels together, e.g. if the forecast window is 512, all 512 items will come from the same channel and the end of the forecast window will not bleed into the next channel.

This behavior is not present in Mono's code as far as I know (Mono's code supports limiting the output to one channel, but not wrap-around), and this modification was done to avoid needing multiple dataloaders for a single dataset.

What the dataloaders do:
- Handle train/val/test splits and scaling. 
- Handle the predefined splits for the ETT-small datasets
- There is code that implements a custom split for the illness dataset to facilitate reproduction with certain papers that have atypical splits for that dataset. This code can be commented out to use the train/val/test split fraction that you provide

# Models

Timeseries Baseline implementations are in models.py
- Base class `TSForecaster` used to unify baseline computation
- `GlobalMean` class always outputs mean of training data
- `MovingAverage` outputs the mean of the last `k` observations, with `k` tunable.
- `AR` is deprecated, ignore it 
- `ARv2` implements the Auto-regressive(`k`) baseline, with `k` tunable, with or without first differencing. It uses the `statsmodels` package as the model fitter, but it uses a custom algorithm I wrote for vectorized multi-step prediction

The previous baselines **do not** use the dataloaders directly, but instead pull the scaled dataframes out of the dataloaders to make things faster. The following methods use the dataloaders
- `DumbMLP4TS` is an MLP with only one fully connected layer (input size -> output size). Training epochs are tuned by stopping when validation loss does not improve for a while.
- `SimpleDLinear` implements the DLinear estimator from the `darts` package. Tuning is done in the `darts` package internally.
- `TimesFM` is not integrated into my framework and is reproduced separately

All tuning is done using the validation split

# Experiments

For each dataset, most papers will run multiple experiments. For each experiment, the input length L is usually fixed, but the forecast horizon h is varied. The current settings are in the [DASH results spreadsheet](https://docs.google.com/spreadsheets/d/1UB1p8p7lmlSQb8ebXLA5AWDiLWwUIVZOlOiucHSivIA/edit?pli=1&gid=1655154626#gid=1655154626). The input length is usually the same for every dataset except for ILI (illness), which typically has a smaller input length. Same for the horizons.

Raw results are stored as slurm output logs. MLP/AR baseline results that I have calculated are [here](https://drive.google.com/file/d/1yZErjLmnC3lgfvt8ngx0_ZE2LOmbRTJK/view?usp=drive_link) and DASH results are [here](https://drive.google.com/file/d/1lIJbOsIWhJKnMkFN4oKg3eakiYdkGgr4/view?usp=sharing). The results are in the last line of the files with `output` in their name. Results in the spreadsheet are MSE, but MAE is also reported in many papers. MAE is available in the raw results for AR and MLP baslines.

The template for a single experiment for MLP and AR (both with and without first differencing) is located in `mlp_ar_baselines.py`. There are some hardcoded parameters (e.g. batch size) that have not been abstracted out.

The script to launch all experiments is in `mlp_ar_experiments.sh`. **Be sure to update the results/repo directory in the script.** Note that this script does not set your conda environment, so be sure to do that beforehand (or modify the script accordingly). 

DASH baselines were launched using `dash_experiments.sh`. **Be sure to update the results/repo directory in the script if you wish to use it**

TimesFM zero-shot results are computed in `times_fm.ipynb`. You can run the last cell to see the results I precomputed, or re-run them yourself.

# Performance Profiles
To produce the performance profiles
- Download the `Dashed dreams preliminary results.xlsx` file from https://docs.google.com/spreadsheets/d/1UB1p8p7lmlSQb8ebXLA5AWDiLWwUIVZOlOiucHSivIA/edit?pli=1&gid=1655154626#gid=1655154626
- Run `performance_profile.ipynb`

# Other files

Other files that might have useful code but otherwise are not relevant
- `wrn_experiments.ipynb`: experiments from when I couldn't get DASH to train. Includes code to train MLP and WRN
- `workbench.ipnyb`: My working notebook