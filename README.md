# MiMe: Multi-Mask Online Ensemble Time Series Forecasting Framework



## Quick Start

### 1. Environment Setup

This project uses Python 3.8 and CUDA 12.6. Specifically : 

`torch==2.3.0+cu121 \
torchvision==0.18.0+cu121 \
torchaudio==2.3.0+cu121 \`

First, install the required Python dependency packages:

`pip install -r requirements.txt`

### 2. Model Training

Run the training script to start the model training process:

`bash online.sh`

This command runs our model and all baselines from the paper on every public dataset included in the study, with forecast lengths set to 12 and 24 steps.

The MSE and MAE results are saved in *result_long_term_forecast.txt* , the detailed information is saved in *result_mse_mae.csv*, models are saved in *checkpoints/*. If you want the prediction results with specific numerical values and visualizations, please uncomment the saving codes in *exp_long_term_forecasting.py* and *exp_MiMe_forecasting.py*.



## Acknowledgements

This project is built upon the excellent framework provided by [Time-Series-Library]([thuml/Time-Series-Library: A Library for Advanced Deep Time Series Models for General Time Series Analysis.](https://github.com/thuml/Time-Series-Library)).

We thank the original authors for making their code publicly available.
