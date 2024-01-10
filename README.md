# Sales Forecasting and Reconciliation Project

## Overview
This project focuses on sales forecasting and reconciliation, utilizing a combination of forecasting algorithms at different aggregation levels. The lower level employs DeepAR, the middle level uses LGBM, and the top level leverages Exponential Smoothing. Forecast reconciliation is performed using the Nixtla package.

## Features
- Data Exploration: Analyze the sales time series dataset to gain insights.
- Data Preparation: Perform necessary preprocessing and transformations for model training.
- Hierarchical Forecasting: Utilize DeepAR, LGBM, and Exponential Smoothing at different aggregation levels.
- Forecast Reconciliation: Combine forecasts using the Nixtla package to obtain a coherent prediction.
- Model Persistence: Save trained models and chosen hyperparameters for reproducibility.
- Results and Metrics: Calculate and analyze forecasting results and metrics. Still missing a Dashboard Script to present results
## Getting Started
### Installation

```
pip install -r requirements.txt
```
## Example Usage
Adjust the code in process.py to set the correct directory and file paths.

Run the following command to execute the entire workflow:

```
python process.py
```

## How it Works
The project uses a combination of DeepAR, LGBM, and Exponential Smoothing for sales forecasting at different levels. The forecasts are reconciled using the Nixtla package to ensure coherence across all levels.

## Acknowledgments
Nixtla: A special thanks to the Nixtla package for providing the foundation for hierarchical forecasting and reconciliation.

## Support
If you encounter any issues or have questions, feel free to open an issue or reach out to the maintainers.
