
# coding: utf-8

# In[ ]:

from __future__ import division
import os as os
 
from IPython.display import HTML
import pandas as pd
import numpy as np
import os as os
from matplotlib import pyplot as plt
import seaborn as sns
from numpy import random as random



from matplotlib.colors import ListedColormap
plt.rcParams['figure.figsize'] = (12, 12)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
#plt.rcParams['savefig.dpi'] = 3*plt.rcParams['savefig.dpi']
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.loc'] = 'center left'
plt.rcParams['axes.linewidth'] = 1

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')
sns.set_style('whitegrid')
plt.close()


import re
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from itertools import count #,izip
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
import pandas as pd
import collections
import plotly.offline as offline
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
from sklearn.ensemble import IsolationForest
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from statsmodels.tsa.seasonal import seasonal_decompose
os.chdir('C:/Users/prana/Desktop/directory/WildAlert_AnamolyModels/')
data_path = 'C:/Users/prana/Desktop/directory/WildAlert_AnamolyModels/data/'



#####################################################################################################################
#####################################################################################################################

"""
def detect_anomalies_with_isolation_forest(series):
    data = series.values.reshape(-1, 1)
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(data)
    
    preds = model.predict(data)
    scores = model.decision_function(data)

    anomalies = pd.Series(preds, index=series.index)
    anomaly_scores = pd.Series(scores, index=series.index)

    positive_anomalies = series[(anomalies == -1) & (series > series.median())]

    if not positive_anomalies.empty:
        threshold_value = positive_anomalies.min()
    else:
        threshold_value = np.nan

    # Fix: create a Series to match ae_case_alert_threshold structure
    threshold_series = pd.Series([threshold_value] * len(series), index=series.index)

    return positive_anomalies, anomaly_scores, threshold_series
"""

def detect_anomalies_with_isolation_forest(series, window_size=10, contamination=0.1):
    from statsmodels.tsa.seasonal import seasonal_decompose
    from sklearn.ensemble import IsolationForest
    import pandas as pd
    import numpy as np

    # Decompose to remove trend and seasonality
    decomposition = seasonal_decompose(series, model='additive', period=window_size)
    residual = decomposition.resid.dropna()
    trend = decomposition.trend.dropna()
    seasonal = decomposition.seasonal.dropna()

    # Fit Isolation Forest on residuals
    data = residual.values.reshape(-1, 1)
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    model.fit(data)

    preds = model.predict(data)
    scores = model.decision_function(data)

    anomalies = pd.Series(preds, index=residual.index)
    anomaly_scores = pd.Series(scores, index=residual.index)

    # Define positive anomalies
    positive_anomalies = residual[(anomalies == -1) & (residual > 0)]

    # Threshold logic (optional: set threshold on original scale)
    if not positive_anomalies.empty:
        residual_threshold = positive_anomalies.min()
    else:
        residual_threshold = np.percentile(residual, 95)

    # Case alert threshold: reconstructed signal + error margin
    reconstructed = trend + seasonal
    # Ensure threshold is numeric
    error_margin = np.sqrt(abs(residual_threshold))
    alert_threshold = reconstructed + error_margin
    alert_threshold_series = pd.Series(alert_threshold, index=series.index)
    
    # Re-align anomalies to full series index
    aligned_anomalies = pd.Series(np.nan, index=series.index)
    aligned_anomalies.loc[positive_anomalies.index] = series.loc[positive_anomalies.index]

    return aligned_anomalies, anomaly_scores.reindex(series.index), alert_threshold_series


#####################################################################################################################
#####################################################################################################################

#####################################################################################################################
#####################################################################################################################
"""
def detect_anomalies_with_autoencoder_seasonality(series, window_size=10, latent_dim=3, epochs=100):
    from statsmodels.tsa.seasonal import seasonal_decompose
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Input
    import pandas as pd
    import numpy as np

    # Decompose the series to extract trend and seasonal components
    decomposition = seasonal_decompose(series, model='additive', period=window_size)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Drop NaN values resulting from decomposition
    residual = residual.dropna()
    trend = trend.dropna()
    seasonal = seasonal.dropna()

    # Prepare the input data
    X = [residual.iloc[i:i + window_size].values for i in range(len(residual) - window_size + 1)]
    X = np.array(X)

    # Define the autoencoder
    model = Sequential([
        Input(shape=(window_size,)),
        Dense(64, activation='relu'),
        Dense(latent_dim, activation='relu'),
        Dense(64, activation='relu'),
        Dense(window_size, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, X, epochs=epochs, verbose=0)

    # Predict and compute MSE
    X_pred = model.predict(X)
    mse = np.mean(np.power(X - X_pred, 2), axis=1)
    threshold = np.percentile(mse, 95)

    mse_series = pd.Series(mse, index=residual.index[window_size - 1:])
    threshold_series = pd.Series([threshold] * len(mse_series), index=mse_series.index)

    # Flatten reconstructed values
    X_pred_flat = np.zeros(len(residual))
    count = np.zeros(len(residual))
    for i in range(len(X_pred)):
        X_pred_flat[i:i + window_size] += X_pred[i]
        count[i:i + window_size] += 1
    X_pred_flat /= count
    X_pred_series = pd.Series(X_pred_flat, index=residual.index)

    # Add back trend + seasonal to get reconstructed total counts
    reconstructed_series = X_pred_series + trend + seasonal

    # Estimate alert threshold: predicted count + sqrt(mse threshold)
    error_margin = np.sqrt(threshold)
    case_alert_threshold = reconstructed_series + error_margin
    case_alert_threshold = case_alert_threshold[mse_series.index]  # align to index

    # Identify anomalies
    residual_trimmed = residual[window_size - 1:]
    anomalies = residual_trimmed[(mse_series >= threshold) & (residual_trimmed > 0)]

    return mse_series, threshold_series, X_pred_series, anomalies, trend, seasonal, case_alert_threshold
"""

def detect_anomalies_with_autoencoder_seasonality(series, window_size=10, latent_dim=3, epochs=100):
    from statsmodels.tsa.seasonal import seasonal_decompose
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Input
    import pandas as pd
    import numpy as np

    # Decompose the series to extract trend and seasonal components
    decomposition = seasonal_decompose(series, model='additive', period=window_size)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Drop NaN values resulting from decomposition
    residual = residual.dropna()
    trend = trend.dropna()
    seasonal = seasonal.dropna()

    # Prepare the input data
    X = [residual.iloc[i:i + window_size].values for i in range(len(residual) - window_size + 1)]
    X = np.array(X)

    # Define the autoencoder
    model = Sequential([
        Input(shape=(window_size,)),
        Dense(64, activation='relu'),
        Dense(latent_dim, activation='relu'),
        Dense(64, activation='relu'),
        Dense(window_size, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, X, epochs=epochs, verbose=0)
    # Predict and compute MSE
    X_pred = model.predict(X)
    mse = np.mean(np.power(X - X_pred, 2), axis=1)
    threshold = np.percentile(mse, 95)
    
    mse_series = pd.Series(mse, index=residual.index[window_size - 1:])
    threshold_series = pd.Series([threshold] * len(mse_series), index=mse_series.index)

    # Flatten reconstructed values
    X_pred_flat = np.zeros(len(residual))
    count = np.zeros(len(residual))
    for i in range(len(X_pred)):
        X_pred_flat[i:i + window_size] += X_pred[i]
        count[i:i + window_size] += 1
    X_pred_flat /= count
    X_pred_series = pd.Series(X_pred_flat,index=residual.index)

    # Add back trend + seasonal to get reconstructed total counts
    reconstructed_series = (X_pred_series + trend + seasonal)

    # ==========================================================
    # <<< CHANGED SECTION START
    # ==========================================================

    reconstruction_error = (series.loc[reconstructed_series.index] - reconstructed_series)
     # rolling SD of reconstruction errors
    rolling_error_sd = (reconstruction_error.rolling(window=window_size, min_periods=max(5, window_size // 2)).std())
    # fill first few values
    rolling_error_sd = rolling_error_sd.fillna(rolling_error_sd.median())
    case_alert_threshold = (reconstructed_series + 2 * rolling_error_sd)
    case_alert_threshold = case_alert_threshold.loc[mse_series.index]
    rolling_error_sd = rolling_error_sd.loc[mse_series.index]

    # ==========================================================
    # <<< CHANGED SECTION END
    # ==========================================================

    # Identify anomalies (unchanged)
    observed_trimmed = observed_series.loc[
        mse_series.index
    ]

    anomalies = observed_trimmed[
        observed_trimmed > case_alert_threshold
    ]


    return mse_series, threshold_series, X_pred_series, anomalies, trend, seasonal, case_alert_threshold, rolling_error_sd

#####################################################################################################################
#####################################################################################################################
def get_anomaly_dataframe(weekly_vo, ae_anomalies_adjusted):
    # Create a DataFrame with dates as the index
    anomaly_df = pd.DataFrame(index=weekly_vo.index)
    anomaly_df['# admissions'] = weekly_vo['# admissions']
    anomaly_df['rolling mean'] = weekly_vo['rolling mean']
    anomaly_df['anomalies'] = weekly_vo['anomalies']
    anomaly_df['rolling std'] = weekly_vo['rolling std']
    
    # Standard Deviation Method
    anomaly_df['Std Dev Anomaly'] = weekly_vo['anomalies'].apply(lambda x: 1 if not pd.isna(x) else 0)
    # Isolation Forest Method
    anomaly_df['Isolation Forest Anomaly'] = weekly_vo['IOF_anomalies'].apply(lambda x: 1 if not pd.isna(x) else 0)
    # Autoencoder with Seasonality Method
    # Creating a binary column where 1 indicates an anomaly
    ae_anomaly_flags = ae_anomalies_adjusted.apply(lambda x: 1 if not pd.isna(x) else 0)
    # Align this with the original timeline
    aligned_ae_anomalies = pd.Series(0, index=weekly_vo.index)
    aligned_ae_anomalies.loc[ae_anomaly_flags.index] = ae_anomaly_flags
    anomaly_df['Autoencoder Anomaly'] = aligned_ae_anomalies
    
    
    return anomaly_df

#####################################################################################################################
#####################################################################################################################

def inject_gradual_trend(series, strength=0.1):
    trend = np.linspace(0, strength * series.max(), len(series))
    return series + trend

from itertools import product

def evaluate_model_sensitivity(data, filename, time_bin='W', sigma=3, 
                                window_sizes=[3, 5, 7, 10, 15, 20],
                                contamination_levels=[0.01, 0.05, 0.1, 0.2],
                                outlier_options=[True, False],
                                trend_options=[True, False]):

    results = []
    base_series = data.resample(time_bin)['case_unique_id'].count().dropna()

    # Generate all combinations
    for ws, contamination, add_outliers, add_trend in product(window_sizes, contamination_levels, outlier_options, trend_options):
        # Copy original series to avoid contamination
        series = base_series.copy()

        if add_outliers:
            outlier_indices = np.random.choice(series.index, size=max(1, int(0.05 * len(series))), replace=False)
            series[outlier_indices] += np.random.randint(10, 50, size=len(outlier_indices))

        if add_trend:
            series = inject_gradual_trend(series)

        try:
            # Autoencoder
            _, _, _, ae_anomalies, *_ = detect_anomalies_with_autoencoder_seasonality(
                series, window_size=ws, latent_dim=3, epochs=50)
            ae_anomaly_count = len(ae_anomalies)

            # Isolation Forest
            iof_anomalies, _, _ = detect_anomalies_with_isolation_forest(
                series, window_size=ws, contamination=contamination)
            iof_anomaly_count = iof_anomalies.dropna().shape[0]

            # Overlap
            overlap = len(set(ae_anomalies.index).intersection(iof_anomalies.dropna().index))

            results.append({
                'filename': filename,
                'window_size': ws,
                'contamination': contamination,
                'add_outliers': add_outliers,
                'add_trend': add_trend,
                'ae_anomaly_count': ae_anomaly_count,
                'iof_anomaly_count': iof_anomaly_count,
                'overlap_count': overlap
            })
        except Exception as e:
            print(f"Skipped combination (ws={ws}, contamination={contamination}, outliers={add_outliers}, trend={add_trend}) due to error: {e}")

    return pd.DataFrame(results)



#####################################################################################################################
#####################################################################################################################

def compare_anomaly_models_with_alert_types_named(raters_df, plot_name="unknown", sensitivity_df=None):
    from sklearn.metrics import cohen_kappa_score, jaccard_score
    import statsmodels.stats.inter_rater as irr

    # Compute metrics
    cohen_kappa = cohen_kappa_score(raters_df['Isolation Forest Anomaly'], raters_df['Autoencoder Anomaly'])
    jaccard = jaccard_score(raters_df['Isolation Forest Anomaly'], raters_df['Autoencoder Anomaly'])

    # Fleiss' Kappa
    counts = pd.DataFrame(0, index=raters_df.index, columns=[0, 1])
    counts[1] = raters_df.sum(axis=1)
    counts[0] = raters_df.shape[1] - counts[1]
    fleiss_kappa = irr.fleiss_kappa(counts.values, method='fleiss')

    # Overlap
    both_agree = ((raters_df['Autoencoder Anomaly'] == 1) & 
                  (raters_df['Isolation Forest Anomaly'] == 1)).sum()
    ae_total = raters_df['Autoencoder Anomaly'].sum()
    iof_total = raters_df['Isolation Forest Anomaly'].sum()
    overlap_pct = both_agree / max(ae_total, iof_total) * 100 if max(ae_total, iof_total) > 0 else 0

    # Core metrics in long format
    long_records = []

    core_metrics = [
        ('Total anomalies detected', ae_total, iof_total),
        ('Common anomalies (both)', both_agree, both_agree),
        ('Jaccard Similarity (IOF vs AE)', jaccard, jaccard),
        ('Cohen’s Kappa (IOF vs AE)', cohen_kappa, cohen_kappa),
        ('Fleiss’ Kappa (All)', fleiss_kappa, fleiss_kappa),
        ('% Overlap in Anomalies', overlap_pct, overlap_pct),
        ('Sensitivity to window size', 'Moderate', 'Low'),
        ('Sensitive to contamination', 'N/A', 'High'),
        ('Good at gradual trends', '✅', '❌'),
        ('Robust to outliers', '✅', '✅')
    ]

    for metric, ae_val, iof_val in core_metrics:
        long_records.append({'Plot Name': plot_name, 'Model': 'Autoencoder', 'Metric': metric, 'Value': ae_val})
        long_records.append({'Plot Name': plot_name, 'Model': 'Isolation Forest', 'Metric': metric, 'Value': iof_val})

    # Add sensitivity analysis results (if provided)
    if sensitivity_df is not None:
        ae_mean = sensitivity_df['ae_anomaly_count'].mean()
        iof_mean = sensitivity_df['iof_anomaly_count'].mean()
        overlap_mean = sensitivity_df['overlap_count'].mean()

        long_records.extend([
            {'Plot Name': plot_name, 'Model': 'Autoencoder', 'Metric': 'Sensitivity Test (Mean Anomalies)', 'Value': round(ae_mean, 2)},
            {'Plot Name': plot_name, 'Model': 'Isolation Forest', 'Metric': 'Sensitivity Test (Mean Anomalies)', 'Value': round(iof_mean, 2)},
            {'Plot Name': plot_name, 'Model': 'Autoencoder', 'Metric': 'Sensitivity Test (Mean Overlap)', 'Value': round(overlap_mean, 2)},
            {'Plot Name': plot_name, 'Model': 'Isolation Forest', 'Metric': 'Sensitivity Test (Mean Overlap)', 'Value': round(overlap_mean, 2)},
        ])

    summary_df_long = pd.DataFrame(long_records)
    return summary_df_long






#####################################################################################################################
#####################################################################################################################

def plot_anomalies(filename, window_size, sigma, time_bin='W', plot=False, save=False, verbose=False, data_loaded = False, data= None, save_anomalies=False):
    if not data_loaded:
        from WildAlertpy import read_data as read_data 
        data = read_data.read_data(filename=filename + ".xlsx")
    else:
        data = data
    
    vo = data
    # ------------------------------------------------------------------
    # Case ID column resolver (schema-safe)
    # ------------------------------------------------------------------
    if 'case_unique_id' in vo.columns:
        case_col = 'case_unique_id'
    elif 'case_number' in vo.columns:
        case_col = 'case_number'
    else:
        raise KeyError(
            "Neither 'case_unique_id' nor 'case_number' found in input data. "
            f"Available columns: {list(vo.columns)}"
        )

    weekly_vo = vo.resample(time_bin)[case_col].count()
    weekly_vo = pd.DataFrame(weekly_vo)
    weekly_vo.columns = ['ID']
    weekly_vo['rolling_mean'] = weekly_vo.ID.rolling(window=window_size, center=False).mean()
    weekly_vo['residual'] = weekly_vo.ID - weekly_vo.rolling_mean
    weekly_vo['std'] = weekly_vo.residual.std(axis=0)
    weekly_vo['testing_std'] = weekly_vo.residual.rolling(window=window_size, center=False).std()
    weekly_vo.testing_std.fillna(weekly_vo.testing_std.mean(), inplace=True)
    series = weekly_vo['ID'].squeeze()
    #print("Original Time-Series len: ", weekly_vo.shape)
    
    def identify_anomalies(c, sigma=sigma):
        if c.ID > c.rolling_mean + (sigma * c.testing_std):
            return c.ID
    
    weekly_vo['anomalies'] = weekly_vo.apply(identify_anomalies, axis=1)
    weekly_vo.columns = ['# admissions', 'rolling mean', 'std', 'residual', 'rolling std', 'anomalies']
    
    # Isolation Forest anomalies
    iof_preds, iof_scores, iof_case_alert_threshold = detect_anomalies_with_isolation_forest(series)
    #print(iof_preds),
    #print(iof_scores), 
    #print(iof_case_alert_threshold)
    weekly_vo["IOF_anomalies"] = iof_preds
    def identify_IOFanomalies(c):
        if c.IOF_anomalies == -1:
            return c['# admissions']
    weekly_vo["IOF_anomalies"] = iof_preds
    
    # Autoencoder with seasonality
    ae_mse, ae_threshold, ae_X_pred, ae_anomalies, trend, seasonal, ae_case_alert_threshold = detect_anomalies_with_autoencoder_seasonality(series, window_size=window_size)
    # Ensure the lengths match for plotting
    aligned_index = weekly_vo.index.intersection(ae_mse.index)
    weekly_vo = weekly_vo.copy()
    weekly_vo['ae_mse'] = ae_mse.reindex(weekly_vo.index)
    weekly_vo['ae_threshold'] = ae_threshold.reindex(weekly_vo.index)
    weekly_vo['ae_X_pred'] = ae_X_pred.reindex(weekly_vo.index)
    
    # Adjust ae_anomalies to include trend and seasonal components
    #ae_anomalies_adjusted = ae_anomalies + trend[ae_anomalies.index] + seasonal[ae_anomalies.index]
    ae_anomalies_adjusted = ae_anomalies + trend[ae_anomalies.index] + seasonal[ae_anomalies.index]
    

    ######################################################################
    ######################################################################
    ######################################################################
    ##### Generating data frame to save with anomalies detected ###########
    anomaly_df = get_anomaly_dataframe(weekly_vo, ae_anomalies_adjusted)
    #print ('merge complete')
    anomaly_df['Autoencoder MSE'] = ae_mse
    anomaly_df['Autoencoder Threshold'] = ae_threshold
    anomaly_df['Autoencoder Case Alert Threshold'] = ae_case_alert_threshold
    anomaly_df['IOF Case Alert Threshold'] = iof_case_alert_threshold
    anomaly_df['AE Intensity'] = ae_mse
    anomaly_df['IOF Intensity'] = iof_scores
    #print(weekly_vo.head())
    #print(anomaly_df.head())
    name = filename.replace(data_path, "").replace("-", " ").title()
    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Base line: # admissions
        anomaly_df['# admissions'].plot(ax=ax, label='# admissions')
    
        # Rolling mean
        if 'rolling mean' in anomaly_df.columns:
            anomaly_df['rolling mean'].plot(ax=ax, label='rolling mean', color='black')
        # Add ±2 * rolling std band (only where rolling std is available)
        if 'rolling std' in anomaly_df.columns:
            upper_band = anomaly_df['rolling mean'] + 2 * anomaly_df['rolling std']
            lower_band = anomaly_df['rolling mean'] - 2 * anomaly_df['rolling std']
            ax.fill_between(anomaly_df.index, upper_band, lower_band, alpha=0.3, color='gray', label='±2 Rolling Std Dev')
        
        # Std Dev Anomalies
        std_anom = anomaly_df[anomaly_df['Std Dev Anomaly'] == 1]
        ax.scatter(std_anom.index, std_anom['# admissions'], color='red', label='Std Dev', marker='o', s=80)
        
        # Isolation Forest Anomalies
        iof_anom = anomaly_df[anomaly_df['Isolation Forest Anomaly'] == 1]
        ax.scatter(iof_anom.index, iof_anom['# admissions'], color='black', label='IOF', marker='X', s=100)
    
        # Autoencoder Anomalies
        ae_anom = anomaly_df[anomaly_df['Autoencoder Anomaly'] == 1]
        ax.scatter(ae_anom.index, ae_anom['# admissions'], color='green', label='ae_anomalies', marker='o', s=100)
    
        # Case Alert Thresholds
        if 'Autoencoder Case Alert Threshold' in anomaly_df.columns:
            anomaly_df['Autoencoder Case Alert Threshold'].plot(
                ax=ax, color='blue', linestyle='--', label='Case Alert Threshold (AE)', lw=3)
        
        if 'IOF Case Alert Threshold' in anomaly_df.columns:
            anomaly_df['IOF Case Alert Threshold'].plot(
                ax=ax, color='orange', linestyle='--', label='Case Alert Threshold (IOF)', lw=3)
    
        # Styling
        ax.set_ylim(0)
        ax.set_ylabel('number of animals')
        ax.set_xlabel('date')
        name = filename.replace(data_path, "").replace("-", " ").title()
        ax.set_title(name)
        ax.legend(loc='upper left')
        plt.tight_layout()
    
        if save:
            plt.savefig(filename + ".png", dpi=600)
            plt.savefig(filename + ".svg", dpi=600)
    
        plt.show()

    # Sum across the columns for each row
    anomaly_sum = anomaly_df.sum(axis=1)
    # Filter the DataFrame to keep only rows where the sum is greater than 0
    filtered_anomaly_df = anomaly_df[anomaly_sum > 0]
    # Add a new column that checks if all methods agree on an anomaly
    #filtered_anomaly_df['agreement'] = filtered_anomaly_df.apply(lambda row: 1 if row.sum() == len(filtered_anomaly_df.columns) else 0, axis=1)
    filtered_anomaly_df.loc[:, 'agreement'] = filtered_anomaly_df.apply(lambda row: 1 if row.sum() == len(filtered_anomaly_df.columns) else 0,   axis=1)
    anomaly_df['IOF Intensity'] = iof_scores
    anomaly_df['AE Intensity'] = ae_mse
    # Display the resulting DataFrame
    import statsmodels.stats.inter_rater as irr
    categories = [0, 1]
    # Just use the binary columns for anomaly detection
    raters_df = anomaly_df[['Std Dev Anomaly', 'Isolation Forest Anomaly', 'Autoencoder Anomaly']].astype(int)
    
    # Now count occurrences of each category per row
    counts = pd.DataFrame(0, index=raters_df.index, columns=[0, 1])
    counts[1] = raters_df.sum(axis=1)
    counts[0] = raters_df.shape[1] - counts[1]
    
    # Check consistency
    assert all((counts[0] + counts[1]) == raters_df.shape[1]), "Each row must sum to number of raters"
    # Compute Fleiss' Kappa
    fleiss_kappa = irr.fleiss_kappa(counts.values, method='fleiss')
    print(f"Fleiss' Kappa: {fleiss_kappa:.3f}")
    from sklearn.metrics import cohen_kappa_score
    # Cohen’s Kappa: pairwise comparison (e.g., IOF vs AE)
    kappa = cohen_kappa_score(raters_df['Isolation Forest Anomaly'], raters_df['Autoencoder Anomaly'])
    print(f"Cohen's Kappa (IOF vs AE): {kappa:.3f}") 
    ## new code 06/28/2025
    # Create and return model comparison summary
    # Run model sensitivity analysis with current data and parameters
    print("running sensitivity")
    sensitivity_df = evaluate_model_sensitivity(data=data,
                                                filename=name, time_bin=time_bin, window_sizes=[5,7,10,15,20],  # or leave default
                                                contamination_levels=[0.01, 0.05, 0.1, 0.2],
                                                outlier_options=[True, False],
                                                trend_options=[True, False])
    
    summary_df = compare_anomaly_models_with_alert_types_named(raters_df, plot_name=name, sensitivity_df=sensitivity_df)
    #print(summary_df)
    
    merged_df = pd.merge(weekly_vo, anomaly_df, left_index=True, right_index=True, how='left')
    if save_anomalies:
        anomaly_filename = f"{filename}_anomalies.csv"
        merged_df.to_csv(anomaly_filename)
        print(f"Anomalies saved to {anomaly_filename}")


    
    # Add mean/summary stats to summary_df
    ae_mean = sensitivity_df['ae_anomaly_count'].mean()
    iof_mean = sensitivity_df['iof_anomaly_count'].mean()
    overlap_mean = sensitivity_df['overlap_count'].mean()

    # Add summary metrics from sensitivity analysis
    #summary_df.loc[len(summary_df)] = [
    #    'Sensitivity Test (Mean Anomalies)',  # Alert Type / Metric
    #    round(ae_mean, 2),
    #    round(iof_mean, 2)
    #]
    #summary_df.loc[len(summary_df)] = [
    #    'Sensitivity Test (Mean Overlap)',
    #    round(overlap_mean, 2),
    #    round(overlap_mean, 2)
    #]

    
    return anomaly_df, summary_df, sensitivity_df
# Example usage:
# plot_anomalies('example_filename', window_size=10, sigma=3, time_bin='W', plot=True, save=False)

#####################################################################################################################





    