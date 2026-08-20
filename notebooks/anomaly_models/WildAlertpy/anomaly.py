
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
import matplotlib.dates as mdates
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
    
    #weekly_vo = weekly_vo.loc[aligned_index]
    weekly_vo = weekly_vo.copy()
    weekly_vo['ae_mse'] = ae_mse.reindex(weekly_vo.index)
    weekly_vo['ae_threshold'] = ae_threshold.reindex(weekly_vo.index)
    weekly_vo['ae_X_pred'] = ae_X_pred.reindex(weekly_vo.index)
    
    #ae_mse = ae_mse.loc[aligned_index]
    #ae_threshold = ae_threshold.loc[aligned_index]
    #ae_X_pred = ae_X_pred.loc[aligned_index]
    
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
    # Normalize intensities
    #print(weekly_vo.head())
    #print(anomaly_df.head())
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
        ax.set_title(filename.replace(data_path, "").replace("-", " ").title())
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
    merged_df = pd.merge(weekly_vo, anomaly_df, left_index=True, right_index=True, how='left')
    if save_anomalies:
        anomaly_filename = f"{filename}_anomalies.csv"
        merged_df.to_csv(anomaly_filename)
        print(f"Anomalies saved to {anomaly_filename}")
    # Save anomaly intensities into full anomaly_df for downstream plotting
    
    return anomaly_df
    #return filtered_anomaly_df
# Example usage:
# plot_anomalies('example_filename', window_size=10, sigma=3, time_bin='W', plot=True, save=False)

#####################################################################################################################

"""
def plot_composite_anomalies(data_dir, title, time_bin='W', sigma=2, window_size=10, max_plots=25, save=False, save_path=None):
    from WildAlertpy import read_data as read_data
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    file_list = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    print(file_list)
    n_files = min(len(file_list), max_plots)
    n_cols = 3
    n_rows = (n_files + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex=False)
    axes = axes.flatten()

    for idx, file in enumerate(file_list[:n_files]):
        file_path = os.path.join(data_dir, file)
        try:
            df = read_data.read_data(filename=file_path)
            short_name = os.path.splitext(os.path.basename(file))[0]

            anomaly_df = plot_anomalies(file_path.replace('.xlsx',''), 
                                        sigma=sigma, window_size=window_size, 
                                        time_bin=time_bin, plot=False, save=False, 
                                        data_loaded=True, data=df)

            ax = axes[idx]

            # Plot base line
            anomaly_df['# admissions'].plot(ax=ax, label='# admissions')

            # Rolling mean
            if 'rolling mean' in anomaly_df.columns:
                anomaly_df['rolling mean'].plot(ax=ax, label='rolling mean', color='black')

            # ±2 Rolling Std Dev
            if 'rolling std' in anomaly_df.columns:
                upper_band = anomaly_df['rolling mean'] + 2 * anomaly_df['rolling std']
                lower_band = anomaly_df['rolling mean'] - 2 * anomaly_df['rolling std']
                ax.fill_between(anomaly_df.index, upper_band, lower_band, alpha=0.3, color='gray', label='±2 Rolling Std Dev')

            # Isolation Forest Anomalies
            iof_anom = anomaly_df[anomaly_df['Isolation Forest Anomaly'] == 1]
            ax.scatter(iof_anom.index, iof_anom['# admissions'], color='black', label='IOF', marker='X', s=100)

            # Autoencoder Anomalies
            ae_anom = anomaly_df[anomaly_df['Autoencoder Anomaly'] == 1]
            ax.scatter(ae_anom.index, ae_anom['# admissions'], color='green', label='AE', marker='o', s=100)

            ax.set_ylim(0)
            ax.set_ylabel('Cases')
            ax.set_xlabel('Date')
            ax.set_title(short_name.replace("-", " ").title(), fontsize=12)

        except Exception as e:
            print(f"Failed to process {file}: {e}")
            axes[idx].axis('off')
            continue

    for ax in axes[n_files:]:
        ax.axis('off')

    # Create custom legend handles
    legend_elements = [
        mlines.Line2D([], [], color='blue', label='# admissions'),
        mlines.Line2D([], [], color='black', label='rolling mean'),
        mpatches.Patch(color='gray', alpha=0.3, label='±2 Rolling Std Dev'),
        mlines.Line2D([], [], color='black', marker='X', linestyle='None', markersize=10, label='Isolation Forest Anomalies'),
        mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=10, label='Autoencoder Anomalies'),
    ]

    # Title and shared legend
    #fig.suptitle(title, fontsize=18, y=1.08)
    #fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=13)
    #fig.subplots_adjust(top=0.88, hspace=0.4)

    if save:
        if save_path is None:
            save_path = os.path.join(data_dir, f"{title.replace(' ', '_').lower()}_composite.png")
        plt.savefig(save_path, dpi=600)
        print(f"Figure saved to {save_path}")

    plt.show()
"""
#####################################################################################################################

def plot_composite_anomalies(data_dir, title, time_bin='W', sigma=2, window_size=10, max_plots=25, save=False, save_path=None):
    import os
    import matplotlib.pyplot as plt
    from WildAlertpy import read_data as read_data
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    file_list = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    print(file_list)
    n_files = min(len(file_list), max_plots)
    n_cols = 3
    n_rows = (n_files + n_cols - 1) // n_cols

    # Wider figure for landscape layout
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 4.5 * n_rows), sharex=False)
    axes = axes.flatten()

    for idx, file in enumerate(file_list[:n_files]):
        file_path = os.path.join(data_dir, file)
        try:
            df = read_data.read_data(filename=file_path)
            short_name = os.path.splitext(os.path.basename(file))[0]

            anomaly_df = plot_anomalies(file_path.replace('.xlsx',''), 
                                        sigma=sigma, window_size=window_size, 
                                        time_bin=time_bin, plot=False, save=False, 
                                        data_loaded=True, data=df)

            ax = axes[idx]

            anomaly_df['# admissions'].plot(ax=ax, label='# admissions')

            if 'rolling mean' in anomaly_df.columns:
                anomaly_df['rolling mean'].plot(ax=ax, label='rolling mean', color='black')

            if 'rolling std' in anomaly_df.columns:
                upper_band = anomaly_df['rolling mean'] + 2 * anomaly_df['rolling std']
                lower_band = anomaly_df['rolling mean'] - 2 * anomaly_df['rolling std']
                ax.fill_between(anomaly_df.index, upper_band, lower_band, alpha=0.3, color='gray', label='±2 Rolling Std Dev')

            iof_anom = anomaly_df[anomaly_df['Isolation Forest Anomaly'] == 1]
            ax.scatter(iof_anom.index, iof_anom['# admissions'], color='black', label='IOF', marker='X', s=100)

            ae_anom = anomaly_df[anomaly_df['Autoencoder Anomaly'] == 1]
            ax.scatter(ae_anom.index, ae_anom['# admissions'], color='green', label='AE', marker='o', s=100)

            ax.set_ylim(0)
            if idx // n_cols != n_rows - 1:
                #ax.set_xticklabels([])
                ax.set_xlabel('')
            else:
                ax.set_xlabel('Date')

            if idx % n_cols != 0:
                #ax.set_yticklabels([])
                ax.set_ylabel('')
            else:
                ax.set_ylabel('Cases')

            ax.set_title(short_name.replace("-", " ").title(), fontsize=12)

        except Exception as e:
            print(f"Failed to process {file}: {e}")
            axes[idx].axis('off')
            continue

    for ax in axes[n_files:]:
        ax.axis('off')

    legend_elements = [
        mlines.Line2D([], [], color='blue', label='# admissions'),
        mlines.Line2D([], [], color='black', label='rolling mean'),
        mpatches.Patch(color='gray', alpha=0.3, label='±2 Rolling Std Dev'),
        mlines.Line2D([], [], color='black', marker='X', linestyle='None', markersize=10, label='Isolation Forest Anomalies'),
        mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=10, label='Autoencoder Anomalies'),
    ]

    # Shared legend and layout adjustment
    #fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=13)
    #fig.suptitle(title, fontsize=18, y=1.08)
    #fig.subplots_adjust(top=0.9, hspace=0.4, wspace=0.2)

    if save:
        if save_path is None:
            save_path = os.path.join(data_dir, f"{title.replace(' ', '_').lower()}_composite.png")
        
        # Capture title and legend for tight saving
        legend = fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.001), ncol=5, fontsize=13)
        title_obj = fig.suptitle(title, fontsize=18, y=1.02)

        # Save with all extra artists
        plt.savefig(save_path, dpi=600, bbox_inches='tight', bbox_extra_artists=[legend, title_obj])
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
#####################################################################################################################

def plot_composite_anomalies(data_dir, title = None, time_bin='W', sigma=2, window_size=10, max_plots=25, save=False, save_path=None):
    import os
    import matplotlib.pyplot as plt
    from WildAlertpy import read_data as read_data
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    file_list = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    print(file_list)
    n_files = min(len(file_list), max_plots)
    n_cols = 3
    n_rows = (n_files + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 4.5 * n_rows), sharex=False, constrained_layout=True)
    axes = axes.flatten()

    for idx, file in enumerate(file_list[:n_files]):
        file_path = os.path.join(data_dir, file)

        try:
            df = read_data.read_data(filename=file_path)
            short_name = os.path.splitext(os.path.basename(file))[0]

            # Capital letter subplot prefix: A), B), C), ...
            subplot_label = f"{chr(65 + idx)}) "   # 65 = 'A'

            anomaly_df = plot_anomalies(
                file_path.replace('.xlsx',''),
                sigma=sigma, window_size=window_size,
                time_bin=time_bin, plot=False, save=False,
                data_loaded=True, data=df
            )

            ax = axes[idx]

            anomaly_df['# admissions'].plot(ax=ax, label='# admissions')

            if 'rolling mean' in anomaly_df.columns:
                anomaly_df['rolling mean'].plot(ax=ax, label='rolling mean', color='black')

            if 'rolling std' in anomaly_df.columns:
                upper_band = anomaly_df['rolling mean'] + 2 * anomaly_df['rolling std']
                lower_band = anomaly_df['rolling mean'] - 2 * anomaly_df['rolling std']
                ax.fill_between(
                    anomaly_df.index,
                    upper_band, lower_band,
                    alpha=0.3, color='gray',
                    label='±2 Rolling Std Dev'
                )

            iof_anom = anomaly_df[anomaly_df['Isolation Forest Anomaly'] == 1]
            ax.scatter(
                iof_anom.index, iof_anom['# admissions'],
                color='black', marker='X', s=100, label='IOF'
            )

            ae_anom = anomaly_df[anomaly_df['Autoencoder Anomaly'] == 1]
            ax.scatter(
                ae_anom.index, ae_anom['# admissions'],
                color='green', marker='o', s=100, label='AE'
            )

            ax.set_ylim(0)

            if idx // n_cols != n_rows - 1:
                ax.set_xlabel('')
            else:
                ax.set_xlabel('Date')

            if idx % n_cols != 0:
                ax.set_ylabel('')
            else:
                ax.set_ylabel('Cases')

            # Final subplot title with prefix
            ax.set_title(subplot_label + short_name.replace("-", " ").title(), fontsize=12)

        except Exception as e:
            print(f"Failed to process {file}: {e}")
            axes[idx].axis('off')
            continue

    # Turn off unused axes
    for ax in axes[n_files:]:
        ax.axis('off')

    legend_elements = [
        mlines.Line2D([], [], color='blue', label='# admissions'),
        mlines.Line2D([], [], color='black', label='rolling mean'),
        mpatches.Patch(color='gray', alpha=0.3, label='±2 Rolling Std Dev'),
        mlines.Line2D([], [], color='black', marker='X', linestyle='None', markersize=10, label='Isolation Forest Anomalies'),
        mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=10, label='Autoencoder Anomalies'),
    ]

    if save:
        if save_path is None:
            save_path = os.path.join(data_dir, f"{title.replace(' ', '_').lower()}_composite.png")

        legend = fig.legend(
            handles=legend_elements,
            loc='upper center', 
            ncol=5, fontsize=13
        )
        title_obj = fig.suptitle(title, fontsize=18)

        plt.savefig(
            save_path,
            dpi=600, bbox_inches='tight',
            bbox_extra_artists=[legend, title_obj]
        )
        print(f"Figure saved to {save_path}")

    plt.show()

#####################################################################################################################

def plot_multiple_anomaly_timelines(filenames,
                                     window_size,
                                     sigma,
                                     time_bin='W',
                                     grey_periods=None,
                                     disease_labels=None,
                                     save=False,
                                     figsize_per_plot=(9, 5),
                                     save_path="wildalert_surveillance_stacked.png",
                                     plot_start_date="2022-01-01",
                                     plot_top_axis=True,
                                     highlight_events=False, plot_map=False,  # <-- new
                                     # dict of GeoDataFrames keyed by filename
                                     state_shapes=None):  # New argument
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as mpatches
    import matplotlib.cm as cm
    import numpy as np
    from collections import defaultdict

    n = len(filenames)
    fig_height = figsize_per_plot[1] * n * 1.0
    fig = plt.figure(figsize=(figsize_per_plot[0], fig_height))

    # Use 3 rows per filename: 1 for intensity (narrow), 2 for main plot
    row_height = 4 if plot_top_axis else 3
    gs = GridSpec(n * row_height, 1, hspace=0.2)

    # Consistent color assignment for disease labels
    unique_labels = sorted(set(disease_labels)) if disease_labels else []
    color_map = cm.get_cmap('Paired', len(unique_labels))
    label_to_color = {label: color_map(i) for i, label in enumerate(unique_labels)}

    for idx, filename in enumerate(filenames):
        anomaly_df = plot_anomalies(filename=filename,
                                    window_size=window_size,
                                    sigma=sigma,
                                    time_bin=time_bin,
                                    plot=False,
                                    save=False,
                                    verbose=False,
                                    data_loaded=False,
                                    data=None,
                                    save_anomalies=False)

        print(f"{filename} date range: {anomaly_df.index.min()} to {anomaly_df.index.max()}")

        # Filter by date
        anomaly_df = anomaly_df[anomaly_df.index >= pd.to_datetime(plot_start_date)]
        if anomaly_df.empty:
            print(f"No data for {filename} after {plot_start_date}")
            continue

        def scale_zero_mean_unit_range(x):
            if x.max() == x.min():
                return x * 0
            x_centered = x - x.mean()
            return x_centered / max(abs(x_centered.min()), abs(x_centered.max()))

        ae_scaled = scale_zero_mean_unit_range(anomaly_df['AE Intensity']) if 'AE Intensity' in anomaly_df else None
        iof_scaled = -1 * (scale_zero_mean_unit_range(anomaly_df['IOF Intensity']) if 'IOF Intensity' in anomaly_df else None)

        if plot_top_axis:
            ax_top = fig.add_subplot(gs[idx * row_height])
            if ae_scaled is not None:
                ae_scaled.plot(ax=ax_top, label='AE Intensity (scaled)', color='green', lw=1.5)
            if iof_scaled is not None:
                iof_scaled.plot(ax=ax_top, label='IOF Intensity (scaled)', color='black', lw=1.5)

            ax_top.set_ylabel("WildAlert\nSignal Intensity")
            ax_top.legend(loc='best', fontsize=7.5)
            ax_top.grid(True)
            ax_top.set_ylim(-1.0, 1.0)
        else:
            ax_top = None

        main_ax_row_start = idx * row_height + (1 if plot_top_axis else 0)
        ax_main = fig.add_subplot(gs[main_ax_row_start:main_ax_row_start + (row_height - 1)],
                                  sharex=ax_top if plot_top_axis else None)

        anomaly_df['# admissions'].plot(ax=ax_main, label='# admissions', lw=1.5)
        if 'rolling mean' in anomaly_df.columns:
            anomaly_df['rolling mean'].plot(ax=ax_main, label='Rolling Mean', color='black')
        if 'rolling std' in anomaly_df.columns:
            upper = anomaly_df['rolling mean'] + 2 * anomaly_df['rolling std']
            lower = anomaly_df['rolling mean'] - 2 * anomaly_df['rolling std']
            ax_main.fill_between(anomaly_df.index, upper, lower, alpha=0.2, color='gray', label='±2 Std Dev')

        if 'Isolation Forest Anomaly' in anomaly_df.columns:
            ax_main.scatter(anomaly_df[anomaly_df['Isolation Forest Anomaly'] == 1].index,
                            anomaly_df[anomaly_df['Isolation Forest Anomaly'] == 1]['# admissions'],
                            color='black', marker='X', s=60, label='IOF Anomaly')
        if 'Autoencoder Anomaly' in anomaly_df.columns:
            ax_main.scatter(anomaly_df[anomaly_df['Autoencoder Anomaly'] == 1].index,
                            anomaly_df[anomaly_df['Autoencoder Anomaly'] == 1]['# admissions'],
                            color='green', marker='^', s=60, label='AE Anomaly')

        outbreak_patches = []
        if grey_periods:
            for i, (start, end) in enumerate(grey_periods):
                if pd.to_datetime(end) < pd.to_datetime(plot_start_date):
                    continue
                color = label_to_color[disease_labels[i]] if disease_labels and i < len(disease_labels) else 'lightgray'
                label = disease_labels[i] if disease_labels and i < len(disease_labels) else f"Outbreak {i+1}"

                # Fill the grey outbreak region
                ax_main.axvspan(pd.to_datetime(start), pd.to_datetime(end), color=color, alpha=0.3)

                # Highlight with red dotted rectangle
                if highlight_events:
                    ax_main.axvline(pd.to_datetime(start), color='red', linestyle='--', linewidth=1.5)
                    ax_main.axvline(pd.to_datetime(end), color='red', linestyle='--', linewidth=1.5)

                outbreak_patches.append(mpatches.Patch(color=color, label=label))

        ax_main.set_ylim(ymin=0)
        ax_main.set_ylabel("Cases")
        ax_main.set_xlabel("Date")
        ax_main.grid(True)

        handles_main, labels_main = ax_main.get_legend_handles_labels()
        unique_legend = dict(zip(labels_main, handles_main))
        for patch in outbreak_patches:
            unique_legend[patch.get_label()] = patch

        ax_main.legend(unique_legend.values(), unique_legend.keys(), loc='upper left', fontsize=9)

    fig.suptitle(save_path.split('.', 1)[0], fontsize=18, y=1.00)
    plt.tight_layout()
    if save:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    plt.show()
    if plot_map:
        plot_state_case_map(save_path, filename, grey_periods, disease_labels, state_shapes)

#####################################################################################################################

def plot_multiple_anomaly_timelines(filenames,
                                     window_size,
                                     sigma,
                                     time_bin='W',
                                     grey_periods=None,
                                     disease_labels=None,
                                     save=False,
                                     figsize_per_plot=(9, 5),
                                     save_path="wildalert_surveillance_stacked.png",
                                     plot_start_date="2022-01-01",
                                     plot_top_axis=True,
                                     highlight_events=False,
                                     plot_map=False,
                                     state_shapes=None):

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as mpatches
    import matplotlib.cm as cm
    from collections import defaultdict
    from scipy.stats import ttest_ind

    def perform_statistical_tests(anomaly_df, grey_periods, column_name):
        anomaly_df = anomaly_df.copy()
        anomaly_df.index = pd.to_datetime(anomaly_df.index)
        in_period_mask = pd.Series(False, index=anomaly_df.index)
        for start, end in grey_periods:
            in_period_mask |= (anomaly_df.index >= pd.to_datetime(start)) & (anomaly_df.index <= pd.to_datetime(end))
        if column_name not in anomaly_df.columns:
            return None, None
        scores_in = anomaly_df.loc[in_period_mask, column_name].dropna()
        scores_out = anomaly_df.loc[~in_period_mask, column_name].dropna()
        if len(scores_in) < 2 or len(scores_out) < 2:
            return None, None
        t_stat, t_pval = ttest_ind(scores_in, scores_out, equal_var=False)
        observed_diff = scores_in.mean() - scores_out.mean()
        combined = np.concatenate([scores_in, scores_out])
        n_in = len(scores_in)
        count = 0
        for _ in range(1000):
            np.random.shuffle(combined)
            perm_diff = combined[:n_in].mean() - combined[n_in:].mean()
            if abs(perm_diff) >= abs(observed_diff):
                count += 1
        perm_pval = count / 1000
        return t_pval, perm_pval

    n = len(filenames)
    fig_height = figsize_per_plot[1] * n * 1.0
    fig = plt.figure(figsize=(figsize_per_plot[0], fig_height))
    row_height = 4 if plot_top_axis else 3
    gs = GridSpec(n * row_height, 1, hspace=0.2)

    unique_labels = sorted(set(disease_labels)) if disease_labels else []
    color_map = cm.get_cmap('Paired', len(unique_labels))
    label_to_color = {label: color_map(i) for i, label in enumerate(unique_labels)}

    for idx, filename in enumerate(filenames):
        anomaly_df = plot_anomalies(filename=filename,
                                    window_size=window_size,
                                    sigma=sigma,
                                    time_bin=time_bin,
                                    plot=False,
                                    save=False,
                                    verbose=False,
                                    data_loaded=False,
                                    data=None,
                                    save_anomalies=False)

        print(f"{filename} date range: {anomaly_df.index.min()} to {anomaly_df.index.max()}")
        anomaly_df = anomaly_df[anomaly_df.index >= pd.to_datetime(plot_start_date)]
        if anomaly_df.empty:
            print(f"No data for {filename} after {plot_start_date}")
            continue

        def scale_zero_mean_unit_range(x):
            if x.max() == x.min():
                return x * 0
            x_centered = x - x.mean()
            return x_centered / max(abs(x_centered.min()), abs(x_centered.max()))

        ae_scaled = scale_zero_mean_unit_range(anomaly_df['AE Intensity']) if 'AE Intensity' in anomaly_df else None
        iof_scaled = -1 * (scale_zero_mean_unit_range(anomaly_df['IOF Intensity']) if 'IOF Intensity' in anomaly_df else None)

        if plot_top_axis:
            ax_top = fig.add_subplot(gs[idx * row_height])
            if ae_scaled is not None:
                ae_scaled.plot(ax=ax_top, label='AE Intensity (scaled)', color='green', lw=1.5)
            if iof_scaled is not None:
                iof_scaled.plot(ax=ax_top, label='IOF Intensity (scaled)', color='black', lw=1.5)
            ax_top.set_ylabel("WildAlert\nSignal Intensity")
            ax_top.legend(loc='best', fontsize=7.5)
            ax_top.grid(True)
            ax_top.set_ylim(-1.0, 1.0)
        else:
            ax_top = None

        main_ax_row_start = idx * row_height + (1 if plot_top_axis else 0)
        ax_main = fig.add_subplot(gs[main_ax_row_start:main_ax_row_start + (row_height - 1)],
                                  sharex=ax_top if plot_top_axis else None)

        anomaly_df['# admissions'].plot(ax=ax_main, label='# admissions', lw=1.5)
        if 'rolling mean' in anomaly_df.columns:
            anomaly_df['rolling mean'].plot(ax=ax_main, label='Rolling Mean', color='black')
        if 'rolling std' in anomaly_df.columns:
            upper = anomaly_df['rolling mean'] + 2 * anomaly_df['rolling std']
            lower = anomaly_df['rolling mean'] - 2 * anomaly_df['rolling std']
            ax_main.fill_between(anomaly_df.index, upper, lower, alpha=0.2, color='gray', label='±2 Std Dev')

        if 'Isolation Forest Anomaly' in anomaly_df.columns:
            ax_main.scatter(anomaly_df[anomaly_df['Isolation Forest Anomaly'] == 1].index,
                            anomaly_df[anomaly_df['Isolation Forest Anomaly'] == 1]['# admissions'],
                            color='black', marker='X', s=60, label='IOF Anomaly')
        if 'Autoencoder Anomaly' in anomaly_df.columns:
            ax_main.scatter(anomaly_df[anomaly_df['Autoencoder Anomaly'] == 1].index,
                            anomaly_df[anomaly_df['Autoencoder Anomaly'] == 1]['# admissions'],
                            color='green', marker='^', s=60, label='AE Anomaly')

        outbreak_patches = []
        if grey_periods:
            for i, (start, end) in enumerate(grey_periods):
                if pd.to_datetime(end) < pd.to_datetime(plot_start_date):
                    continue
                color = label_to_color[disease_labels[i]] if disease_labels and i < len(disease_labels) else 'lightgray'
                label = disease_labels[i] if disease_labels and i < len(disease_labels) else f"Outbreak {i+1}"
                ax_main.axvspan(pd.to_datetime(start), pd.to_datetime(end), color=color, alpha=0.3)
                if highlight_events:
                    ax_main.axvline(pd.to_datetime(start), color='red', linestyle='--', linewidth=1.5)
                    ax_main.axvline(pd.to_datetime(end), color='red', linestyle='--', linewidth=1.5)
                outbreak_patches.append(mpatches.Patch(color=color, label=label))

            # Run statistical tests for AE and IOF
            ae_t_pval, ae_perm_pval = perform_statistical_tests(anomaly_df, grey_periods, 'AE Intensity')
            iof_t_pval, iof_perm_pval = perform_statistical_tests(anomaly_df, grey_periods, 'IOF Intensity')
            print(f"{filename} — AE t-test p={ae_t_pval:.4f}, AE permutation p={ae_perm_pval:.4f}")
            print(f"{filename} — IOF t-test p={iof_t_pval:.4f}, IOF permutation p={iof_perm_pval:.4f}")
    
            # Annotate p-values on plot
            annotation_text = f"AE t={ae_t_pval:.3f}, perm={ae_perm_pval:.3f}\nIOF t={iof_t_pval:.3f}, perm={iof_perm_pval:.3f}"
            ax_main.text(0.99, 0.95, annotation_text, transform=ax_main.transAxes,
                         fontsize=9, verticalalignment='top', horizontalalignment='right',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        ax_main.set_ylim(ymin=0)
        ax_main.set_ylabel("Cases")
        ax_main.set_xlabel("Date")
        ax_main.grid(True)

        handles_main, labels_main = ax_main.get_legend_handles_labels()
        unique_legend = dict(zip(labels_main, handles_main))
        for patch in outbreak_patches:
            unique_legend[patch.get_label()] = patch
        ax_main.legend(unique_legend.values(), unique_legend.keys(), loc='upper left', fontsize=9)

    fig.suptitle(save_path.split('.', 1)[0], fontsize=18, y=1.00)
    plt.tight_layout()
    if save:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    plt.show()

    if plot_map:
        plot_state_case_map(save_path, filename, grey_periods, disease_labels, state_shapes)


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def plot_state_case_map(save_path, filename, grey_periods, disease_labels, state_shapes):
    import geopandas as gpd
    import geoplot as gplt
    import geoplot.crs as gcrs
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import to_hex
    from WildAlertpy import read_data as read_data  # Your custom reader

    # Detect state from save_path
    states_to_check = ['California', 'Florida', 'Arizona', 'Washington']
    found_state = next((state for state in states_to_check if state in save_path), None)
    if not found_state:
        print("No recognized state found in save_path for mapping.")
        return

    # Load and geocode the case data
    df = read_data.read_data(filename=filename + ".xlsx")
    gdf = make_geocoded_gdf(df)
    gdf['date'] = pd.to_datetime(gdf['found_date'])

    # Remove points outside the selected state boundary
    target_state_geom = state_shapes[found_state].to_crs("EPSG:4326")
    gdf = gdf[gdf.within(target_state_geom.unary_union)]

    if gdf.empty:
        print(f"No geocoded cases found within {found_state} boundaries.")
        return

    # Build label-to-color mapping consistent with timeline plot
    from matplotlib.patches import Patch
    unique_labels = list(dict.fromkeys(disease_labels))  # Preserves order
    print(unique_labels)
    label_colors = {label: to_hex(plt.cm.Paired(i / len(unique_labels))) for i, label in enumerate(unique_labels)}
    gdf['period_color'] = 'grey'  # Default color
    

    for (start, end), label in zip(grey_periods, disease_labels):
        mask = (gdf['date'] >= pd.to_datetime(start)) & (gdf['date'] <= pd.to_datetime(end))
        gdf.loc[mask, 'period_color'] = label_colors[label]
    print(gdf['period_color'].unique())
    # A4 size portrait in inches
    fig, ax = plt.subplots(figsize=(8.27, 11.69), subplot_kw={'projection': gcrs.AlbersEqualArea()})

    # Plot US background and target state
    #gplt.polyplot(state_shapes['USA'], edgecolor='#d3d3d3', linewidth=0.6, ax=ax)
    gplt.polyplot(state_shapes[found_state], edgecolor='#555555', linewidth=1.0, ax=ax, facecolor='white')

    # Plot cases
    gplt.pointplot(gdf, hue='period_color', cmap='Paired',
                   ax=ax, s=3, alpha=0.5, legend=False)

    # Title and legend
    #ax.set_title(f"Wildlife Cases in {found_state}", fontsize=14, pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Custom legend for outbreak labels
    legend_handles = [Patch(facecolor=label_colors[label], label=label) for label in unique_labels]
    #ax.legend(handles=legend_handles, title='Outbreak Periods', loc='lower left', fontsize=9, title_fontsize=10)

    # Save
    map_out_path = save_path.replace(".png", f"_{found_state}_map.png")
    plt.tight_layout()
    plt.savefig(map_out_path, dpi=600)
    print(f"Map saved to: {map_out_path}")
    plt.show()


import geopandas as gpd

def prepare_us_state_shapes(shapefile_path):
    USA = gpd.read_file(shapefile_path)
    outside_states = ['Guam', "Hawaii", 'Commonwealth of the Northern Mariana Islands',
                      'American Samoa', 'United States Virgin Islands', 'Puerto Rico', "Alaska"]
    USA_continent = USA[~USA.NAME.isin(outside_states)]
    
    state_shapes = {'USA': USA_continent[['geometry']].dissolve()}
    for state in ['California', 'Florida', 'Arizona', 'Washington']:
        state_gdf = USA_continent[USA_continent.NAME == state]
        state_shapes[state] = state_gdf
    return state_shapes


def make_geocoded_gdf(case_df, lon_col='longitude_found', lat_col='latitude_found'):
    import geopandas as gpd
    from shapely.geometry import Point
    import pandas as pd

    case_df = case_df.dropna(subset=[lon_col, lat_col])
    geometry = [Point(xy) for xy in zip(case_df[lon_col], case_df[lat_col])]
    gdf = gpd.GeoDataFrame(case_df, geometry=geometry, crs="EPSG:4326")
    return gdf

def plot_state_case_map(save_path, filename, grey_periods, disease_labels, state_shapes):
    import geopandas as gpd
    import geoplot as gplt
    import geoplot.crs as gcrs
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import to_hex, ListedColormap
    from matplotlib.patches import Patch
    from WildAlertpy import read_data as read_data  # Your custom reader
    # Detect state from save_path
    states_to_check = ['California', 'Florida', 'Arizona', 'Washington']
    found_state = next((state for state in states_to_check if state in save_path), None)
    if not found_state:
        print("No recognized state found in save_path for mapping.")
        return

    # Load and geocode the case data
    df = read_data.read_data(filename=filename + ".xlsx")
    gdf = make_geocoded_gdf(df)
    gdf['date'] = pd.to_datetime(gdf['found_date'])

    # Remove points outside the selected state boundary
    target_state_geom = state_shapes[found_state].to_crs("EPSG:4326")
    gdf = gdf[gdf.within(target_state_geom.unary_union)]

    if gdf.empty:
        print(f"No geocoded cases found within {found_state} boundaries.")
        return

    # Create custom colormap (grey + Paired)
    unique_labels = list(dict.fromkeys(disease_labels))  # Preserves order
    base_colors = [to_hex(plt.cm.Paired(i / max(1, len(unique_labels)))) for i in range(len(unique_labels))]
    label_colors = dict(zip(unique_labels, base_colors))
    label_colors['grey'] = '#808080'  # Default color for non-outbreak cases

    # Assign period labels to gdf
    gdf['period_label'] = 'grey'  # default
    for (start, end), label in zip(grey_periods, disease_labels):
        mask = (gdf['date'] >= pd.to_datetime(start)) & (gdf['date'] <= pd.to_datetime(end))
        gdf.loc[mask, 'period_label'] = label

    # Create colormap and plotting order
    color_levels = list(label_colors.keys())  # Order: ['label1', 'label2', ..., 'grey']
    cmap = ListedColormap([label_colors[label] for label in color_levels])

    # Create A4 size map
    fig, ax = plt.subplots(figsize=(8.27, 11.69), subplot_kw={'projection': gcrs.AlbersEqualArea()})

    # Plot base state shape
    gplt.polyplot(state_shapes[found_state], edgecolor='#555555', linewidth=1.0, ax=ax, facecolor='white')

    # Plot points
    gplt.pointplot(
        gdf,
        hue='period_label',
        cmap=cmap,
        ax=ax,
        s=3,
        alpha=0.5,
        legend=False,
        
    )

    # Custom legend
    #legend_handles = [Patch(facecolor=label_colors[label], label=label) for label in color_levels]
    #ax.legend(handles=legend_handles, title='Outbreak Periods', loc='lower left', fontsize=9, title_fontsize=10)

    # Clean up plot aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save map
    map_out_path = save_path.replace(".png", f"_{found_state}_map.png")
    plt.tight_layout()
    plt.savefig(map_out_path, dpi=600)
    print(f"Map saved to: {map_out_path}")
    plt.show()

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

def plot_stacked_wildalert_timelines(
    filenames,
    window_size,
    sigma,
    time_bin='W',
    plot_mode="both",   # "intensity", "cases", or "both"
    plot_start_date="2020-01-01",
    grey_periods=None,
    disease_labels=None,
    share_x=True,
    save=False,
    save_path=None
):
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as mpatches
    import matplotlib.cm as cm
    import numpy as np

    if save and (save_path is None or save_path == "wildalert_stacked_timelines.png"):
        suffix_map = {
            "intensity": "intensity",
            "cases": "cases",
            "both": "both"
        }
    save_path = f"wildalert_stacked_timelines_{suffix_map[plot_mode]}.png"
    n = len(filenames)

    # -------------------------------
    # Height logic (per timeline)
    # -------------------------------
    if plot_mode == "intensity":
        heights = [2] * n
    elif plot_mode == "cases":
        heights = [4] * n
    else:  # both
        heights = [6] * n

    fig = plt.figure(figsize=(10, sum(heights)))

    # Row heights inside each timeline
    if plot_mode == "both":
        row_heights = []
        for _ in range(n):
            row_heights.extend([2, 4])
    else:
        row_heights = heights

    gs = GridSpec(len(row_heights), 1, height_ratios=row_heights, hspace=0.25)

    # Grey-period color mapping
    unique_labels = list(dict.fromkeys(disease_labels)) if disease_labels else []
    cmap = cm.get_cmap("Paired", len(unique_labels))
    label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels)}

    def scale_zero_mean_unit_range(x):
        if x.max() == x.min():
            return x * 0
        x = x - x.mean()
        return x / max(abs(x.min()), abs(x.max()))

    axes = []
    row_cursor = 0

    for i, filename in enumerate(filenames):

        anomaly_df = plot_anomalies(
            filename=filename,
            window_size=window_size,
            sigma=sigma,
            time_bin=time_bin,
            plot=False,
            save=False
        )

        anomaly_df = anomaly_df[anomaly_df.index >= pd.to_datetime(plot_start_date)]
        if anomaly_df.empty:
            continue

        # -------------------------
        # INTENSITY PANEL
        # -------------------------
        if plot_mode in ["intensity", "both"]:
            ax_int = fig.add_subplot(
                gs[row_cursor],
                sharex=axes[0] if (share_x and axes) else None
            )
            axes.append(ax_int)

            intensity_vals = []

            if "AE Intensity" in anomaly_df:
                ae_scaled = scale_zero_mean_unit_range(anomaly_df["AE Intensity"])
                ae_scaled.plot(ax=ax_int, lw=1.3, color="green", label="AE")
                intensity_vals.append(ae_scaled)

            if "IOF Intensity" in anomaly_df:
                iof_scaled = -scale_zero_mean_unit_range(anomaly_df["IOF Intensity"])
                iof_scaled.plot(ax=ax_int, lw=1.3, color="black", label="IOF")
                intensity_vals.append(iof_scaled)

            # Dynamic Y-limits
            if intensity_vals:
                combined = pd.concat(intensity_vals)
                pad = 0.1 * (combined.max() - combined.min())
                ax_int.set_ylim(combined.min() - pad, combined.max() + pad)

            ax_int.set_ylabel("WildAlert\nSignal")
            ax_int.grid(True)
            ax_int.legend(fontsize=8)

            if plot_mode == "intensity":
                ax_int.set_title(filename.split("/")[-1].replace("-", " ").title())

            row_cursor += 1

        # -------------------------
        # CASES PANEL
        # -------------------------
        if plot_mode in ["cases", "both"]:
            ax = fig.add_subplot(
                gs[row_cursor],
                sharex=axes[0] if (share_x and axes) else None
            )
            axes.append(ax)

            anomaly_df["# admissions"].plot(ax=ax, lw=1.4, label="Cases")

            if "rolling mean" in anomaly_df:
                anomaly_df["rolling mean"].plot(ax=ax, color="black", label="Rolling Mean")

            if "rolling std" in anomaly_df:
                upper = anomaly_df["rolling mean"] + 2 * anomaly_df["rolling std"]
                lower = anomaly_df["rolling mean"] - 2 * anomaly_df["rolling std"]
                ax.fill_between(anomaly_df.index, upper, lower, color="gray", alpha=0.25)

            if "Isolation Forest Anomaly" in anomaly_df:
                ax.scatter(
                    anomaly_df[anomaly_df["Isolation Forest Anomaly"] == 1].index,
                    anomaly_df[anomaly_df["Isolation Forest Anomaly"] == 1]["# admissions"],
                    marker="X", s=60, color="black", label="IOF"
                )

            if "Autoencoder Anomaly" in anomaly_df:
                ax.scatter(
                    anomaly_df[anomaly_df["Autoencoder Anomaly"] == 1].index,
                    anomaly_df[anomaly_df["Autoencoder Anomaly"] == 1]["# admissions"],
                    marker="^", s=60, color="green", label="AE"
                )

            # Grey periods
            patches = []
            if grey_periods:
                for j, (start, end) in enumerate(grey_periods):
                    color = label_to_color[disease_labels[j]] if disease_labels else "lightgray"
                    label = disease_labels[j] if disease_labels else f"Event {j+1}"
                    ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color=color, alpha=0.3)
                    patches.append(mpatches.Patch(color=color, label=label))

            ax.set_ylabel("Cases")
            ax.set_ylim(bottom=0)
            ax.grid(True)

            title = filename.split("/")[-1].replace("-", " ").title()
            ax.set_title(title)

            handles, labels = ax.get_legend_handles_labels()
            legend = dict(zip(labels, handles))
            for p in patches:
                legend[p.get_label()] = p
            ax.legend(legend.values(), legend.keys(), fontsize=8, loc="upper left")

            row_cursor += 1

    # -------------------------
    # X-axis formatting
    # -------------------------
    for ax in axes[:-1]:
        ax.set_xlabel("")

    if axes:
        axes[-1].set_xlabel("Date")

    plt.tight_layout()
    if save:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()

    