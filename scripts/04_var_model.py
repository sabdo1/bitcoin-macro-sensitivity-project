!pip install statsmodels --upgrade -q
import importlib
import statsmodels
importlib.reload(statsmodels)
print(statsmodels.__version__)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.stattools import durbin_watson

# STEP 1: LOAD DATA


url = 'https://raw.githubusercontent.com/sabdo1/bitcoin-macro-sensitivity-project/main/data/data_clean.csv'

df = pd.read_csv(url, index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# Keep only the columns we need
cols = ['BTC-USD', '^GSPC', '^VIX', 'DX-Y.NYB', 'US10Y', 'BREAKEVEN']
df = df[cols].copy()

print("Raw data loaded:")
print(f"  Shape: {df.shape}")
print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"  Missing values:\n{df.isnull().sum()}")
print(df.head())

# STEP 2: TRANSFORM TO STATIONARY SERIES

transformed = pd.DataFrame(index=df.index)

transformed['BTC_ret']    = np.log(df['BTC-USD'] / df['BTC-USD'].shift(1))
transformed['SP500_ret']  = np.log(df['^GSPC']   / df['^GSPC'].shift(1))
transformed['VIX_ret']    = np.log(df['^VIX']    / df['^VIX'].shift(1))
transformed['DXY_ret']    = np.log(df['DX-Y.NYB']/ df['DX-Y.NYB'].shift(1))
transformed['US10Y_diff']     = df['US10Y'].diff()
transformed['BREAKEVEN_diff'] = df['BREAKEVEN'].diff()

# Drop the first row (NaN from differencing)
transformed = transformed.dropna()

transformed = transformed[['US10Y_diff', 'BREAKEVEN_diff', 'VIX_ret', 'SP500_ret', 'DXY_ret', 'BTC_ret']]

print("\nTransformed data (first 5 rows):")
print(transformed.head())
print(f"\nShape after transformation: {transformed.shape}")
print(f"Date range: {transformed.index[0].date()} to {transformed.index[-1].date()}")

# STEP 3: ADF STATIONARITY TESTS


def run_adf(series, name):
    result = adfuller(series.dropna(), autolag='AIC')
    adf_stat  = result[0]
    p_value   = result[1]
    n_lags    = result[2]
    crit_1pct = result[4]['1%']
    crit_5pct = result[4]['5%']
    stationary = 'YES ✓' if p_value < 0.05 else 'NO ✗'
    print(f"  {name:<15} | ADF: {adf_stat:8.4f} | p-value: {p_value:.4f} | "
          f"Lags: {n_lags:2d} | 5% crit: {crit_5pct:.4f} | Stationary: {stationary}")

print("\n--- ADF Test Results (after transformation) ---")
print(f"  {'Series':<15} | {'ADF Stat':>8} | {'p-value':>8} | "
      f"{'Lags':>4} | {'5% crit':>8} | Stationary")
print("  " + "-"*75)

for col in transformed.columns:
    run_adf(transformed[col], col)

print("\nInterpretation: p < 0.05 → reject unit root → series is stationary")
print("All series should be stationary after our transformations above.")

# STEP 4: LAG LENGTH SELECTION (AIC and BIC)


model_select = VAR(transformed)
lag_results  = model_select.select_order(maxlags=10)

print("\n--- Lag Length Selection ---")
print(lag_results.summary())

aic_lag = lag_results.aic
bic_lag = lag_results.bic

print(f"\nAIC selects: {aic_lag} lag(s)")
print(f"BIC selects: {bic_lag} lag(s)")
print(f"\nWe will use AIC-selected lag = {aic_lag} for the full-sample VAR.")
print("(BIC tends to select more parsimonious models on large samples — "
      "we report both but proceed with AIC as our data may be misspecified)")

chosen_lag = aic_lag

# STEP 5: FIT VAR MODEL (FULL SAMPLE)


var_model  = VAR(transformed)
var_result = var_model.fit(chosen_lag)

print("\n--- VAR Model Summary (Full Sample) ---")
print(var_result.summary())

# Durbin-Watson statistics for residual autocorrelation check
print("\n--- Durbin-Watson Residual Diagnostics ---")
print("(Values close to 2.0 indicate no autocorrelation in residuals)")
dw = durbin_watson(var_result.resid)
for col, d in zip(transformed.columns, dw):
    flag = '✓' if 1.5 < d < 2.5 else '⚠ CHECK'
    print(f"  {col:<15}: DW = {d:.4f}  {flag}")

# VAR Stability Check
print("\n--- VAR Stability Check ---")
print(f"Full sample VAR stable: {var_result.is_stable()}")


# STEP 6: IMPULSE RESPONSE FUNCTIONS (IRFs) WITH CONFIDENCE INTERVALS


irf = var_result.irf(20)

# Get column indices
btc_idx   = list(transformed.columns).index('BTC_ret')
us10y_idx = list(transformed.columns).index('US10Y_diff')
vix_idx   = list(transformed.columns).index('VIX_ret')
sp500_idx = list(transformed.columns).index('SP500_ret')

# Extract IRF values
irf_us10y = irf.orth_irfs[:, btc_idx, us10y_idx]
irf_vix   = irf.orth_irfs[:, btc_idx, vix_idx]
irf_sp500 = irf.orth_irfs[:, btc_idx, sp500_idx]

# Compute confidence intervals via bootstrap (100 runs)
irf_ci = var_result.irf(20, var_decomp=None)
lower_band, upper_band = irf.errband_mc(orth=True, repl=1000, signif=0.05)

# Extract bands for BTC response
lower_us10y = lower_band[:, btc_idx, us10y_idx]
upper_us10y = upper_band[:, btc_idx, us10y_idx]
lower_vix   = lower_band[:, btc_idx, vix_idx]
upper_vix   = upper_band[:, btc_idx, vix_idx]
lower_sp500 = lower_band[:, btc_idx, sp500_idx]
upper_sp500 = upper_band[:, btc_idx, sp500_idx]

days = range(21)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Impulse Response Functions: Effect on Bitcoin Returns\n(Full Sample 2017–2026)',
             fontsize=13, fontweight='bold')

for ax, irf_vals, lower, upper, title in zip(
    axes,
    [irf_us10y, irf_vix, irf_sp500],
    [lower_us10y, lower_vix, lower_sp500],
    [upper_us10y, upper_vix, upper_sp500],
    ['Shock to US 10Y Yield', 'Shock to VIX', 'Shock to S&P 500']
):
    ax.plot(days, irf_vals, color='steelblue', linewidth=2)
    ax.fill_between(days, lower, upper, alpha=0.2, color='steelblue',
                    label='95% CI')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f'{title}\n→ Response of Bitcoin Returns', fontsize=10)
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Response')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('irf_full_sample.png', dpi=150, bbox_inches='tight')
plt.show()
print("IRF plot saved as 'irf_full_sample.png'")

# STEP 7: FORECAST ERROR VARIANCE DECOMPOSITION (FEVD)


fevd        = var_result.fevd(20)
fevd_df     = pd.DataFrame(
    fevd.decomp[transformed.columns.get_loc('BTC_ret')],
    columns=transformed.columns
)
fevd_df.index.name = 'Horizon (days)'

print("\n--- FEVD: Share of Bitcoin Return Variance Explained by Each Variable ---")
print("(Rows = forecast horizon in trading days)")
print(fevd_df.round(4).to_string())

# Plot FEVD
fig, ax = plt.subplots(figsize=(12, 6))
fevd_df.plot(
    kind='area',
    stacked=True,
    ax=ax,
    colormap='tab10',
    alpha=0.85
)
ax.set_title('Forecast Error Variance Decomposition: Bitcoin Returns\n(Full Sample 2017–2026)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Forecast Horizon (Trading Days)')
ax.set_ylabel('Proportion of Variance Explained')
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('fevd_full_sample.png', dpi=150, bbox_inches='tight')
plt.show()
print("FEVD plot saved as 'fevd_full_sample.png'")


# STEP 8: SPLIT SAMPLE VAR (PRE vs POST 2020)

pre_2020  = transformed[transformed.index < '2020-01-01']
post_2020 = transformed[transformed.index >= '2020-01-01']

print(f"Pre-2020:  {pre_2020.index[0].date()} to {pre_2020.index[-1].date()} ({len(pre_2020)} obs)")
print(f"Post-2020: {post_2020.index[0].date()} to {post_2020.index[-1].date()} ({len(post_2020)} obs)")

lag_pre  = max(1, VAR(pre_2020).select_order(maxlags=10).aic)
var_pre  = VAR(pre_2020).fit(lag_pre)

lag_post  = max(1, VAR(post_2020).select_order(maxlags=10).aic)
var_post  = VAR(post_2020).fit(lag_post)

fevd_pre = var_pre.fevd(20)
fevd_pre_df = pd.DataFrame(
    fevd_pre.decomp[pre_2020.columns.get_loc('BTC_ret')],
    columns=pre_2020.columns
)

fevd_post = var_post.fevd(20)
fevd_post_df = pd.DataFrame(
    fevd_post.decomp[post_2020.columns.get_loc('BTC_ret')],
    columns=post_2020.columns
)

print(f"Pre-2020 VAR: AIC selected {lag_pre} lag(s)")
print(f"Post-2020 VAR: AIC selected {lag_post} lag(s)")

print(f"Pre-2020 VAR stable: {var_pre.is_stable()}")
print(f"Post-2020 VAR stable: {var_post.is_stable()}")

irf_pre  = var_pre.irf(20)
irf_post = var_post.irf(20)

btc_idx = list(pre_2020.columns).index('BTC_ret')
shocks  = ['US10Y_diff', 'VIX_ret', 'SP500_ret']
labels  = ['US 10Y Yield', 'VIX', 'S&P 500']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('IRF Comparison: Bitcoin Response to Macro Shocks\nPre-2020 vs Post-2020',
             fontsize=13, fontweight='bold')

periods = irf_pre.irfs.shape[0]

for col, (shock, label) in enumerate(zip(shocks, labels)):
    shock_idx = list(pre_2020.columns).index(shock)

    irf_pre_vals  = irf_pre.orth_irfs[:,  btc_idx, shock_idx]
    irf_post_vals = irf_post.orth_irfs[:, btc_idx, shock_idx]

    axes[0, col].plot(range(periods), irf_pre_vals,
                      color='steelblue', linewidth=2)
    axes[0, col].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0, col].set_title(f'PRE-2020\nShock to {label} → BTC', fontsize=10)
    axes[0, col].set_xlabel('Trading Days')

    axes[1, col].plot(range(periods), irf_post_vals,
                      color='darkorange', linewidth=2)
    axes[1, col].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1, col].set_title(f'POST-2020\nShock to {label} → BTC', fontsize=10)
    axes[1, col].set_xlabel('Trading Days')

plt.tight_layout()
plt.savefig('irf_split_sample.png', dpi=150, bbox_inches='tight')
plt.show()
print("Split sample IRF plot saved as 'irf_split_sample.png'")



# STEP 9: SUMMARY TABLE FOR REPORT


print("\n" + "="*60)
print("SUMMARY FOR REPORT")
print("="*60)
print(f"Full sample VAR lag order (AIC): {chosen_lag}")
print(f"Pre-2020 VAR lag order (AIC):    {lag_pre}")
print(f"Post-2020 VAR lag order (AIC):   {lag_post}")
print("\nFEVD at 20-day horizon (Full Sample):")
print(fevd_df.iloc[-1].round(4).to_string())
print("\nFEVD at 20-day horizon (Pre-2020):")
print(fevd_pre_df.iloc[-1].round(4).to_string())
print("\nFEVD at 20-day horizon (Post-2020):")
print(fevd_post_df.iloc[-1].round(4).to_string())
print("\nOutputs saved: irf_full_sample.png, fevd_full_sample.png,")
print("               irf_split_sample.png, fevd_split_sample.png")


# Regenerate and save FEVD split sample plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('FEVD Comparison: Share of Bitcoin Variance Explained by Macro Shocks\n'
             'Pre-2020 vs Post-2020', fontsize=13, fontweight='bold')

fevd_pre_df.plot(kind='area', stacked=True, ax=axes[0],
                 colormap='tab10', alpha=0.85)
axes[0].set_title('Pre-2020', fontsize=11)
axes[0].set_xlabel('Forecast Horizon (Trading Days)')
axes[0].set_ylabel('Proportion of Variance Explained')
axes[0].set_ylim(0, 1)
axes[0].legend(fontsize=8)

fevd_post_df.plot(kind='area', stacked=True, ax=axes[1],
                  colormap='tab10', alpha=0.85)
axes[1].set_title('Post-2020', fontsize=11)
axes[1].set_xlabel('Forecast Horizon (Trading Days)')
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('fevd_split_sample.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved!")
