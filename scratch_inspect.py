import pandas as pd

df = pd.read_csv(r'data\raw\ibtracs.NI.list.v04r01.csv', skiprows=[1], low_memory=False)
print('Shape:', df.shape)
print('Subbasins:', df['SUBBASIN'].unique())

# Check recent storms with good data
for y in range(2015, 2025):
    yr = df[df['SEASON'].astype(str) == str(y)]
    n_storms = yr['SID'].nunique()
    # How many have wind data
    yr_wind = yr[yr['WMO_WIND'].notna() & (yr['WMO_WIND'].astype(str).str.strip() != '')]
    n_wind = yr_wind['SID'].nunique()
    print(f"  {y}: {n_storms} storms total, {n_wind} with WMO_WIND")

# Look at 2018-2023 storms in detail
target = df[(df['SEASON'].astype(int) >= 2018) & (df['SEASON'].astype(int) <= 2023)]
target_wind = target[target['WMO_WIND'].notna() & (target['WMO_WIND'].astype(str).str.strip() != '')]
storms = target_wind.groupby('SID').size().reset_index(name='obs_count')
storms = storms.sort_values('obs_count', ascending=False)
print("\nStorms 2018-2023 with wind data (obs count):")
print(storms.to_string(index=False))
