import holidays
import numpy as np
import pandas as pd

from models import MinMaxScaler

raw_csv = 'data/1_raw'
rows_to_skip = 17
start_year = 2012
end_year = 2025
unscaled_csv = 'data/2_unscaled/unscaled.csv'
scaled_csv = 'data/3_scaled/scaled.csv'

UK_HOLIDAYS = pd.to_datetime(
    list(holidays.country_holidays('UK', years=range(2011, 2026)).keys())
)

raw_feature_name, target_feature_name = zip(*[
    ('Date', 'date'),
    ('Time', 'time'),
    ('Ozone', 'O3'),
    ('Nitric oxide', 'NO'),
    ('Nitrogen dioxide', 'NO2'),
    ('Carbon monoxide', 'CO'),
    ('Modelled Wind Direction', 'wind_dir'),
    ('Modelled Wind Speed', 'wind_speed'),
    ('Modelled Temperature', 'temp'),
    ('PM10 particulate matter (Hourly measured)', 'PM10'),
    ('PM2.5 particulate matter (Hourly measured)', 'PM2.5')
])

is_weekend = lambda date: date.dt.day_name().isin(['Saturday', 'Sunday'])
is_holiday = lambda date, holidays: date.isin(holidays)
is_off_day = lambda date, holidays: is_weekend(date) | is_holiday(date, holidays)


def get_day_category(date, holiday_dates):
    """Calculate day category: 0=workday, 1=weekend, 2=day before weekend"""
    next_date = date + pd.Timedelta(days=1)
    return pd.Categorical(np.select(
        [is_off_day(date, holiday_dates), is_off_day(next_date, holiday_dates)],
        [1, 2],
        default=0
    ))


def add_day_category(df, holiday_dates):
    category = get_day_category(df['date'], holiday_dates)
    return df.assign(day_category=lambda x: category)


def add_sin_cos(df, col_name):
    return df.assign(
        **{f'{col_name}_sin': lambda x, c=col_name: np.sin(np.radians(x[c])).round(4),
           f'{col_name}_cos': lambda x, c=col_name: np.cos(np.radians(x[c])).round(4)}
    )


def process(csv_year):
    print(f"Processing {csv_year}")
    return (
        pd.read_csv(
            f"{raw_csv}/{csv_year}.csv",
            na_values=['No data'],
            parse_dates=['Date'],
            skiprows=rows_to_skip,
            skipfooter=1,
            usecols=raw_feature_name,
            engine='python'
        )
        .rename(columns=dict(zip(raw_feature_name, target_feature_name)))
        .assign(time=lambda df: df['time'].str[:2].astype(int))
        .pipe(add_day_category, UK_HOLIDAYS)
        .drop(columns=['CO'])

    )


combined_raw = (
    pd.concat([process(year) for year in range(start_year, end_year + 1)])
    .sort_values(['date', 'time'])
)

combined_raw = combined_raw[~combined_raw['date'].dt.year.isin([2020, 2021, 2022])]

cols = ['O3', 'NO', 'NO2', 'PM10', 'PM2.5']
mask = (combined_raw[cols] >= 0) | combined_raw[cols].isna()
combined_raw = combined_raw[mask.all(axis=1)]

combined_raw = (combined_raw
 .pipe(add_sin_cos, 'wind_dir')
 .drop(columns=['wind_dir']))

combined_raw.to_csv(unscaled_csv, index=False)

cleaned = combined_raw.dropna()

EXCLUDE = ['wind_dir_sin', 'wind_dir_cos', 'day_category', 'time']
train_raw = cleaned[cleaned['date'].dt.year < 2025]
test_raw  = cleaned[cleaned['date'].dt.year == 2025]

scaler = MinMaxScaler()
scaler.fit(train_raw, exclude=EXCLUDE)

train_scaled = scaler.transform(train_raw)
test_scaled  = scaler.transform(test_raw)

scaled = pd.concat([train_scaled, test_scaled]).sort_values(['date', 'time'])
scaled.to_csv(scaled_csv, index=False)
