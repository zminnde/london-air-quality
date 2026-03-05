import holidays
import numpy
import numpy as np
import pandas

UK_HOLIDAYS = pandas.to_datetime(list(holidays.country_holidays('UK', years=range(2011, 2026)).keys()))

raw_csv_directory = 'data/1_raw'
output_directory = 'data/3_scaled'
start_year = 2011
end_year = 2025
rows_to_skip = 17
raw_features, target_features = zip(*[
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


def add_day_category(df, country):
    current_date = df['date']
    next_date = current_date + pandas.Timedelta(days=1)
    is_off_day = lambda date: is_weekend(date) | is_holiday(date, country)

    category = pandas.Categorical(numpy.select(
        [is_off_day(current_date), is_off_day(next_date)],
        [1, 2],
        default=0
    ))

    df.insert(df.columns.get_loc('date') + 1, 'day_category', category)
    return df


def is_weekend(date):
    return date.dt.day_name().isin(['Saturday', 'Sunday'])


def is_holiday(date, country_holidays):
    return date.isin(country_holidays)


def apply_min_max(df, exclude=None):
    x = df.select_dtypes(include='number')
    if exclude:
        x = x.drop(columns=exclude, errors='ignore')
    df[x.columns] = (2 * (x - x.min()) / (x.max() - x.min()) - 1).round(4)
    return df


def process(csv_year):
    print(f"Processing {csv_year}")
    return (
        pandas.read_csv(
            f"{raw_csv_directory}/{csv_year}.csv",
            na_values=['No data'],
            parse_dates=['Date'],
            skiprows=rows_to_skip,
            skipfooter=1,
            usecols=raw_features,
            engine='python'
        )
        .rename(columns=dict(zip(raw_features, target_features)))
        .pipe(add_day_category, UK_HOLIDAYS)
        .assign(wind_dir_sin=lambda df: np.sin(np.radians(df['wind_dir'])).round(4),
                wind_dir_cos=lambda df: np.cos(np.radians(df['wind_dir'])).round(4))
        .drop(columns=['wind_dir'])
    )


combined = (pandas.concat([process(year) for year in range(start_year, end_year + 1)])
 .pipe(apply_min_max, exclude=['wind_dir_sin', 'wind_dir_cos', 'day_category'])
 .sort_values(['date', 'time']))

combined.to_csv(f"{output_directory}/scaled.csv", index=False)
