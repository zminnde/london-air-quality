import pandas
import numpy
import holidays

raw_csv_directory = 'data/1_raw'
output_directory = 'data/2_unscaled_features'
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


def add_day_category(df):
    current_date = pandas.to_datetime(df['date'])
    next_date = current_date + pandas.Timedelta(days=1)
    is_off_day = lambda date: is_weekend(date) | is_holiday(date)

    category = numpy.select(
        [is_off_day(current_date), is_off_day(next_date)],
        [1, 2],
        default=0
    )

    df.insert(df.columns.get_loc('date') + 1, 'day_category', category)
    return df


def is_weekend(date):
    return date.dt.day_name().isin(['Saturday', 'Sunday'])


def is_holiday(date):
    return False


for year in range(start_year, end_year + 1):
    print(f"Processing {year}")
    (pandas.read_csv(
        f"{raw_csv_directory}/{year}.csv",
        skiprows=rows_to_skip,
        skipfooter=1,
        usecols=raw_features,
        engine='python')
     .rename(columns=dict(zip(raw_features, target_features)))
     .reindex(columns=target_features)
     .pipe(add_day_category)
     .to_csv(f"{output_directory}/{year}_unscaled.csv", index=False))
