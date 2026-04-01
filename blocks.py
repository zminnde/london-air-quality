import numpy as np

def make_blocks(df, input_features, output_features, input_hours, forecast_hours):
    all_hours = list(input_hours) + list(forecast_hours)
    n_input = len(input_hours)

    results = [
        (date, *_make_block(group, input_features, output_features, n_input))
        for date, group in df[df['time'].isin(all_hours)].groupby('date')
        if len(group) == len(all_hours)
    ]

    if not results:
        raise ValueError(f"No complete blocks found for hours={all_hours}")

    dates, inputs, outputs = zip(*results)
    return np.array(inputs), np.array(outputs), np.array(dates)


def _make_block(group, input_features, output_features, n_input):
    sorted_group = group.sort_values('time')
    inputs = sorted_group.iloc[:n_input][input_features].values.flatten()
    outputs = sorted_group.iloc[n_input:][output_features].values.flatten()
    return inputs, outputs
