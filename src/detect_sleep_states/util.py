import pandas as pd


def clean_events(events_path: str):
    events = pd.read_csv(events_path)
    events = events[~events['step'].isna()]

    # Remove nights where there isn't a complete record
    event_counts = events.groupby(['series_id', 'night']).size()
    event_counts = event_counts.reset_index().rename(
        columns={0: 'count'})
    invalid = event_counts[event_counts['count'] < 2]
    invalid_series_nights = invalid['series_id'].str.cat(
        invalid['night'].astype(str), sep='_')
    series_nights = events['series_id'].str.cat(events['night'].astype(str),
                                                sep='_')
    events = events[~series_nights.isin(invalid_series_nights)]

    events = events.set_index('series_id')

    # protect against sequence going past the data
    events_ = []
    for series_id in events.index.unique():
        events_.append(events.loc[series_id].iloc[:-1])
    events = pd.concat(events_)
    return events
