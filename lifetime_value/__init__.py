"""Lifetime Value (LTV) function.
Function that measures a cumalative average value through time for a population derived from events. Loosely based on
the Kaplan-Meier estimator.

Created: 2021-03-30 (Merijn)
Updated: 2021-04-12 (Merijn)
"""


# ----------------------------------------------------------------------------------------------------------------------
# Import libraries
# ----------------------------------------------------------------------------------------------------------------------
import itertools
import datetime
import pandas as pd
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------------------------------------------
def _accept_dataframe(name: str, df: pd.DataFrame, col_specs: dict) -> pd.DataFrame:
    """Check if the necessary columns are present in the input dataframe and correct the data type.

    :param name: str, original name of the dataframe `df`.
    :param df: pd.DataFrame, input dataframe.
    :param col_specs: dict[str, type], column specifications with column name and corresponding data type.
    :return: pd.DataFrame, the dataframe in the desired format.
    """
    df2 = df.copy()
    for col in col_specs.keys():
        if col not in df2.columns:
            raise ValueError(f'DataFrame `{name}` must contain the column `{col}.')
        # Correct format:
        if col_specs[col] == datetime.date:
            if any([not isinstance(item, datetime.date) for item in df2[col]]):
                raise ValueError(f'The column {col} in DataFrame `{name}` must be of type `datetime.date`.')
        else:
            df2.loc[:, col] = df2[col].astype(col_specs[col])
    df2 = df2[col_specs.keys()]  # Remove irrelevant columns.
    return df2


# ----------------------------------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------------------------------
def from_subjects_and_events_dataframe(subjects: pd.DataFrame, events: pd.DataFrame, confidence_level: float = None) -> pd.DataFrame:
    """Calculate the average lifetime value (LTV) of a group of subjects through time. The actual estimated lifetime
    value would be the value if time goes to infinity.

    :param subjects: pd.DataFrame, every subject should have exactly one row with the columns`subject_id` and
    `lifetime`. Where `lifetime` represents the time the subject is under study from start of its tracking until now or
    a prior known end date of the tracking (examples: end of subscription, death).
    :param events: pd.DataFrame, all events with times and values.
    :param confidence_level: float in (0,1), the confidence level for the bootstrapped confidence interval for the mean.
    :return: pd.DataFrame, with columns `time` and `value`, and optionally a confidence interval.
    """
    # Check if the `subjects` dataframe is well-defined.
    column_specs = {
        'subject_id': str,
        'lifetime': int
    }
    subjects = _accept_dataframe('subjects', subjects, column_specs)
    if not subjects.subject_id.is_unique:
        raise ValueError(
            'The column `subject_id` in DataFrame `subjects` must only contain unique values.'
        )
    if min(subjects.lifetime) < 0:
        raise ValueError(
            'The values in column `lifetime` in DataFrame `subjects` must be greater or equal to 0.'
        )

    # Check if the `events` dafaframe is well-defined
    column_specs = {
        'subject_id': str,
        'time': int,
        'value': float,
    }
    events = _accept_dataframe('events', events, column_specs)
    if min(events.time) < 0:
        raise ValueError(
            'The values in column `time` in DataFrame `events` must be greater or equal to 0.'
        )

    # Check if the `confidence_level` is well-defined.
    if confidence_level:
        if not (0 < confidence_level < 1):
            raise ValueError('`confidence_level` must lay in between 0 and 1.')

    # Base dataframe, with all times a subject has been in until the minimum lifetime of the subjects in the group.
    min_lifetime = min(subjects.lifetime)
    n_subjects = subjects.shape[0]
    df = pd.DataFrame({
        'subject_id': list(
            itertools.chain.from_iterable(itertools.repeat(subj, min_lifetime + 1) for subj in subjects.subject_id)
        ),
        'time': list(range(min_lifetime + 1)) * n_subjects,
    })

    # Join events on base.
    events_time = events.groupby(['subject_id', 'time'])['value'].agg('sum').reset_index()
    df = pd.merge(df, events_time, how='left', on=['subject_id', 'time'])
    df['value'] = df['value'].fillna(0)
    df['value'] = df.groupby('subject_id')['value'].cumsum()

    # Dataframe with list of ordered cumsum value per subject_id.
    df_subj = df.groupby(['subject_id']).agg({'value': lambda x: x.tolist()}).reset_index()

    # Final dataframe.
    df_final = pd.DataFrame({
        'time': list(range(0, min_lifetime + 1)),
        'value': np.mean(np.array(df_subj.value.values.tolist()), axis=0),
    })

    # Bootstrap confidence interval (non-parametric resampling).
    # https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
    if confidence_level:
        n_samples = 1000
        sample_size = min(n_subjects, 10000)
        ltv_samples = np.empty((0, min_lifetime + 1), float)
        for i in range(n_samples):
            df_sample = df_subj.sample(n=sample_size, replace=True)
            arr_sample = np.array(df_sample.value.values.tolist())
            ltv_sample = np.mean(arr_sample, axis=0).reshape(1, min_lifetime + 1)
            ltv_samples = np.append(ltv_samples, ltv_sample, axis=0)

        df_final['confidence_interval_left_bound'] = np.quantile(ltv_samples, (1 - confidence_level)/2, axis=0)
        df_final['confidence_interval_right_bound'] = np.quantile(ltv_samples, (1 + confidence_level)/2, axis=0)

    return df_final


def from_event_log(event_log: pd.DataFrame, end_date: datetime.date = None, start_event_type: str = None,
                   confidence_level: float = None) -> pd.DataFrame:
    """Calculate the average lifetime value (LTV) of a group of subjects through time from a single event log DataFrame,
    that holds events with a value. The actual estimated lifetime value would be the value if time goes to infinity.

    :param event_log: pd.DataFrame, contains events as rows with a 'date', 'subject_id', 'type' and a 'value'.
    :param end_date: datetime.date, optional, the right bound for the dates of events.
    :param start_event_type: str, optional, the event type name as starting point. Example: 'registration'.
    :param confidence_level: float in (0,1), the confidence level for the bootstrapped confidence interval for the mean.
    :return: pd.DataFrame, with columns `time` and `value`, and optionally a confidence interval.
    """

    column_specs = {
        'subject_id': str,
        'date': datetime.date,
        'type': str,
        'value': float,
    }
    event_log = _accept_dataframe('event_log', event_log, column_specs)

    # Set right bound for date.
    if not end_date:
        end_date = max(event_log.date)

    # Form `subjects` dataframe.
    if start_event_type:
        subjects = event_log.loc[event_log.type == start_event_type]
    else:
        subjects = event_log.copy()
    subjects = subjects[['subject_id', 'date']].groupby('subject_id').agg('min').reset_index()
    subjects = subjects.rename({'date': 'date_first_activity'}, axis='columns')
    subjects['lifetime'] = [(end_date - d).days for d in subjects.date_first_activity]

    # Form `events` dataframe.
    events = event_log.copy()
    events = pd.merge(events, subjects[['subject_id', 'date_first_activity']], how='left', on=['subject_id'])
    events = events.reset_index()
    events = events.loc[events.date >= events.date_first_activity]
    events['time'] = [(row.date - row.date_first_activity).days for index, row in events.iterrows()]

    return from_subjects_and_events_dataframe(subjects, events, confidence_level=confidence_level)
