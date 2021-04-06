"""Lifetime Value (LTV) function.
Function that measures a cumalative average value through time for a population derived from events. Loosely based on
the Kaplan-Meier estimator.

Created: 2021-03-30 (Merijn)
Updated: 2021-04-06 (Merijn)
"""


# ----------------------------------------------------------------------------------------------------------------------
# Import libraries
# ----------------------------------------------------------------------------------------------------------------------
import itertools
import pandas as pd
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------------------------------
def lifetime_value(subjects: pd.DataFrame, events: pd.DataFrame, confidence_level: float = None) -> pd.DataFrame:
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
    subjects = subjects.copy()
    columns = {
        'subject_id': str,
        'lifetime': int
    }
    subjects = subjects[columns.keys()]  # Remove irrelevant columns.
    input_columns = subjects.columns
    for col in columns.keys():
        if col not in input_columns:
            raise ValueError(f'DataFrame `subjects` must contain the column `{col}.')
        subjects.loc[:, col] = subjects[col].astype(columns[col])  # Correct format.
    if not subjects.subject_id.is_unique:
        raise ValueError(
            'The column `subject_id` in DataFrame `subjects` must only contain unique values.'
        )
    if min(subjects.lifetime) < 0:
        raise ValueError(
            'The values in column `lifetime` in DataFrame `subjects` must be greater or equal to 0.'
        )

    # Check if the `events` dafaframe is well-defined
    events = events.copy()
    columns = {
        'subject_id': str,
        'time': int,
        'value': float,
    }
    events = events[columns.keys()]  # Remove irrelevant columns.
    input_columns = events.columns
    for col in columns.keys():
        if col not in input_columns:
            raise ValueError(f'DataFrame `events` must contain the column `{col}.')
        events.loc[:, col] = events[col].astype(columns[col])
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
