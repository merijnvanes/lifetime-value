"""Lifetime Value (LTV) function.
Versatile function that measures a cumalative average value of events through time. Loosely based on the Kaplan-Meier
estimator.

Created: 2021-03-30 (Merijn)
Updated: 2021-04-05 (Merijn)
"""


# ----------------------------------------------------------------------------------------------------------------------
# Import libraries
# ----------------------------------------------------------------------------------------------------------------------
import itertools
import pandas as pd


# ----------------------------------------------------------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------------------------------------------------------
def lifetime_value(subjects: pd.DataFrame, events: pd.DataFrame):
    """Calculate the average lifetime value (LTV) of a group of subjects through time. The actual estimated lifetime
    value would be the value if time goes to infinity.

    :param subjects: pd.DataFrame, holds information on the subject.
    :param events: pd.DataFrame, all events with times and values.
    :return: pd.DataFrame,
    """
    # Check if subjects dataframe is correct.
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
            f'The column `{list(columns.keys())[0]}` in DataFrame `subjects` must only contain unique values.'
        )

    # Check if events dafaframe is correct.
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

    # Final aggregation.
    df = df.groupby('time').agg({
        'value': 'sum',
        'subject_id': 'nunique'
    }).reset_index()
    df = df.rename(columns={'value': 'value_avg', 'subject_id': 'n_subject'})
    df['value_avg'] = df.value_avg / df.n_subject

    return df
