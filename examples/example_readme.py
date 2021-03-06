"""Example for getting a lifetime value dataframe.

Created: 2021-04-06 (Merijn)
Updated: 2021-04-12 (Merijn)
"""


# ----------------------------------------------------------------------------------------------------------------------
# Import libraries
# ----------------------------------------------------------------------------------------------------------------------
import datetime
import pandas as pd


# ----------------------------------------------------------------------------------------------------------------------
# Import internal modules
# ----------------------------------------------------------------------------------------------------------------------
import lifetime_value as ltv


# ----------------------------------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------------------------------
event_log_df = pd.DataFrame({
    'subject_id': ['user_a', 'user_a', 'user_a', 'user_b', 'user_b', 'user_a'],
    'date': ['2021-01-04', '2021-01-04', '2021-01-10', '2021-01-05', '2021-01-07', '2021-01-07'],
    'type': ['registration', 'conversion', 'conversion', 'registration', 'conversion', 'conversion'],
    'value': [0, 10, 5, 0, 7, 1],
})
event_log_df['date'] = [datetime.datetime.strptime(item, "%Y-%m-%d").date() for item in event_log_df.date]

df_result = ltv.from_event_log(event_log_df, confidence_level=0.8)
print(df_result)

