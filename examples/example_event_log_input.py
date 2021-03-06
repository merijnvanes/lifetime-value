"""Example of common event log input.
First we generate a data sample of events 'registration' and 'conversion' per user for a given date and with a
corresponding conversion value. Then we transform the data into a format the 'lifetime-value' library can work with.
After the transformation is done, the 'lifetime_value()' function is called and the results are visualised using the
'matplotlib' library.

Example event log input lines:
user_id                                     date          type      value
e9b0016a-50f4-4a5d-a245-59b8bf856793  2020-04-30    conversion  18.686502
551cb997-8900-4075-8bbe-4ce9b1694485  2020-06-01  registration        NaN
e14dc6c3-57ab-4e65-a1f7-5f6abdcf3f51  2020-06-04    conversion   1.381172

Created: 2021-04-09 (Merijn)
Updated:
"""


# ----------------------------------------------------------------------------------------------------------------------
# Import libraries
# ----------------------------------------------------------------------------------------------------------------------
import datetime
import random
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
# Import internal modules
# ----------------------------------------------------------------------------------------------------------------------
import lifetime_value as ltv


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------
N_USERS = 100
DATA_EXTRACTION_DATE = '2020-12-31'
REGISTRATION_DATE_START = '2020-01-01'
REGISTRATION_DATE_END = '2020-01-31'


# ----------------------------------------------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------------------------------------------
def generate_random_date(start_date: datetime.date, end_date: datetime.date) -> datetime.date:
    days_between_dates = (end_date - start_date).days
    random_number_of_days = random.randrange(days_between_dates)
    return start_date + datetime.timedelta(days=random_number_of_days)


# ----------------------------------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------------------------------
# Format input dates.
registration_start_date = datetime.datetime.strptime(REGISTRATION_DATE_START, "%Y-%m-%d").date()
registration_end_date = datetime.datetime.strptime(REGISTRATION_DATE_END, "%Y-%m-%d").date()
end_date = datetime.datetime.strptime(DATA_EXTRACTION_DATE, "%Y-%m-%d").date()

# Generate event log data.
column_names = ['user_id', 'date', 'type', 'value']
event_log = pd.DataFrame(columns=column_names)
for i in range(N_USERS):
    user_id = uuid.uuid4()
    registration_date = generate_random_date(registration_start_date, registration_end_date)
    event_log = event_log.append(pd.Series(
            [
                user_id,
                registration_date,
                'registration',
                np.nan,
            ],
            index=column_names
        ),
        ignore_index=True
    )
    n = np.random.poisson(lam=2, size=None)
    if n > 0:
        for j in range(n):
            event_log = event_log.append(pd.Series(
                    [
                        user_id,
                        generate_random_date(registration_date, end_date),
                        'conversion',
                        np.random.normal(loc=15, scale=5),
                    ],
                    index=column_names
                ),
                    ignore_index=True
            )
print(event_log)

# Get dataframe of empirical lifetime value through time.
event_log = event_log.rename({'user_id': 'subject_id'}, axis='columns')
df_result = ltv.from_event_log(event_log, confidence_level=0.8)

# Generate a graph using the 'matplotlib' library.
x = df_result.time
plt.plot(
    x, df_result.value, 'k',
    x, df_result.confidence_interval_left_bound, 'k--',
    x, df_result.confidence_interval_right_bound, 'k--',
)
plt.xlabel('time')
plt.ylabel('value')
plt.show()
