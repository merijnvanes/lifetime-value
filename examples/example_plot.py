"""Example of generating a graph of the results.
Using the library 'matplotlib', make sure to install this in order to run this example.

Created: 2021-04-09 (Merijn)
Updated:
"""


# ----------------------------------------------------------------------------------------------------------------------
# Import libraries
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
# Import internal modules
# ----------------------------------------------------------------------------------------------------------------------
import lifetime_value as ltv


# ----------------------------------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------------------------------
df_subjects = pd.DataFrame({
    'subject_id': ['a', 'b', 'c'],
    'lifetime': [6, 6, 4],
})

df_events = pd.DataFrame({
    'subject_id': ['a', 'a', 'b', 'c', 'c', 'a'],
    'time': [3, 1, 5, 1, 4, 3],
    'value': [12.3, 0.5, 1.5, 3.3, 34.3, 1.2]
})

df_result = ltv.lifetime_value(df_subjects, df_events, confidence_level=0.8)

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
