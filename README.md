# lifetime-value

A library to help finding the lifetime value of a group of subjects 
by calculating the average values through time.

## Installation

You can install `lifetime-value` from 
[PyPI](https://pypi.org/project/lifetime-value/) 
using `pip` like this:
```commandline
pip install lifetime-value
```

## Usage

The following example code:
```python
import datetime
import pandas as pd
import lifetime_value as ltv

event_log_df = pd.DataFrame({
    'subject_id': ['user_a', 'user_a', 'user_a', 'user_b', 'user_b', 'user_a'],
    'date': ['2021-01-04', '2021-01-04', '2021-01-10', '2021-01-05', '2021-01-07', '2021-01-07'],
    'type': ['registration', 'conversion', 'conversion', 'registration', 'conversion', 'conversion'],
    'value': [0, 10, 5, 0, 7, 1],
})
event_log_df['date'] = [datetime.datetime.strptime(item, "%Y-%m-%d").date() for item in event_log_df.date]

df_result = ltv.from_event_log(event_log_df, confidence_level=0.8)
print(df_result)
```

Will return: 
```commandline
   time  value  confidence_interval_left_bound  confidence_interval_right_bound
0     0    5.0                             0.0                             10.0
1     1    5.0                             0.0                             10.0
2     2    8.5                             7.0                             10.0
3     3    9.0                             7.0                             11.0
4     4    9.0                             7.0                             11.0
5     5    9.0                             7.0                             11.0

```
Note that the results of the confidence intervals could vary, because they
are estimated with a probabilistic resampling technique.
 