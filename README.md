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
import pandas as pd
import lifetime_value as ltv

df_subjects = pd.DataFrame({
    'subject_id': ['a', 'b', 'c'],
    'lifetime': [6, 6, 4],
})

df_events = pd.DataFrame({
    'subject_id': ['a', 'a', 'b', 'c', 'c', 'a'],
    'time': [3, 1, 5, 1, 4, 3],
    'value': [12.3, 0.5, 1.5, 3.3, 34.3, 1.2]
})

df_result = ltv.from_subjects_and_events_dataframe(df_subjects, df_events, confidence_level=0.8)
print(df_result)
```

Will return: 
```commandline
   time      value  confidence_interval_left_bound  confidence_interval_right_bound
0     0   0.000000                        0.000000                         0.000000
1     1   1.266667                        0.166667                         2.366667
2     2   1.266667                        0.166667                         2.366667
3     3   5.766667                        1.100000                        10.433333
4     4  17.200000                        4.666667                        29.733333
```
Note that the results of the confidence intervals could vary, because they
are estimated with a probabilistic resampling technique.
 