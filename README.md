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

df_result = ltv.lifetime_value(df_subjects, df_events, confidence_level=0.8)
print(df_result)
```