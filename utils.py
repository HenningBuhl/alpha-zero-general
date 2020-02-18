import numpy as np


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

intervals = (
    #('weeks', 'w', 60 * 60 * 24 * 7),
    ('days', 'd', 60 * 60 * 24),
    ('hours', 'h', 60 * 60),
    ('minutes', 'm', 60),
    ('seconds', 's', 1),
    #('milli seconds', 'millis', 1e-3),
    #('micro seconds', 'micros', 1e-6),
    #('nano seconds', 'ns', 1e-9),
    #('pico seconds', 'ps', 1e-12),
    #('femto seconds', 'fs', 1e-15),
    #('atto seconds', 'as', 1e-18),
    #('zepto seconds', 'zs', 1e-21),
    #('yocto seconds', 'ys', 1e-24),
    )

def display_time(seconds, granularity=2, use_long_names=False, join=', ', separator=''):
    '''
    Args:
        seconds: Elapsed seconds.
        granularity: Number of values to include into the result (default 2). Use -1 to display all.
        use_long_names: Whether to use long instead of short format names (default False).
        join: Used to join all results (default ', ').
        separator: Used to separate the value and the name ({value}{separator}{name}) (default '').

    Returns:
        A formatted time string.
    '''
    result = []
    for long_name, short_name, count in intervals:
        name = long_name if use_long_names else short_name
        value = seconds // count
        if value: # Skip 0 values.
            seconds -= value * count
            if use_long_names and value == 1: name = name.rstrip('s') # remove plural 's'.
            result.append(f'{int(value)}{separator}{name}')
    if granularity == -1:
        granularity = len(intervals) + 1
    return join.join(result[:granularity])

def ewma(data, window_size=100): # Add parameters for [E]xponentially [W]eighted moving average
    '''
    Args:
        data: The data to operate on.
        window_size: The windows size of the moving average (default 100).
    
    Returns:
        The exponentially weighted moving average.
    '''
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec
