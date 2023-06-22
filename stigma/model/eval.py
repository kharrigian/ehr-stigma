
#######################
### Imports
#######################

## External Libraries
import numpy as np

#######################
### Functions
#######################

def std_error(x):
    """
    
    """
    ## Non Null entries
    xnn = [i for i in x if not np.isnan(i)]
    ## Return Logic
    if len(xnn) == 0:
        return np.nan
    elif len(xnn) == 1:
        return 0
    return np.std(xnn) / np.sqrt(len(xnn) - 1)
