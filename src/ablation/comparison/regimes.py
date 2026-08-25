# Define a function to label dates based on regimes
import pandas as pd
def assign_regime_labels(dates, regimes):
    """
    Given a list of dates and a dictionary of regimes with {'regime_name': (start_date, end_date), ...},
    this function returns a list of labels indicating to which regime each date belongs.

    Args:
        dates (list or pd.DatetimeIndex): List of dates to be labeled.
        regimes (dict): Dictionary with regime names as keys and tuples of (start_date, end_date) as values.

    Returns:
        list: List of regime labels for each date.
    """
    labels = []
    for date in dates:
        label_found = False
        for regime_name, (start_date, end_date) in regimes.items():
            if pd.to_datetime(start_date) <= date <= pd.to_datetime(end_date):
                labels.append(regime_name)
                label_found = True
                break
        if not label_found:
            labels.append("Unknown")
    return labels
