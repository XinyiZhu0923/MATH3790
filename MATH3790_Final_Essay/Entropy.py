import numpy as np
def calculate_entropy(series):
    p = series.value_counts(normalize=True)
    entropy = -np.sum(p * np.log2(p))
    return entropy