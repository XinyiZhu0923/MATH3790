import Entropy as En
def calculate_information_gain(df, feature, target):
    entropy_parent = En.calculate_entropy(df[target])
    unique_values = df[feature].unique()
    entropy_children = 0
    for value in unique_values:
        df_subset = df[df[feature] == value]
        entropy_subset = En.calculate_entropy(df_subset[target])
        weight = len(df_subset) / len(df)
        entropy_children += weight * entropy_subset
    information_gain = entropy_parent - entropy_children
    return information_gain