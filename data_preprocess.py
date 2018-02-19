import pandas as pd
import numpy as np


def find_correlation(df, thresh):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove and the count of collums
    Parameters
    ----------
    df : pd.DataFrame
    thresh : correlation threshold, will remove one of pairs of features with
              a correlation greater than this value
    Returns
    -------
    set : features to remove
    int : count of collums
    """

    corrMatrix = df.corr()
    corrMatrix.loc[:, :] = np.tril(corrMatrix, k=-1)

    already_in = set()
    result = []

    for col in corrMatrix:
        perfect_corr = corrMatrix[col][corrMatrix[col] > thresh].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)

    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    
    select_flat = set(select_flat)

    if len(select_flat) > 0:
      print('Correlations found between features with correlation threshols {}. {} out of {} features will be removed...'.format(
          thresh, len(select_flat), len(corrMatrix)))
    else:
      print('No correlations found with correlation threshols {}. All {} features are used in the training...'.format(
          thresh, len(corrMatrix)))
    return select_flat, len(corrMatrix)

def feature_reduction(df, thresh):
    """
    Given a pandas data frame with data, this will find highly correlated features
    and return a list of only relevant features as index of the columns,
    Parameters
    ----------
    df : a pandas data frame with data
    thresh : correlation threshold, will remove one of pairs of features with
              a correlation greater than this value

    Returns
    -------
    list : indexes of useful feature columns to use in the classification
    """

    print('Feature reduction:\nChecking if there is any correlation between the features in the training data...')

    columns_to_take_away, count_of_orig_columns = find_correlation(df, thresh)
    columns = set(range(count_of_orig_columns))

    columns = columns - columns_to_take_away

    return columns