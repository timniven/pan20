"""Gradient boosting."""
import xgboost as xgb


def build_inputs(df):
    # need a csv for train and test with label, probs
    # df is loaded from the input files and has a row for each tweet

    # here if we defined a list of classifiers, and a function for each,
    # where we input the df above and output is, for each author, the
    # probability of them being a spreader.

    # then here we are joining these probabilities into a csv input file
    pass
