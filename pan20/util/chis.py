"""Statistical helpers.

The main reference papers is (p. 9):
https://web.stanford.edu/~gentzkow/research/biasmeas.pdf

The target counts denote the total number of times phrases/tokens are used by
each class label.

The `n` for each class label is the total number of phrases that are not the
target. (So, if doing intersection, `n` should still include all others).
"""
import pandas as pd


def chi_squared(c1, c2, cn1, cn2):
    """Compute chi-squared for a target.

    Args:
      c1: number of times target used by group 1.
      c2: number of times target used by group 2.
      cn1: number of non-targets used by group 1.
      cn2: number of non-targets used by group 2.

    Returns:
      Float.
    """
    if c1 == 0:
        c1 = 1e-16
    if c2 == 0:
        c2 = 1e-16
    num = (c1*cn2 - c2*cn1)**2
    den = (c1 + c2) * (c1 + cn1) * (c2 + cn2) * (cn1 + cn2)
    return num / den


def get_chis(counts1, counts2):
    # assume intersection already occurred if desired
    n1 = len(counts1)
    n2 = len(counts2)
    chis = []
    for key in counts1.keys():
        c1 = counts1[key]
        c2 = counts2[key]
        cn1 = n1 - c1
        cn2 = n2 - c2
        chi = chi_squared(c1, c2, cn1, cn2)
        chis.append({
            'token': key,
            'c1': c1,
            'c2': c2,
            'n1': n1,
            'n2': n2,
            'chi': chi,
        })
    chis = pd.DataFrame(chis).sort_values(by='chi', ascending=False)
    return chis
