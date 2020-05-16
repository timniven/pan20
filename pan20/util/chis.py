"""Statistical helpers."""
import pandas as pd


def chi_squared(c1, c2, cn1, cn2):
    if c1 == 0:
        c1 = 1e-12
    if c2 == 0:
        c2 = 1e-12
    num = (c1*cn2 - c2*cn1)**2
    den = (c1 + c2) * (c1 * cn1) + (c2 * cn2) * (cn1 * cn2)
    return num / den


def chi_squareds(f1, f2):
    chis = {}
    n1 = f1['word_count']
    n2 = f2['word_count']
    key_set = set(f1['counts'].keys()).union(set(f2['counts'].keys()))
    for key in list(key_set):
        c1 = f1['counts'][key] if key in f1['counts'].keys() else 0
        c2 = f2['counts'][key] if key in f2['counts'].keys() else 0
        cn1 = n1 - c1
        cn2 = n2 - c2
        chi = chi_squared(c1, c2, cn1, cn2)
        chis[key] = {
            'c1': c1,
            'c2': c2,
            'chi': chi,
        }
    return chis


def get_chis(counts1, counts2, n1, n2):
    # assume intersection already occurred
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
