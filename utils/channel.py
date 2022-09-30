#/usr/bin/env python3
import pandas as pd

def from_data(df, secret, qids = None):
    # If no QID is informed, we consider all features except the secret
    # as QIDs, following the same approach as PIF (i.e. an strong adversary
    # that knows all but one of the features in the dataset).
    if qids is None:
        qids = [feature for feature in df.columns if feature != secret]

    # Compute the joint matrix so we can get the marginal on X (prior)
    # and the channel (just divide the joint by the prior).
    # J = {x: {y: 0 for y in df.groupby(qids).groups.keys()}
    #      for x in df[secret].unique()}

    # Count the occurrences of each row (i.e. combinations of secret + qid)
    count_row = df.value_counts().to_dict()

    # Count the occurrences of the elements of the sensitive attribute
    count_x = {x[0]: count for x, count in df.value_counts([secret]).to_dict().items()}

    C = {}
    cache_y = {}

    for record in df.drop_duplicates().to_dict('records'):
        row = tuple(record.values())
        x = record[secret]
        y = cache_y[row] = cache_y.get(
            row, tuple(record[f] for f in record if f != secret))
        if y not in C: C[y] = {}
        C[y][x] = count_row[row] / count_x[x]

    pi = {x: count / len(df.index) for x, count in count_x.items()}
    return pi, pd.DataFrame(C).fillna(0)
