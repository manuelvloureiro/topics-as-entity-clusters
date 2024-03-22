from topic_inference.typing import *

import pandas as pd
import numpy as np

NUMWORDS = 500
MINSCORE = 1e-4


def read_topics_excel(path: PathType, max_column=NUMWORDS):
    df = pd.read_excel(path, engine='openpyxl')

    # standardize column names
    column_names = dict(zip(df.columns, [o.title() for o in df.columns]))
    df = df.rename(columns=column_names)
    df = df.rename(columns={'Label': 'Topic Label'})

    # filter rows, this is mostly used with the Keep column
    for column in ['Topic No.', 'Topic Label', 'Keep']:
        try:
            filter_column = list(df.columns).index(column)
        except ValueError:
            pass
        else:
            column = df.iloc[:, filter_column]
            df = df[~column.replace(False, np.nan).isnull()]

    # filter columns
    pattern = 'Id {}' if 'Id 1' in df.columns else 'Word {}'
    try:
        columns = df.columns[list(df.columns).index(pattern.format(1)):]
    except ValueError as e:
        print(f"Couldn't find column '{pattern.format(1)}'."
              f" These are the first columns: {list(df.columns)[:10]}")
        raise e
    last_column = pattern.format(max_column + 1)
    if last_column in df.columns:
        columns = df.columns[:list(df.columns).index(last_column)]
    meta = df[[o for o in df.columns if o not in columns]]
    df = df[columns]

    return meta, df
