"""
    Creating string tables
"""

import pandas as pd
import itertools
import numpy as np
from typing import (  # NOQA
    Iterable, List, Dict, Any)


def string_table(
        table_rows: List[Iterable],
        header: List[str] = None,
        col_formats: Iterable[str] = itertools.repeat('{}'),
        col_alignments: Iterable[str] = itertools.repeat('<'),
        pad=0,
            ) -> str:
    """ Revisiting the string tables creation"""
    table_rows_s = [[cf.format(i)
        for i, cf in zip(row, col_formats)]
        for row in table_rows]
    if header is not None:
        table_rows_s = [header] + table_rows_s
    widths = []
    for x in zip(*table_rows_s):
        widths.append(max([len(y) for y in x]))
    formats = [f'{{:{a}{w}}}' for w, a in zip(widths, col_alignments)]
    formats = [f'{f:^{pad+len(f)}}' for f in formats]  # Apply padding
    row_format = '|' + '|'.join(formats) + '|'
    table = [row_format.format(*row) for row in table_rows_s]
    return '\n'.join(table)


def df_to_table_v1(df, pad=0):
    return string_table(
            df.reset_index().values,
            ['', ]+df.columns.tolist(), pad=pad)


def df_to_table_v2(df: pd.DataFrame, indexname=None) -> str:
    # Header
    if indexname is None:
        indexname = df.index.name
    if indexname is None:
        indexname = 'index'
    header = [indexname, ] + [str(x) for x in df.columns]
    # Col formats
    col_formats = ['{}']
    for dt in df.dtypes:
        form = '{}'
        if dt in ['float32', 'float64']:
            form = '{:.2f}'
        col_formats.append(form)

    table = string_table(
            np.array(df.reset_index()),
            header=header,
            col_formats=col_formats,
            pad=2)
    return table
