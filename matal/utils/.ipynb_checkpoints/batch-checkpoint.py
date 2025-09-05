from typing import Union, List

import pandas as pd

from ..configs import BATCH_DIR


def tag_to_str(tags: Union[List[str], str]):
    if isinstance(tags, str):
        return tags
    elif isinstance(tags, list):
        return '__'.join(tags)


def load_batch(batch_sid: str, label: Union[str, None] = None, sid_as_index=True):
    if label:
        fn = BATCH_DIR / f'{batch_sid}.{label}.csv'
    else:
        fn = BATCH_DIR / f'{batch_sid}.csv'
    with open(fn, 'r') as f:
        batch_df = pd.read_csv(f)
    
    if sid_as_index:
        batch_df.set_index('SampleID', drop=True, inplace=True, verify_integrity=True)

    return batch_df



