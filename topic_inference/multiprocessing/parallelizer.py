from . import N_JOBS

from tqdm import tqdm

import multiprocessing as mp
from functools import partial


def parallelizer(fn, iterator, n_jobs=None, verbose=False, desc=None,
                 *args, **kwargs):
    n_jobs = n_jobs or min(N_JOBS, len(iterator))
    pool = mp.Pool(n_jobs)
    fn = partial(fn, *args, **kwargs)
    iterator = tqdm(iterator, desc=desc) if verbose else iterator
    output = pool.map(fn, iterator)
    pool.close()
    pool.join()
    return output
