from . import N_JOBS
import multiprocessing as mp
import bz2
from functools import partial
from tqdm import tqdm
from pathlib import Path

__manager = mp.Manager()
__q = __manager.Queue()


def __worker(input_tuple):
    """Process the data and send it to the listener"""
    f, x = input_tuple
    out = f(x)
    if out is not None:
        __q.put(out)


def __listener(output_path):
    """Listens to processed data and saves it in the output file"""
    with open(output_path, 'w') as f:
        while True:
            m = str(__q.get())
            f.write(m + '\n')
            f.flush()


def parallelizer2file(input_path, output_path, fn, n_jobs=None, chunksize=1000,
                      verbose=False, *args, **kwargs):
    input_path = Path(input_path)
    output_path = Path(output_path)
    open_fn = bz2.BZ2File if 'bz2' in input_path.suffix else open

    fn = partial(fn, *args, **kwargs)

    desc = f"{input_path.name} > {output_path.name}"
    iterator = ((fn, o) for o in open_fn(input_path, 'r'))
    iterator = tqdm(iterator, desc=desc) if verbose else iterator

    n_jobs = n_jobs or N_JOBS
    pool = mp.Pool(n_jobs)

    # start the listener
    pool.apply_async(__listener, (output_path,))

    # parallelize
    try:
        list(pool.imap_unordered(__worker, iterator, chunksize=chunksize))
    except OSError:
        # this seems to show up with some files.
        # we can ignore this error as the processing is already complete
        pass
