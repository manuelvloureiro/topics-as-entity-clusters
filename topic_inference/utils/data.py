from topic_inference.typing import *
from .general import chunks

from tqdm import tqdm

from pathlib import Path
import pickle
import bz2
import json
from functools import partial
import csv


# Lines
################################################################################
def iteratelines(path: PathType) -> Iterable[str]:
    with open(path, encoding="utf8") as f:
        for o in f:
            yield o.rstrip('\n')


def readlines(path: PathType) -> List[str]:
    with open(path, encoding="utf8") as f:
        return [o.rstrip('\n') for o in f]


def writelines(obj: Iterable, path: PathType, append: bool = False,
               buffer: int = 10_000):
    path = Path(path)
    if not append and path.exists():
        path.unlink()
    with open(path, 'a') as f:
        for chunk in chunks(obj, buffer):
            f.write('\n'.join(chunk) + '\n')
        f.truncate(f.tell() - 1)  # truncates last '\n'


# CSV
################################################################################

def readcsv(path: PathType, quotechar='"', delimiter=',',
            quoting=csv.QUOTE_MINIMAL, doublequote=True,
            skipinitialspace=True) -> List[List[str]]:
    with open(path, encoding="ISO-8859-1") as f:
        lines = (o.rstrip('\n') for o in f)
        csv_ = [o for o in
                csv.reader(lines, quotechar=quotechar, delimiter=delimiter,
                           quoting=quoting, doublequote=doublequote,
                           skipinitialspace=skipinitialspace)]
    return csv_


# Pickle
################################################################################

def readpickle(path: PathType) -> PickleType:
    with open(path, 'rb') as f:
        return pickle.load(f)


def writepickle(obj: PickleType, path: PathType):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def getpickle(path: PathType, fn: Callable, *args, **kwargs) -> PickleType:
    path = Path(path)
    if path.exists():
        return readpickle(path)
    obj = fn(*args, **kwargs)
    writepickle(obj, path)
    return obj


# json
################################################################################
def readjson(path: PathType) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def writejson(obj: JsonType, path: PathType):
    with open(path, 'w') as f:
        json.dump(obj, f)


# jsonl
################################################################################

def iteratejsonl(path: PathType, strict: bool = True) -> List[dict]:
    with open(path, 'r') as f:
        for o in f:
            try:
                yield json.loads(o.rstrip(), strict=strict)
            except (json.decoder.JSONDecodeError, TypeError):
                pass


def readjsonl(path: PathType, strict: bool = True) -> List[Union[dict, list]]:
    with open(path, 'r') as f:
        obj = [json.loads(o.rstrip(), strict=strict) for o in f.readlines()]
    return obj


def writejsonl(obj: JsonType, path: PathType, append: bool = False,
               buffer: int = 10_000):
    path = Path(path)
    if not append and path.exists():
        path.unlink()
    with open(path, 'a') as f:
        for chunk in chunks(obj, buffer):
            f.write('\n'.join([json.dumps(o) for o in chunk]))
            f.write('\n')


# BZ2
################################################################################

def iteratebz2(path: PathType, rstrip: str = ',\n') -> Iterator[str]:
    for line in bz2.BZ2File(path, 'r'):
        yield line.decode().strip().rstrip(rstrip)


# Advanced
################################################################################

def process2file(input_path: PathType, output_path: PathType, fn: Callable,
                 verbose: bool = False, *args, **kwargs):
    input_path = Path(input_path)
    output_path = Path(output_path)
    open_fn = bz2.BZ2File if 'bz2' in input_path.suffix else open

    fn = partial(fn, *args, **kwargs)

    desc = f"{input_path.name} > {output_path.name}"
    iterator = open_fn(input_path, 'r')
    iterator = tqdm(iterator, desc=desc) if verbose else iterator

    with open(output_path, 'w') as f:
        for line in iterator:
            result = fn(line)
            f.write(str(result).rstrip('\n'))
            f.write('\n')
        f.truncate(f.tell() - 1)  # truncates last '\n'
