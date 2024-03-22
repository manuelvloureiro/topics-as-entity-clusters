import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
import logging
from collections import Counter
from topic_inference.utils import get_logger

logger = get_logger(stderr_level=logging.ERROR)

COLUMNS = ['relevant', 'event']


def count_by_property(wikidata_summary, output, properties, verbose=False):
    if output.suffix:  # if has suffix it is a file, get the folder
        output = output.parent

    entities = {}
    counters = {o: Counter() for o in properties}

    iterator = enumerate(open(wikidata_summary, 'r'))
    iterator = tqdm(iterator, desc='Counting values') if verbose else iterator

    for i, row in iterator:
        try:
            row = json.loads(row)
        except json.JSONDecodeError:
            continue
        # our working language is english so we ignore all those entities that
        # are not in this language as it is unlikely that an entity property
        # will not have an english label
        if not row['labels'].get('en'):
            continue
        # we get a dict with id, english label pairs that will then be used to
        # extract labels for those used as entity properties
        entities[row['id']] = row['labels']['en']
        for prop in properties:
            counters[prop].update(row['properties'].get(prop))

    logger.info('Counts completed. Generating files...')

    for prop in properties:
        filename = output / f'{prop}_counts.csv'
        save_property_csv(filename, entities, counters[prop])


def save_property_csv(filename, entities, counter):
    df = pd.DataFrame([(o[0], entities.get(o[0], o[0]), o[1])
                       for o in counter.most_common()])
    df.columns = ['id', 'label', 'count']

    if filename.exists():
        logger.info(f'File {filename} already exists. Rewriting...')

        old_df = pd.read_csv(filename)
        for column in COLUMNS:
            if column in old_df:
                old_df[column] = old_df[column].fillna(0).astype(int)
                logger.info(f"Found {sum(old_df[column] == 1)} values in old "
                            f"{filename.name} file for column {column}.")
                df = df.merge(old_df[['id', column]], on='id', how='left')
                df[column] = ['1' if o > 0 else '' for o in df[column]]
    else:
        logger.info(f'File {filename} does not exist. Writing...')

    df.to_csv(filename, index=False)
    logger.info(f'Saved file: {filename}')


def main(args):
    global logger
    if args.verbose:
        logger = get_logger(stderr_level=logging.INFO, message=None)

    count_by_property(wikidata_summary=args.input, output=args.output,
                      properties=args.properties, verbose=args.verbose)

    logger.info('Finished counting Wikidata by: ' + ', '.join(args.properties))


def parse_args():
    parser = argparse.ArgumentParser("Retrieve wikidata counts by property")

    parser.add_argument("input", help="Wikidata summary file")
    parser.add_argument("output", help="Wikidata property counts file folder")
    parser.add_argument("--properties", "-p", nargs='*', required=True,
                        help="List of properties to process")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show verbose output")

    args = parser.parse_args()

    args.input = args.input
    args.output = Path(args.output)

    return args


if __name__ == '__main__':
    main(parse_args())
