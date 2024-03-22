"""
Example usage:
--------------
python paper_experiments/wikidata/translations/extract_labels_for_translation.py \
    --input paper_experiments/assets/entitizer.pkl \
    --output data/interim/wikidatamatcher/labels_to_translate_tr.txt \
    --language tr
"""
from topic_inference.utils import readpickle, writelines

from tqdm import tqdm

from pathlib import Path
import argparse

HUMANS = {'Q5', 'Q95074', 'Q15632617', 'Q28020127'}
BUSINESSES = ['Q4830453', 'Q6881511', 'Q167037', 'Q431289', 'Q43229']


def extract_labels_for_translation(tokenizer_path, output, lang, add_humans):
    tokenizer = readpickle(tokenizer_path)

    ignore = set(BUSINESSES)
    if not add_humans:
        ignore = ignore.union(HUMANS)

    labels = []
    for v in tqdm(tokenizer.knowledge_base.values(), desc="Checking entities"):
        if not v['labels'].get(lang):
            if ignore.intersection(set(v['properties'].get('instance_of', []))):
                continue
            if v['labels'].get('en'):
                labels.append(v['labels']['en'])
                instance_of = set(v['properties'].get('instance_of', []))
                is_human = bool(HUMANS.intersection(instance_of))
                if not is_human:
                    labels.extend(v['aliases'].get('en', []))

    labels = [o for o in tqdm(labels, desc="Selecting labels") if len(o) >= 4
              and ' ' in o and not o.isupper()]  # and not o.istitle()]

    print('\nTotal number of labels:', len(labels))

    writelines(labels, output)


def main(args):
    extract_labels_for_translation(tokenizer_path=args.input,
                                   output=args.output, lang=args.language,
                                   add_humans=args.add_humans)


def parse_args():
    parser = argparse.ArgumentParser(
        "Extract english labels that require translation to a target language")

    parser.add_argument("--input", "-i", required=True, type=Path,
                        help="path to the tokenizer object")
    parser.add_argument("--output", "-o", required=True, type=Path,
                        help="path to save the list of labels")
    parser.add_argument("--language", "-l", required=True,
                        help="language to capture labels from")
    parser.add_argument("--add-humans", action='store_true',
                        help="include names of humans")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_args())
