"""
Example usage:
--------------
python paper_experiments/wikidata/translations/merge_labels_with_translations.py \
    --labels data/interim/wikidatamatcher/labels_to_translate_tr.txt \
    --translations data/interim/wikidatamatcher/translated_labels_tr.txt \
    --language tr \
    --output data/interim/wikidatamatcher/labels_pairs_tr.csv
"""
from topic_inference.utils import readlines, writelines
from topic_inference.lemmatizers import lemmatizers

from tqdm import tqdm

from pathlib import Path

import argparse


def merge_labels_with_translations(labels_path,
                                   translations_path, lang, output):
    def f(text):
        text = ' '.join([o.split("'")[0] for o in text.split()])
        return lemmatizers.get(lang, lambda x: x)(text)

    keys = (o.replace(',', ' ') for o in readlines(labels_path))
    values = (o.replace(',', ' ') for o in readlines(translations_path))

    iterator = tqdm(zip(keys, values), desc="Preprocessing translations")
    obj = ['label,translation'] + [','.join([k, f(v)]) for k, v in iterator]

    writelines(obj, output)


def main(args):
    merge_labels_with_translations(labels_path=args.labels,
                                   translations_path=args.translations,
                                   lang=args.language, output=args.output)


def parse_args():
    parser = argparse.ArgumentParser("Merge extracted english labels with"
                                     " translations after lemmatizing")

    parser.add_argument("--labels", required=True, type=Path,
                        help="path to the list of labels")
    parser.add_argument("--translations", required=True, type=Path,
                        help="path to the list of translations")
    parser.add_argument("--language", required=True,
                        help="language to capture labels from")
    parser.add_argument("--output", "-o", required=True, type=Path,
                        help="path to save the labels and translations pairs")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_args())
