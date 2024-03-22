from topic_inference.utils import as_list, get_logger
from topic_inference.multiprocessing.parallelizer2file import N_JOBS, \
    parallelizer2file

import argparse
import logging
from functools import partial
import json

logger = get_logger(stderr_level=logging.ERROR)

LANGUAGES = ('en', 'fr', 'de', 'pl', 'ru', 'tr',
             'es', 'pt', 'ms', 'ar', 'it', 'th')

# add more properties as needed
PROPERTIES = {
    'citizenship': 'P27',
    'country': 'P17',
    'occupation': 'P106',
    'employer': 'P108',
    'field_of_work': 'P101',
    'position_held': 'P39',
    'member_of': 'P463',
    'owner_of': 'P1830',
    'owned_by': 'P127',
    'founded_by': 'P112',
    'ceo': 'P169',
    'chairperson': 'P488',
    'stock_exchange': 'P414',
    'product': 'P1056',
    'notable_work': 'P800',
    'instance_of': 'P31',
    'part_of': 'P361',
    'subclass_of': 'P279',
    'topic': 'P910',
    'follows': 'P155',
    'followed_by': 'P156',
    'manufacturer': 'P176',
    'industry': 'P452',
    'cuisine': 'P2012',
    'facet_of': 'P1269',
    'brand': 'P1716',
    'political_party': 'P102',
    'sports_league': 'P118',
    'sports_season_of': 'P3450',
    'sport': 'P641',
    'sports_country': 'P1532',
    'sports_team': 'P54',
    'participant_in': 'P1344',
    'candidate_in_election': 'P3602',
    'has_effect': 'P1542',
    'immediate_cause_of': 'P1536',
    'art_movement': 'P135',
    'genre': 'P136',
    'part_of_series': 'P179',
    'fictional_universe': 'P1434',
    'has_part': 'P527',
    'form_of_creative_work': 'P7937',
    'performer': 'P175',
    'creator': 'P170',
    'present_in_work': 'P1441',
    'significant_event': 'P793',
    'medical_condition': 'P1050',
    'field_of_occupation': 'P425',
    'location_of_formation': 'P740',
    'location_of_headquarters': 'P159',
    'practiced_by': 'P3095',
    'defendant': 'P1591',
    'end_time': 'P582',
    'has_cause': 'P828',
    'has_contributing_factor': 'P1479',
    'has_immediate_cause': 'P1478',
    'in_opposition_to': 'P5004',
    'inception': 'P571',
    'legislated_by': 'P467',
    'located_in': 'P131',
    'located_terrain_feature': 'P706',
    'location': 'P276',
    'objective': 'P3712',
    'official_date_opening': 'P1619',
    'organizer': 'P664',
    'partially_coincident': 'P1382',
    'participant': 'P710',
    'perpetrator': 'P8031',
    'point_in_time': 'P585',
    'signatory': 'P1891',
    'significant_person': 'P3342',
    'start_time': 'P580',
    'target': 'P533',
    'time_period': 'P2348',
    'time_spacecraft_landing': 'P620',
    'victim': 'P8032',
    'candidate': 'P726',
    'office_contested': 'P541',
    'same_as': 'P460',
    'short_name': 'P1813',
    'subsidiary': 'P355',
    'capital_of': 'P1376',
    'continent': 'P30',
    'official_name': 'P1448',
    'chief_executive_officer': 'P169',
    'cast_member': 'P161',
    'director': 'P57',
    'screenwriter': 'P58',
    'author': 'P50',
    'country_of_origin': 'P495',
    'producer': 'P162',
    'composer': 'P86',
    'place_of_birth': 'P19',
    'child': 'P40',
    'parent_organization': 'P749',
    'developer': 'P178',
}


def get_properties(w):
    all_claims = w.get('claims', {})
    if not all_claims:
        return {}

    output = {}
    for property_label, property_id in PROPERTIES.items():
        values = []
        for claim in all_claims.get(property_id, []):
            try:
                values.append(claim['mainsnak']['datavalue']['value']['id'])
                continue
            except KeyError:
                pass
            try:
                values.append(claim['mainsnak']['datavalue']['amount'])
                continue
            except KeyError:
                pass

        if values:
            output[property_label] = values
    output = {k: v for k, v in output.items() if v}
    return output


def get_entities(line, languages=LANGUAGES):
    languages = as_list(languages)
    global logger
    line = line.decode('utf-8').rstrip()
    try:
        w = json.loads(line.rstrip(',\n').strip('"'))
    except json.decoder.JSONDecodeError:
        if line.strip() != '[' and line.strip() != ']':
            logger.info(f'Can\'t decode line of length {len(line)} '
                        f'starting with "{line[:40]}"')
        return

    aliases = {o: [] for o in languages}
    for lang, aliases_per_language in w['aliases'].items():
        for alias in aliases_per_language:
            if lang in languages:
                aliases[lang].append(alias['value'])
    for key in list(aliases.keys()):
        if not aliases[key]:
            del aliases[key]

    try:
        row = {
            'id': w['id'],
            'labels': {lang: w['labels'].get(lang, {'value': None})['value']
                       for lang in LANGUAGES if
                       w['labels'].get(lang, {'value': None})['value']},
            'descriptions': {
                lang: w['descriptions'].get(lang, {'value': None})['value']
                for lang in LANGUAGES if
                w['descriptions'].get(lang, {'value': None})['value']
            },
            'aliases': aliases,
            'properties': get_properties(w)
        }

        enwiki = w.get('sitelinks', {}).get('enwiki', {}).get('title')
        if enwiki:
            row['enwiki'] = enwiki

        # ignore those rows that have no label
        if not any(row['labels'].values()):
            return

    except KeyError as e:
        logger.info(f'Missing key {e} on line of length {len(line)} '
                    f'starting with "{line[:40]}"')
        return
    return json.dumps(row)


def main(args):
    global logger
    if args.verbose:
        logger.setLevel(logging.INFO)

    parallelizer2file(
        input_path=args.input,
        output_path=args.output,
        fn=partial(get_entities, languages=args.languages),
        n_jobs=args.njobs,
        chunksize=args.chunksize,
        verbose=args.verbose
    )

    logger.info('Finished preprocessing Wikidata')


def parse_args():
    parser = argparse.ArgumentParser("Retrieve key properties from wikidata")

    parser.add_argument("input", nargs=1, help="")
    parser.add_argument("output", nargs=1, help="")
    parser.add_argument("--languages", "-l", nargs='*', default=LANGUAGES,
                        help="")
    parser.add_argument("--njobs", "--n_jobs", "-n", type=int, default=N_JOBS,
                        help="")
    parser.add_argument("--chunksize", "-c", type=int, default=1000,
                        help="")
    parser.add_argument("--verbose", "-v", action="store_true", help="")

    args = parser.parse_args()

    args.input = args.input[0]
    args.output = args.output[0]

    return args


if __name__ == '__main__':
    main(parse_args())
