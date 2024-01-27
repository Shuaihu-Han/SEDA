#! /usr/bin/env python3

import gzip
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Tuple

from tqdm import tqdm

from dataclass_parser import parse_into_dataclass
from circumst_event.preprocessing.dnee_adapter import dnee_event_chains
from circumst_event.preprocessing.models import EventChain


def make_predicate_string(predicate: List[str]):
    return " ".join(predicate)


VerbIndex = Dict[str, List[Tuple[str, int]]]


def merge_indexes(*verb_indexes: VerbIndex) -> VerbIndex:
    verb_index = defaultdict(list)
    for index in verb_indexes:
        for predicate, document_ids in index.items():
            verb_index[predicate].extend(document_ids)
    return verb_index


def build_predicate_index(event_chain: EventChain):
    return {
        make_predicate_string(event["predicate"]): [event_chain["document_id"]]
        for event in event_chain["events"]
    }


def build_verb_index(chain_file_path: str, jobs: int):
    chain_iter = map(json.loads, tqdm(gzip.open(chain_file_path, "rt")))
    indexes = map(build_predicate_index, chain_iter)
    verb_index = merge_indexes(*indexes)
    return verb_index


@dataclass(frozen=True)
class ExtractEventChain:
    __program__ = __file__

    annotation_folder_path: str
    chain_folder_path: str
    output_verb_index_file: str = None


def main():
    args = parse_into_dataclass(ExtractEventChain)
    annotation_paths = [
        os.path.join(args.annotation_folder_path, file_name)
        for file_name in os.listdir(args.annotation_folder_path)
    ]
    raw_event_chains = chain.from_iterable(
        dnee_event_chains(ann_file_path=ann_file_path)
        for ann_file_path in annotation_paths
    )
    with gzip.open(args.chain_folder_path, "wt") as gfp:
        chains = [c for c in raw_event_chains]
        for c in tqdm(chains):
            json.dump(c, gfp)
            gfp.write("\n")

    if args.output_verb_index_file is not None:
        verb_index = merge_indexes(*map(build_predicate_index, chains))
        predicate_counter = Counter(
            {key: len(value) for key, value in verb_index.items()}
        )
        with open(args.output_verb_index_file, "w") as fp:
            for item, count in predicate_counter.most_common():
                fp.write("%s\t%d\n" % (item, count))


if __name__ == "__main__":
    main()