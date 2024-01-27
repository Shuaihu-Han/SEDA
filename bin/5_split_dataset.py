#! /usr/bin/env python3

import gzip
import json
import os
from dataclasses import dataclass
from sys import stderr

from tqdm import tqdm

from dataclass_parser import parse_into_dataclass
from circumst_event.preprocessing.models import IndexedChain


def load_gz_list(list_filepath: str):
    example_ids_with_txt = gzip.open(list_filepath, "rt").read().splitlines()
    return {
        os.path.splitext(ex_id_with_txt)[0] for ex_id_with_txt in example_ids_with_txt
    }


@dataclass
class _SplitArguments:
    __program__ = __file__
    chain_file_path: str
    dev_gz_list: str
    test_gz_list: str
    split_file_path: str


def main():

    args = parse_into_dataclass(_SplitArguments)
    os.makedirs(args.split_file_path, exist_ok=True)

    dev_document_ids = load_gz_list(args.dev_gz_list)
    print("Find %d dev Example" % len(dev_document_ids), file=stderr)
    test_document_ids = load_gz_list(args.test_gz_list)
    print("Find %d test Example" % len(test_document_ids), file=stderr)

    chain_iter = map(json.loads, tqdm(gzip.open(args.chain_file_path, "rt")))
    event_chain: IndexedChain
    with gzip.open(
        os.path.join(args.split_file_path, "train.json.gz"), "wt"
    ) as train_fp, gzip.open(
        os.path.join(args.split_file_path, "dev.json.gz"), "wt"
    ) as dev_fp, gzip.open(
        os.path.join(args.split_file_path, "test.json.gz"), "wt"
    ) as test_fp:
        for event_chain in chain_iter:
            if event_chain["document_id"] in test_document_ids:
                json.dump(event_chain, test_fp)
                test_fp.write("\n")
            elif event_chain["document_id"] in dev_document_ids:
                json.dump(event_chain, dev_fp)
                dev_fp.write("\n")
            else:
                json.dump(event_chain, train_fp)
                train_fp.write("\n")

        pass


if __name__ == "__main__":
    main()