#! /usr/bin/env python3
"""
This script is used to annotate text to corenlp json output from multiple corenlp server.
It use `stanza` for corenlp client and `joblib` for parallelism.
"""

import gzip
import json
import logging
import os
from argparse import ArgumentParser
from typing import List
import sys

from joblib import Parallel, delayed
from more_itertools import chunked, distribute
from requests import ReadTimeout
from stanza.server import CoreNLPClient, StartServer
from stanza.server.client import TimeoutException

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_file_size(filepath: str) -> int:
    return os.stat(filepath).st_size


def batch_annotate(example_paths: List[str], endpoint: str, output_path: str):
    client = CoreNLPClient(
        start_server=StartServer.DONT_START,
        endpoint=endpoint,
        output_format="json",
    )
    client.ensure_alive()
    for example_path in sorted(
        example_paths, key=get_file_size
    ):  # small file first
        basename = os.path.basename(example_path)
        example_id, _ = os.path.splitext(basename)
        try:
            with open(example_path) as fp:
                text = fp.read()
            resp = client.annotate(text)
        except (TimeoutException, ReadTimeout):
            logger.warning("Timeout: %s" % example_id)
            continue

        except Exception as e:
            logger.exception(e)
            continue

        with open(os.path.join(output_path, "%s.json" % example_id), "w") as fp:
            json.dump(resp, fp)


def host_annotate(
    endpoint: str,
    n_jobs: int,
    example_paths: List[str],
    output_path: str,
):
    CoreNLPClient(
        StartServer.DONT_START,
        endpoint=endpoint,
    ).ensure_alive()
    print(
        "Process %s at %s. " % (len(example_paths), endpoint),
        file=sys.stderr,
    )
    Parallel(n_jobs=n_jobs, verbose=11)(
        delayed(batch_annotate)(paths, endpoint, output_path)
        for paths in chunked(example_paths, 101)
    )


ENDPOINTS = [
    ("https://192.168.190.189:9000", 24),
    ("https://192.168.190.186:9000", 24),
]


def remove_suffix(string: str, suffix: str):
    if string.endswith(suffix):
        string = string[: -len(suffix)]
    return string


def load_list_file(
    list_file_path: str,
    suffix: str = None,
):
    with open(list_file_path) as fp:
        ids = fp.readlines()
    if suffix:
        ids = [remove_suffix(i, suffix) for i in ids]
    return ids


def main():
    logging.basicConfig(level=logging.DEBUG)
    parser = ArgumentParser()
    parser.add_argument(
        "--corpus-path",
        help="the path of directory that contains text file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--annotation-path",
        "-o",
        help="annotation path to store the annotated documents",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--gz_list",
        help="Optional argument. You can use it to process on a small dataset. such as dev or test.list.gz file given by (2016)",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--duplicate_list",
        help="Path to the duplicate",
        type=str,
        required=False,
        default=None,
    )

    args = parser.parse_args()

    example_txt_files = (
        gzip.open(args.gz_list, "wt").read().splitlines()
        if args.gz_list
        else os.listdir(args.corpus_path)
    )
    example_files = [
        remove_suffix(example_txt_file, ".txt")
        for example_txt_file in example_txt_files
    ]
    print("Find %d examples" % len(example_files), file=sys.stderr)

    if args.duplicate_list:
        duplicates = set(
            gzip.open(args.duplicate_list, mode="rt").read().splitlines()
        )
        print("Load %d NYT duplicates" % len(duplicates), file=sys.stderr)
        nyt_duplicates = {
            dup for dup in duplicates if dup.startswith("NYT_ENG")
        }
        print("Load %d NYT duplicates" % len(nyt_duplicates), file=sys.stderr)
        example_files = [
            example_file
            for example_file in example_files
            if example_file not in nyt_duplicates
        ]
        print(
            "Remained %d after removing duplicates" % len(example_files),
            file=sys.stderr,
        )

    doc_file_paths = [
        os.path.join(args.corpus_path, "%s.txt" % file)
        for file in example_files
        if not os.path.exists(
            os.path.join(args.annotation_path, "%s.json" % file)
        )
    ]
    print(
        "Remained %d after removing processed" % len(doc_file_paths),
        file=sys.stderr,
    )

    Parallel(len(ENDPOINTS))(
        delayed(host_annotate)(
            endpoint, n_jobs, list(paths), args.annotation_path
        )
        for (endpoint, n_jobs), paths in zip(
            ENDPOINTS, distribute(len(ENDPOINTS), doc_file_paths)
        )
    )


if __name__ == "__main__":
    main()