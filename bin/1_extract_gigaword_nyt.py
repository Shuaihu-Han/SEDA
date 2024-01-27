#! /usr/bin/env python3

import gzip
import os
import re
from argparse import ONE_OR_MORE, ArgumentParser
from typing import Iterable, List

import bs4
from joblib import Parallel, cpu_count, delayed


def make_nyt_re(years: List[str]):
    return re.compile(r"nyt_eng_(%s)\d{2}\.gz" % ("|".join(years)))


def parse_tgz_to_xml(tgz_filepath: str, output_path: str):
    with gzip.open(tgz_filepath) as gfp:
        xml_data = gfp.read()
    soup = bs4.BeautifulSoup(xml_data, features="html.parser")
    all_docs = soup.find_all("doc", attrs={"type": "story"})
    for doc in all_docs:
        doc_id = doc["id"]
        sentences = [
            str.replace(p.text, "\n", "").strip() for p in doc.find_all("p")
        ]
        with open(os.path.join(output_path, "%s.txt" % doc_id), "w") as fp:
            fp.write("\n".join(sentences))


def list_tgz_files(gigaword_filepath: str, years: List[str]) -> Iterable[str]:
    nyt_re = make_nyt_re(years)
    for filename in os.listdir(gigaword_filepath):
        if nyt_re.match(filename):
            yield os.path.join(gigaword_filepath, filename)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--gigaword-nyt", help="path to the gigaword file path", required=True
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--years", nargs=ONE_OR_MORE, default=DEFAULT_YEARS)
    parser.add_argument("--jobs", "-j", type=int, default=cpu_count() // 2)
    args = parser.parse_args()
    tgz_iter = list_tgz_files(args.gigaword_nyt, args.years)
    os.makedirs(args.output_path, exist_ok=True)
    Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(parse_tgz_to_xml)(tgz_path, args.output_path)
        for tgz_path in tgz_iter
    )


DEFAULT_YEARS = [str(int_year) for int_year in range(1994, 2004)]

if __name__ == "__main__":
    main()