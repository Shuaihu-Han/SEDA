#! /usr/bin/env python3

import gzip
import json
from dataclasses import dataclass
from itertools import chain

import numpy
from tqdm import tqdm

from dataclass_parser import parse_into_dataclass
from circumst_event.preprocessing.event_indexer import VocabEventIndexer
from circumst_event.preprocessing.models import EventChain
from torch_utils import load_glove_into_dict


@dataclass
class _IndexArguments:
    chain_file_path: str
    indexed_chain_file_path: str
    index_type: str
    argument_length: int
    max_sequence_length: int
    pretrained_path: str
    vocab_save_path: str
    jobs: int = 64


def main():
    args = parse_into_dataclass(_IndexArguments)

    def word_iter(event_chain: EventChain):
        for sentence in event_chain["sentences"]:
            yield from map(str.lower, sentence)

    chain_iter = map(json.loads, tqdm(gzip.open(args.chain_file_path, "rt")))
    chain_words = set(chain.from_iterable(map(word_iter, chain_iter)))
    glove_dict = load_glove_into_dict(args.pretrained_path)
    words = list(chain_words.intersection(glove_dict))
    indexer = VocabEventIndexer(
        words,
        missing_token="[UNK]",
        padding_token="[PAD]",
        special_tokens={"[UNK]": 1, "[PAD]": 0},
    )
    avg_embedding = numpy.row_stack(list(glove_dict.values())).mean(axis=0)
    with open(args.vocab_save_path, "w") as fp:
        for word in indexer.word2id:
            embedding_string = " ".join(map(str, glove_dict.get(word, avg_embedding)))
            fp.write(f"{word} {embedding_string}\n")

    chain_iter = map(json.loads, tqdm(gzip.open(args.chain_file_path, "rt")))
    indexed_chains = [
        indexer.index_chain(c, args.argument_length, args.max_sequence_length)
        for c in chain_iter
    ]
    with gzip.open(args.indexed_chain_file_path, "wt") as gfp:
        for e_chain in indexed_chains:
            json.dump(e_chain, gfp)
            gfp.write("\n")


if __name__ == "__main__":
    main()