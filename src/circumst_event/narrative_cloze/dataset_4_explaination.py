import gzip
import json
import os
import random
from typing import Collection, Iterator, List, Optional, Union

import pytorch_lightning as pl
import torch
from more_itertools import islice_extended
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from circumst_event.narrative_cloze.choice_generator import EventPoolChoiceGenerator
from circumst_event.narrative_cloze.models import (
    MCNCWithSentence,
    MCNCWithSentenceBatch,
)
from circumst_event.preprocessing.models import IndexedChain, IndexedEvent
from matplotlib import pyplot as plt


def get_events_tensor(
    indexed_events: Collection[IndexedEvent],
    argument_length: int,
    num_to_pad: int,
):
    def handleAllZerosPredicate(predicate_ids):
        if predicate_ids[0] == 0:
            predicate_ids[0] = 1
        return predicate_ids

    def _event_as_tensor(indexed_event: IndexedEvent):
        predicate_ids = handleAllZerosPredicate(indexed_event["predicate_ids"][:argument_length])
        assert len(predicate_ids) == argument_length

        subject_ids = indexed_event["subject_ids"][:argument_length]
        assert len(subject_ids) == argument_length

        object_ids = indexed_event["object_ids"][:argument_length]
        assert len(object_ids) == argument_length

        return torch.as_tensor(
            [predicate_ids, subject_ids, object_ids], dtype=torch.long
        )

    event_tensors = [_event_as_tensor(event) for event in indexed_events]

    pad_event_tensors = torch.stack(
        event_tensors
        + [torch.ones_like(event_tensors[0], dtype=torch.long)] * num_to_pad,
    )

    mask_event_tensors = torch.as_tensor(pad_event_tensors != 0, dtype=torch.long)
    return (pad_event_tensors, mask_event_tensors)


class MCNCDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        num_chains: int = -1,
        num_choices: int = 5,
        argument_length: int = 15,
        context_events_length: int = 8,
        *,
        small: bool = False,
        use_part: bool = False,
        data_segmentation_num=4,
        data_current_segmentation=0,
        isRandom: bool = False,
        indicator = [],
        sample_count = 3
    ):
        super(MCNCDataset, self).__init__()
        self._data_file = data_file
        self._sample_count = sample_count
        self.isRandom = isRandom
        self.indicator = indicator
        chain_iter: Iterator[IndexedChain] = map(
            json.loads,
            tqdm(gzip.open(data_file, "rt"), desc="Loading Examples"),
        )
        if use_part:
            chain_iter = islice_extended(chain_iter, data_current_segmentation, None if num_chains == -1 else num_chains, data_segmentation_num)
        else:
            chain_iter = islice_extended(chain_iter, 0, None if num_chains == -1 else num_chains, 1)
        self._examples = []
        for chain in chain_iter:
            if len(self._examples) >= 150:
                break
            if len(chain["events"]) >= context_events_length:
                self._examples.append(chain)
        self.examples_len = len(self._examples)
        print(f"Here are {self.examples_len} examples.")

        self._num_choices = num_choices
        self._argument_length = argument_length
        self._context_events_length = context_events_length
        self._choice_generator = EventPoolChoiceGenerator.from_chains(self._examples)
        self._choices_array = [[] for _ in range(self.examples_len)] 

    def __getitem__(self, item: int) -> MCNCWithSentence:
        if self.isRandom:
            return self.__getRandomitem__(item)
        indexed_chain: IndexedChain = self._examples[item]
        events = indexed_chain["events"][: self._context_events_length + 1]
        context_events: List[IndexedEvent]
        [*context_events, target_event] = events
        num_to_pad_event = self._context_events_length - len(context_events)
        events_mask_tensor = torch.as_tensor(
            [True] * len(context_events) + [False] * num_to_pad_event,
            dtype=torch.bool,
        )

        choices = self._choice_generator.generate_choices(
            target_event, num_choices=self._num_choices
        )
        random.shuffle(choices)
        label = choices.index(target_event)
        events_tensor = get_events_tensor(
            context_events,
            argument_length=self._argument_length,
            num_to_pad=num_to_pad_event,
        )

        sentence_ids = torch.as_tensor(
            indexed_chain["sentences_ids"], dtype=torch.long
        )[indexed_chain["event_sentence_indexes"][: self._context_events_length]]

        sentences_ids_mask = torch.as_tensor(sentence_ids != 0, dtype=torch.long)

        event_sentence_indexes = torch.as_tensor(
            indexed_chain["event_sentence_indexes"][: self._context_events_length]
            ,
            dtype=torch.long,
        )

        event_sent_mat = torch.zeros(
            self._context_events_length,
            self._context_events_length,
            dtype=torch.float,
        )
        diags = [0] * self._context_events_length
        for sent_id in event_sentence_indexes:
            if sent_id >= self._context_events_length:
                continue
            diags[sent_id] += 1
        count = 0
        for c in diags:
            event_sent_mat[count : count + c, count : count + c] = 1
            count += c

        return MCNCWithSentence(
            events_tensor=events_tensor,
            events_mask_tensor=events_mask_tensor,
            choices_tensor=get_events_tensor(
                choices, argument_length=self._argument_length, num_to_pad=0
            ),
            label_tensor=torch.as_tensor(label, dtype=torch.long),
            sentences_tensor=sentence_ids,
            sentences_ids_mask=sentences_ids_mask,
            event_sent_mat=event_sent_mat,
        )

    def construct_chain(self, chains, origin_chain, event_len):
        my_chain = {}
        events = []
        event_sentence_indexes = []
        sentences_ids = []

        origin_chain_len = len(origin_chain['events'])
        r_indexs = []
        if origin_chain_len == event_len:
            r_indexs = random.sample(range(event_len - 1), event_len - 1)
        else:
            r_indexs = random.sample(range(event_len), event_len)

        for i, node in enumerate(zip(r_indexs, chains)):
            index = node[0]
            item = node[1]
            if self.indicator[i]:
                events.append(origin_chain['events'][i])
                event_sentence_indexes.append(i)
                sentences_ids.append(origin_chain['sentences_ids'][origin_chain["event_sentence_indexes"][i]])
            else:
                events.append(item['events'][index])
                event_sentence_indexes.append(i)
                sentences_ids.append(item['sentences_ids'][item["event_sentence_indexes"][index]])

        if origin_chain_len == event_len:
            events.append(origin_chain['events'][event_len - 1])
            event_sentence_indexes.append(event_len - 1)
            sentences_ids.append(origin_chain['sentences_ids'][origin_chain["event_sentence_indexes"][event_len - 1]])
        else:
            events.append(origin_chain['events'][event_len])
            event_sentence_indexes.append(event_len)
            sentences_ids.append(origin_chain['sentences_ids'][origin_chain["event_sentence_indexes"][event_len]])

        my_chain['events'] = events
        my_chain['event_sentence_indexes'] = event_sentence_indexes
        my_chain['sentences_ids'] = sentences_ids
        return my_chain
    
    def __getRandomitem__(self, item: int) -> MCNCWithSentence:
        real_item = item % self.examples_len
        allIndex = range(self.examples_len)
        randomIndex = random.sample(allIndex, self._context_events_length)
        random_indexed_chains: IndexedChain = []
        for i in randomIndex:
            random_indexed_chains.append(self._examples[i])

        indexed_chain = self.construct_chain(random_indexed_chains, self._examples[real_item], self._context_events_length)
        events = indexed_chain["events"][: self._context_events_length + 1]
        context_events: List[IndexedEvent]
        [*context_events, target_event] = events
        num_to_pad_event = self._context_events_length - len(context_events)
        events_mask_tensor = torch.as_tensor(
            [True] * len(context_events) + [False] * num_to_pad_event,
            dtype=torch.bool,
        )
        if self._choices_array[real_item]:
            choices = self._choices_array[real_item][0]
            label = self._choices_array[real_item][1]
        else:
            choices = self._choice_generator.generate_choices(
                target_event, num_choices=self._num_choices
            )
            random.shuffle(choices)
            label = choices.index(target_event)
        
            self._choices_array[real_item].append(choices)
            self._choices_array[real_item].append(label)
        events_tensor = get_events_tensor(
            context_events,
            argument_length=self._argument_length,
            num_to_pad=num_to_pad_event,
        )

        sentence_ids = torch.as_tensor(
            indexed_chain["sentences_ids"], dtype=torch.long
        )[indexed_chain["event_sentence_indexes"][: self._context_events_length]]

        sentences_ids_mask = torch.as_tensor(sentence_ids != 0, dtype=torch.long)

        event_sentence_indexes = torch.as_tensor(
            indexed_chain["event_sentence_indexes"][: self._context_events_length]
            ,
            dtype=torch.long,
        )

        event_sent_mat = torch.zeros(
            self._context_events_length,
            self._context_events_length,
            dtype=torch.float,
        )
        diags = [0] * self._context_events_length
        for sent_id in event_sentence_indexes:
            if sent_id >= self._context_events_length:
                continue
            diags[sent_id] += 1
        count = 0
        for c in diags:
            event_sent_mat[count : count + c, count : count + c] = 1
            count += c

        return MCNCWithSentence(
            events_tensor=events_tensor,
            events_mask_tensor=events_mask_tensor,
            choices_tensor=get_events_tensor(
                choices, argument_length=self._argument_length, num_to_pad=0
            ),
            label_tensor=torch.as_tensor(label, dtype=torch.long),
            sentences_tensor=sentence_ids,
            sentences_ids_mask=sentences_ids_mask,
            event_sent_mat=event_sent_mat,
        )

    @staticmethod
    def collate_fn(item_list: List[MCNCWithSentence]):
        return MCNCWithSentenceBatch(
            events_tensor=torch.stack([mcnc.events_tensor for mcnc in item_list]),
            events_mask_tensor=torch.stack(
                [mcnc.events_mask_tensor for mcnc in item_list]
            ),
            choices_tensor=torch.stack([mcnc.choices_tensor for mcnc in item_list]),
            label_tensor=torch.stack([mcnc.label_tensor for mcnc in item_list]),
            sentences_tensor=torch.stack([mcnc.sentences_tensor for mcnc in item_list]),
            sentences_ids_mask=torch.stack(
                [mcnc.sentences_ids_mask for mcnc in item_list]
            ),
            event_sent_mat=torch.stack([mcnc.event_sent_mat for mcnc in item_list]),
        )

    def __len__(self):
        return len(self._examples) * self._sample_count


class MCNCDataModule(pl.LightningDataModule):
    def prepare_data(self, *args, **kwargs):
        pass

    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        num_choices: int,
        argument_length: int,
        context_events_length: int,
        num_workers: int,
        pin_memory: bool = True,
        small: bool = False,
        data_segmentation_num = 4,
        data_current_segmentation=0,
        chain_indicator = []
    ):
        super().__init__()
        self._dataset_path = dataset_path
        self._batch_size = batch_size
        self._num_choices = num_choices
        self._argument_length = argument_length
        self._context_events_length = context_events_length
        self._chain_indicator = chain_indicator if chain_indicator else [True] * self._context_events_length
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._small = small
        self._data_segmentation_num = data_segmentation_num
        self._data_current_segmentation = data_current_segmentation

        self._train = None
        self._dev = None
        self._test = None

    def _new_dataset(self, data_file: str, use_part: bool = False):
        return MCNCDataset(
            os.path.join(self._dataset_path, data_file),
            num_chains=-1 if not self._small else 1000,
            num_choices=self._num_choices,
            argument_length=self._argument_length,
            context_events_length=self._context_events_length,
            use_part=use_part,
            data_segmentation_num=self._data_segmentation_num,
            data_current_segmentation=self._data_current_segmentation,
            isRandom=True,
            indicator=self._chain_indicator
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "test" or stage is None:
            self._test = self._test or self._new_dataset("test.json.gz")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return [
            DataLoader(
                dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
            )
            for dataset in [self._dev, self._test]
        ]

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self._test,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            shuffle=False
        )
