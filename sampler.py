import torch
import math
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def get_token_sum(g):
    sum = 0
    for i in g:
        sum += i[2]
    return sum


def get_vit_num(g):
    vit_num = 0
    for _ in g:
        vit_num += _[1]
    return vit_num


class RandomSampler:
    def __init__(self,
                 total_samples,
                 consumed_samples,
                 micro_batch_size,
                 data_parallel_rank,
                 data_parallel_size):
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        g = torch.Generator()
        g.manual_seed(self.epoch)
        random_idx = torch.randperm(bucket_size, generator=g).tolist()
        idx_range = [start_idx + x for x in random_idx[bucket_offset:]]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []


class LengthGroupSampler:
    def __init__(self,
                 total_samples,
                 consumed_samples,
                 micro_batch_size,
                 data_parallel_rank,
                 data_parallel_size,
                 lengths=None,
                 seed=233):

        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

        self.lengths = lengths
        self.num_samples = math.ceil(
            len(self.lengths) / (data_parallel_size * self.micro_batch_size))
        self.total_size = self.num_samples * data_parallel_size * self.micro_batch_size
        self.seed = seed
        self.buffer = None

    # copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L38
    def split_to_even_chunks(self, indices, lengths, num_chunks):
        """
        Split a list of indices into `chunks` chunks of roughly equal lengths.
        """
        if len(indices) % num_chunks != 0:
            return [indices[i::num_chunks] for i in range(num_chunks)]

        num_indices_per_chunk = len(indices) // num_chunks

        chunks = [[] for _ in range(num_chunks)]
        chunks_lengths = [0 for _ in range(num_chunks)]
        for index in indices:
            shortest_chunk = chunks_lengths.index(min(chunks_lengths))
            chunks[shortest_chunk].append(index)
            chunks_lengths[shortest_chunk] += lengths[index]
            if len(chunks[shortest_chunk]) == num_indices_per_chunk:
                chunks_lengths[shortest_chunk] = float('inf')

        return chunks

    # copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L88
    def get_length_grouped_indices(self, lengths, batch_size, world_size, generator=None, merge=True):
        # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
        indices = torch.randperm(len(lengths), generator=generator)
        megabatch_size = world_size * batch_size
        megabatches = [indices[i: i + megabatch_size].tolist()
                       for i in range(0, len(lengths), megabatch_size)]
        megabatches = [sorted(megabatch, key=lambda i: lengths[i],
                              reverse=True) for megabatch in megabatches]
        megabatches = [self.split_to_even_chunks(
            megabatch, lengths, world_size) for megabatch in megabatches]

        return [i for megabatch in megabatches for batch in megabatch for i in batch]

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        if self.buffer is None:
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = self.get_length_grouped_indices(
                self.lengths, self.micro_batch_size, self.data_parallel_size, generator=g)
            self.buffer = indices
        else:
            indices = self.buffer
        active_indices = indices + indices[: (self.total_size - len(indices))]
        active_indices_split = []
        start = self.data_parallel_rank * self.micro_batch_size
        stop = (self.data_parallel_rank + 1) * self.micro_batch_size
        while start < len(active_indices):
            if stop <= len(active_indices):
                active_indices_split += active_indices[start:stop]
            else:
                active_indices_split += active_indices[start:]
            start += self.data_parallel_size * self.micro_batch_size
            stop += self.data_parallel_size * self.micro_batch_size

        current_epoch_samples = self.consumed_samples % len(active_indices)
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        bucket_offset = current_epoch_samples // self.data_parallel_size

        idx_range = active_indices_split[bucket_offset:]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []


class GroupRandomBatchSamper:
    def __init__(self,
                 total_samples,
                 consumed_samples,
                 micro_batch_size,
                 data_parallel_rank,
                 data_parallel_size,
                 lengths=None):
        if isinstance(total_samples, list):
            self.total_samples_list = total_samples
            self.total_samples = sum(self.total_samples_list)
        else:
            self.total_samples_list = [total_samples]
            self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size
        self.consumed_iter = self.consumed_samples // self.micro_batch_times_data_parallel_size
        self.epoch = 0
        self.text_lengths = lengths

        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)
        self.generate_indices()
        self.indices = self.indices[self.consumed_samples:]
        self.build_balanced_indices()

    def build_balanced_indices(self):
        bin_size_group = self.build_bin_size_group()
        sorted_keys = sorted(list(bin_size_group.keys()))
        new_indices = []
        for k in sorted_keys:
            new_indices.extend(bin_size_group[k])
        return new_indices

    def build_bin_size_group(self):
        bin_size_group = {}
        bin_size = 64
        for idx in self.indices:
            length = self.text_lengths[idx]
            if (length // bin_size) not in bin_size_group:
                bin_size_group[(length // bin_size)] = []
            bin_size_group[(length // bin_size)].append(idx)

        return bin_size_group

    def __len__(self):
        return self.total_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.generate_indices()

    def _generate_indices(self, samples, accu_length):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        random_idx = torch.randperm(samples, generator=g).tolist()
        # random_idx = list(range(samples))
        random_idx = [item + accu_length for item in random_idx]

        align_length = int(math.ceil(samples * 1.0 / (self.micro_batch_size * self.data_parallel_size))) * self.micro_batch_size * self.data_parallel_size
        padding_size = align_length - samples
        if padding_size <= len(random_idx):
            random_idx += random_idx[:padding_size]
        else:
            random_idx += (random_idx * math.ceil(padding_size / len(random_idx)))[:padding_size]
        assert len(random_idx) == align_length
        t_size = len(random_idx) // self.micro_batch_times_data_parallel_size
        indices_group = []
        for i in range(t_size):
            indices_group.append(random_idx[i * self.micro_batch_times_data_parallel_size:(i + 1) * self.micro_batch_times_data_parallel_size])
        return indices_group

    def generate_indices(self):
        self.indices_group = []
        accu_length = [0]
        self.indices = []
        for i in self.total_samples_list:
            accu_length.append(i)
        for idx, samples in enumerate(self.total_samples_list):
            self.indices_group.extend(self._generate_indices(samples, accu_length=accu_length[idx]))

        g = torch.Generator()
        g.manual_seed(self.epoch + 10)
        random_idx = torch.randperm(len(self.indices_group), generator=g).tolist()
        for idx in random_idx:
            self.indices.extend(self.indices_group[idx])

        self.total_length = len(self.indices)

    def __iter__(self):
        batch = []
        self.epoch = self.consumed_samples // self.total_length
        if self.consumed_samples % self.total_length == 0 and self.consumed_samples > 0:
            self.generate_indices()
        for idx in self.indices[self.data_parallel_rank::self.data_parallel_size]:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []
