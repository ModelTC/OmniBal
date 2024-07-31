from torch.utils.data import DataLoader
import torch
import math
import json
from torch.utils.data import Dataset
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


class BalancedDataset(Dataset):
    def __init__(self,
                 json_files,
                 max_seq_length=4096,
                 llm_thresh=4050,
                 iter_time=10,
                 vit_packed_length=9,
                 init=True,
                 fast_group=False,
                 vit_packed_thresh=None,
                 seed=1024,
                 multi_group=False):
        super(BalancedDataset, self).__init__()
        if vit_packed_thresh is None:
            vit_packed_thresh = vit_packed_length
        self.multi_group = multi_group
        self.token_lengths_list = self.load_wrap_meta(json_files)
        self.vit_packed_thresh = self._convert2list(vit_packed_thresh)
        self.seed = self._convert2list(seed)
        self.vit_packed_length = self._convert2list(vit_packed_length)
        self.llm_packed_length = self._convert2list(max_seq_length)
        self.llm_thresh = self._convert2list(llm_thresh)
        self.iter_time = self._convert2list(iter_time)
        self.pack_group = []
        self.group_lengths = []

        if init:
            for idx, token_lengths in enumerate(self.token_lengths_list):
                pack_group = []
                if fast_group:
                    from fast_isf import fast_random_group
                    pack_group = fast_random_group.fast_process_random_groups(token_lengths,
                                                                              self.seed[idx],
                                                                              self.vit_packed_thresh[idx],
                                                                              self.vit_packed_length[idx],
                                                                              self.llm_thresh[idx],
                                                                              self.llm_packed_length[idx],
                                                                              iter_time=self.iter_time[idx])
                else:
                    pack_group = self.iter_random_groups(token_lengths,
                                                         self.seed[idx],
                                                         self.vit_packed_thresh[idx],
                                                         self.vit_packed_length[idx],
                                                         self.llm_thresh[idx],
                                                         self.llm_packed_length[idx],
                                                         iter_time=self.iter_time[idx])
                self.group_lengths.append(len(pack_group))
                self.pack_group.extend(pack_group)
            print(json.dumps(self.collect_packed_info(self.pack_group), indent=4, sort_keys=True))

            lengths = []
            for g in self.pack_group:
                temp = 0
                for item in g:
                    temp += item[2]
                lengths.append(temp)
            self.lengths = lengths

    def _convert2list(self, item):
        if self.multi_group:
            target_len = len(self.token_lengths_list)
        else:
            target_len = 1
        if not isinstance(item, list):
            item = [item] * target_len
        return item

    def get_display_bin_num(self, llm_num, display_bin_size):
        mod, div = llm_num // display_bin_size, llm_num % display_bin_size
        if div > 0:
            mod += 1
        return mod * display_bin_size

    def collect_origin_info(self, patch_set):
        info = {}
        llm_info = {}
        all_vit_length = []
        all_llm_length = []
        act_length = []
        for item in patch_set:
            vit_num, llm_num = item[1], item[2]
            if vit_num not in info:
                info[vit_num] = 0
            info[vit_num] += 1

            llm_num_bin = self.get_display_bin_num(llm_num, 256)
            if llm_num_bin not in llm_info:
                llm_info[llm_num_bin] = 0
            llm_info[llm_num_bin] += 1
            all_vit_length.append(vit_num)
            all_llm_length.append(llm_num_bin)
            act_length.append(llm_num)
        info['vit_length_mean'] = np.mean(all_vit_length)
        info['vit_length_var'] = np.var(all_vit_length)
        info['vit_length_std'] = np.std(all_vit_length)

        info['llm_length_mean'] = np.mean(all_llm_length)
        info['llm_length_var'] = np.var(all_llm_length)
        info['llm_length_std'] = np.std(all_llm_length)

        print("origin vit info")
        print(json.dumps(info, indent=4))
        print("origin llm info")
        print(json.dumps(llm_info, indent=4, sort_keys=True))
        self.origin_info = info

    def load_wrap_meta(self, json_files):
        token_lengths_list = []
        idx = 0
        for wrap_file in json_files:
            token_lengths = []
            with open(wrap_file) as f:
                data_info = json.load(f)
            for data_name in data_info.keys():
                if "token_lengths" in data_info[data_name]:
                    token_length_path = data_info[data_name]['token_lengths']
                    with open(token_length_path, "r") as f:
                        token_length = json.load(f)
                    for item in token_length:
                        token_lengths.append(
                            [idx, item['vit_num'], item['token_num']])
                        idx += 1
            print(f"{wrap_file} data info")
            self.collect_origin_info(token_lengths)
            token_lengths_list.append(token_lengths)
        if not self.multi_group:
            merge_token_lengths = []
            for item in token_lengths_list:
                merge_token_lengths.extend(item)
            print(f"merged data info")
            self.collect_origin_info(merge_token_lengths)
            token_lengths_list = [merge_token_lengths]
        return token_lengths_list

    def _random_groups(self, token_lengths, seed=None, vit_max=None, llm_max=None):
        """
        tokens_length: [(idx, vit_img_num, llm_token_len)]
        """
        rng = np.random.RandomState(seed)
        index = list(range(len(token_lengths)))
        rng.shuffle(index)

        pack_groups = []
        vit_token_length_sum, llm_token_length_sum = 0, 0
        each_group = []
        for idx, sample_id in enumerate(index):
            vit_sample_length, llm_sample_length = token_lengths[
                sample_id][1], token_lengths[sample_id][2]
            if vit_sample_length > vit_max or llm_sample_length > llm_max:
                continue
            vit_token_length_sum += vit_sample_length
            llm_token_length_sum += llm_sample_length
            if vit_token_length_sum > vit_max or llm_token_length_sum > llm_max:
                pack_groups.append(each_group)
                vit_token_length_sum = vit_sample_length
                llm_token_length_sum = llm_sample_length
                each_group = [token_lengths[sample_id]]
            else:
                each_group.append(token_lengths[sample_id])
            if idx == len(token_lengths) - 1:
                if len(each_group) > 0:
                    pack_groups.append(each_group)
        return pack_groups

    def iter_random_groups(self,
                           groups,
                           seed=None,
                           vit_thresh=None,
                           vit_max=None,
                           llm_thresh=None,
                           llm_max=None,
                           iter_time=300):
        groups = self._random_groups(groups, seed=seed, vit_max=vit_max, llm_max=llm_max)
        if iter_time == 1:
            return groups
        output = []
        for i in range(iter_time - 1):
            print(f"iter_random_groups {i} / {iter_time - 1}")
            need_process_groups = []
            for g in groups:
                vit_num = get_vit_num(g)
                llm_num = get_token_sum(g)
                if vit_num == vit_thresh or llm_num >= llm_thresh:
                    output.append(g)
                else:
                    need_process_groups.extend(g)
            if len(need_process_groups) >= 0:
                groups = self._random_groups(need_process_groups, seed + i, vit_max, llm_max)
            else:
                break
            print(i, len(groups), len(need_process_groups))
        if len(need_process_groups) > 0:
            output.extend(self._random_groups(need_process_groups, seed + i, vit_max, llm_max))
        return output

    def collect_packed_info(self, packed_groups):
        info_dict = {}
        info_dict['vit_num_info'] = {}
        vit_num_min = 10000000
        vit_num_max = 0
        llm_num_min = 10000000
        llm_num_max = 0
        vit_ave_num = 0
        llm_ave_num = 0
        sample_num = 0
        for group in packed_groups:
            vit_num = get_vit_num(group)
            llm_num = get_token_sum(group)
            if vit_num not in info_dict['vit_num_info']:
                info_dict['vit_num_info'][vit_num] = 0
            info_dict['vit_num_info'][vit_num] += 1
            vit_num_min = min(vit_num_min, vit_num)
            vit_num_max = max(vit_num_max, vit_num)
            llm_num_min = min(llm_num_min, llm_num)
            llm_num_max = max(llm_num_max, llm_num)
            vit_ave_num += vit_num
            llm_ave_num += llm_num
            sample_num += len(group)
        info_dict['vit_num_min'] = vit_num_min
        info_dict['vit_num_max'] = vit_num_max
        info_dict['llm_num_max'] = llm_num_max
        info_dict['llm_num_min'] = llm_num_min
        info_dict['vit_ave_num'] = vit_ave_num / float(len(packed_groups))
        info_dict['llm_ave_num'] = llm_ave_num / float(len(packed_groups))
        info_dict['sample_num'] = sample_num
        info_dict['packed_group_num'] = len(packed_groups)
        info_dict['ave_bs'] = sample_num / float(len(packed_groups))
        self.info_dict = info_dict
        return info_dict

    def __getitem__(self, idx):
        groups = self.pack_group[idx]
        vit_num = 0
        llm_num = 0
        for item in groups:
            vit_num += item[1]
            llm_num += item[2]
        return (llm_num, vit_num)

    def __len__(self):
        """
        Returns dataset length
        """
        return len(self.pack_group)


class BaseDataset(Dataset):
    def __init__(self,
                 json_files):
        super(BaseDataset, self).__init__()
        self.load_wrap_meta(json_files)

    def load_wrap_meta(self, json_files):
        lengths = []
        vit_nums = []
        for wrap_file in json_files:
            with open(wrap_file) as f:
                data_info = json.load(f)
            for data_name in data_info.keys():
                if "token_lengths" in data_info[data_name]:
                    token_length_path = data_info[data_name]['token_lengths']
                    with open(token_length_path, "r") as f:
                        token_length = json.load(f)
                    for item in token_length:
                        lengths.append(item['token_num'])
                        vit_nums.append(item['vit_num'])
        self.lengths = lengths
        self.vit_nums = vit_nums

    def __getitem__(self, idx):
        return (self.lengths[idx], self.vit_nums[idx])

    def __len__(self):
        """
        Returns dataset length
        """
        return len(self.lengths)


class BatchCollector(object):
    def __init__(self,
                 ignore_idx=-100,
                 max_seq_length=2048):
        self.ignore_idx = ignore_idx
        self.max_seq_length = max_seq_length

    def __call__(self, instances):
        vit = []
        llm = []
        for instance in instances:
            vit.append(instance[1])
            llm.append(instance[0])
        max_length = max(llm)
        pad_list = []
        seq_len = []
        vit_num = []
        for idx, instance in enumerate(llm):
            pad_list.append(max_length - instance)
            seq_len.append(instance)
            vit_num.append(vit[idx])

        return dict(
            max_length=max_length,
            pad_list=pad_list,
            seq_len=seq_len,
            vit_num=vit_num
        )


def get_pad_dist_ratio(dataset, sampler_type='random', dp_size=16, micro_bs=4):
    random_pad_token = 0
    random_all_token = 0
    random_act_token = 0
    dp_output = []
    vit_output = []
    iteration = 0
    for rank in range(dp_size):
        print("random rank", rank)
        if sampler_type == 'random':
            batch_sampler = RandomSampler(
                len(dataset), 0, micro_bs, rank, dp_size)
        if sampler_type == 'length_group':
            batch_sampler = LengthGroupSampler(
                len(dataset), 0, micro_bs, rank, dp_size, lengths=dataset.lengths)
        if sampler_type == 'group_random':
            batch_sampler = GroupRandomBatchSamper(dataset.group_lengths, 0, micro_bs, rank, dp_size, lengths=dataset.lengths)
        batch_collator = BatchCollector()
        dataloader = DataLoader(dataset,
                                batch_sampler=batch_sampler,
                                num_workers=0,
                                collate_fn=batch_collator)
        temp_llm = []
        temp_vit = []
        for data in dataloader:
            iteration += 1
            random_pad_token += sum(data['pad_list'])
            random_all_token += data['max_length'] * micro_bs
            random_act_token += sum(data['seq_len'])
            temp_llm.append(data['max_length'])
            temp_vit.append(sum(data['vit_num']))
        dp_output.append(temp_llm)
        vit_output.append(temp_vit)
    device_group = []
    device_vit_group = []

    for idx in range(len(temp_llm)):
        temp_data = []
        temp_vit_data = []
        for i in range(dp_size):
            temp_data.append(dp_output[i][idx])
            temp_vit_data.append(vit_output[i][idx])
        llm_max_v = max(temp_data)
        llm_waste_v = 0
        for item in temp_data:
            llm_waste_v += (llm_max_v - item)
        llm_waste_ratio = llm_waste_v / float(llm_max_v * dp_size)
        vit_max_v = max(temp_vit_data)
        vit_waste_v = 0
        for item in temp_vit_data:
            vit_waste_v += (vit_max_v - item)
        vit_waste_ratio = vit_waste_v / max(float(vit_max_v * dp_size), 1)
        device_group.append(llm_waste_ratio)
        device_vit_group.append(vit_waste_ratio)
    pad_ratio = random_pad_token / float(random_all_token)
    vit_dist_ratio = sum(device_vit_group) / len(device_vit_group)
    llm_dist_ratio = sum(device_group) / len(device_group)
    print("pad ratio", pad_ratio)
    print('dist ratio llm', llm_dist_ratio)
    print('dist ratio vit', vit_dist_ratio)
    return pad_ratio, vit_dist_ratio, llm_dist_ratio


def get_init_llm_vit_len(json_files):
    o_dataset = BalancedDataset(json_files=json_files, init=False)
    o_vit_bs_mean = o_dataset.origin_info['vit_length_mean']
    o_llm_len_mean = o_dataset.origin_info['llm_length_mean']
    init_ave_bs_llm_len = ((o_llm_len_mean // max(o_vit_bs_mean, 1) + 1) // 16) * 16

    init_llm_len = 4096
    init_vit_bs = init_llm_len // init_ave_bs_llm_len
    return init_vit_bs, init_llm_len, init_ave_bs_llm_len


def search_arguments(json_files, dp_size=8, itertime=50, micro_bs=4, multi_group=False):
    best_r = 100
    result_list = []
    vit_bs, llm_len, init_ave_bs_llm_len = get_init_llm_vit_len(json_files)
    step = 1
    max_ave_bs = 0

    while True:
        print(vit_bs, llm_len)
        threshs = [llm_len - 128]
        for thresh in threshs:
            balanced_dataset = BalancedDataset(
                json_files, llm_len, thresh, itertime, vit_bs, fast_group=False, multi_group=multi_group)
            ave_bs = balanced_dataset.info_dict['ave_bs']
            max_ave_bs = max(ave_bs, max_ave_bs)
            pad_r, v_dist_r, l_dist_r = get_pad_dist_ratio(
                balanced_dataset, sampler_type='random', dp_size=dp_size, micro_bs=1)
            if sum([pad_r, v_dist_r, l_dist_r]) < best_r:
                best_r = sum([pad_r, v_dist_r, l_dist_r])
            if ave_bs >= micro_bs and ave_bs <= micro_bs + 1:
                result_list.append(
                    [sum([pad_r, v_dist_r, l_dist_r]), v_dist_r, l_dist_r, vit_bs, llm_len, thresh, ave_bs])
        if max_ave_bs > micro_bs + 1:
            if len(result_list) > 0:
                break
            else:
                vit_bs -= step
                llm_len -= step * init_ave_bs_llm_len
        else:
            vit_bs += step
            llm_len += step * init_ave_bs_llm_len
    print("ratio, v_dist_ratio, l_dist_ratio, vit_bs, llm_len, llm_thresh, ave_bs")
    result_list = sorted(result_list)
    print(result_list)
    return result_list[0][3:6]


def test_for_vlm(multi_group=False, vit_bs=None, llm_len=None, thresh=None):
    json_files = ['./internvl_sft_1.2M.json']
    dp_size = 8
    itertime = 50
    micro_bs = 4
    if vit_bs is None:
        vit_bs, llm_len, thresh = search_arguments(json_files, dp_size, itertime, micro_bs)
    balanced_dataset = BalancedDataset(
        json_files, llm_len, thresh, itertime, vit_bs, fast_group=False, multi_group=multi_group)
    ave_bs = balanced_dataset.info_dict['ave_bs']
    pad_r, v_dist_r, l_dist_r = get_pad_dist_ratio(
        balanced_dataset, sampler_type='random', dp_size=dp_size, micro_bs=1)
    print(pad_r, v_dist_r, l_dist_r, ave_bs)


def test_for_vlm_llm(multi_group=False, vit_bs=None, llm_len=None, thresh=None):
    json_files = ['./internvl_sft_1.2M.json', './pure_llm_1.0M.json']
    dp_size = 8
    itertime = 50
    micro_bs = 4
    if vit_bs is None:
        vit_bs_list = []
        llm_len_list = []
        thresh_list = []
        if multi_group:
            for json_file in json_files:
                vit_bs, llm_len, thresh = search_arguments([json_file], dp_size, itertime, micro_bs)
                vit_bs_list.append(vit_bs)
                llm_len_list.append(llm_len)
                thresh_list.append(thresh)
            sampler_type = 'group_random'
        else:
            vit_bs_list, llm_len_list, thresh_list = search_arguments(json_files, dp_size, itertime, micro_bs)
            sampler_type = 'random'
    else:
        vit_bs_list = vit_bs
        llm_len_list = llm_len
        thresh_list = thresh
        if multi_group:
            sampler_type = 'group_random'
        else:
            sampler_type = 'random'

    balanced_dataset = BalancedDataset(
        json_files, llm_len_list, thresh_list, itertime, vit_bs_list, fast_group=False, multi_group=multi_group)
    ave_bs = balanced_dataset.info_dict['ave_bs']
    pad_r, v_dist_r, l_dist_r = get_pad_dist_ratio(
        balanced_dataset, sampler_type=sampler_type, dp_size=dp_size, micro_bs=1)
    print(pad_r, v_dist_r, l_dist_r, ave_bs)


if __name__ == "__main__":

    # test with search arguments
    logging.info("test for vlm with search arguments")
    test_for_vlm()
    # test with fix arguments
    logging.info("test for vlm with fix arguments")
    test_for_vlm(vit_bs=9, llm_len=4096, thresh=3968)

    # test for vlm + llm with search arguments, without multi group
    # test_for_vlm_llm(multi_group=False)

    logging.info("test for vlm + llm with fix arguments, without multi group")
    # test for vlm + llm with fix arguments , without multi group
    test_for_vlm_llm(multi_group=False, vit_bs=9, llm_len=4096, thresh=3968)

    # test for vlm + llm with search arguments , with multi group
    # test_for_vlm_llm(multi_group=True)

    # test for vlm + llm with search arguments , with multi group
    logging.info("test for vlm + llm with fix arguments, with multi group")
    test_for_vlm_llm(multi_group=True, vit_bs=[9, 9], llm_len=[4096, 4096], thresh=[3968, 3968])
