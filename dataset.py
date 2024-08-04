import json
from torch.utils.data import Dataset
import numpy as np
import math

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


def get_sp_groups(pack_group, sp_num):

    # padding to sp_num
    align_length = int(math.ceil(len(pack_group) * 1.0 / sp_num)) * sp_num
    padding_size = align_length - len(pack_group)
    if padding_size <= len(pack_group):
        pack_group += pack_group[:padding_size]
    else:
        pack_group += (pack_group * math.ceil(padding_size / len(pack_group)))[:padding_size]

    lengths = []
    for idx, g in enumerate(pack_group):
        temp = 0
        for item in g:
            temp += item[2]
        lengths.append((temp , idx))
    lengths = sorted(lengths)

    sp_groups = []
    target_len = align_length // sp_num
    for i in range(target_len):
        temp = []
        for j in range(sp_num):
            g_idx = lengths[i * sp_num + j][1]
            temp.append(pack_group[g_idx])
        sp_groups.append(temp)
    return sp_groups


def get_sp_dist_pad_ratio(sp_groups):
    sp_vit_dist_ratio = []
    sp_llm_dist_ratio = []
    act_token = 0
    all_token = 0
    pad_token = 0
    sp_num = len(sp_groups[0])
    for groups in sp_groups:
        vit_bs = []
        llm_len = []
        for g in groups:
            vit_bs_sum = 0
            llm_len_sum = 0
            for item in g:
                vit_bs_sum += item[1]
                llm_len_sum += item[2]
            vit_bs.append(vit_bs_sum)
            llm_len.append(llm_len_sum)
        max_vit_bs = max(vit_bs)
        max_llm_len = max(llm_len)
        all_token += (max_llm_len * sp_num)
        for item in vit_bs:
            waste_vit_ratio = (max_vit_bs - item) / max(max_vit_bs, 1)
            sp_vit_dist_ratio.append(waste_vit_ratio)
        for item in llm_len:
            waste_llm_ratio = (max_llm_len - item) / max(max_llm_len, 1)
            pad_token += (max_llm_len - item)
            act_token += item
            sp_llm_dist_ratio.append(waste_llm_ratio)
    ave_sp_vit_dist_ratio = np.mean(sp_vit_dist_ratio)
    ave_sp_llm_dist_ratio = np.mean(sp_llm_dist_ratio)
    ave_sp_llm_pad_ratio = pad_token / all_token
    print(f"ave_sp_vit_dist_ratio: {ave_sp_vit_dist_ratio}")
    print(f"ave_sp_llm_dist_ratio: {ave_sp_llm_dist_ratio}")
    print(f"ave_sp_llm_pad_ratio: {ave_sp_llm_pad_ratio}")


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
                 multi_group=False,
                 sp_num=1):
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
        self.sp_num = sp_num
        self.sp_groups = []

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
                if self.sp_num > 1:
                    self.sp_groups.extend(get_sp_groups(pack_group, self.sp_num))
                    e_pack_group = []
                    for item in self.sp_groups:
                        temp = []
                        for j in item:
                            temp.extend(j)
                        e_pack_group.append(temp)
                    pack_group = e_pack_group
                self.group_lengths.append(len(pack_group))
                self.pack_group.extend(pack_group)
            if self.sp_num > 1:
                get_sp_dist_pad_ratio(self.sp_groups)
            print(json.dumps(self.collect_packed_info(self.pack_group), indent=4, sort_keys=True))

            lengths = []
            if self.sp_num > 1:
                for sp_g in self.sp_groups:
                    temp = 0
                    for g in sp_g:
                        for item in g:
                            temp += item[2]
                    lengths.append(temp)
            else:
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
        if self.sp_num > 1:
            sp_groups = self.sp_groups[idx]
            sp_out = []
            for groups in sp_groups:
                vit_num = 0
                llm_num = 0
                for item in groups:
                    vit_num += item[1]
                    llm_num += item[2]
                sp_out.append((vit_num, llm_num))
            return sp_out
        else:
            groups = self.pack_group[idx]
            vit_num = 0
            llm_num = 0
            for item in groups:
                vit_num += item[1]
                llm_num += item[2]
        return [(vit_num, llm_num)]

    def __len__(self):
        """
        Returns dataset length
        """
        if self.sp_num > 1:
            return len(self.sp_groups)
        else:
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
