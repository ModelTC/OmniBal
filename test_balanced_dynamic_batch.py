from torch.utils.data import DataLoader
from dataset import BalancedDataset
from sampler import RandomSampler, GroupRandomBatchSamper, LengthGroupSampler

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


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
            if len(instance) == 1:
                vit.append(instance[0][0])
                llm.append(instance[0][1])
            else:
                temp = []
                for sp_instance in instance:
                    temp.append(sp_instance[1])
                max_llm = max(temp)
                sp_llm = []
                sp_vit = []
                # pad for llm
                for sp_instance in instance:
                    if sp_instance[1] < max_llm:
                        sp_llm.append(max_llm)
                    else:
                        sp_llm.append(sp_instance[1])
                    # sp vit num = 1, fake vit num = 1
                    if sp_instance[0] == 0:
                        sp_vit.append(1)
                    else:
                        sp_vit.append(sp_instance[0])
                vit.append(sum(sp_vit))
                llm.append(sum(sp_llm))

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


def test_for_vlm(multi_group=False, vit_bs=None, llm_len=None, thresh=None, sp_num=1, sampler='random'):
    json_files = ['./internvl_sft_1.2M.json']
    dp_size = 8
    itertime = 50
    micro_bs = 4
    if vit_bs is None:
        vit_bs, llm_len, thresh = search_arguments(json_files, dp_size, itertime, micro_bs)
    balanced_dataset = BalancedDataset(
        json_files, llm_len, thresh, itertime, vit_bs, fast_group=False, multi_group=multi_group, sp_num=sp_num)
    ave_bs = balanced_dataset.info_dict['ave_bs']
    pad_r, v_dist_r, l_dist_r = get_pad_dist_ratio(
        balanced_dataset, sampler_type=sampler, dp_size=dp_size, micro_bs=1)
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

    # test with sp
    logging.info("test for vlm for sp")
    test_for_vlm(vit_bs=9, llm_len=4096, thresh=3968, sp_num=4, sampler='group_random')
