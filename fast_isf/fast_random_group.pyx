import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nogil
cdef _random_group(list token_lengths, long seed, long vit_packed_length, long llm_packed_length):
    cdef list pack_groups
    cdef list each_group
    cdef long vit_token_length_sum
    cdef long llm_token_length_sum
    cdef long idx
    cdef long sample_id
    cdef list temp
    cdef np.ndarray[np.int64_t, ndim=1] index

    pack_groups = []
    np.random.seed(seed)
    index = np.random.permutation(len(token_lengths))
    vit_token_length_sum, llm_token_length_sum = 0, 0
    each_group = []

    for idx, sample_id in enumerate(index):
        temp = token_lengths[sample_id]
        if temp[1] > vit_packed_length or temp[2] > llm_packed_length:
            continue
        vit_token_length_sum += temp[1]
        llm_token_length_sum += temp[2]
        if vit_token_length_sum > vit_packed_length or llm_token_length_sum > llm_packed_length:
            pack_groups.append(each_group)
            vit_token_length_sum = temp[1]
            llm_token_length_sum = temp[2]
            each_group = [temp]
        else:
            each_group.append(temp)
        if idx == len(token_lengths) - 1:
            if len(each_group) > 0:
                pack_groups.append(each_group)
    return pack_groups


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nogil
cdef get_token_sum(list g):
    cdef long sum
    cdef list i
    sum = 0
    for i in g:
        sum += i[2]
    return sum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nogil
cdef get_vit_num(list g):
    cdef long vit_num
    cdef list _
    vit_num = 0
    for _ in g:
        vit_num += _[1]
    return vit_num


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nogil
cdef _process_random_groups(list patch_set, long seed, long vit_min, long vit_max, long llm_min, long llm_max, long iter_time):
    cdef list groups
    cdef list output
    cdef list need_process_groups
    cdef long i
    cdef list g

    groups = _random_group(patch_set, seed, vit_max, llm_max)
    if iter_time == 1:
        return groups
    output = []
    need_process_groups = []
    for i in range(iter_time - 1):
        need_process_groups = []
        for g in groups[:]:
            vit_num = get_vit_num(g)
            llm_num = get_token_sum(g)
            if vit_num == vit_min or llm_num >= llm_min:
                output.append(g)
            else:
                need_process_groups.extend(g)
        if len(need_process_groups) >= 0:
            groups = _random_group(need_process_groups, seed + i, vit_max, llm_max)
        else:
            break
        print(i, len(output), len(need_process_groups))
    if len(need_process_groups) > 0:
        output.extend(_random_group(need_process_groups, seed + i, vit_max, llm_max))
    return output

def random_group(token_lengths, seed=None, vit_packed_length=14, llm_packed_length=4096):
    return _random_group(token_lengths, seed, vit_packed_length, llm_packed_length)

def fast_process_random_groups(patch_set, seed, vit_min, vit_max, llm_min, llm_max, iter_time=3):
    return _process_random_groups(patch_set, seed, vit_min, vit_max, llm_min, llm_max, iter_time)