import random


class cached_property(object):
    """
    Descriptor (non-data) for building an attribute on-demand on first use.
    """
    def __init__(self, factory):
        """
        <factory> is called such: factory(instance) to build the attribute.
        """
        self._attr_name = factory.__name__
        self._factory = factory

    def __get__(self, instance, owner):
        # Build the attribute.
        attr = self._factory(instance)

        # Cache the value; hide ourselves.
        setattr(instance, self._attr_name, attr)
        return attr


def get_inds(expected_num, clu_nums, cid2clu, seq2ind):
    cur_len, cur_idx, query_cids, query_idx = 0, 0, [], []
    while cur_len < expected_num:
        cid, l = clu_nums[cur_idx % (len(clu_nums))]
        cur_idx += 1
        # check if this cluster has been selected
        if cid in query_cids:
            continue
        if random.random() > 0.5:
            for seq in cid2clu[cid]:
                # seq2ind: ensure it is in limited lengths
                if seq in seq2ind.keys():
                    query_idx.append(seq2ind[seq])
                    cur_len += 1

            query_cids.append(cid)
    return query_cids, query_idx


def get_num(N, valid_num=100):
    train_n, valid_n = int(0.9 * N), min(valid_num, int(0.05 * N))
    test_n = N - train_n - valid_n
    return train_n, valid_n, test_n


def get_full_inds(expected_num, clu_nums, cid2clu, full_seq2ind):
    cur_len, cur_idx, query_cids, query_idx = 0, 0, [], {}
    # build query_idx for each dataset
    for dataname in full_seq2ind.keys():
        if dataname not in query_idx.keys():
            query_idx[dataname] = []
    cur_idx_lst = list(range(len(clu_nums)))
    while cur_len < expected_num:
        cur_idx = random.choice(cur_idx_lst)
        cid, l = clu_nums[cur_idx]
        # check if this cluster has been selected
        if cid in query_cids:
            continue
        for seq in set(cid2clu[cid]):
            # seq2ind: ensure it is in limited lengths
            for dataname in full_seq2ind.keys():
                if seq in full_seq2ind[dataname].keys():
                    query_idx[dataname].append(full_seq2ind[dataname][seq])
                    cur_len += 1
        query_cids.append(cid)
        cur_idx_lst.remove(cur_idx)
    return query_cids, query_idx


def get_inds(expected_num, clu_nums, cid2clu, seq2ind):
    cur_len, query_cids, query_idx = 0, [], []
    cur_idx_lst = list(range(len(clu_nums)))
    while cur_len < expected_num:
        try:
            cur_idx = random.choice(cur_idx_lst)
            cid, l = clu_nums[cur_idx]
            # check if this cluster has been selected
            if cid in query_cids:
                continue

            # check if this cluster is too big
            pre = abs(expected_num - cur_len)
            aft = abs(cur_len + l - expected_num)
            if pre < aft:
                continue

            for seq in cid2clu[cid]:
                # seq2ind: ensure it is in limited lengths
                if seq in seq2ind.keys():
                    query_idx.append(seq2ind[seq])
                    cur_len += 1
            query_cids.append(cid)
            cur_idx_lst.remove(cur_idx)
        except:
            break
    return query_cids, query_idx
