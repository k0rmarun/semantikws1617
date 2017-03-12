import numpy as np


def prepare_biskip(text_splitter):
    """
    Generates forward and reverse dictionary for BiSkip algorithm
    :param text_splitter: function function that yields word aligned sentences

    :return forw_dict: dict lookup table lang1 => idx1
    :returns rev_dict: dict lookup table idx2 => lang2
    :returns biskip_batch: function function to generate BiSkip batch data
    """
    forw_dict = {}
    inv_rev_dict = {} # inverse reverse dict: lang2 => idx2

    # sentence = [("one":"ein"),("apple", "Apfel"), ("a day", "den Tag"), ...]
    for sentence in text_splitter():
        for word in sentence:
            if word[0] not in forw_dict.keys():
                forw_dict[word[0]] = len(forw_dict)
            if word[1] not in inv_rev_dict.keys():
                inv_rev_dict[word[1]] = len(inv_rev_dict)
    rev_dict = dict(zip(inv_rev_dict.values(), inv_rev_dict.keys()))

    def biskip_batch(batch_size):
        """
        Generate the data for a single batch in BiSkip algorithm
        :param batch_size: int number of entries in a BiSKip batch
        """
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        # How many words left and right
        window = 4

        # How many elements in batch. Yield and reset at :batch_size
        batchidx = 0
        while True:
            for sentence in text_splitter():
                l = len(sentence)
                for i in range(l):
                    forw_word = forw_dict[sentence[i][0]]
                    # Get words left/right inside sentence, skip where idx1 = idx2
                    for j in range(max(0, i-window), min(l, i+window+1)):
                        if i == j:
                            continue
                        back_word = inv_rev_dict[sentence[j][1]]

                        batch[batchidx] = forw_word
                        labels[batchidx, 0] = back_word

                        batchidx += 1
                        # yield and reset as soon as we reach batch_size entries
                        if batchidx == batch_size:
                            yield batch, labels
                            batchidx = 0

    return forw_dict, rev_dict, biskip_batch

if __name__ == "__main__":
    from pprint import pprint

    def test():
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i in range(0, 52, 4):
            yield list(zip(chars[i:i+4], [i, i+1, i+2, i+3]))


    d, rd, func = prepare_biskip(test)
    pprint(d)
    pprint(rd)
    el = func(10)
    for i in range(10):
        pprint(next(el))
