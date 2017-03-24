#!/usr/bin/env python3
tempdir = "/media/windows2/tmp/"


def post_GIZA_from(line: str):
    line = line.split()
    return tuple(zip(line, range(1, 1 + len(line))))


def post_GIZA_to(line: str):
    line = line.split()
    out_data = {}
    inner_line_type = 2  # 0=word, 1=opened positions brackets, 2=closed positions brackets
    last_idx = -1
    last_word = ""
    cur_idx = -1
    ignore = False
    for word in line:
        word = word.strip()
        if word == "NULL":
            ignore = True
            continue
        if ignore:
            if word == "})":
                ignore = False
                inner_line_type = 2
            continue

        if inner_line_type is 2:
            last_word = word
            inner_line_type = 0
        elif inner_line_type is 0 and word == "({":
            inner_line_type = 1
            cur_idx = -1
        elif inner_line_type is 1:
            if word == "})":
                inner_line_type = 2
                if cur_idx is -1:
                    if (last_idx + 1) not in out_data.keys():
                        out_data[last_idx + 1] = []
                    out_data[last_idx + 1].append(last_word)
                    last_idx += 1
                else:
                    last_idx = cur_idx
            else:
                cur_idx = int(word)
                if cur_idx not in out_data.keys():
                    out_data[cur_idx] = []
                out_data[cur_idx].append(last_word)
                # print(out_data)
    return tuple(zip(map(lambda x: " ".join(x), out_data.values()), out_data.keys()))

from pympler import asizeof
def post_GIZA():
    line_type = None  # 0=comment, 1=from, 2=to
    out_from = []
    out_to = []
    lidx = 0
    line = ""
    with open(tempdir+"result") as f:
        while True:  # Parse GIZA result file *A3* using a simple state machine
            line = f.readline()
            if not line:
                break
            lidx += 1
            if lidx > 10:
                break
            if lidx % 100000 == 0:
                print(lidx / 10000000 * 100, "%")
            # if lidx % 1000000 == 0:
            #     print(lidx)
            #     break
            # if lidx > 100000:
            #     break
            line = line.strip()
            if line.startswith("#"):  # Comment line, ignore + reset state machine
                line_type = 0
            elif line_type is 0:  # Line from FROM language. Is always in order
                out_from.append(post_GIZA_from(line))
                line_type = 1
            elif line_type is 1:  # Line from TO language. Is aligned in respect to FROM language
                out_to.append(post_GIZA_to(line))
                line_type = 2
    return out_from, out_to

if __name__ == "__main__":
    f, t = post_GIZA()
    for i in range(len(f)):
        if len(f[i]) is not len(t[i]):
            print("mismatch in line {}, length {} != {}".format(i, len(f[i]), len(t[i])))
            print(f[i])
            print(t[i])
    input()