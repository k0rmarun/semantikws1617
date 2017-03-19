#!/usr/bin/env python3
import re
import os
import sys

import time
import xml.etree.ElementTree as ET
import gzip
import subprocess
from pprint import pprint


corpus_name = "EUbookshop"
corpus_path = "/home/niels/Downloads/"+corpus_name+"/xml/"
corpus_en = corpus_path + "en/"
corpus_de = corpus_path + "de/"
outpath = "/media/windows2/"+corpus_name+"/result/"
tempdir = "/media/windows2/tmp/"
giza_skript = "/home/niels/PycharmProjects/semantikws1617/GIZA.sh"


def prepareSentence(sentence)->str:
    # get best name
    def mapper(elem):
        return elem.text

    # Strip special chars
    def filt(elem):
        if len(elem) is 1 and elem in ",.;:-_#'+*~!^°\"§$%&/()=?`{[]}\\¸<>|":
            return False
        return True
    return " ".join(list(filter(filt, map(mapper, sentence.findall(".//w")))))+"."


def readGZIPXML(filename: str)->dict:
    with gzip.open(filename) as f:
        content = f.read()
    root = ET.fromstring(content)
    data = {}
    for sentence in root.findall(".//s"):
        if "id" not in sentence.attrib:
            continue
        idx = sentence.attrib["id"]
        words = prepareSentence(sentence)
        data[idx] = words
    return data


def exec(name: str, arg: str):
    print("Running", name)
    p = subprocess.Popen(arg, shell=True, cwd=tempdir)
    p.wait()


def clean_GIZA():
    exec("cleanup giza", "rm -rf "+tempdir+"/*")


def GIZA(fromDoc: str, toDoc: str, links: list):
    try:
        from_data = readGZIPXML(corpus_path + fromDoc)
        to_data = readGZIPXML(corpus_path + toDoc)

        from_prep = []
        to_prep = []

        for link in links:
            from_links = link[0]
            to_links = link[1]

            for flink in from_links:
                if flink not in from_data.keys():
                    continue
                for tlink in to_links:
                    if tlink not in to_data.keys():
                        continue
                    from_prep.append(from_data[flink])
                    to_prep.append(to_data[tlink])
    except ET.ParseError as err:
        print("Caught Exeption", err)
    else:
        with open(tempdir+"from", "a") as f:
            f.write("\r\n".join(from_prep))
        with open(tempdir+"to", "a") as f:
            f.write("\r\n".join(to_prep))


def format_time(delta: int)->str:
    h = delta // 3600
    delta -= (h*3600)

    m = delta // 60
    s = delta - m*60
    return "{}:{}:{}".format(int(h), int(m), int(s))


def MetaGIZA(metafile: str):
    grpRegex = re.compile(r"<linkGrp (.*?)>")
    fromDoc = re.compile(r"fromDoc=\"(.*?)\"")
    toDoc = re.compile(r"toDoc=\"(.*?)\"")
    grpEndRegex = re.compile(r"</linkGrp>")
    lnkRegex = re.compile(r"<link (.*?)/>")
    xtarRegex = re.compile(r"xtargets=\"(.*?)\"")

    data = []

    def parse_target(elem)->tuple:
        elems = elem.split(";")
        return elems[0].split(" "), elems[1].split(" ")

    print("calculating line numbers")
    num_lines = 0
    with gzip.open(metafile) as f:
        for i in f:
            num_lines += 1

    print("starting giza pre processing")
    start_time = time.time()

    with gzip.open(metafile) as f:
        from_doc = ""
        to_doc = ""
        links = []
        i = 0
        active = False
        for line in f:
            i += 1
            #if i > 10000:
            #    break
            if i % 1000 == 0:
                perc = i/num_lines
                delta = time.time()-start_time
                print("Completed {}% in {}. ETA: {}".format(int(perc*100), format_time(delta), format_time(delta*(1/perc))))

            line = line.decode("utf-8")
            match = grpRegex.search(line)
            if match:
                if active:  # Workaround for broken XML meta file
                    print("tried to open a linkGroup without closing previous on line", i)
                    GIZA(from_doc, to_doc, links)
                    from_doc = ""
                    to_doc = ""
                    links = []

                inner = match.groups()[0]

                try:
                    from_doc = fromDoc.search(inner).groups()[0]
                    to_doc = toDoc.search(inner).groups()[0]
                except AttributeError as err:
                    print(err)
                    active = False
                else:
                    active = True

            match = lnkRegex.search(line)
            if match:
                if not active:
                    continue
                inner = match.groups()[0]
                links.append(parse_target(xtarRegex.search(inner).groups()[0]))

            match = grpEndRegex.search(line)
            if match:
                if not active:
                    print("tried to close a linkGroup without opening it on line", i)

                    continue
                GIZA(from_doc, to_doc, links)
                from_doc = ""
                to_doc = ""
                links = []
                active = False

from array import array

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
            if lidx % 1000000 == 0:
                print(lidx, asizeof.asized(out_from))
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
    if not os.path.exists(tempdir+"result"):
        clean_GIZA()
        MetaGIZA(corpus_path+"de-en.xml.gz")
        exec("giza pipeline", giza_skript)
    else:
        f, t = post_GIZA()
        for i in range(len(f)):
            if len(f[i]) is not len(t[i]):
                print("mismatch in line {}, length {} != {}".format(i, len(f[i]), len(t[i])))
                print(f[i])
                print(t[i])