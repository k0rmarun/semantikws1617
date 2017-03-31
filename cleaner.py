import xml.etree.ElementTree as ET
from collections import defaultdict
from os import listdir
from sqlite3 import connect
import json


def cleaner(num_lines):
    pathPrefix = 'germanet-11.0/GN_V110_XML/'
    typeDict = {'adj': 'ADJ', 'nomen': 'NOUN', 'verben': 'VERB'}
    polysemous = defaultdict(lambda: 0)
    for path in [path for path in listdir(pathPrefix)]:
        if ('nomen' in path or 'verben' in path or 'adj' in path) and not 'wiktionary' in path:
            tree = ET.parse(pathPrefix + path).getroot()
            for synset in tree:
                for word in synset:
                    if word.tag == 'lexUnit':
                        orthos = []
                        for orthform in word:
                            if orthform.tag in ['orthForm', 'orthVar', 'oldOrthForm', 'oldOrthVar']:
                                orthos.append(
                                    (typeDict[synset.attrib['category']], orthform.text.lower(), word.attrib['sense']))
                        polysemous[orthos[0][1], orthos[0][0]] += 1
    polysemous = {key for key in polysemous if polysemous[key] > 1}
    corpus = []

    lidx = 0
    with open('mapping.txt') as f:
        while True:
            lidx +=1
            if lidx > num_lines:
                break
            sentence_ = json.loads(f.readline())
            sentence = []
            for word in sentence_:
                if word[3] not in ['', -1]:
                    if (word[2], word[1]) in polysemous:
                        sentence.append(word)
                    else:
                        sentence.append([word[1], word[2], '', -1])
                else:
                    sentence.append(word)
            yield sentence