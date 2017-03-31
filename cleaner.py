import xml.etree.ElementTree as ET
from collections import defaultdict
from os import listdir
from sqlite3 import connect

pathPrefix = 'germanet-11.0/GN_V110_XML/'
typeDict = {'adj':'ADJ', 'nomen':'NOUN', 'verben':'VERB'}
polysemous = defaultdict(lambda: 0)
for path in [path for path in listdir(pathPrefix)]:
    if ('nomen' in path or 'verben' in path or 'adj' in path) and not 'wiktionary' in path:
        tree = ET.parse(pathPrefix+path).getroot()
        for synset in tree:
            for word in synset:
                if word.tag == 'lexUnit':
                    orthos = []
                    for orthform in word:
                        if orthform.tag in ['orthForm','orthVar','oldOrthForm','oldOrthVar']:
                            orthos.append((typeDict[synset.attrib['category']], orthform.text.lower(), word.attrib['sense']))
                    polysemous[orthos[0][1],orthos[0][0]] += 1
polysemous = {key for key in polysemous if polysemous[key] > 1}

corpus = []

for sentence in open('mapping-inv.txt').readlines()[:5]:
    sentence = []
    for word in sentence:
        if word[3] not in ['',-1]:
            if (word[1],word[2]) in polysemous:
                sentence.append(word)
            else:
                sentence.append([word[0],word[1],'',''])
        else:
            sentence.append(word)


print(corpus)
