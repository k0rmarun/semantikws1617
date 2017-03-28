import xml.etree.ElementTree as ET
from collections import defaultdict
from os import listdir
from sqlite3 import connect

def xml_to_germanet (pathPrefix):
    typeDict = {'adj':'ADJ', 'nomen':'NOUN', 'verben':'VERB'}
    synsets = {}
    words = defaultdict(set)
    for path in [path for path in listdir(pathPrefix)]:
        if ('nomen' in path or 'verben' in path or 'adj' in path) and not 'wiktionary' in path:
            tree = ET.parse(pathPrefix+path).getroot()
            for synset in tree:
                for word in synset:
                    if word.tag == 'lexUnit':
                        synsets[word.attrib['id']] = synset.attrib['id']
                        for orthform in word:
                            if orthform.tag in ['orthForm','orthVar','oldOrthForm','oldOrthVar']:
                                words[synset.attrib['id']].add((typeDict[synset.attrib['category']], orthform.text.lower(), word.attrib['sense']))
    return synsets,dict(words)

def db_to_wordnet (path):
    typeDict = {'s':'ADJ', 'a':'ADJ', 'n':'NOUN', 'v':'VERB'}
    synsets = defaultdict(set)
    connection = connect(path)
    cursor = connection.cursor()
    cursor.execute('SELECT synsetid, sensenum, lemma, pos FROM words NATURAL JOIN senses NATURAL JOIN synsets')
    for synsetid, sense, lemma, pos in cursor.fetchall():
        if pos in 'nvas':
            synsets[synsetid].add((typeDict[pos],lemma,sense))
    return dict(synsets)

def create_mapping (iliPath, wordnet_path, germanet_path, lang='en'):
    wordnet = db_to_wordnet(wordnet_path)
    g_syns, g_words = xml_to_germanet(germanet_path)
    def clean (string):
        dictio = {'n':'1','v':'2','a':'3','s':'3'}
        return int(dictio[string[-1]]+string[6:-2])
    tagDict = defaultdict(lambda: defaultdict(set))
    tree = ET.parse(iliPath).getroot()
    for entry in tree:
        if entry.tag == 'iliRecord':
            if  all([stop not in entry.attrib['pwn30Id'] for stop in ['00000000', '0000null', '-r']]):
                try:
                    for english, german in ((english,german) for english in wordnet[clean(entry.attrib['pwn30Id'])] for german in g_words[g_syns[entry.attrib['lexUnitId']]]):
                        if lang == 'en':
                            tagDict[english[0]][(english[1],german[1])].add(english[2])
                        elif lang == 'de':
                            tagDict[german[0]][(english[1],german[1])].add(german[2])
                except KeyError:
                    pass
    return {k1:{k2:list(tagDict[k1][k2])[0] for k2 in tagDict[k1] if len(tagDict[k1][k2]) == 1} for k1 in tagDict}

if __name__ == '__main__':
    dictio = create_mapping('interLingualIndex_DE-EN_GN110.xml', 'sqlite-30.db', 'germanet-11.0/GN_V110_XML/')
    print(sum([len(dictio[k1]) for k1 in dictio]))
