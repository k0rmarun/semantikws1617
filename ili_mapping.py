import xml.etree.ElementTree as ET
from collections import defaultdict
from os import listdir
from sqlite3 import connect

def xml_to_germanet (pathPrefix):
    typeDict = {'adj':'ADJ', 'nomen':'NOUN', 'verben':'VERB'}
    synsets = {}
    words = defaultdict(set)
    polysemous = defaultdict(lambda: 0)
    for path in [path for path in listdir(pathPrefix)]:
        if ('nomen' in path or 'verben' in path or 'adj' in path) and not 'wiktionary' in path:
            tree = ET.parse(pathPrefix+path).getroot()
            for synset in tree:
                for word in synset:
                    if word.tag == 'lexUnit':
                        synsets[word.attrib['id']] = synset.attrib['id']
                        orthos = []
                        for orthform in word:
                            if orthform.tag in ['orthForm','orthVar','oldOrthForm','oldOrthVar']:
                                orthos.append((typeDict[synset.attrib['category']], orthform.text.lower(), word.attrib['sense']))
                        words[synset.attrib['id']].add(tuple(orthos))
                        polysemous[orthos[0][1],orthos[0][0]] += 1
    polysemous = {key for key in polysemous if polysemous[key] > 1}
    return synsets,dict(words), polysemous

def db_to_wordnet (path):
    typeDict = {'s':'ADJ', 'a':'ADJ', 'n':'NOUN', 'v':'VERB'}
    synsets = defaultdict(set)
    polysemous = defaultdict(lambda: 0)
    connection = connect(path)
    cursor = connection.cursor()
    cursor.execute('SELECT synsetid, sensenum, lemma, pos FROM words NATURAL JOIN senses NATURAL JOIN synsets')
    for synsetid, sense, lemma, pos in cursor.fetchall():
        if pos in 'nvas':
            synsets[synsetid].add((typeDict[pos],lemma,sense))
            polysemous[(lemma,typeDict[pos])] += 1
    polysemous = {key for key in polysemous if polysemous[key] > 1}
    return dict(synsets), polysemous

def create_mapping (iliPath, wordnet_path, germanet_path, lang='en'):
    def isPolysemous(english,german,wordtype):
        if lang == 'en':
            if (english,wordtype) in polysemous_e:
                return True
        elif lang == 'de':
            if (german,wordtype) in polysemous_g:
                return True
        return False
    wordnet, polysemous_e = db_to_wordnet(wordnet_path)
    g_syns, g_words, polysemous_g = xml_to_germanet(germanet_path)
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
                            tagDict[english[0]][(english[1],tuple(word[1] for word in german))].add(english[2])
                        elif lang == 'de':
                            tagDict[german[0][0]][(english[1],tuple(word[1] for word in german))].add(german[0][2])
                except KeyError:
                    pass
    language = 'english' if lang == 'en' else 'german'
    print('Linked %s nouns:'%(language),len(tagDict['NOUN']))
    print('Linked %s verbs:'%(language),len(tagDict['VERB']))
    print('Linked %s adjectives:'%(language),len(tagDict['ADJ']))
    tagDict= {wordtype:{(english,german):tagDict[wordtype][(english,german)] for english,german in tagDict[wordtype] if isPolysemous(english,german[0],wordtype)} for wordtype in tagDict}
    print('Polysemous linked %s nouns:'%(language),len(tagDict['NOUN']))
    print('Polysemous linked %s verbs:'%(language),len(tagDict['VERB']))
    print('Polysemous linked %s adjectives:'%(language),len(tagDict['ADJ']))
    tagDict = {wordtype:{wordpair:tagDict[wordtype][wordpair] for wordpair in tagDict[wordtype] if len(tagDict[wordtype][wordpair])==1} for wordtype in tagDict}
    print('Polysemous linked %s nouns that can be disambiguated:'%(language),len(tagDict['NOUN']))
    print('Polysemous linked %s verbs that can be disambiguated:'%(language),len(tagDict['VERB']))
    print('Polysemous linked %s adjectives that can be disambiguated:'%(language),len(tagDict['ADJ']))
    tagDict = {wordtype:{(english,german):tagDict[wordtype][english,g_set] for english, g_set in tagDict[wordtype] for german in g_set} for wordtype in tagDict}
    return dict(tagDict), polysemous_e, polysemous_g

if __name__ == '__main__':
    dictio,_,_ = create_mapping('interLingualIndex_DE-EN_GN110.xml', 'sqlite-30.db', 'germanet-11.0/GN_V110_XML/')
    print('Entries after expanding german orthography:',sum([len(dictio[k1]) for k1 in dictio]))
    dictio,_,_ = create_mapping('interLingualIndex_DE-EN_GN110.xml', 'sqlite-30.db', 'germanet-11.0/GN_V110_XML/','de')
    print('Entries after expanding german orthography:',sum([len(dictio[k1]) for k1 in dictio]))
