# Aufgabe

Ziel unseres Projektes ist es, die mehrsprachige Word Sense Disambiguation zu untersuchen.

Während unseres Projekts untersuchen wir ob
1. Das ILI eine geeignete Ressource ist um mehrdeutige Wörter zu disambiguieren
2. Word2Vec geeignet ist um mehrdeutige Senses zu disambiguieren.

Beispiele einer mehrsprachigen Disambiguierung:
- E: duty_1 --> G: Pflicht_1

  E: duty_2 --> G: Steuer_1

  In diesem Beispiel kann eine Disambiguierung anhand der Übersetzung eindeutig durchgeführt werden.
- E: sharp_1 --> G: scharf_1

  E: sharp_2 --> G: scharf_2

  In diesem Beispiel ist die Übersetzung mehrdeutig und ist deshalb nicht zur Disambiguierung geeignet. Bei Betrachtung des Kontextes (SKipGram / Word2Vec) kann jedoch eine Disambiguierung weiterhin sinvoll sein.

# Vorgehen
In diesem Projekt verwenden wir den Inter Linugal Index (ILI) um Senses zwischen WordNet (Englisch) und GermaNet (Deutsch) zu projizieren.

Das ILI besteht aus einzelnen Einträgen ("iliRecords") von denen jeder ein SymSet in WordNet (mit allen dazugehörigen Bedeutungen) auf ein SynSet in GermaNet abbildet.

Zum Training verwenden wir den parallelen Korpus "EUBookShop" der OPUS Kollektion.
Dieser Korpus umfasst ca. 10 Millionen Sätze die sowohl im Deutschen als auch im Englischen vorliegen.
Jedoch ist dieser Korpus nur Satz-Aliniert, so dass wir ihn in einem Vorverarbeitungsschritt mittels GIZA++ (bzw. der auf moderne Systeme ausgelegten Version mgiza) Wort-Aliniert haben.

Weiterhin haben wir einen parallelen Korpus von Hand annotiert und umfasst 75 Sätze. Dieser Korpus wird als weiterhin als "Test-Korpus" bezeichnet und dient vorrangig zur (Qualitäts-)Auswertung der Embeddings aus Task 2.

## Task 1:
Zunächst werden die folgenden Statistiken (nach PoS Kategorie getrennt) berechnet:
1. Anteil der Wörter mit einer / mit mehreren Bedeutungen
2. Anteil der mehrdeutigen Wörter die mittels Übersetzung eindeutig unterschieden werden können.
3. Anteil der mehrdeutigen Wörter die nicht durch ihre Übersetzung disambiguiert werden können.

Anschließend werden die Wörter des Trainings-Korpus die bereits disambiguiert werden können, mit entsprechenden Sense-Tags versehen.
Auch hierüber werden Statistiken erstellt (nach PoS Kategorie getrennt):
1. Wieviele Wörter sind mehrdeutig
2. Wieviele Wörter sind eindeutig disambiguierbar
3. Wieviele Wörter sind im Korpus enthalten


## Task 2:
In diesem Schritt wird Word2Vec (mit Skipgram) an dem bereits teilweise disambiguierte Korpus (in nur einer Sprache, z.B. Englisch) trainiert und ein Embedding generiert.
Dieses Embedding wird wiederum auf unserem Test-Korpus angewendet und gibt für den Kontext eines jeden Wortes den wahrscheinlichsten Sense zurück.
Diese Senses werden nun mithilfe von ILI in das entsprechende Wort der anderen Sprache überführt und dort wiederum mit dem erwarteten Wort des parallelen Korpus verglichen und so die Qualität der Disambiguierung bestimmt.

Der von uns in diesem Schritt verwendete Word2Vec / SkipGram Algorithmus wurde von uns selbst mit Hilfe der TensorFlow Bibliothek geschrieben und basiert auf einem Turorial von "Thushan Ganegedaras" welches unter <link>(http://www.thushv.com/natural_language_processing/word2vec-part-1-nlp-with-deep-learning-with-tensorflow-skip-gram/) gefunden werden kann.
Wir haben uns für diese Vorgehensweise entschieden, weil wir Embeddings erzeugen wollten die
1. verschiedene Senes mit verschiedenen Vektoren repräsentieren und
2. die Entscheidung des besten Senses anhand der Lemmata der Kontext Wörter fällen können und
3. deren Trainingsdaten durch einen Generator gespeißt werden können.

# Evaluation

## Einschränkungen
In vielen Texten des EUBookShop weichen die Übersetzer mit steigender Textlänge weiter von dem Englischen Original-Text ab, bis hin zu komplett unterschiedlichen Paragraphen und Kapiteln.

Eine weitere Fehlerquelle ist die Wort-Alinierung mittels GIZA. Hier können mehrere verschiedene Fehler entstehen:
1. GIZA trennt Wörter immer an Leerzeichen, so dass häufig mehrteilige Wörter nicht korrekt zugeordnet werden.
2. GIZA kommt nicht mit Satzzeichen zurecht, so dass diese in einem Vorverarbeitungsschritt entfernt werden müssen. Dies macht sich besonders bei Wörter mit Apostrophen (z.B. in don't) bemerkbar
3. Allgemeine Fehler bei der Alinierung

# Resultate

# Erkentnisse

# References
- Kaveh Taghipour; Hwee Tou Ng (2015): One Million Sense-Tagged Instances for Word Sense Disambiguation and Induction, CoNLL 2015.
- Hong Jin Kang; Tao Chen; Muthu Kumar Chandrasekaran; Min-Yen Kan A (2016): Comparison of Word Embeddings for English and Cross-Lingual Chinese Word Sense Disambiguation, BEA.
- Roberto Navigli. 2009. Word sense disambiguation: A survey. ACM Comput. Surv. 41, 2, Article 10 (February 2009), 69 pages. DOI=http://dx.doi.org/10.1145/1459352.1459355
- Maarten van Gompel: Proceedings of the 5th International Workshop on Semantic Evaluation, ACL 2010, pages 238–241, UvT-WSD1: a Cross-Lingual Word Sense Disambiguation system
- Els Lefever; Véronique Hoste: Second Joint Conference on Lexical and Computational Semantics (*SEM), Volume 2: Seventh International Workshop on Semantic
Evaluation (SemEval 2013), pages 158–166, Atlanta, Georgia, June 14-15, 2013: SemEval-2013 Task 10: Cross-lingual Word Sense Disambiguation
- Bahareh Sarrafzadeh, Nikolay Yakovets, Nick Cercone, and Aijun An: Cross-Lingual Word Sense Disambiguation for Languages with Scarce Resources, 2014
- Minh-Thang Luong, Hieu Pham, Christopher D. Manning:Bilingual Word Representations with Monolingual Quality in Mind
- ella.cl.uni-heidelberg.de
- GIZA-pp / mgiza (https://github.com/moses-smt/mgiza)
- Vorlesungsfolien
# Aufgabenverteilung

Verena Mengen:
- Handannotierter Korpus

M. Moslemi:
- Handannotierter Korpus

Julian Rodriquez:
- WordNet / GermaNet / ILI
- Disambiguation via ILI

Niels Bernlöhr:
- Wort-Alinierung mittels GIZA
- Word2Vec / SkipGram