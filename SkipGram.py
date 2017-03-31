import math
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from scipy.spatial.distance import cosine
import json
from memory_profiler import profile
import pickle


__licence__ = "GPLv3"
__author__ = "Niels Bernlöhr (kormarun)"
__attributions__ = ["Thushan Ganegedaras"]


class SkipGramTF:
    """
    Implementation of the Skip-Gram algorithm using TensorFlow
    """
    num_steps = 10001  # Number of learning iterations. Should be much larger (10-1000) than len(text)/batch_size
    batch_size = 10  # Number of samples to learn at a time. The higher the faster
    embedding_size = 300  # Dimension of the embedding vector.
    num_sampled = 5  # Number of negative examples to sample.

    @staticmethod
    def get_preferred_device():
        """
        Select name of preferred device for TensorFlow.
        If a GPU is available use this.
        :return: Device to select
        :rtype: str
        """
        local_devices = device_lib.list_local_devices()
        local_devices = [x.name for x in local_devices if x.device_type == 'GPU']

        if len(local_devices) > 0:
            return local_devices[0]
        return "/cpu:0"

    def __init__(self, vocabulary_size: int, corpus_size: int = None, load: bool = False):
        """
        Initialize data and neural network
        :param vocabulary_size: number of words in vocabulary
        :type vocabulary_size: int
        """
        if not load:
            if corpus_size:
                self.num_steps = 10 * corpus_size // self.batch_size
            self.device = self.get_preferred_device()
            print("using device", self.device)

            self.generate_batch = None
            self.showLoss = True

            self.vocabulary_size = vocabulary_size
            self.trained_embeddings = None

            self.graph = tf.Graph()
            self.train_dataset = None
            self.train_labels = None
            self.embeddings = None
            self.loss = None
            self.average_loss = None
            self.optimizer = None
            self.normalized_embeddings = None
            self.setupNN()
        else:
            self.load()

    def setupNN(self):
        """
        Setup layout of the neural network
        Based on Thushan Ganegedaras' TensorFlow embedding tutorial
        """
        with self.graph.as_default(), tf.device(self.device):  # Select graph and device as default

            # Input data.
            self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            # Variables.
            # embedding, vector for each word in the vocabulary
            self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
            softmax_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                                              stddev=1.0 / math.sqrt(self.embedding_size)))
            softmax_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Model.
            # Look up embeddings for inputs.
            # this might efficiently find the embeddings for given ids (traind dataset)
            # manually doing this might not be efficient given there are 50000 entries in embeddings
            embed = tf.nn.embedding_lookup(self.embeddings, self.train_dataset)
            print("Embed size: %s" % embed.get_shape().as_list())
            # Compute the softmax loss, using a sample of the negative labels each time.
            # inputs are embeddings of the train words
            # with this loss we optimize weights, biases, embeddings
            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases,
                                                                  self.train_labels, embed,
                                                                  self.num_sampled,
                                                                  self.vocabulary_size))

            # Optimizer.
            # Note: The optimizer will optimize the softmax_weights AND the embeddings.
            # This is because the embeddings are defined as a variable quantity and the
            # optimizer's `minimize` method will by default modify all variable quantities
            # that contribute to the tensor it is passed.
            # See docs on `tf.train.Optimizer.minimize()` for more details.
            # Adagrad is required because there are too many things to optimize
            self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            self.normalized_embeddings = self.embeddings / norm

    #@profile
    def train(self, generate_batch):
        """
        Train neural network on data yielded from generate batch
        due to technical constraints "generate_batch" must yield keys and label in the following format
        - keys = [7, 24, 3, ...]
        - labels = [[46],[156],[128]]
        :param generate_batch: generator which provides data & labels for a single batch
        :type generate_batch: generator
        """
        # Initialize generator
        self.generate_batch = generate_batch(self.batch_size)

        # Allow for fuzzy device selection
        config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(graph=self.graph, config=config) as session:  # bind session as default
            tf.global_variables_initializer().run()
            session.graph.finalize()
            print('Starting training')

            self.average_loss = 0
            # Do num_steps minibatches
            for step in range(self.num_steps):
                self.average_loss += self.train_batch(session)
                self.post_train_batch(step)

            # Save learned embeddings for later
            self.trained_embeddings = self.embeddings.eval()

    def train_batch(self, session):
        """
        Train neural network on a single minibatch
        :param session: TensorFlow Session to use. Must be passed for technical reasons
        :type session: tf.Session
        :return: Neural network loss in minibatch
        :rtype: float
        """
        # Fetch new data from generator
        batch_data, batch_labels = next(self.generate_batch)

        # Advance training with new data
        feed_dict = {
            self.train_dataset: batch_data,
            self.train_labels: batch_labels
        }
        _, l = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
        return l

    def post_train_batch(self, step: int):
        """
        Perform display of loss and evaluation after n training steps
        :param step: Training step which was recently executed
        :type step: int
        """
        # Display neural network loss every 2k steps
        if self.showLoss and step % 10000 == 0:
            if step > 0:
                # The average loss is an estimate of the loss over the last 2000 minibatches.
                print('Average loss at step %d: %f' % (step, self.average_loss / 2000))
                self.average_loss = 0

    def word2vec(self, word: int) -> np.array:
        """
        Fetch word vector for given word
        :param word: wordID to fetch vector from
        :type word: int
        :return: word vector
        :rtype: np.array
        """
        return self.trained_embeddings[word, :]

    def save(self):
        np.save("./embeddings.npy", self.trained_embeddings)

    def load(self):
        self.trained_embeddings = np.load("./embeddings.npy")

class SkipGram:
    """
    Simple wrapper for Skip-Gram algorithm
    Performs translation from corpus data to Skip-Gram indices, trains Skip-Gram algorithm and
    can be queried to disambiguate words from their contexts
    """

    def __prepare(self):
        """
        Translate sense tagged corpus data into internal representation used in the Skip-Gram algorithm
        """
        it = self.__corpus()
        for sentence in it:
            for word in sentence:
                if isinstance(word, str):
                    typ, lemma, sid = "Unk", word, -1
                else:
                    typ, lemma, sid = word

                if lemma not in self.__lemmacount:
                    self.__lemmacount[lemma] = 0
                self.__lemmacount[lemma] += 1

        it = self.__corpus()
        for sentence in it:
            for word in sentence:
                if isinstance(word, str):
                    typ, lemma, sid = "Unk", word, -1
                else:
                    typ, lemma, sid = word

                # Ignore all broken words
                if self.__lemmacount[lemma] < 2:
                    continue
                self.__corpus_size += 1
                if self.__corpus_size % 10000 == 0:
                    print(self.__corpus_size)
                if lemma not in self.__lemma2sense:
                    self.__lemma2sense[lemma] = []
                if word not in self.__lemma2sense[lemma]:
                    self.__lemma2sense[lemma].append(word)

                if word not in self.__sensedict:
                    self.__sensedict[word] = len(self.__sensedict)
        self.__inv_sensedict = dict(zip(self.__sensedict.values(), self.__sensedict.keys()))

    def __batch(self, batch_size: int):
        """
        Generate the data for a single batch in BiSkip algorithm
        :param batch_size: number of entries in a BiSKip batch
        :type batch_size: int
        """
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        # How many words left and right
        window = 4

        # How many elements in batch. Yield and reset at :batch_size
        batchidx = 0
        while True:
            it = self.__corpus()
            for sentence in it:
                l = len(sentence)
                for i in range(l):
                    try:
                        forw_word = self.__sensedict[sentence[i]]
                    except KeyError:
                        continue
                    # Get words left/right inside sentence, skip where idx1 = idx2
                    for j in range(max(0, i - window), min(l, i + window + 1)):
                        if i == j:
                            continue
                        try:
                            back_word = self.__sensedict[sentence[j]]
                        except KeyError:
                            continue

                        batch[batchidx] = forw_word
                        labels[batchidx, 0] = back_word

                        batchidx += 1
                        # yield and reset as soon as we reach batch_size entries
                        if batchidx == batch_size:
                            yield batch, labels
                            batchidx = 0

    #@profile
    def __init__(self, corpus, load=False):
        """
        Translate and train Skip-Gram algorithm
        :param corpus: corpus to use in training
        :type corpus: Function -> Generator->List[List[(type, lemma, senseID)]]
        """
        self.__corpus = corpus
        self.__lemma2sense = {}
        self.__sensedict = {}
        self.__inv_sensedict = {}
        self.__lemmacount = {}
        self.__corpus_size = 0
        if load:
            self.load()
        else:
            self.__prepare()
            self.__sg = SkipGramTF(len(self.__sensedict), self.__corpus_size)
            self.__sg.train(self.__batch)

    def save(self):
        with open("lemma2sense.pickle", "wb") as f:
            f.write(pickle.dumps(self.__lemma2sense))
        with open("sensedict.pickle", "wb") as f:
            f.write(pickle.dumps(self.__sensedict))
        self.__sg.save()

    def load(self):
        with open("lemma2sense.pickle", "rb") as f:
            self.__lemma2sense = pickle.load(f)
        with open("sensedict.pickle", "rb") as f:
            self.__sensedict = pickle.load(f)
        self.__inv_sensedict = dict(zip(self.__sensedict.values(), self.__sensedict.keys()))
        self.__sg = SkipGramTF(self.__corpus_size, load=True)

    def choose(self, context, choice: str):
        """
        Choose the best fitting choice in the given context
        Selects the sense from choices that has the highest similarity to the mean context vector
        :param context: Untagged context lemmas
        :type context: Function -> Generator
        :param choice: Sense tagged choices
        :type choice: List[(type, lemma, senseID)]
        :return: Best choice
        :rtype: Union{None, (type, lemma, senseID)}
        """

        try:
            context_vectors = np.zeros(self.__sg.embedding_size)
            context_cnt = 0
            for word in context:
                if word not in self.__lemma2sense:
                    continue
                senses = self.__lemma2sense[word]
                for sense in senses:
                    tmp = self.__sg.word2vec(
                        self.__sensedict[sense]
                    )
                    if np.any(np.isnan(tmp)):
                        print("fuck")
                    context_vectors += tmp
                    context_cnt += 1
            context_vectors /= context_cnt  # calculate mean

            if choice not in self.__lemma2sense:
                return None
            choices = self.__lemma2sense[choice]

            cosines = np.zeros(len(choices))
            print("Got", len(choices), "choices for word", choice,"are", choices)
            for i in range(len(choices)):
                choice_ = choices[i]
                choice_vector = self.__sg.word2vec(self.__sensedict[choice_])
                cosines[i] = cosine(context_vectors, choice_vector)
            return choices[cosines.argmax()]#, choices, len(choices)
        except ValueError as e:
            print(e)
        except KeyError as e:
            print("keyerror")
            raise e
            #return None


if __name__ == "__main__":
    import collections
    import numpy as np
    import os
    import random
    import tensorflow as tf
    import zipfile

    from six.moves import range
    from six.moves.urllib.request import urlretrieve

    url = 'http://mattmahoney.net/dc/'


    def maybe_download(filename, expected_bytes):
        """Download a file if not present, and make sure it's the right size."""
        if not os.path.exists(filename):
            print("downloading ", filename)
            filename, _ = urlretrieve(url + filename, filename)
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified %s' % filename)
        else:
            print(statinfo.st_size)
            raise Exception(
                'Failed to verify ' + filename + '. Can you get to it with a browser?')
        return filename


    filename = maybe_download('text8.zip', 31344016)


    def read_data(filename):
        """Extract the first file enclosed in a zip file as a list of words"""
        with zipfile.ZipFile(filename) as f:
            data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        return data


    words = read_data(filename)
    print('Data size %d' % len(words))


    def make_sentenced(words):
        i = 0
        sentence = 1
        sentences = []
        for word in words:
            if len(sentences) < sentence:
                sentences.append([])
            sentences[-1].append((1, word, 1))
            i += 1
            if i % 50 == 0:
                i = 0
                sentence += 1
        return sentences


    data = make_sentenced(words)

    sg = SkipGram(data)
    print("best: ", sg.choose(
        ["one", "two", "three"],
        "six"
    ))
    print("best: ", sg.choose(
        ["one", "two", "three"],
        "bank"
    ))
