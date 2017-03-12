import math
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from scipy.spatial.distance import cosine


class BiSkip:
    """
    Implementation of the Bilingual Skip-Gram algorithm
    """
    num_steps = 10001     # Number of learning iterations. Should be much larger (10-1000) than len(text)/batch_size
    batch_size = 128      # Number of samples to learn at a time. The higher the faster
    num_sampled = 64      # Number of negative examples to sample.
    embedding_size = 300  # Dimension of the embedding vector.
    eval_k = 8            # number of nearest neighbors during inplace evaluation

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

    def __init__(self, dictionary: dict, reverse_dictionary: dict, evaluation_data: np.array=None):
        """
        Initialize data and neural network
        :param dictionary: Lookup table lang1 => idx1
        :type dictionary: dict[str, int]
        :param reverse_dictionary: Lookup table idx2 => lang2
        :type reverse_dictionary: dict[int, str]
        :param evaluation_data: Data to display evaluation performance during training
        :type evaluation_data: np.array
        """
        self.device = self.get_preferred_device()
        print("using device", self.device)

        self.generate_batch = None
        self.showLoss = True
        self.showEval = True

        self.dict = dictionary
        self.rev_dict = reverse_dictionary
        self.vocabulary_size = max(len(self.dict), len(self.rev_dict))
        self.evaluation_data = evaluation_data
        self.trained_embeddings = None

        self.graph = tf.Graph()
        self.train_dataset = None
        self.train_labels = None
        self.eval_dataset = None
        self.embeddings = None
        self.loss = None
        self.average_loss = None
        self.optimizer = None
        self.normalized_embeddings = None
        self.similarity = None
        self.setupNN()

    def setupNN(self):
        """
        Setup layout of the neural network
        Based on Thushan Ganegedaras' TensorFlow embedding tutorial
        """
        with self.graph.as_default(), tf.device(self.device):  # Select graph and device as default

            # Input data.
            self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            if self.evaluation_data is not None:
                self.eval_dataset = tf.constant(self.evaluation_data, dtype=tf.int32)

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
            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                                                             self.train_labels, self.num_sampled, self.vocabulary_size))

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

            # Compute the similarity between minibatch examples and all embeddings.
            # We use the cosine distance:
            if self.evaluation_data is not None:
                eval_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, self.eval_dataset)
                self.similarity = tf.matmul(eval_embeddings, tf.transpose(self.normalized_embeddings))

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
            tf.initialize_all_variables().run()
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

    def post_train_batch(self, step):
        """
        Perform display of loss and evaluation after n training steps
        :param step: Training step which was recently executed
        :type step: int
        """
        # Display neural network loss every 2k steps
        if self.showLoss and step % 2000 == 0:
            if step > 0:
                # The average loss is an estimate of the loss over the last 2000 minibatches.
                print('Average loss at step %d: %f' % (step, self.average_loss / 2000))
                self.average_loss = 0

        # Display neural network performance on example data every 10k steps
        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if self.showEval and step % 10000 == 0 and self.evaluation_data is not None:
            sim = self.similarity.eval()
            for wIdx in range(len(self.evaluation_data)):
                predict_word = self.rev_dict[self.evaluation_data[wIdx]]
                log = 'Nearest to {}:'.format(predict_word)
                eval = (-sim[wIdx, :]).argsort()[1:self.eval_k + 1]
                for predicted_idx in eval:
                    predicted_word = self.rev_dict[predicted_idx]
                    log += " "+predicted_word
                print(log)

    def wordvec(self, word):
        """
        Get word2vec vector for a given word
        :param word: word to fetch
        :type word: int | str
        :return: word2vec vector
        :rtype: np.array
        """
        if isinstance(word, str):
            word = self.dict[word]
        return self.trained_embeddings[word, :]

    def predict_one(self, word, k=1):
        """
        Predict k words most similar to given word in lang1
        :param word: word to predict similarities
        :type word: int | str
        :param k: number of similar words to predict
        :type k: int
        :return: similar words in lang2
        :rtype: [int | str]
        """
        was_str = False
        if isinstance(word, str):
            was_str = True
            word = self.dict[word]

        a = self.trained_embeddings[word, :]
        cos = np.zeros(self.vocabulary_size)
        for i in range(self.vocabulary_size):
            cos[i] = cosine(a, self.trained_embeddings[i, :])
        cos_sort = cos.argsort()
        word = []
        for i in range(k):
            if was_str:
                word.append((self.rev_dict[cos_sort[1+i]], cos[cos_sort[1+i]]))
            else:
                word.append((cos_sort[1 + i], cos[cos_sort[1 + i]]))
        return word

    def predict_context(self, words):
        """
        Predict most similar word from given context words in lang1
        :param words: context words
        :type words: [int | str]
        :return: most probable word in lang2
        :rtype: str
        """
        # Calculate average of word vectors of context words
        contexts = np.zeros(self.embedding_size)
        for i in range(len(words)):
            contexts += self.wordvec(words[i])
        contexts /= len(words)

        # Find trained word with most similar vector
        cos = np.zeros(self.vocabulary_size)
        for i in range(self.vocabulary_size):
            cos[i] = cosine(contexts, self.trained_embeddings[i, :])
        cos = cos.argsort()
        return self.rev_dict[cos[1]]

