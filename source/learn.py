import tensorflow as tf
import numpy as np

#TODO Complete MLP

class MLP_Classifier(object):
    """
    This class is used for the multilayer perceptron architecture. It is used to process only the features
    and not the images
    :param structure: It defines the structure of the convolutional network part
    :type structure: An iterable of dictionaries..
    :param nb_classes: The number of labels in the training/validation/test sets
    :type nb_classes: Integer
    :param img_rows: Number of vertical pixels in the image
    :type img_rows: Integer
    :param img_cols: Number of horizontal pixels in the image
    :type img_cols: Integer
    :param nb_hidden: Defines the structure of the fully connected layers
    :type nb_hidden: Iterable of integers (at the moment only 2 layers so it will be a single integer)
    :param nb_features: Number of features added at the first fully connected layer
    :type nb_feature: Integer
    """
    def __init__(self, structure=None, nb_classes=10, nb_features=100):
        self.__dict__.update(locals())
        self.graph = tf.Graph()
        self.saver = None
        self.logger = {"training_error" : [], "validation_error" : [], "test_error" : []}


    def _define_placeholders(self):
        """
        Defines all the placeholders needed during the computation. Two placeholders are used
        for the input images and input features, and one for the labels.
        """
        self.x = tf.placeholder(tf.float32, shape=[None, self.nb_features])
        self.y = tf.placeholder(tf.float32, shape=[None, self.nb_classes])

    def _initialize_variables(self):
        """
        Initializes weigth and bias variables.
        """

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        self.W = [weight_variable([self.nb_features, self.structure[0]])]
        self.b = [bias_variable([self.structure[0]])]

        for i in range(1, len(self.structure)):
            self.W.append(weight_variable([self.structure[i-1], self.structure[i]]))
            self.b.append(bias_variable([self.structure[i]]))

        self.W_final = weight_variable([self.structure[-1], self.nb_classes])
        self.b_final = bias_variable([self.nb_classes])

    def _model(self, X_image, dropout=1.0):
        """
        Builds the network structure.
        """
        data = X_image

        for i in range(len(self.structure)):
            data = tf.nn.relu(tf.matmul(data, self.W[i]) + self.b[i])
        data = tf.nn.dropout(data, dropout)
        return tf.matmul(data, self.W_final) + self.b_final

    def _accuracy(self, predictions, actual):
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(actual, 1))/predictions.shape[0]


    def fit(self, X, y, batch_size = 128, nb_epochs = 100,
            p_dropout = 0.5, logging_info = 10, optimizer={"type" : "Adagrad"}, save_path=None, seed=None):

        """
        Fits the network using AdagradOptimizer #TODO setting up the parameters for
        :param X: The feature vector for the training set
        :param y: The target vector for the trainin set
        :param batch_size: The size of batch fed for each epoch
        :param nb_epochs: The number of epochs in a minibatch framework
        :param p_dropout: Dropout probability (only applied once)
        :param logging_info: An integer managing logging informations
        :param save_path: The path of the saved model. If None, the model is not saved
        :param seed: The seed of the tensorflow random generator. Used to set up the initial weights of the network
        """

        def next_batch(X, y, length):
            if (self.batch_init + 1) * length <= len(y):
                init = self.batch_init * length
                fin = (self.batch_init + 1) * length
                self.batch_init += 1
                return X[init: fin], y[init: fin]
            else:
                init = self.batch_init * length
                self.batch_init = 0
                return X[init:], y[init:]

        def prepare_dict(batch):
            dic = {self.x: batch[0], self.y : batch[1]}
            return dic

        def _optimizer_type(optimizer, loss):
            if optimizer["type"] == "Adagrad":
                learning_rate = tf.train.exponential_decay(0.05, tf.Variable(0), 10000, 0.95)
                return tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=tf.Variable(0))
            elif optimizer["type"] == "Adadelta":
                return tf.train.AdadeltaOptimizer(optimizer["learning_rate"]).minimize(loss)

        with self.graph.as_default():
            if seed is not None:
                tf.set_random_seed(seed)
            self._define_placeholders()
            self._initialize_variables()
            self.saver = tf.train.Saver()

            logits = self._model(self.x, p_dropout)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.y))

            optimizer = _optimizer_type(optimizer, loss)
            prediction = tf.nn.softmax(self._model(self.x,  1.0))


        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            self.batch_init = 0
            for step in range(nb_epochs):
                batch = next_batch(X, y, batch_size)
                _, l, results = session.run([optimizer, loss, prediction], feed_dict=prepare_dict(batch))
                if step % logging_info == 0:
                    print("Minibatch loss value at step {}: {:.2f}".format(step+1, l))
                    minibatch_accuracy = self._accuracy(results, batch[1])
                    print("Minibatch accuracy: {:.1f}%".format(minibatch_accuracy))
                    #self.logger["training_error"].append(np.array([minibatch_accuracy_digits, minibatch_accuracy_full]))


            if save_path is not None:
                self.saver.save(session, save_path)
                print ("Model saved in {}".format(save_path))

    #def score(self, X, y, restore_path=None):
    #    with tf.Session(graph=self.graph) as session:
    #        tf.initialize_all_variables().run()
    #        if restore_path is not None:
    #            self.saver.restore(session, restore_path)
     #       predictions = tf.nn.softmax(self._model(X.reshape(-1, self.img_rows, self.img_cols, 1), 1.0))
     #       predictions = predictions.eval(feed_dict={})
     #       return self._accuracy(predictions, y), self._accuracy_full(predictions, y)

    #def predict(self, X, restore_path=None):
    #    with tf.Session(graph=self.graph) as session:
    #        tf.initialize_all_variables().run()
     #       if restore_path is not None:
     #           self.saver.restore(session, restore_path)
     ##       predictions = tf.pack([tf.nn.softmax(self._model(X.reshape(-1, self.img_rows, self.img_cols, 1), 1.0)[i]) for i in range(self.max_nb_digits)])
     #       predictions = predictions.eval(feed_dict={})
     #       return np.argmax(predictions, 2).T


class CNN_Classifier(object):
    """
    This class is used for the deep learning architecture. It defines a specific convolution neural network
    architecture which takes input images and process them. At the level of the fully connected structure
    additional features can be added and serialized.
    :param structure: It defines the structure of the convolutional network part
    :type structure: An iterable of dictionaries...
    :param nb_channels: The number of channels on the input images (default = 1)
    :type nb_channels: Integer
    :param nb_classes: The number of labels in the training/validation/test sets
    :type nb_classes: Integer
    :param img_rows: Number of vertical pixels in the image
    :type img_rows: Integer
    :param img_cols: Number of horizontal pixels in the image
    :type img_cols: Integer
    :param nb_hidden: Defines the structure of the fully connected layers
    :type nb_hidden: Iterable of integers (at the moment only 2 layers so it will be a single integer)
    :param nb_features: Number of features added at the first fully connected layer
    :type nb_feature: Integer
    """
    def __init__(self, structure=None, nb_channels=1, nb_classes=10, img_rows=32, img_cols=32, nb_hidden=(1024,), nb_features=100):
        self.__dict__.update(locals())
        self.graph = tf.Graph()
        self.structure = structure
        self.saver = None
        self.logger = {"training_error" : [], "validation_error" : [], "test_error" : []}


    def _define_placeholders(self):
        """
        Defines all the placeholders needed during the computation. Two placeholders are used
        for the input images and input features, and one for the labels.
        """
        self.x_cnn = tf.placeholder(tf.float32, shape=[None, self.img_rows, self.img_cols, self.nb_channels])
        self.x_fully = tf.placeholder(tf.float32, shape=[None, self.nb_features])
        self.y = tf.placeholder(tf.float32, shape=[None, self.nb_classes])

    def _initialize_variables(self):
        """
        Initializes weigth and bias variables.
        """

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        self.W_conv = []
        self.b_conv = []
        self.W_dense = []
        self.b_dense = []
        self.W_fc = []
        self.b_fc = []

        #pool_prod = 1
        n_neurons = 0
        for s in self.structure:
            if s["type"] == "conv":
                params = s["params"]
                self.W_conv.append(weight_variable([params["patch_x"], params["patch_y"],
                                                          params["channels"], params["depth"]]))
                self.b_conv.append(bias_variable([params["depth"]]))
            #    last_depth = params["depth"]
            #elif s["type"] == "pool":
            #    pool_prod *= s["params"]["side"]

            elif s["type"] == "dense":
                params = s["params"]
                self.W_dense.append(weight_variable([params["n_input"],params["n_neurons"]]))
                self.b_dense.append(bias_variable([params["n_neurons"]]))
                n_neurons = params["n_neurons"]

        #TODO At the moment works only if a dense layer is present at the end of the conv/pool architecture
        #TODO before adding the external variables.

        n_neurons += self.nb_features
        for n in self.nb_hidden:
            self.W_fc.append(weight_variable([n_neurons, n]))
            self.b_fc.append(bias_variable([n]))
            n_neurons = n

        self.W_fcfinal = weight_variable([n_neurons, self.nb_classes])
        self.b_fcfinal = bias_variable([self.nb_classes])

    def _model(self, X_image, X_features, dropout=1.0):
        """
        Builds the network structure.
        """
        data = X_image
        def add_conv_layer(data, W, b):
            activation = tf.nn.conv2d(data, W, strides=[1, 1, 1, 1], padding='SAME') + b
            return tf.nn.local_response_normalization(tf.nn.relu(activation))

        def add_pool_layer(data, params):
            return tf.nn.max_pool(data, ksize=[1, params["side"], params["side"], 1],
                                  strides=[1, params["stride"], params["stride"], 1], padding='SAME')

        def add_dense_layer(data, W, b):
            shape = data.get_shape()
            try:
                data = tf.reshape(data, [-1, int(shape[1]*shape[2]*shape[3])])
            except:
                pass
            return tf.nn.relu(tf.matmul(data, W) + b)

        conv = 0
        dense = 0
        for s in self.structure:
            if s["type"] == "conv":
                data = add_conv_layer(data, self.W_conv[conv], self.b_conv[conv])
                conv += 1
            elif s["type"] == "pool":
                data = add_pool_layer(data, s["params"])
            elif s["type"] == "dense":
                data = add_dense_layer(data, self.W_dense[dense], self.b_dense[dense])
                dense += 1

        try:
            im_shape = data.get_shape()
            data = tf.reshape(data, [-1, int(im_shape[1]*im_shape[2]*im_shape[3])])
        except:
            pass
        data = tf.concat(1, [data, X_features])
        for i in range(len(self.nb_hidden)):
            data = tf.nn.relu(tf.matmul(data, self.W_fc[i]) + self.b_fc[i])
            data = tf.nn.dropout(data, dropout)
        return tf.matmul(data, self.W_fcfinal) + self.b_fcfinal

    def _accuracy(self, predictions, actual):
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(actual, 1))/predictions.shape[0]


    def fit(self, X, y, batch_size = 128, nb_epochs = 100,
            p_dropout = 0.5, logging_info = 10, optimizer={"type" : "Adagrad"}, save_path=None, seed=None):

        """
        Fits the network using AdagradOptimizer #TODO setting up the parameters for
        :param X: The feature vector for the training set
        :param y: The target vector for the trainin set
        :param batch_size: The size of batch fed for each epoch
        :param nb_epochs: The number of epochs in a minibatch framework
        :param p_dropout: Dropout probability (only applied once)
        :param logging_info: An integer managing logging informations
        :param save_path: The path of the saved model. If None, the model is not saved
        :param seed: The seed of the tensorflow random generator. Used to set up the initial weights of the network
        """

        def next_batch(X, y, length):
            if (self.batch_init + 1) * length <= len(y):
                init = self.batch_init * length
                fin = (self.batch_init + 1) * length
                self.batch_init += 1
                return X["image"][init: fin], X["features"][init: fin], y[init: fin]
            else:
                init = self.batch_init * length
                self.batch_init = 0
                return X["image"][init:], X["features"][init:] ,y[init:]

        def prepare_dict(batch):
            dic = {self.x_cnn: batch[0].reshape(-1, self.img_rows, self.img_cols, self.nb_channels),
                   self.x_fully: batch[1],
                   self.y : batch[2]}
            return dic

        def _optimizer_type(optimizer, loss):
            if optimizer["type"] == "Adagrad":
                learning_rate = tf.train.exponential_decay(0.05, tf.Variable(0), 10000, 0.95)
                return tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=tf.Variable(0))
            elif optimizer["type"] == "Adadelta":
                return tf.train.AdadeltaOptimizer(optimizer["learning_rate"]).minimize(loss)

        with self.graph.as_default():
            if seed is not None:
                tf.set_random_seed(seed)
            self._define_placeholders()
            self._initialize_variables()
            self.saver = tf.train.Saver()

            logits = self._model(self.x_cnn, self.x_fully, p_dropout)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.y))

            optimizer = _optimizer_type(optimizer, loss)
            prediction = tf.nn.softmax(self._model(self.x_cnn, self.x_fully, 1.0))


        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            self.batch_init = 0
            for step in range(nb_epochs):
                batch = next_batch(X, y, batch_size)
                _, l, results = session.run([optimizer, loss, prediction], feed_dict=prepare_dict(batch))
                if step % logging_info == 0:
                    print("Minibatch loss value at step {}: {:.2f}".format(step+1, l))
                    minibatch_accuracy = self._accuracy(results, batch[2])
                    print("Minibatch accuracy: {:.1f}%".format(minibatch_accuracy))
                    #self.logger["training_error"].append(np.array([minibatch_accuracy_digits, minibatch_accuracy_full]))


            if save_path is not None:
                self.saver.save(session, save_path)
                print ("Model saved in {}".format(save_path))

    #def score(self, X, y, restore_path=None):
    #    with tf.Session(graph=self.graph) as session:
    #        tf.initialize_all_variables().run()
    #        if restore_path is not None:
    #            self.saver.restore(session, restore_path)
     #       predictions = tf.nn.softmax(self._model(X.reshape(-1, self.img_rows, self.img_cols, 1), 1.0))
     #       predictions = predictions.eval(feed_dict={})
     #       return self._accuracy(predictions, y), self._accuracy_full(predictions, y)

    #def predict(self, X, restore_path=None):
    #    with tf.Session(graph=self.graph) as session:
    #        tf.initialize_all_variables().run()
     #       if restore_path is not None:
     #           self.saver.restore(session, restore_path)
     ##       predictions = tf.pack([tf.nn.softmax(self._model(X.reshape(-1, self.img_rows, self.img_cols, 1), 1.0)[i]) for i in range(self.max_nb_digits)])
     #       predictions = predictions.eval(feed_dict={})
     #       return np.argmax(predictions, 2).T