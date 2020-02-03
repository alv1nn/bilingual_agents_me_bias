import tensorflow as tf


class RSASpeaker0(tf.keras.Model):
    """ Literal speaker: selects the message with the highest lexicon value given a state so its policy is simply the
        row in the lexicon belonging to the respective state.

        Inherits from the keras.Model class, so it must implement init and call. """

    def __init__(self, n_states, n_messages, lexicon):
        """ Constructor.

        :param n_states: number of states
        :param n_messages: number of messages
        :param lexicon: a trainable randomly initialized matrix of the shape n_states x n_messages, encodes how appropriate
                        a message is for a state (with all values >0).
        """
        super(RSASpeaker0, self).__init__()
        self.n_states = n_states
        self.n_messages = n_messages
        self.lexicon = lexicon

    def call(self, inputs, **kwargs):
        """ Call function: returns the message policies for the input states.

        :param inputs:  states encoded as matrices where the row of the state in question is filled with ones 
                        and all other entries are zero, shape (batch_size, n_states, n_messages)
        :param kwargs
        :return:        returns the policies: the distributions over messages given the input states (not normalized),
                        shape (batch_size, n_messages)
        """
        x = inputs * self.lexicon               # shape batch_size x n_states x n_messages
        x = tf.reduce_sum(x, axis=1)            # shape batch_size x n_messages
        # Note: x is not normalized by the row sum as tf.random.categorical (see get_messages) takes the un-normalized
        # log probabilities as input. Otherwise a "return x / tf.reshape(tf.reduce_sum(x, axis=1), (-1, 1))" would be
        # necessary.
        return x                                # shape batch_size x n_messages

    def get_messages(self, data):
        """ Calculates the message policies for given states and then samples the messages from this distribution.

        :param data:    states encoded as matrices where the row of the state in question is filled with 
                        ones and all other entries are zero, shape (batch_size, n_states, n_messages)
        :return:        messages, so one-hot vectors encoding the selected messages for the given states,
                        shape (batch_size, n_messages)
        """
        policy = self.call(data)
        log_policy = tf.math.log(policy)
        actions = tf.squeeze(tf.one_hot(tf.random.categorical(log_policy, 1), depth=self.n_messages))
        return policy, actions


class RSASpeaker1(tf.keras.Model):
    """ Pragmatic speaker: reasons about a literal listener and selects the message that will most likely make the
        literal listener select the right state.

        Inherits from the keras.Model class, so it must implement init and call. """

    def __init__(self, n_states, n_messages, lexicon, alpha=1.):
        """ Constructor.

        :param n_states: number of states
        :param n_messages: number of messages
        :param lexicon: a trainable randomly initialized matrix of the shape n_states x n_messages, encodes how
                        appropriate a message is of a state (with all values >0).
        :param alpha:   rationality parameter, positive real value. alpha is similar to a temperature parameter 
                        in a softmax function. if alpha=0 choices are random between the messages regardless of 
                        their truth values, if alpha=1 the exact probabilities are used to sample an action. 
        """
        super(RSASpeaker1, self).__init__()
        self.n_states = n_states
        self.n_messages = n_messages
        self.lexicon = lexicon
        self.alpha = alpha

    def call(self, inputs, **kwargs):
        """ Call function: returns the message policies for the input states.

        :param inputs:  states encoded as matrices where the row of the state in question is filled with ones and all other 
                        entries are zero, shape (batch_size, n_states, n_messages)
        :param kwargs
        :return:        returns the policies: the distribution over messages given the input states (not normalized),
                        shape (batch_size, n_messages)
        """
        filtered_input = inputs * self.lexicon
        numerator = tf.reduce_sum(filtered_input, axis=1)                    # shape (batch_size, n_messages)
        denominator = tf.reduce_sum(self.lexicon, axis=0)                    # shape (n_messages,)
        message_utils = tf.math.pow(numerator/denominator, self.alpha)       # shape (batch_size, n_messages)
        return message_utils

    def get_messages(self, data):
        """ Calculates the message policies for given states and then samples the messages from these distributions.

        :param data:    states encoded as matrices where the row of the state in question is filled with ones and all 
                        other entries are zero, shape (batch_size, n_states, n_messages)
        :return:        messages, so one-hot vectors encoding the selected messages for the given states,
                        shape (batch_size, n_messages)
        """
        policy = self.call(data)
        log_probabilities = tf.math.log(policy)
        actions = tf.squeeze(tf.one_hot(tf.random.categorical(log_probabilities, 1), depth=self.n_messages))
        return policy, actions


class RSAListener0(tf.keras.Model):
    """ Literal listener: similar to the literal speaker it selects a state given a message based on the truth of the
        message for the state. So its policy is simply the column of the lexicon belonging to the state.

        Inherits from the keras.Model class, so it must implement init and call."""

    def __init__(self, n_states, n_messages, lexicon):
        """ Constructor.

        :param n_states: number of states
        :param n_messages: number of messages
        :param lexicon: a trainable randomly initialized matrix of the shape n_states x n_messages, encodes 
                        how appropriate a message is for a state (with all values >0).
        """
        super(RSAListener0, self).__init__()
        self.n_states = n_states
        self.n_messages = n_messages
        self.lexicon = lexicon

    def call(self, inputs, **kwargs):
        """ Call function: returns the state / or selection policies for the input messages.

        :param inputs:  messages encoded as matrices where the row of the message in question is filled with ones
                        and all other entries are zero, shape (batch_size, n_messages, n_states)
        :param kwargs
        :return:        returns the policies: the distributions over states given the input messages (not normalized),
                        shape (batch_size, n_messages)
        """
        x = inputs * tf.transpose(self.lexicon)     # shape batch_size * n_messages * n_states
        x = tf.reduce_sum(x, axis=1)                # shape batch_size * n_messages
        # The same note as for the RSASpeaker0 holds: a normalization of the probabilities is not necessary as
        # tf random categorical uses the un-normalized log probabilities.
        return x

    def get_states(self, data):
        """ Calculates the state policies for given messages and then samples the states from these distributions.

        :param data:    states encoded as matrices where the row of the state in question is filled with ones 
                        and all other entries are zero, shape (batch_size, n_states, n_messages)
        :return:        states, so one-hot vectors encoding the selected states for the given messages,
                        shape (batch_size, n_messages)
        """
        policy = self.call(data)
        log_probabilities = tf.math.log(policy)
        actions = tf.squeeze(tf.one_hot(tf.random.categorical(log_probabilities, 1), depth=self.n_states))
        return policy, actions


class RSAListener1(tf.keras.Model):
    """ Pragmatic listener: reasons about a pragmatic speaker and selects the state that was most likely intended
        by that speaker.

        Inherits from the keras.Model class, so it must implement init and call. """

    def __init__(self, n_states, n_messages, lexicon, alpha=1.):
        """ Constructor.

        :param n_states: number of states
        :param n_messages: number of messages
        :param lexicon: a trainable randomly initialized matrix of the shape n_states x n_messages, encodes how 
                        appropriate a message is of a state (with all values >0).
        :param alpha:   rationality parameter, positive real value. alpha is similar to a temperature parameter 
                        in a softmax function. if alpha=0 choices are random between the messages regardless of 
                        their truth values, if alpha=1 the exact probabilities are used to sample an action.
        """
        super(RSAListener1, self).__init__()
        self.n_states = n_states
        self.n_messages = n_messages
        self.lexicon = lexicon
        self.alpha = alpha

    def call(self, inputs, **kwargs):
        """ Call function: returns the state/selection policies for the input messages.

        :param inputs:  messages encoded as matrices where the row of the message in question is filled with ones,
                        and all other entries are zero, shape (batch_size, n_messages, n_states)
        :param kwargs
        :return:        returns the policies: the distribution over states given the input messages (not normalized),
                        shape (batch_size, n_messages)
        """

        filtered_input = inputs * tf.transpose(self.lexicon)
        filtered_input = tf.reduce_sum(filtered_input, axis=1)
        # note that numberator and denominator refers to the fraction in the numerator of the RSA model
        numerator = filtered_input / (tf.expand_dims(tf.reduce_sum(filtered_input, axis=1), axis=1))
        numerator = tf.math.pow(numerator, self.alpha)
        denominator_inside_sum = tf.math.pow(self.lexicon / tf.reduce_sum(self.lexicon, axis=0), self.alpha)
        denominator = tf.reduce_sum(denominator_inside_sum, axis=1)
        state_utils = numerator/denominator
        return state_utils

    def get_states(self, data):
        """ Calculates the state policies for given messages and then samples the states from these distributions.

        :param data:    states encoded as matrices where the row of the state in question is filled with ones 
                        and all other entries are zero, shape (batch_size, n_states, n_messages)
        :return:        states, so one-hot vectors encoding the selected states for the given messages,
                        shape (batch_size, n_messages)
        """
        policy = self.call(data)
        log_probabilities = tf.math.log(policy)
        actions = tf.squeeze(tf.one_hot(tf.random.categorical(log_probabilities, 1), depth=self.n_states))
        return policy, actions
