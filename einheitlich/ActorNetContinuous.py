# Description: This file contains the ActorNetContinuous class, which represents the stochastic-policy (policy = weights from the nn) for continuous action spaces.
import tensorflow as tf


# # first implementation
# # *************************************************MODIFIED************************************************** 
# class ActorNetContinuous(tf.keras.Model):    # represents/ approximates the stochastic-policy (policy = weights from the nn)
#     def __init__(self, units=(400, 300), n_actions=1, **kwargs):    # input = observation shape(batchsize, observation_shape) -> same as in discrete action space 
#         super(ActorNetContinuous, self).__init__(**kwargs)
#         self._layers = []
#         n_outputs = n_actions*2 # one continuous output distribution contains values std and mean for gaussian 
#         for i, u in enumerate(units):
#             self._layers.append(tf.keras.layers.Dense(u, activation='relu'))
#         self._layers.append(tf.keras.layers.Dense(n_outputs, activation = 'tanh'))   # output = ?? shape(batchsize, n_outputs)
#         # modify output dimension to n_actions * 2 (= 1 for MountainCarCont) -> output is now std and mean of continuous gaussian distribution
#         # modify output layer activation function -> use no activation/ linear activation a(x) = x to output the estimated values directly
#         # if custom clipping is necessary, use tanh as output activation function to clip[-1,1]
#         # in discrete action space 'softmax' exp(x) / tf.reduce_sum(exp(x)) calculates the value of each output vector in that way, the output can be interpreted as a discrete probability distribution (sum vectors = 1)
        
#     # forward pass through the network
#     def call(self, inputs, **kwargs):
#         outputs = inputs
#         for l in self._layers:
#             outputs = l(outputs)
#         # if last layer is reached, prepare the output to return mean and std
#         mean, log_std = tf.split(outputs, 2, axis=-1)  # Split the output(Tensor shape(batchsize, n_outputs)) into 2 tensors (mean and log_std (ln!)) along the last axis(collums)
#         print('mean', mean,'log_std',log_std)
#         return mean, log_std



# modified trys
# https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c
# *************************************************MODIFIED************************************************** 
# class ActorNetContinuous(tf.keras.Model):    # represents/ approximates the stochastic-policy (policy = weights from the nn)
#     def __init__(self, units=(400, 300), n_actions=None, **kwargs):    # input = observation shape(batchsize, observation_shape) -> same as in discrete action space 
#         super(ActorNetContinuous, self).__init__(**kwargs)
#         self._hiddenlayers = []
#         for i, u in enumerate(units):
#             self._hiddenlayers.append(tf.keras.layers.Dense(u, activation= tf.nn.leaky_relu)) # eyplanation for leaky_relu see below
#         # output layer (parallel!) kernel_initializer=tf.keras.initializers.glorot_normal()
#         self._mean = tf.keras.layers.Dense(n_actions, activation = 'tanh')  # linear activation f(x) = x -> direct pass
#         self._std = tf.keras.layers.Dense(n_actions, activation = tf.nn.softplus) # softplus activation f(x) = log(exp(x) + 1) -> only positive returns
#         # output = ?? shape(batchsize, n_outputs)
#         # modify output layer activation function -> use no activation/ linear activation a(x) = x to output the estimated values directly
#         # in discrete action space 'softmax' exp(x) / tf.reduce_sum(exp(x)) calculates the value of each output vector in that way, the output can be interpreted as a discrete probability distribution (sum vectors = 1)

#         # explanation to relu vs. leaky_relu:
#         # relu: f(x) = (0,x) --> returns x for positive input, 0 for negative input --> 0 could lead to numerical instability --> nan error
#         # leaky_relu: f(x) = (ax, x) --> returns x for positive input, a*x for negative input (default a = 0.3) --> can handle negative inputs
#         # weights to zero, because the inputs are to often negative "deaktiavates" some units which can lead to major stability problems
        
#     # forward pass through the network
#     def call(self, inputs, **kwargs):
#         outputs = inputs
#         for l in self._hiddenlayers:
#             outputs = l(outputs)
#         # if last layer is reached, prepare the output to return mean and std
#         mean = self._mean(outputs)
#         log_std = tf.math.log(self._std(outputs)) # log for numerical stability
#         #print('mean', mean,'log_std',log_std)
#         return mean, log_std

##------------------------try learnable std, instead of predicting it via the nn---------------------------

class ActorNetContinuous(tf.keras.Model):    # represents/ approximates the stochastic-policy (policy = weights from the nn)
    def __init__(self, units=(400, 300), n_actions=None, **kwargs):    # input = observation shape(batchsize, observation_shape) -> same as in discrete action space 
        super(ActorNetContinuous, self).__init__(**kwargs)
        self._hiddenlayers = []

        # Define the trainable log standard deviation
        self.log_std = tf.Variable(tf.zeros([1, n_actions], dtype=tf.float32), trainable=True)

        for i, u in enumerate(units):
            self._hiddenlayers.append(tf.keras.layers.Dense(u, activation= tf.nn.leaky_relu)) # eyplanation for leaky_relu see below
        # output layer (parallel!) kernel_initializer=tf.keras.initializers.glorot_normal()
        self._mean = tf.keras.layers.Dense(n_actions, activation = 'tanh')  
        
    # forward pass through the network
    def call(self, inputs, **kwargs):
        outputs = inputs
        for l in self._hiddenlayers:
            outputs = l(outputs)
        # if last layer is reached, prepare the output to return mean and std
        mean = self._mean(outputs)
        log_std = self.log_std
        #print('mean', mean,'log_std',log_std)
        return mean, log_std

