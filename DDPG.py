
import gym
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
sys.path.append('/media/veracrypt1/DigitalTwin/modules2import/')
from import_modules import *
from helper_functions import *
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

class Environment:
    def __init__(self,env=None,nOfa=None,nOfs=None,
                 ub=None, lb=None):
        env                  = env
        self.numberOfActions = nOfa
        self.numberOfStates  = nOfs
        self.upperBound      = ub
        self.lowerBound      = lb

        if env == 'Pendulum-v0':
            self.sim             = gym.make(env)
            self.numberOfActions = self.sim.action_space.shape[0]
            self.numberOfStates  = self.sim.observation_space.shape[0]
            self.upperBound      = self.sim.action_space.high[0]
            self.lowerBound      = self.sim.action_space.low[0]
        
        print("Size of State Space ->  {}".format(self.numberOfStates))
        print("Size of Action Space ->  {}".format(self.numberOfActions))
        print("Max Value of Action ->  {}".format(self.upperBound))
        print("Min Value of Action ->  {}".format(self.lowerBound))

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta     = theta
        self.mean      = mean
        self.std_dev   = std_deviation
        self.dt        = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt \
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        # Store x into x_prev, Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Buffer:
    def __init__(self, num_states, num_actions, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        self.batch_size      = batch_size
        self.buffer_counter  = 0
        # Instead of list of tuples as the exp.replay concept go, We use different np.arrays for each tuple element
        self.state_buffer      = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer     = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer     = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded, replacing old records
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index]      = obs_tuple[0]
        self.action_buffer[index]     = obs_tuple[1]
        self.reward_buffer[index]     = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter           = self.buffer_counter + 1

    

# This update target parameters slowly, based on rate `tau`, which is much less than one.
class Agent(Environment):
    def __init__(self,actor_network  = {'nn'          :[60,30],
                                        'activation'  :'relu',
                                        'initializer' :glorot_normal,
                                        'optimizer'   :Adam(lr=0.001)}, 
                      critic_network = {'nn'          :[16,32],
                                        'concat'      :[16,60,30],
                                        'activation'  :'relu',
                                        'initializer' :glorot_normal,
                                        'optimizer'   :Adam(lr=0.002)},
                      gamma           = 0.99,
                      tau             = 0.005,
                      buffer_capacity = 50000, 
                      batch_size      = 64,
                      environment     = None,
                      numberOfActions = None,
                      numberOfStates  = None,
                      upperBound      = None, 
                      lowerBound      = None):

        Environment.__init__(self,env=environment, nOfa=numberOfActions,
                                  nOfs=numberOfStates, ub=upperBound, lb=lowerBound)

        Buffer.__init__(self,buffer_capacity=buffer_capacity, batch_size=batch_size,
                             num_actions=self.numberOfActions, num_states=self.numberOfStates)
        
        self.actor_network   = actor_network
        self.critic_network  = critic_network
        self.gamma           = gamma
        self.tau             = tau
        self.noise           = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))
        self.observation     = (None,None,None,None)

        self.main_actor_model    = self.get_actor(name='MainActorModel')
        self.target_actor_model  = self.get_actor(name='TargetActorModel')
        self.main_critic_model   = self.get_critic(name='MainCriticModel')
        self.target_critic_model = self.get_critic(name='TargetCriticModel')
        # Making the weights equal initially
        self.target_actor_model.set_weights(self.main_actor_model.get_weights())
        self.target_critic_model.set_weights(self.main_critic_model.get_weights())
        
    def record_buffer(self):
        Buffer.record(self,self.observation)

    def get_actor(self,name='Actor'):
        # Initialize weights between -3e-3 and 3-e3
        L1_inp  = Input(shape=(self.numberOfStates,),name=name+'_Stateinput')
        L1      = Dense(self.actor_network['nn'][0], activation=self.actor_network['activation'], kernel_initializer= self.actor_network['initializer'])(L1_inp)
        L1      = BatchNormalization()(L1)
        for ii in range(1,len(self.actor_network['nn'])):
            L1 = Dense(self.actor_network['nn'][ii], activation=self.actor_network['activation'], kernel_initializer= self.actor_network['initializer'])(L1)
            L1 = BatchNormalization()(L1)
        Lout = Dense(self.numberOfActions, activation="tanh", kernel_initializer=self.actor_network['initializer'])(L1)
        # Our upper bound is 2.0 for Pendulum.
        Lout = Lout * self.upperBound
        model = Model(L1_inp, Lout)
        plot_model(model,to_file=name+'.png', show_layer_names=True,show_shapes=True)
        return model

    def get_critic(self,name='Critic'):
        # State as input
        L1_state_inp  = Input(shape=(self.numberOfStates,),name=name+'_Stateinput')
        L1_state      = Dense(self.critic_network['nn'][0], activation=self.critic_network['activation'], kernel_initializer=self.critic_network['initializer'])(L1_state_inp)
        L1_state      = BatchNormalization()(L1_state)
        for ii in range(1,len(self.critic_network['nn'])):
            L1_state  = Dense(self.critic_network['nn'][ii], activation=self.critic_network['activation'], kernel_initializer=self.critic_network['initializer'])(L1_state)
            L1_state  = BatchNormalization()(L1_state)
        # Action as input
        L1_action_inp = Input(shape=(self.numberOfActions),name=name+'_Actioninput')
        L1_action     = Dense(self.critic_network['concat'][0], activation="relu",kernel_initializer=self.critic_network['initializer'])(L1_action_inp)
        L1_action     = BatchNormalization()(L1_action)
        # Both are passed through seperate layer before concatenating
        L1            = Concatenate()([L1_state, L1_action])
        for ii in range(1,len(self.critic_network['concat'])):
            L1        = Dense(self.critic_network['concat'][ii], activation=self.critic_network['activation'],kernel_initializer=self.critic_network['initializer'])(L1)
            L1        = BatchNormalization()(L1)
        Lout          = Dense(self.numberOfActions,kernel_initializer=self.critic_network['initializer'])(L1)
        # Outputs single value for give state-action
        model         = Model([L1_state_inp, L1_action_inp], Lout)
        plot_model(model,to_file=name+'.png', show_layer_names=True,show_shapes=True)
        return model

    def policy(self,state):
        sampled_actions = tf.squeeze(self.main_actor_model(state))
        noise           = self.noise()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise
        # We make sure action is within bounds
        legal_action    = np.clip(sampled_actions, self.lowerBound, self.upperBound)
        self.action     = [np.squeeze(legal_action)]

        return self.action

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range  = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)
        # Convert to tensors
        state_batch      = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch     = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch     = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch     = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        with tf.GradientTape() as tape:
            target_actions = self.target_actor_model(next_state_batch)
            y              = reward_batch + self.gamma * self.target_critic_model([next_state_batch, target_actions])
            critic_value   = self.main_critic_model([state_batch, action_batch])
            critic_loss    = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, self.main_critic_model.trainable_variables)
        self.critic_network['optimizer'].apply_gradients(zip(critic_grad, self.main_critic_model.trainable_variables))
        
        with tf.GradientTape() as tape: 
            actions      = self.main_actor_model(state_batch)
            critic_value = self.main_critic_model([state_batch, actions])
            actor_loss   = -tf.math.reduce_mean(critic_value) # Used `-value` as we want to maximize the value given by the critic for our actions
        actor_grad = tape.gradient(actor_loss, self.main_actor_model.trainable_variables)
        self.actor_network['optimizer'].apply_gradients(zip(actor_grad, self.main_actor_model.trainable_variables))

        self.update_target()

    def update_target(self):
        new_weights = []
        target_variables = self.target_critic_model.weights
        for i, variable in enumerate(self.main_critic_model.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))

        self.target_critic_model.set_weights(new_weights)

        new_weights = []
        target_variables = self.target_actor_model.weights
        for i, variable in enumerate(self.main_actor_model.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))

        self.target_actor_model.set_weights(new_weights)


