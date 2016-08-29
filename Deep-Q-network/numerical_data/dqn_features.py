import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

import matplotlib.finance as finance
import matplotlib.mlab as mlab
import datetime


day_len = 10    # numbers of days for every data
ticker = '2330.TW'  # stock TSMC

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch

# Hyper Parameters for main function
EPISODE = 10000  # total episode
STEP = 10000  # Step limitation in an episode, it must be equal or larger than the length of your training data
TEST = 10 #10 # The number of experiment test every 100 episode


# Get stock data with matplotlib.finance, and remove the data with zero volume
def get_stock(ticker, startdate, enddate):
    fh = finance.fetch_historical_yahoo(ticker, startdate, enddate)
    # a numpy record array with fields: date, open, high, low, close, volume, adj_close)
    r = mlab.csv2rec(fh)
    fh.close()
    r.sort()
    print 'the length of data:', len(r.close)
    get_stock_data = []
    for i in xrange(0, len(r.close)-1):
        if (r.volume[i] != 0):
            get_stock_data.append(r.close[i].tolist())
    print 'after removing the datas with zero volume, the length of data:', len(get_stock_data)
    return get_stock_data

train = get_stock(ticker, datetime.date(2011, 1, 1), datetime.date(2015, 12, 31))
test = get_stock(ticker, datetime.date(2016, 1, 1), datetime.date.today())


# Calculate MA5, MA20, MA5-MA20
def get_moving_average(x, n, type_str):
    #compute an n period moving average.
    #type is 'simple' | 'exponential'
    my_list=[]
    x = np.asarray(x)
    if type_str == 'simple':
        weights = np.ones(n)
    elif type_str == 'exponential':
        weights = np.exp(np.linspace(-1., 0., n))
    elif type_str == 'weight':
        weights = np.flipud(np.arange(1,n+1, dtype=float))
    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    for i in xrange (0, len(a), 1):
        my_list.append(np.array_str(a[i]))
    return my_list

MA5_train = get_moving_average(train, 5, type_str='simple')
MA5_test = get_moving_average(test, 5, type_str='simple')
MA20_train = get_moving_average(train, 20, type_str='simple')
MA20_test = get_moving_average(test, 20, type_str='simple')
MA_diff_train = []
for i in xrange (0, len(MA5_train)):
    MA_diff_train.append(float(MA5_train[i]) - float(MA20_train[i]))
MA_diff_test = []
for i in xrange (0, len(MA5_test)):
    MA_diff_test.append(float(MA5_test[i]) - float(MA20_test[i]))


# Calculate RSI9, RSI15
def relative_strength(prices, n):
    my_list = []
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1. + rs)

    for i in range(0, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n - 1) + upval)/n
        down = (down*(n - 1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)
        
        my_list.append(np.array_str(rsi[i]))
        
    return my_list

RSI9_train = relative_strength(train, 9)
RSI9_test = relative_strength(test, 9)
RSI15_train = relative_strength(train, 15)
RSI15_test = relative_strength(test, 15)


# training data and testing data 
# (close price, MA5, MA20, MA5-MA20, RSI9, RSI15)
my_train = np.zeros((len(train)-day_len, day_len*6), dtype = np.float)
my_test = np.zeros((len(test)-day_len, day_len*6), dtype=np.float)

for i in range(0, len(my_train)):
    for j in range(0, day_len):
        my_train[i,j] = train[i+j]
        my_train[i,10+j] = MA5_train[i+j]
        my_train[i,20+j] = MA20_train[i+j]
        my_train[i,30+j] = MA_diff_train[i+j]
        my_train[i,40+j] = RSI9_train[i+j]
        my_train[i,50+j] = RSI15_train[i+j]
        
for i in range(0, len(my_test)):
    for j in range(0, day_len):
        my_test[i,j] = test[i+j]
        my_test[i,10+j] = MA5_test[i+j]
        my_test[i,20+j] = MA20_test[i+j]
        my_test[i,30+j] = MA_diff_test[i+j]
        my_test[i,40+j] = RSI9_test[i+j]
        my_test[i,50+j] = RSI15_test[i+j]

class TWStock():
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.stock_index = 0
    
    def render(self):
        return 
    
    def reset(self):
        self.stock_index = 0
        return self.stock_data[self.stock_index]

    def step(self, action): 
        self.stock_index += 1
        action_reward = self.stock_data[self.stock_index][day_len-1] - self.stock_data[self.stock_index][day_len-2] 
        if (action == 0):
            action_reward = 0
        if (action == 2):
            action_reward = -1 * action_reward

        stock_done = False
        if self.stock_index >= len(self.stock_data)-1:
            stock_done = True
        else:
            stock_done = False
        return self.stock_data[self.stock_index], action_reward, stock_done, 0


class DQN():
    # DQN Agent
    def __init__(self, env):
        # init experience replay
        self.replay_buffer = deque()

        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
    
        self.state_dim = day_len*6
        self.action_dim = 3

        self.create_Q_network()
        self.create_training_method()

        # create session
        self.t_session = tf.InteractiveSession()
        self.t_session.run(tf.initialize_all_variables())


    def create_Q_network(self):
        # input layer
        self.state_input = tf.placeholder(tf.float32,[None,self.state_dim])
        x_image = tf.reshape(self.state_input, [-1, 6, day_len, 1])
        
        # network weights
        W_conv1 = self.weight_variable([6, 10, 1, 20])
        b_conv1 = self.bias_variable([20])
        
        W_conv2 = self.weight_variable([6, 10, 20, 10])
        b_conv2 = self.bias_variable([10])
        
        W_conv3 = self.weight_variable([6, 10, 10, 10])
        b_conv3 = self.bias_variable([10])
        
        W_fc1 = self.weight_variable([self.state_dim*10, 40])
        b_fc1 = self.bias_variable([40])
        
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2) + b_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3) + b_conv3)
        h_conv3_flat = tf.reshape(h_conv3, [-1, self.state_dim*10])
        
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
        W_fc2 = tf.Variable(tf.truncated_normal([40, self.action_dim], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[self.action_dim]))

        # Q Value layer
        self.Q_value = tf.matmul(h_fc1, W_fc2) + b_fc2
        

    def create_training_method(self):
        self.action_input = tf.placeholder(tf.float32,[None,self.action_dim])
        # one hot presentation
        self.y_input = tf.placeholder(tf.float32,[None])
        Q_action = tf.reduce_sum(tf.mul(self.Q_value,self.action_input),reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)


    def perceive(self,state,action,reward,next_state,done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))

        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()


    def train_Q_network(self):
        self.time_step += 1

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})

        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
          self.y_input: y_batch,
          self.action_input: action_batch,
          self.state_input: state_batch
          })
        

    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict = {
          self.state_input:[state]})[0]
        if random.random() <= self.epsilon:
            return random.randint(0,self.action_dim - 1)
        else:
            return np.argmax(Q_value)

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000


    def action(self,state):
        return np.argmax(self.Q_value.eval(feed_dict = {
          self.state_input:[state]})[0])


    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)
    
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")


def main():

    # initialize OpenAI Gym env and dqn agent
    env = TWStock(my_train) 
    agent = DQN(env)

    print 'Start!'
    for episode in xrange(EPISODE):
    
        # initialize task
        state = env.reset()

        # Train
        for step in xrange(STEP):
            action = agent.egreedy_action(state) # e-greedy action for trai

            next_state,reward,done,_ = env.step(action)

            # Define reward for agent
            reward_agent = -1 if done else 0.1
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
    

        # Test every 100 episodes
        if episode % 10 == 0:
            env_test = TWStock(my_test)
            total_reward = 0

            for i in xrange(TEST):
                state = env_test.reset()

                for j in xrange(STEP):
                    env_test.render()
                    action = agent.action(state)   # direct action for test
                    state,reward,done,_ = env_test.step(action)
                    total_reward += reward
                    if done:
                        break

            ave_reward = total_reward/TEST
            print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
            

if __name__ == '__main__':
    main()
