import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

import matplotlib.finance as finance
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from yahoo_finance import Share
import datetime


day_len = 10  # numbers of days for every data
ticker = '2330.TW'  # stock TSMC
save_figure = False  # true if you want to renew figure

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch

# Hyper Parameters for main function
EPISODE = 10000  # total episode
STEP = 10000  # Step limitation in an episode, it must be equal or larger than the length of your training data
TEST = 10 #10 # The number of experiment test every 100 episode


# Get stock data with matplotlib.finance, and remove the data with zero volume
def get_stock(ticker, startdate, enddate):
    fh = finance.fetch_historical_yahoo(ticker, startdate, enddate)
    # a numpy record array with fields: (date, open, high, low, close, volume, adj_close)
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
test = get_stock(ticker, datetime.date(2016, 1, 1), datetime.date(2016, 8, 17))


# Calculate relative price
def get_relative_data(stock_data):
    relative_data = []
    for i in xrange(1, len(stock_data)):
        relative_price_change = (stock_data[i] - stock_data[i-1]) / stock_data[i-1]
        relative_data.append(relative_price_change)
    return relative_data

relative_train = get_relative_data(train)
relative_test = get_relative_data(test)


# plot and save relative price line chart
max_ylim = max(max(relative_train), max(relative_test))
min_ylim = min(min(relative_train), min(relative_test))

def save_pic(data, filename):
    for i in xrange (0, len(data)-day_len):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(1, 1)
        ax.plot([i, i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8, i+9], [data[i], data[i+1], data[i+2], data[i+3], data[i+4], data[i+5], data[i+6], data[i+7], data[i+8], data[i+9]])
        ax.set_ylim([min_ylim, max_ylim])
        plt.axis('off')
        fig.savefig('/home/carine/Desktop/2330/relative_price/'+filename+'/'+filename+'_'+str(i)+'.png', dpi=80)
        fig.clear()
        plt.close(fig)

if save_figure == True:
    save_pic(relative_train, "train")
    save_pic(relative_test, "test")


# load the figure
def get_image(file_dir):
    img = mpimg.imread(file_dir)
    return img

image = []
for i in xrange(0, len(relative_train)-day_len):
    file_dir = "/home/carine/Desktop/2330/relative_price/train/train_" + str(i) + ".png"
    image.append(get_image(file_dir))
my_train = np.asarray(image)

image = []
for i in xrange(0, len(relative_test)-day_len):
    file_dir = "/home/carine/Desktop/2330/relative_price/test/test_" + str(i) + ".png"
    image.append(get_image(file_dir))
my_test = np.asarray(image)

# recolor the figure from RGBA to (0,1)
my_train_image = np.ndarray(shape=(len(my_train), 80, 80), dtype = np.int_)
for i in xrange (0, len(my_train)):
    for j in xrange(0, 80):
        for k in xrange(0, 80):
            if my_train[i][j][k][0] != 1 or my_train[i][j][k][1] != 1 or my_train[i][j][k][2] != 1 or my_train[i][j][k][3] != 1:
                my_train[i][j][k][0] = 0
            my_train_image[i][j][k] = my_train[i][j][k][0]

my_test_image = np.ndarray(shape=(len(my_test), 80, 80), dtype = np.int_)
for i in xrange (0, len(my_test)):
    for j in xrange(0, 80):
        for k in xrange(0, 80):
            if my_test[i][j][k][0] != 1 or my_test[i][j][k][1] != 1 or my_test[i][j][k][2] != 1 or my_test[i][j][k][3] != 1:
                my_test[i][j][k][0] = 0
            my_test_image[i][j][k] = my_test[i][j][k][0]


my_stock_train = np.zeros((len(relative_train)-day_len, day_len), dtype=np.float)
my_stock_test = np.zeros((len(relative_test)-day_len, day_len), dtype=np.float)
for i in xrange(0, len(my_stock_train)):
    for j in xrange(0, day_len):
        my_stock_train[i,j] = train[i+j+1]

for i in xrange(0, len(my_stock_test)):
    for j in xrange(0, day_len):
        my_stock_test[i,j] = test[i+j+1]


class TWStock():
    def __init__(self, image_data, stock_price):
        self.image_data = image_data
        self.stock_price = stock_price
        self.stock_index = 0
    
    def render(self):
        return 
    
    def reset(self):
        self.stock_index = 0
        return self.image_data[self.stock_index]
    
    # 0:do nothing, 1:buy and then sell, 2:sell and then buy
    def step(self, action): 
        self.stock_index += 1
        action_reward = self.stock_price[self.stock_index][day_len-1] - self.stock_price[self.stock_index][day_len-2] 
        if (action == 0):
            action_reward = 0
        #if (action == 2):
        #    action_reward = -1 * action_reward

        stock_done = False
        if self.stock_index >= len(self.image_data)-1:
            stock_done = True
        else:
            stock_done = False
        return self.image_data[self.stock_index], action_reward, stock_done, 0


class DQN():
    # DQN Agent
    def __init__(self, env):
        # init experience replay
        self.replay_buffer = deque()

        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
    
        self.state_dim = [None, 80, 80]
        self.action_dim = 2

        self.create_Q_network()
        self.create_training_method()

        self.t_session = tf.InteractiveSession()
        self.t_session.run(tf.initialize_all_variables())


    def create_Q_network(self):
        # input layer
        self.state_input = tf.placeholder(tf.float32,[None ,80, 80])
        x_image = tf.reshape(self.state_input, [-1, 80, 80, 1]) 
        
        # network weights
        W_conv1 = self.weight_variable([8, 8, 1, 32])
        b_conv1 = self.bias_variable([32])
        
        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])
        
        
        W_fc1 = self.weight_variable([1600, 512])
        b_fc1 = self.bias_variable([512])
        
        W_fc2 = self.weight_variable([512, self.action_dim])
        b_fc2 = self.bias_variable([self.action_dim])
        
        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1, 4) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
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
    
    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")


def main():

    # initialize OpenAI Gym env and dqn agent
    env = TWStock(my_train_image,my_stock_train) 
    agent = DQN(env)
    
    
    print 'Start!'
    for episode in xrange(EPISODE):
    
        # initialize task
        state = env.reset()
        train_reward = 0
        # Train
        for step in xrange(STEP):
            
            action = agent.egreedy_action(state) # e-greedy action for train

            next_state,reward,done,_ = env.step(action)
            train_reward += reward
            # Define reward for agent
            reward_agent = -1 if done else 0.1
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break  
        if episode % 10 == 0:
            print 'training episode:', episode, 'Evalutaion Average Reward:', train_reward

        # Test every 10 episodes
        if episode % 10 == 0:
            env_test = TWStock(my_test_image, my_stock_test)
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
            print 'testing episode: ',episode,'Evaluation Average Reward:',ave_reward


if __name__ == '__main__':
    main()
