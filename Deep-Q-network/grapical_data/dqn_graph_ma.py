import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

import matplotlib.finance as finance
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime


day_len = 10    # numbers of days for every data
ticker = '2330.TW'  # stock TSMC


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
_stock_data = get_stock(ticker, datetime.date(2011, 1, 1), datetime.date(2015, 8, 17))

# label ( rise over 10% : [0,1], fall over 5% : [1,0], neither : [0,0] )
def get_label(stock_data):
    my_label = np.zeros((len(stock_data)), dtype = np.int_)
    for i in xrange(0, len(stock_data)):
        my_label[i] = 0
        for j in xrange(i+1, len(stock_data)):
            if (float(stock_data[j]) >= 1.10*float(stock_data[i])):
                my_label[i] = 1
                break
            if (float(stock_data[j]) <= 0.95*float(stock_data[i])):
                my_label[i] = 2
                break
    return my_label
    
my_stock_label = get_label(_stock_data)
my_train_label = my_stock_label[:len(train)]
my_test_label = my_stock_label[len(train):]


# load the figure
def get_image(file_dir):
    img = mpimg.imread(file_dir)
    return img

image = []
for i in xrange(0, len(train)-day_len-1):
    file_dir = "/home/carine/Desktop/2330_2011-2016/MA/train/train_" + str(i) + ".png"
    image.append(get_image(file_dir))
my_train = np.asarray(image)

image = []
for i in xrange(0, len(test)-day_len-1):
    file_dir = "/home/carine/Desktop/2330_2011-2016/MA/test/test_" + str(i) + ".png"
    image.append(get_image(file_dir))
my_test = np.asarray(image)


# recolor the figure from RGBA to (0,1)
def image_initialize(data):
    my_image = np.ndarray(shape=(len(data), 80, 80, 2), dtype = np.int_)
    for i in xrange (0, len(data)):
        for j in xrange(0, 80):
            for k in xrange(0, 80):
                my_image[i][j][k][0] = 0
                my_image[i][j][k][1] = 0
    return my_image

my_train_image = image_initialize(my_train)
my_test_image = image_initialize(my_test)

def image_recolor(data, my_image):
    for i in xrange (0, len(data)):
        for j in xrange (0, 80):
            for k in xrange (0, 80):
                if data[i][j][k][0] != 1 and data[i][j][k][1] != 1:
                    my_image[i][j][k][0] = 1
                if data[i][j][k][1] != 1 and data[i][j][k][2] != 1:
                    my_image[i][j][k][1] = 1
    return my_image

my_train_image = image_recolor(my_train, my_train_image)
my_test_image = image_recolor(my_test, my_test_image)


my_stock_train = np.ndarray(shape=(len(my_train_label)-day_len-1, day_len), dtype = np.int_)
my_stock_test = np.ndarray(shape=(len(my_test_label)-day_len-1, day_len), dtype = np.int_)

#initialize
for i in xrange(0, len(my_stock_train)):
    for j in xrange(0, day_len):
        my_stock_train[i][j] = 0
            
for i in xrange(0, len(my_stock_test)):
    for j in xrange(0, day_len):
        my_stock_test[i][j] = 0

for i in xrange(0, len(my_stock_train)):
    for j in xrange(0, day_len):
        my_stock_train[i,j] = my_train_label[i+j+2]
        
for i in xrange(0, len(my_stock_test)):
    for j in xrange(0, day_len):
        my_stock_test[i,j] = my_test_label[i+j+2]


class TWStock():
    def __init__(self, image_data, stock_label):
        self.image_data = image_data
        self.stock_label = stock_label
        self.stock_index = 0
    
    def render(self):
        return 
    
    def reset(self):
        self.stock_index = 0
        return self.image_data[self.stock_index]
    

    def step(self, action): 
        self.stock_index += 1
       
        if (self.stock_label[self.stock_index][day_len-1] == 1):
            action_reward = -10
        elif (self.stock_label[self.stock_index][day_len-1] == 2):
            action_reward = 5
        else:
            action_reward = 0
        
        
        if (action == 1):
            if (self.stock_label[self.stock_index][day_len-1] == 1):
                action_reward = 10
            elif (self.stock_label[self.stock_index][day_len-1] == 2):
                action_reward = -20
            else:
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
    
    
        self.state_dim = [None, 80, 80, 2]
        self.action_dim = 2
        self.label_dim = 149

        self.create_Q_network()
        self.create_training_method()

        self.t_session = tf.InteractiveSession()
        self.t_session.run(tf.initialize_all_variables())

    def create_Q_network(self):
        # input layer
        self.state_input = tf.placeholder(tf.float32,[None ,80, 80, 2])
        
        # network weights
        W_conv1 = self.weight_variable([8, 8, 2, 32])
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
        h_conv1 = tf.nn.relu(self.conv2d(self.state_input, W_conv1, 2) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 2) + b_conv3)
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
        # Q Value layer
        self.Q_value = tf.matmul(h_fc1, W_fc2) + b_fc2
        print self.Q_value.get_shape()
    def create_training_method(self):
        self.action_input = tf.placeholder(tf.float32,[None,self.action_dim])
        # one hot presentation
        self.y_input = tf.placeholder(tf.float32,[None])
        Q_action = tf.reduce_sum(tf.mul(self.Q_value,self.action_input),reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)
        self.prediction = tf.equal(tf.argmax(self.Q_value, 1), tf.argmax(self.y_input, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.prediction, "float"))

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
            _action = random.randint(0,self.action_dim - 1)
        else:
            _action = np.argmax(Q_value)
        str_action = str(_action)
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000
        return _action, str_action

    def action(self,state):
        _action = np.argmax(self.Q_value.eval(feed_dict = {
          self.state_input:[state]})[0])
        str_action = str(_action)
        return _action, str_action


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
    #env = gym.make(ENV_NAME)
    env = TWStock(my_train_image,my_stock_train) 
    agent = DQN(env)
    #sess,merged,R,writer = agent.get_summ()
    train_output = ""
    test_output = ""
    action_output = ""
    train_action = ""
    
    print 'Start!'
    for episode in xrange(EPISODE):
        

        
        # initialize task
        state = env.reset()
        train_reward = 0
        # Train
        for step in xrange(STEP):
            
            action, str_action = agent.egreedy_action(state) # e-greedy action for trai
            train_action += str_action + ","
            
            next_state,reward,done,_ = env.step(action)
            train_reward += reward
            # Define reward for agent
            reward_agent = -1 if done else 0.1
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        train_action += '\n'
        train_output += 'training episode:' + str(episode) + 'Evalutaion Average Reward:' + str(train_reward) + '\n'
        if episode % 1 == 0:
            print 'training episode:', episode, 'Evalutaion Average Reward:', train_reward

        # Test every 100 episodes
        if episode % 1 == 0:
            env_test = TWStock(my_test_image, my_stock_test)
            total_reward = 0

            for i in xrange(TEST):
                state = env_test.reset()

                for j in xrange(STEP):
                    env_test.render()
                    action, str_action = agent.action(state)   # direct action for test
                    action_output += str_action + ","
                    state,reward,done,_ = env_test.step(action)
                    total_reward += reward
                    if done:
                        break
            action_output += '\n'
            ave_reward = total_reward/TEST
            test_output += 'testing episode: ' + str(episode) + ' Evaluation Average Reward: ' + str(ave_reward) + '\n'
            print 'testing episode: ',episode,'Evaluation Average Reward:',ave_reward
            
        # save result
        file_dir = '/home/carine/tensorflow/result/'
        f_train = open(file_dir + "DQN_MA_LineChart_train.txt","w")
        f_train.write(train_output)
        f_train.close()
        f_test = open(file_dir + "DQN_MA_LineChart_test.txt","w")
        f_test.write(test_output)
        f_test.close()
        f_train_action = open(file_dir + "DQN_MA_LineChart_train_action.txt","w")
        f_train_action.write(train_action)
        f_train_action.close()
        f_action = open(file_dir + "DQN_MA_LineChart_action.txt","w")
        f_action.write(action_output)
        f_action.close()


if __name__ == '__main__':
    main()
        
