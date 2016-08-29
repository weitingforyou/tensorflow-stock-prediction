# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Distributed MNIST training and validation, with model replicas.
A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on two parameter servers (ps), while the
ops are defined on a worker node. The TF sessions also run on the worker
node.
Multiple invocations of this script can be done in parallel, with different
values for --worker_index. There should be exactly one invocation with
--worker_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.
The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import re
import tempfile
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import layers
import tensorflow.contrib.learn as skflow

flags = tf.app.flags
flags.DEFINE_string("data_dir", "./MNIST_data",
                    "Directory for storing mnist data")
flags.DEFINE_boolean("download_only", False,
                     "Only perform downloading of data; Do not proceed to "
                     "session preparation, model definition or training")
#air
flags.DEFINE_string("job_name", "","One of 'ps' or 'worker'")


'''flags.DEFINE_string("ps_hosts", "10.81.103.124:7777","List of hostname:port for ps jobs."
		    "This string should be the same on every host!!")
flags.DEFINE_string("worker_hosts", "10.81.103.124:2224,10.81.103.122:2222,10.81.103.118:2218,10.81.103.119:2219,10.81.103.120:2220,10.81.103.121:2221","List of hostname:port for worker jobs."
		    "This string should be the same on every host!!")'''


flags.DEFINE_string("ps_hosts", "10.81.103.122:7777","List of hostname:port for ps jobs."
		    "This string should be the same on every host!!")
flags.DEFINE_string("worker_hosts", "10.81.103.122:2222,  10.81.103.122:2226","List of hostname:port for worker jobs."
		    "This string should be the same on every host!!")


#air

flags.DEFINE_integer("worker_index", None,
                     "Ps task index or worker task index, should be >= 0. worker_index=0 is "
                     "the master worker task that performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_workers", 2,
                     "Total number of workers (must be >= 1)")
flags.DEFINE_integer("num_parameter_servers", 1,
                     "Total number of parameter servers (must be >= 1)")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
#flags.DEFINE_integer("grpc_port", 5555,
#                     "TensorFlow GRPC port")
flags.DEFINE_integer("hidden_units", 1024,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("train_steps", 150,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_string("worker_grpc_url", None,
                    "Worker GRPC URL (e.g., grpc://1.2.3.4:2222, or "
                    "grpc://tf-worker0:2222)")
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
FLAGS = flags.FLAGS


IMAGE_PIXELS = 28

PARAM_SERVER_PREFIX = "tf-ps"  # Prefix of the parameter servers' domain names
WORKER_PREFIX = "tf-worker"  # Prefix of the workers' domain names


def get_device_setter(num_parameter_servers, num_workers):
  """Get a device setter given number of servers in the cluster.
  Given the numbers of parameter servers and workers, construct a device
  setter object using ClusterSpec.
  Args:
    num_parameter_servers: Number of parameter servers
    num_workers: Number of workers
  Returns:
    Device setter object.
  """
  # air
  '''
  ps_spec = []
  for j in range(num_parameter_servers):
    ps_spec.append("%s%d:%d" % (PARAM_SERVER_PREFIX, j, FLAGS.grpc_port))

  worker_spec = []
  for k in range(num_workers):
    worker_spec.append("%s%d:%d" % (WORKER_PREFIX, k, FLAGS.grpc_port))

  cluster_spec = tf.train.ClusterSpec({
      "ps": ps_spec,
      "worker": worker_spec})
  '''
  ps_hosts = re.findall(r'[\w\.:]+', FLAGS.ps_hosts)
  worker_hosts = re.findall(r'[\w\.:]+', FLAGS.worker_hosts)

  assert num_parameter_servers == len(ps_hosts)
  assert num_workers == len(worker_hosts)

  cluster_spec = tf.train.ClusterSpec({"ps":ps_hosts,"worker":worker_hosts})
  
  # air
  
  # Get device setter from the cluster spec
  return tf.train.replica_device_setter(cluster=cluster_spec)


def main(unused_argv):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  if FLAGS.download_only:
    sys.exit(0)


  # Sanity check on the number of workers and the worker index
  #if FLAGS.worker_index >= FLAGS.num_workers:
  # raise ValueError("Worker index %d exceeds number of workers %d " %
  #                   (FLAGS.worker_index, FLAGS.num_workers))

  # Sanity check on the number of parameter servers
  if FLAGS.num_parameter_servers <= 0:
    raise ValueError("Invalid num_parameter_servers value: %d" %
                     FLAGS.num_parameter_servers)
  # air
  ps_hosts = re.findall(r'[\w\.:]+', FLAGS.ps_hosts)
  worker_hosts = re.findall(r'[\w\.:]+', FLAGS.worker_hosts)
  server = tf.train.Server({"ps":ps_hosts,"worker":worker_hosts}, job_name = FLAGS.job_name, task_index = FLAGS.worker_index)
  
  print("Worker GRPC URL: %s" % server.target)
  print("Worker index = %d" % FLAGS.worker_index)
  print("Number of workers = %d" % FLAGS.num_workers)

  if FLAGS.job_name == "ps":
    server.join()
  
  # air
  else:
    is_chief = (FLAGS.worker_index == 0)

  if FLAGS.sync_replicas:
    if FLAGS.replicas_to_aggregate is None:
      replicas_to_aggregate = FLAGS.num_workers
    else:
      replicas_to_aggregate = FLAGS.replicas_to_aggregate

  # Construct device setter object
  device_setter = get_device_setter(FLAGS.num_parameter_servers,
                                    FLAGS.num_workers)

  # The device setter will automatically place Variables ops on separate
  # parameter servers (ps). The non-Variable ops will be placed on the workers.
  with tf.device(device_setter):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    '''
    # Variables of the hidden layer
    hid_w = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                            stddev=1.0 / IMAGE_PIXELS), name="hid_w")
    hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

    # Variables of the softmax layer
    sm_w = tf.Variable(
        tf.truncated_normal([FLAGS.hidden_units, 10],
                            stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
        name="sm_w")
    sm_b = tf.Variable(tf.zeros([10]), name="sm_b")
    '''
    #air
    '''
    W1 = tf.Variable(tf.truncated_normal([784,1024], stddev=0.01))
    b1 = tf.Variable(tf.zeros([1024]))
    W2 = tf.Variable(tf.truncated_normal([1024,1024], stddev=0.01))
    b2 = tf.Variable(tf.zeros([1024]))
    W3 = tf.Variable(tf.truncated_normal([1024,512], stddev=0.01))
    b3 = tf.Variable(tf.zeros([512]))
    W4 = tf.Variable(tf.truncated_normal([512,10], stddev=0.01))
    b4 = tf.Variable(tf.zeros([10])) '''
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
        # target 10 output classes
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
        prob = tf.placeholder(tf.float32, name='keep_prob')
        x_image = tf.reshape(x, [-1,28,28,1])

    stack1_conv1 = layers.convolution2d(x_image,
                                            64,
                                            [3,3],
                                            weights_regularizer=layers.l2_regularizer(0.1),
                                            biases_regularizer=layers.l2_regularizer(0.1),
                                            scope='stack1_Conv1')
    stack1_conv2 = layers.convolution2d(stack1_conv1,
                                            64,
                                            [3,3],
                                            weights_regularizer=layers.l2_regularizer(0.1),
                                            biases_regularizer=layers.l2_regularizer(0.1),
                                            scope='stack1_Conv2')
    stack1_pool = layers.max_pool2d(stack1_conv2,
                                        [2,2],
                                        padding='SAME',
                                        scope='stack1_Pool')
    stack3_pool_flat = layers.flatten(stack1_pool, scope='stack3_pool_flat')
    fcl1 = layers.fully_connected(stack3_pool_flat, 
                                      512, 
                                      weights_regularizer=layers.l2_regularizer(0.1), 
                                      biases_regularizer=layers.l2_regularizer(0.1), 
                                      scope='FCL1')
    fcl1_d = layers.dropout(fcl1, keep_prob=prob, scope='dropout1')
    fcl2 = layers.fully_connected(fcl1_d, 
                                      128, 
                                      weights_regularizer=layers.l2_regularizer(0.1), 
                                      biases_regularizer=layers.l2_regularizer(0.1), 
                                      scope='FCL2')
    fcl2_d = layers.dropout(fcl2, keep_prob=prob, scope='dropout2')
    y, cross_entropy = skflow.models.logistic_regression(fcl2_d, y_, init_stddev=0.01)

    '''with tf.name_scope('Softmax'):
        fcl_softmax = layers.fully_connected(fcl2_d, 
                                      10, 
                                      weights_regularizer=layers.l2_regularizer(0.1), 
                                      biases_regularizer=layers.l2_regularizer(0.1), 
                                      scope='Softmax')
        y = tf.nn.softmax(fcl_softmax, name='y-output')
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=[1]))'''

        

    with tf.name_scope('train'):
        start_l_rate = 0.001
        decay_step = 1000
        decay_rate = 0.5
        learning_rate = tf.train.exponential_decay(start_l_rate, global_step, decay_step, decay_rate, staircase=False)
        grad_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        '''rep_op = tf.train.SyncReplicasOptimizer(grad_op, 
                                                    replicas_to_aggregate=len(workers),
                                                    replica_id=FLAGS.task_index, 
                                                    total_num_replicas=len(workers))'''
        train_op = tf.contrib.layers.optimize_loss(loss=cross_entropy, 
                                                       global_step=global_step, 
                                                       learning_rate=0.001, 
                                                       optimizer=grad_op, 
                                                       clip_gradients=1)
    #air

    # Ops: located on the worker specified with FLAGS.worker_index
    #x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    #y = tf.placeholder(tf.float32, [None, 10])
    #y_ = None

    '''
    hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
    hid = tf.nn.relu(hid_lin)

    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
    cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
    '''
    #air
    '''
    h1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    h1d = tf.nn.dropout(h1, 0.7)
    h2 = tf.nn.sigmoid(tf.matmul(h1d, W2) + b2)
    h2d = tf.nn.dropout(h2, 0.7)
    h3 = tf.nn.sigmoid(tf.matmul(h2d, W3) + b3)
    h3d = tf.nn.dropout(h3, 0.7)
    y_ = tf.nn.softmax(tf.matmul(h3d, W4) + b4)

    cost = -tf.reduce_sum(y*tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))
    #air
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)'''
    '''if FLAGS.sync_replicas:
      opt = tf.train.SyncReplicasOptimizer(
          opt,
          replicas_to_aggregate=replicas_to_aggregate,
          total_num_replicas=FLAGS.num_workers,
          replica_id=FLAGS.worker_index,
          name="mnist_sync_replicas")'''

    '''train_step = opt.minimize(cost,
                              global_step=global_step)'''

    '''if FLAGS.sync_replicas and is_chief:
      # Initial token and chief queue runners required by the sync_replicas mode
      chief_queue_runner = opt.get_chief_queue_runner()
      init_tokens_op = opt.get_init_tokens_op()'''

#air
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#air

    init_op = tf.initialize_all_variables()
    #train_dir = tempfile.mkdtemp()
    sv = tf.train.Supervisor(is_chief=is_chief,
                             #logdir=train_dir,
                             init_op=init_op,
                             recovery_wait_secs=1,
                             global_step=global_step)

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.worker_index])

    # The chief worker (worker_index==0) session will prepare the session,
    # while the remaining workers will wait for the preparation to complete.
    if is_chief:
      print("Worker %d: Initializing session..." % FLAGS.worker_index)
    else:
      print("Worker %d: Waiting for session to be initialized..." %
            FLAGS.worker_index)

    '''sess = sv.prepare_or_wait_for_session(FLAGS.worker_grpc_url,
                                          config=sess_config)'''
    sess = sv.prepare_or_wait_for_session(server.target,
                                          config=sess_config)

    print("Worker %d: Session initialization complete." % FLAGS.worker_index)

    '''if FLAGS.sync_replicas and is_chief:
      # Chief worker will start the chief queue runner and call the init op
      print("Starting chief queue runner and running init_tokens_op")
      sv.start_queue_runners(sess, [chief_queue_runner])
      sess.run(init_tokens_op)'''

    # Perform training
    time_begin = time.time()
    print("Training begins @ %s" % time.ctime(time_begin))

    local_step = 1
    while True:
      # Training feed
      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
      train_feed = {x: batch_xs,
                    y_: batch_ys,
                    prob: 0.8}

      _, step, loss = sess.run([train_op, global_step, cross_entropy], feed_dict=train_feed)

      now = time.time()
      if(local_step % 2 == 0):
          print("%s: Worker %d: training step %d done (global step: %d), loss: %.6f" %
              (time.ctime(now), FLAGS.worker_index, local_step, step+1, loss))

      if step+1 >= FLAGS.train_steps:
        break
      local_step += 1

    time_end = time.time()
    print("Training ends @ %s" % time.ctime(time_end))
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)


    acc_acu = 0.
    for i in xrange(int(10000/1000)):
        test_x, test_y = mnist.test.next_batch(1000)
        #print(test_x.shape)
        acc_batch = sess.run(accuracy, feed_dict={x: test_x, y_: test_y, prob: 1.0})
        print(acc_batch)
        acc_acu += acc_batch
    acc = acc_acu/10.0
    print ("test accuracy %g" % acc)
    sv.stop()

if __name__ == "__main__":
  tf.app.run()

