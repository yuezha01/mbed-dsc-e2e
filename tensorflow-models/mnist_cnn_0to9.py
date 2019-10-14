import warnings
warnings.filterwarnings("ignore")
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util as gu
from tensorflow.python.framework.graph_util import remove_training_nodes
import json

# load the dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# set up parameters
img_size, num_channels, num_classes = 28, 1, 10

pooling_ksize = [1, 2, 2, 1]
pooling_strides = [1, 2, 2, 1]
kernel_shape_1 = [5, 5, 1, 16]
bias_shape_1 = [16]
kernel_shape_2 = [5, 5, 16, 32]
bias_shape_2 = [32]
kernel_shape_3 = [5, 5, 32, 64]
bias_shape_3 = [64]
fc_neuron_1 = 128

batch_size = 50
n_epochs = 2
n_batches = int(mnist.train.num_examples / batch_size)
display_step = 10

save_ckps_dir = "./saved_models/chkps_mnist_cnn/"
save_pb_dir = "./saved_models/pb_mnist_cnn/"
chkp_fd_name = "chkp_config1/mnist_cnn"
pb_file_name = "mnist_cnn_config1.pb"
training_dir = "./training_logs/"
config_file_name = "training_config1.json"
result_file_name = "training_log1.json"

config = {"pooling_ksize": pooling_ksize,
          "pooling_strides": pooling_strides,
          "kernel_shape": [kernel_shape_1, kernel_shape_2, kernel_shape_3],
          "fc_shape": [fc_neuron_1],
          "batch_size": batch_size, 
          "n_epochs": n_epochs}
results = {"init_loss": [], "init_acc": [],
           "val_loss": [], "val_acc": [],
           "train_loss": [], "train_acc": [],
           "test_loss": [], "test_acc": []}


# build the model
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_')

with tf.variable_scope("covn1") as scope:
    kernel = tf.get_variable(name="weights", shape=kernel_shape_1, dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer_conv2d(), trainable=True)
    bias = tf.get_variable(name="bias", shape=bias_shape_1, dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    conv = tf.nn.conv2d(x, filters=kernel, strides=[1, 1, 1, 1], padding="SAME", name="conv_map")
    pre_activation = tf.add(conv, bias)
    conv1 = tf.nn.relu(pre_activation, name="relu")

pool1 = tf.nn.max_pool(conv1, ksize=pooling_ksize, strides=pooling_strides, padding="SAME", name="max_pool1")
print(x)
print(kernel)
print(conv)
print(conv1)
print(pool1)

with tf.variable_scope("conv2") as scope:
    kernel = tf.get_variable(name="weights", shape=kernel_shape_2, dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer_conv2d(), trainable=True)
    bias = tf.get_variable(name="bias", shape=bias_shape_2, dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding="SAME", name="conv_map")
    pre_activation = tf.add(conv, bias)
    conv2 = tf.nn.relu(pre_activation, name="relu")

pool2 = tf.nn.max_pool(conv2, ksize=pooling_ksize, strides=pooling_strides, padding="SAME", name="max_pool2")
print(kernel)
print(conv)
print(conv2)
print(pool2)

with tf.variable_scope("conv3") as scope:
    kernel = tf.get_variable(name="weights", shape=kernel_shape_3, dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer_conv2d(), trainable=True)
    bias = tf.get_variable(name="bias", shape=bias_shape_3, dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding="SAME", name="conv_map")
    pre_activation = tf.add(conv, bias)
    conv3 = tf.nn.relu(pre_activation, name="relu")

pool3 = tf.nn.max_pool(conv3, ksize=pooling_ksize, strides=pooling_strides, padding="SAME", name="max_pool3")
print(kernel)
print(conv)
print(conv3)
print(pool3)

pool3_flat = tf.reshape(pool3, [-1, pool3.get_shape()[1]*pool3.get_shape()[2]*pool3.get_shape()[3]], name="flatten")
print(pool3_flat)

with tf.variable_scope("fc1") as scope:
    weights = tf.get_variable(name="weights", shape=[pool3_flat.get_shape()[1], fc_neuron_1], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    bias = tf.get_variable(name="bias", shape=[fc_neuron_1], dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    fc = tf.matmul(pool3_flat, weights, name="matmul")
    pre_activation = tf.add(fc, bias)
    fc1 = tf.nn.relu(pre_activation, name="relu")

print(weights)
print(fc1)

with tf.variable_scope("fc2") as scope:
    weights = tf.get_variable(name="weights", shape=[fc_neuron_1, num_classes], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    bias = tf.get_variable(name="bias", shape=[num_classes], dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    fc = tf.matmul(fc1, weights, name="matmul")
    logits = tf.add(fc, bias, name="logits")
    y_pred = tf.argmax(logits, axis=1, name="y_pred")

print(weights)
print(logits)
print(y_pred)

with tf.name_scope("loss"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits, name="cross_entropy")
    loss = tf.reduce_mean(cross_entropy, name="cross_entropy_loss")
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, name="train_step")
print(cross_entropy)
print(loss)

with tf.name_scope("eval"):
    correct = tf.equal(y_pred, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
print(correct)
print(accuracy)

# training
sess = tf.Session()
# Initialize the variables (i.e. assign their default value)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# initialization
train_loss, train_accuracy = sess.run([loss, accuracy], 
                                      feed_dict={x: mnist.train.images.reshape(mnist.train.num_examples, img_size, img_size, num_channels), 
                                                 y_: mnist.train.labels})
print('Epoch %d, training loss: %g, training accuracy: %g' % (0, train_loss, train_accuracy))
val_loss, val_accuracy = sess.run([loss, accuracy], 
                                  feed_dict={x: mnist.validation.images.reshape(mnist.validation.num_examples, img_size, img_size, num_channels),
                                             y_: mnist.validation.labels})
print('Epoch %d, validation loss: %g, validation accuracy %g' % (0, val_loss, val_accuracy))
results["init_loss"].append(val_loss)
results["init_acc"].append(val_accuracy)
max_accuracy = val_accuracy

for i in range(n_epochs):
    for j in range(n_batches):
        batch_images, batch_labels = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_images.reshape(batch_size, img_size, img_size, num_channels), 
                                        y_: batch_labels})
        
    train_loss, train_accuracy = sess.run([loss, accuracy], 
                                          feed_dict={x: mnist.train.images.reshape(mnist.train.num_examples, img_size, img_size, num_channels), 
                                                     y_: mnist.train.labels})
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_accuracy)

    val_loss, val_accuracy = sess.run([loss, accuracy], 
                                      feed_dict={x: mnist.validation.images.reshape(mnist.validation.num_examples, img_size, img_size, num_channels),
                                                 y_: mnist.validation.labels})
    results["val_loss"].append(val_loss)
    results["val_acc"].append(val_accuracy)
    
    if val_accuracy > max_accuracy:
        print("Save the current model!")
        max_accuracy = val_accuracy
        saver.save(sess, save_ckps_dir + chkp_fd_name)
    
    if (i+1) % display_step == 0:
        print('Epoch %d, training loss: %g, training accuracy: %g' % (i, train_loss, train_accuracy))
        print('Epoch %d, validation loss: %g, validation accuracy %g' % (i, val_loss, val_accuracy))
        

test_loss, test_accuracy =  sess.run([loss, accuracy], 
                                     feed_dict={x: mnist.test.images.reshape(mnist.test.num_examples, img_size, img_size, num_channels), 
                                                y_: mnist.test.labels})
results["test_loss"].append(test_loss)
results["test_acc"].append(test_accuracy)
print("Test accuracy: %g" % test_accuracy)


# write the graph to graph_def
out_nodes = [y_pred.op.name]
print(out_nodes)
sub_graph_def = remove_training_nodes(sess.graph_def)
sub_graph_def = gu.convert_variables_to_constants(sess, sub_graph_def, out_nodes)
graph_path = tf.train.write_graph(sub_graph_def,
                                  save_pb_dir,
                                  pb_file_name,
                                  as_text=False)

print('written graph to: %s' % graph_path)

with open(training_dir+config_file_name, 'w') as f:
    json.dump(config, f)

with open(training_dir+result_file_name, 'w') as f:
    json.dump(config, f)
    
sess.close()