
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


import tensorflow as tf
from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt


# In[5]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[110]:


tf.reset_default_graph()


# In[111]:


img_size, num_channels, num_classes = 28, 1, 10
pooling_ksize = [1, 2, 2, 1]
pooling_strides = [1, 2, 2, 1]
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_')


# In[112]:


with tf.variable_scope('conv1') as scope:
    kernel_shape = [5, 5, 1, 16]
    strides = [1, 1, 1, 1]
    bias_shape = kernel_shape[-1]
    kernel = tf.get_variable("weights", shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), 
                             dtype=tf.float32, trainable=True)
    conv = tf.nn.conv2d(x, kernel, strides=strides, padding="SAME", name="conv_map")
    bias = tf.get_variable("bias", shape=bias_shape, initializer=tf.contrib.layers.xavier_initializer(), 
                           dtype=tf.float32, trainable=True)
    pre_activation = tf.add(conv, bias)
    conv1 = tf.nn.relu(pre_activation, name="activation")


pool1 = tf.nn.max_pool(value=conv1, ksize=pooling_ksize, strides=pooling_strides, padding="SAME", name="max_pool1")

print(x)
print(kernel)
print(conv)
print(conv1)
print(pool1)


# In[113]:


with tf.variable_scope("conv2") as scope:
    kernel_shape = [5, 5, 16, 32]
    strides = [1, 1, 1, 1]
    bias_shape = kernel_shape[-1]
    kernel = tf.get_variable(name="weights", shape=kernel_shape, dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    conv = tf.nn.conv2d(input=pool1, filters=kernel, strides=strides, padding="SAME", name='conv_map')
    bias = tf.get_variable(name="bias", shape=bias_shape, dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    pre_activation = tf.add(conv, bias)
    conv2 = tf.nn.relu(pre_activation, name="relu")

pool2 = tf.nn.max_pool(value=conv2, ksize=pooling_ksize, strides=pooling_strides, padding="SAME", name="max_pooling2")

print(kernel)
print(conv)
print(conv2)
print(pool2)


# In[114]:


with tf.variable_scope("conv3") as scope:
    kernel_shape = [5, 5, 32, 64]
    strides = [1, 1, 1, 1]
    bias_shape = kernel_shape[-1]
    kernel = tf.get_variable(name="weights", shape=kernel_shape, dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer_conv2d(), trainable=True)
    conv = tf.nn.conv2d(input=pool2, filters=kernel, strides=strides, padding="SAME", name="conv_map")
    bias = tf.get_variable(name="bias", shape=bias_shape, dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    pre_activation = tf.add(conv, bias)
    conv3 = tf.nn.relu(pre_activation, name="relu")

pool3 = tf.nn.max_pool(value=conv3, ksize=pooling_ksize, strides=pooling_strides, padding="SAME", name="max_pooling3")

print(kernel)
print(conv)
print(conv3)
print(pool3)


# In[115]:


pool3_shape = pool3.shape.as_list()
pool3_flat = tf.reshape(pool3, [-1, pool3_shape[1]*pool3_shape[2]*pool3_shape[3]], name="flatten")
print(pool3_flat)


# In[116]:


pool3_flat.shape.as_list()


# In[117]:


pool3_flat.get_shape()[1].value


# In[118]:


with tf.variable_scope("fc1") as scope:
    weights_shape = [pool3_flat.shape.as_list()[1], 128]
    bias_shape = weights_shape[-1]
    weights = tf.get_variable(name="weights", shape=weights_shape, dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    bias = tf.get_variable(name="bias", shape=bias_shape, dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    fc = tf.matmul(pool3_flat, weights, name="matmul")
    pre_activation = tf.add(fc, bias)
    fc1 = tf.nn.relu(fc, name="relu")

print(weights)
print(fc1)


# In[119]:


with tf.variable_scope("fc2") as scope:
    weights_shape = [fc1.get_shape()[1].value, num_classes]
    bias_shape = weights_shape[-1]
    weights = tf.get_variable(name="weights", shape=weights_shape, dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    bias = tf.get_variable(name="bias", shape=bias_shape, dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    fc = tf.matmul(fc1, weights, name="matmul")
    logits = tf.add(fc, bias, name="logits")
    y_pred = tf.argmax(logits, axis=1, name="y_pred")

print(weights)
print(logits)
print(y_pred)


# In[122]:


global_step = tf.contrib.framework.get_or_create_global_step()


# In[124]:


with tf.name_scope("Loss"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, 
                                                               logits=logits)
    loss = tf.reduce_mean(cross_entropy, name="cross_entropy_loss")
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step, name="train_step")
  
# Here we specify the output as "Prediction/y_pred", this will be important later
with tf.name_scope("Prediction"): 
    correct_prediction = tf.equal(y_pred, 
                                  tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")


# In[125]:


print(cross_entropy)
print(loss)
print(accuracy)


# In[130]:


batch_size = 100
n_epochs = 2
n_batches = int(mnist.train.num_examples / batch_size)
end_step = n_epochs * n_batches
print(n_batches)
print(end_step)


# In[129]:


#pruning_hparams = pruning.get_pruning_hparams()
#print(pruning_hparams)


# In[131]:


# pruning_hparams.begin_pruning_step = 0
# pruning_hparams.end_pruning_step = end_step
# pruning_hparams.frequency = 100
# pruning_hparams.target_sparsity = 0.9


# In[132]:


# pruning_obj = pruning.Pruning(pruning_hparams, global_step=global_step)
# make_update_op = pruning_obj.conditional_mask_update_op()


# In[133]:


sess = tf.Session()
# Initialize the variables (i.e. assign their default value)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


# In[134]:


train_loss, train_accuracy = sess.run([loss, accuracy], 
                                      feed_dict={x: mnist.train.images.reshape(mnist.train.num_examples, img_size, img_size, num_channels), 
                                                 y_: mnist.train.labels})
print('Epoch %d, training loss: %g, training accuracy: %g' % (0, train_loss, train_accuracy))
val_loss, val_accuracy = sess.run([loss, accuracy], 
                                  feed_dict={x: mnist.validation.images.reshape(mnist.validation.num_examples, img_size, img_size, num_channels),
                                             y_: mnist.validation.labels})
print('Epoch %d, validation loss: %g, validation accuracy %g' % (0, val_loss, val_accuracy))


# In[135]:


#print("Weight sparsities: ", sess.run(tf.contrib.model_pruning.get_weight_sparsity()))


# In[138]:


for i in range(n_epochs):
    for j in range(n_batches):
        batch_images, batch_labels = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_images.reshape(batch_size, img_size, img_size, num_channels), 
                                        y_: batch_labels})
        #sess.run(make_update_op)
    if i % 2 == 0:
        train_loss, train_accuracy = sess.run([loss, accuracy], 
                                              feed_dict={x: mnist.train.images.reshape(mnist.train.num_examples, img_size, img_size, num_channels), 
                                                         y_: mnist.train.labels})
        print('Epoch %d, training loss: %g, training accuracy: %g' % (i, train_loss, train_accuracy))
        val_loss, val_accuracy = sess.run([loss, accuracy], 
                                          feed_dict={x: mnist.validation.images.reshape(mnist.validation.num_examples, img_size, img_size, num_channels),
                                                     y_: mnist.validation.labels})
        print('Epoch %d, validation loss: %g, validation accuracy %g' % (i, val_loss, val_accuracy))
        #print("Weight sparsity: ", sess.run(tf.contrib.model_pruning.get_weight_sparsity()))


# In[192]:


print('test accuracy %g' % sess.run(accuracy, 
                                    feed_dict={x: mnist.validation.images.reshape(mnist.validation.num_examples, img_size, img_size, num_channels), 
                                               y_: mnist.validation.labels}))


# In[142]:


saver.save(sess, './chkps_mnist_weight_noprune/mnist_noprune')


# In[193]:


from tensorflow.python.framework.graph_util import remove_training_nodes
from tensorflow.tools.graph_transforms import TransformGraph
sub_graph_def = remove_training_nodes(sess.graph_def)


# In[195]:


output_nodes_list = [sess.graph.get_operation_by_name("fc2/y_pred").name]
print(output_nodes_list)


# In[196]:


from tensorflow.python.framework import graph_util as gu
sub_graph_def = gu.convert_variables_to_constants(sess=sess, 
                                                  input_graph_def=sub_graph_def,
                                                  output_node_names=output_nodes_list)


# In[197]:


print([node.name for node in sub_graph_def.node])


# In[198]:


tf.train.write_graph(sub_graph_def, "./mnist_cnn_0to9", "mnist_cnn_weight_noprune.pb", as_text=False)


sess.close()

