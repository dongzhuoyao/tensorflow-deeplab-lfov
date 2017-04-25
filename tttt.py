import tensorflow as tf

import tensorflow as tf

initializer = tf.constant_initializer(value=1)
w_append = tf.get_variable(name="filter_of_attention_map", shape=[3, 3, 1, 64])
#var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)

#with tf.variable_scope("a_name_scope", reuse=True):
#    for i in xrange(3):
initializer = tf.constant_initializer(value=1)
w_append = tf.get_variable(name="filter_of_attention_map", shape=[3, 3, 1, 64])
#var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    #print(var1.name)        # var1:0
    #print(sess.run(var1))   # [ 1.]
    print(var2.name)        # a_name_scope/var2:0
    print(sess.run(var2))   # [ 2.]
    print(var21.name)       # a_name_scope/var2_1:0
    print(sess.run(var21))  # [ 2.0999999]
    print(var22.name)       # a_name_scope/var2_2:0
    print(sess.run(var22))  # [ 2.20000005]