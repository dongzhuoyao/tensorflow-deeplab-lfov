import tensorflow as tf

import tensorflow as tf

tt = u'12345'
if u'123' in tt:
    print ("ok")

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
    for v in tf.trainable_variables():
        print("{}:  {}".format(v.name, v.get_shape()))