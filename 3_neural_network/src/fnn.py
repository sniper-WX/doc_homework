import tensorflow as tf

hello = tf.constant('hello world!')
with tf.Session() as sess:
    result = sess.run(hello)
    print (result)