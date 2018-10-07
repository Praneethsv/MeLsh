import tensorflow as tf
sess = tf.InteractiveSession()
#a = tf.Variable(1.0)
# a = tf.get_variable(shape=(2,1), initializer=tf.constant_initializer(0.0), dtype=tf.float64,name='a')
# #b = tf.Variable(10.0)
# b = tf.get_variable(shape=(2,1), initializer=tf.constant_initializer(0.0), dtype=tf.float64,name='b')
# #c = tf.Variable([10.0, 5.0, 4.0])
# c = tf.get_variable(shape=(), initializer=tf.constant_initializer(0.0), dtype=tf.float64,name='c')
# new_a = tf.get_variable(shape=(2,1), initializer=tf.constant_initializer(1.0), dtype=tf.float64,name='newa')
# new_b = tf.get_variable(shape=(2,1), initializer=tf.constant_initializer(0.0), dtype=tf.float64,name='newb')
# new_c = tf.get_variable(shape=(), initializer=tf.constant_initializer(10.0), dtype=tf.float64,name='newc')
# #updates = [tf.assign_add(a, new_a), tf.assign_add(c, new_c)]
# #ch = [tf.assign_add(b, new_b)]
#  # this is the group op
# sess.run(tf.global_variables_initializer()) # initialize c each time
# updates = [tf.assign_add(a, new_a), tf.assign_add(b, new_b), tf.assign_add(c, new_c)]
# grp = tf.group(*updates)
# sess.run(grp) # run the group op
# #print(sess.run(*ch))
# print(sess.run(updates))
# print(sess.run(updates))

#import torch as t
import numpy as np
# shape = (2, 1)
# epsilon = 0.0
# epsilon1 = 10.0
# sum_ = t.zeros(shape, dtype=t.float64)
# sumsq = t.FloatTensor(np.full(shape, epsilon))
# count = t.FloatTensor(np.full((), epsilon))
#
# newsum_ = t.ones(shape, dtype=t.float64)
# newsumsq = t.FloatTensor(np.full(shape, epsilon))
# newcount = t.FloatTensor(np.full((), epsilon1))
#
# refsum = np.add(sum_, newsum_)
# refsum = t.add(refsum, newsum_)
# refsumsq = np.add(sumsq, newsumsq)
# refsumsq = t.add(refsumsq, newsumsq)
# refcount = np.add(count, newcount)
# refcount = t.add(refcount, newcount)

x = np.random.randn(3,4)

x_clip = np.clip(x, -1.0, 1.0)
sess.run(tf.global_variables_initializer()) # initialize c each time
t = tf.convert_to_tensor(x)
t_clip = tf.clip_by_value(t, -1.0, 1.0)
#updates = [ t.add(sum_ + newsum_ + newsum_), t.add(sumsq + newsum_ + newsumsq), t.add(count + newcount + newcount)]

print(x_clip)
print(type(x_clip))

print(sess.run(t_clip))
print(type(sess.run(t_clip)))