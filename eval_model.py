from __future__ import print_function

import numpy as np
from numpy.random import randn, randint

import tensorflow as tf

from data import sym, assym

trainX = np.array(sym[0])
validX = np.array(sym[1])
testX = np.array(sym[2])

# 1. read data -> triplets
# 2. build theano graph with slices
# 3. train

embedding_size = 5
entity_count = 10


ereals = tf.Variable(tf.random_normal([entity_count, embedding_size], stddev=1.0/embedding_size), name="ereals")
ecomplex = tf.Variable(tf.random_normal([entity_count, embedding_size], stddev=1.0/embedding_size), name="ecomplex")

relation_count = 2

wreals = tf.Variable(tf.random_normal([relation_count, embedding_size], stddev=1.0/embedding_size), name="wreals")
wcomplex = tf.Variable(tf.random_normal([relation_count, embedding_size], stddev=1.0/embedding_size), name="wcomplex")


batch_size = 1 # TODO needs tensorflow multidim slicing, which didn't work as expected


si = tf.placeholder(tf.int32, name="subject_index")
ri = tf.placeholder(tf.int32, name="relation_index")
oi = tf.placeholder(tf.int32, name="object_index")

Y = tf.placeholder(tf.float32, name="Y")

def pred(si,ri,oi):
    # formula 11 from paper
    return tf.reduce_sum(wreals[ri,:]*ereals[si,:]*ereals[oi,:], 0) \
        + tf.reduce_sum(wreals[ri,:]*ecomplex[si,:]*ecomplex[oi,:], 0) \
        + tf.reduce_sum(wcomplex[ri,:]*ereals[si,:]*ecomplex[oi,:], 0) \
        + tf.reduce_sum(wcomplex[ri,:]*ecomplex[si,:]*ereals[oi,:], 0)

def regul(si, ri, oi, lmbda=0.03): # TODO from paper
    return lmbda*(tf.reduce_mean(ereals[si]) + tf.reduce_mean(ecomplex[si]) + \
                  tf.reduce_mean(wreals[ri]) + tf.reduce_mean(wcomplex[ri]) + \
                  tf.reduce_mean(ereals[oi]) + tf.reduce_mean(ecomplex[oi]))

def loss(si, ri, oi, Y, activation=lambda Y, pred: tf.log(1+tf.exp(-Y*pred))):
    p = pred(si, ri, oi)

    l = activation(Y,p) + regul(si, ri, oi)

    return l


def train(epochs=10, learning_rate=0.001):
    l = loss(si, ri, oi, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(l)
    init = tf.initialize_all_variables().run()

    triple_count=10

    for i in range(epochs):
        for si_ in range(10):
            for oi_ in range(10):
                if(trainX[si_, oi_] != 0):
                    sess.run(optimizer, {si: si_, ri: 0, oi: oi_, Y: trainX[si_, oi_]})

        #print([pred(si, ri, oi).eval({si: si_, ri: 0, oi: 0}, sess)
        #       for si_ in range(5)])
        #print([trainX[si_, 0] for si_ in range(5)])
        print([pred(si, ri, oi).eval({si: 6, ri: 0, oi: 0}, sess)], -1)
        print([pred(si, ri, oi).eval({si: 9, ri: 0, oi: 2}, sess)], 1)

    # TODO accuracy on training set
    # TODO accuracy on valid set

sess = tf.InteractiveSession()

train(epochs=100)

