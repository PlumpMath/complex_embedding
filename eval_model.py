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

wreals = tf.Variable(tf.random_normal([relation_count, embedding_size],
                                      stddev=1.0/embedding_size), name="wreals")
wcomplex = tf.Variable(tf.random_normal([relation_count, embedding_size],
                                        stddev=1.0/embedding_size), name="wcomplex")


si = tf.placeholder(tf.int32, name="subject_index")
ri = tf.placeholder(tf.int32, name="relation_index")
oi = tf.placeholder(tf.int32, name="object_index")

Ys = tf.placeholder(tf.float32, name="Ys")

def pred(esr, esc, wr, wc, eor, eoc):
    # formula 11 from paper
    return tf.reduce_sum(wr*esr*eor, 1) \
        + tf.reduce_sum(wr*esc*eoc, 1) \
        + tf.reduce_sum(wc*esr*eoc, 1) \
        + tf.reduce_sum(wc*esc*eor, 1)

def regul(esr, esc, wr, wc, eor, eoc, lmbda=0.03): # TODO from paper
    return lmbda*(tf.reduce_mean(esr) + tf.reduce_mean(esc) + \
                  tf.reduce_mean(wr) + tf.reduce_mean(wc) + \
                  tf.reduce_mean(eor) + tf.reduce_mean(eoc))

def loss(esr, esc, wr, wc, eor, eoc, Ys, activation=lambda Ys, pred: tf.log(1+tf.exp(-Ys*pred))):
    p = pred(esr, esc, wr, wc, eor, eoc)

    l = activation(Ys,p) + regul(esr, esc, wr, wc, eor, eoc)

    return p, l


def train(epochs=10, learning_rate=0.001, batch_size=2):
    # prepare slicing
    sub_indices = tf.placeholder(tf.int32, shape=(batch_size), name="sub_indices")
    rel_indices = tf.placeholder(tf.int32, shape=(batch_size), name="rel_indices")
    obj_indices = tf.placeholder(tf.int32, shape=(batch_size), name="obj_indices")

    esr = tf.gather(ereals, sub_indices)
    esc = tf.gather(ereals, sub_indices)
    wr = tf.gather(wreals, rel_indices)
    wc = tf.gather(wcomplex, rel_indices)
    eor = tf.gather(ereals, obj_indices)
    eoc = tf.gather(ereals, obj_indices)


    # calculate loss
    p, l = loss(esr, esc, wr, wc, eor, eoc, Ys)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(l)
    init = tf.initialize_all_variables().run()

    for i in range(epochs):
        for o in range(0, entity_count, batch_size):
            print(o, batch_size)
            sess.run(optimizer, {sub_indices: [0,1],
                                 rel_indices: [0,0],
                                 obj_indices: [2,3],
                                 Ys: [0,1]})


#        for si_ in range(10):
#            for oi_ in range(10):
#                if(trainX[si_, oi_] != 0):

        #print([pred(si, ri, oi).eval({si: si_, ri: 0, oi: 0}, sess)
        #       for si_ in range(5)])
        #print([trainX[si_, 0] for si_ in range(5)])
        #print([pred(si, ri, oi).eval({si: 6, ri: 0, oi: 0}, sess)], -1)
        #print([pred(si, ri, oi).eval({si: 9, ri: 0, oi: 2}, sess)], 1)

    # TODO accuracy on training set
    # TODO accuracy on valid set

#sess = tf.InteractiveSession()
#train(epochs=100)
