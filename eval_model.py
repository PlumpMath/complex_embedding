from __future__ import print_function

import numpy as np
from numpy.random import randn, randint

import tensorflow as tf

from data import sym, assym

# 1. read data -> triplets
# 2. build theano graph with slices
# 3. train

trainX = np.array(sym[0])
validX = np.array(sym[1])
testX = np.array(sym[2])

trainX = np.array([[i,0,j] for i in range(10) for j in range(10) if sym[0][i][j]!=0])
trainY = np.array([sym[0][i][j] for i in range(10) for j in range(10) if sym[0][i][j]!=0])

validX = np.array([[i,0,j] for i in range(10) for j in range(10) if sym[1][i][j]!=0])
validY = np.array([sym[1][i][j] for i in range(10) for j in range(10) if sym[1][i][j]!=0])


embedding_size = 5
entity_count = trainY.shape[0]


ereals = tf.Variable(tf.random_normal([entity_count, embedding_size],
                                      stddev=1.0/embedding_size),
                     name="ereals")
ecomplex = tf.Variable(tf.random_normal([entity_count, embedding_size],
                                        stddev=1.0/embedding_size),
                       name="ecomplex")

relation_count = 2

wreals = tf.Variable(tf.random_normal([relation_count, embedding_size],
                                      stddev=1.0/embedding_size), name="wreals")
wcomplex = tf.Variable(tf.random_normal([relation_count, embedding_size],
                                        stddev=1.0/embedding_size), name="wcomplex")


si = tf.placeholder(tf.int32, name="subject_index")
ri = tf.placeholder(tf.int32, name="relation_index")
oi = tf.placeholder(tf.int32, name="object_index")


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

    l = activation(Ys, p) + regul(esr, esc, wr, wc, eor, eoc)

    return p, l


def accuracy(msg, dataX, dataY):
    trainX = dataX
    trainY = dataY
    entity_count = dataY.shape[0]
    tsub_indices = tf.placeholder(tf.int32, shape=(entity_count), name="sub_indices")
    trel_indices = tf.placeholder(tf.int32, shape=(entity_count), name="rel_indices")
    tobj_indices = tf.placeholder(tf.int32, shape=(entity_count), name="obj_indices")

    tesr = tf.gather(ereals, tsub_indices)
    tesc = tf.gather(ereals, tsub_indices)
    twr = tf.gather(wreals, trel_indices)
    twc = tf.gather(wcomplex, trel_indices)
    teor = tf.gather(ereals, tobj_indices)
    teoc = tf.gather(ereals, tobj_indices)

    tYs = tf.placeholder(tf.float32, shape=trainY.shape, name="tYs")
    tp, tl = loss(tesr, tesc, twr, twc, teor, teoc, tYs)

    correct_prediction = tf.cast(tp*tYs >= 0., tf.bool)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print(msg, accuracy.eval({tsub_indices: trainX.T[0],
                              trel_indices: trainX.T[1],
                              tobj_indices: trainX.T[2],
                              tYs: trainY}, sess))


def train(sess, trainX, trainY, validX, validY,
          epochs=10, learning_rate=0.001, batch_size=3, accuracy_step=10):
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


    Ys = tf.placeholder(tf.float32, name="Ys")
    # calculate loss
    p, l = loss(esr, esc, wr, wc, eor, eoc, Ys)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(l)
    init = tf.initialize_all_variables().run()

    for epoch in range(epochs):
        for o in range(0, trainY.shape[0]-1, batch_size):
            sess.run(optimizer, {sub_indices: trainX.T[0,o:o+batch_size],
                                 rel_indices: trainX.T[1,o:o+batch_size],
                                 obj_indices: trainX.T[2,o:o+batch_size],
                                 Ys: trainY[o:o+batch_size]})

        if epoch % accuracy_step == 0:
            accuracy("Training accuracy:", trainX, trainY)
            accuracy("Validation accuracy:", validX, validY)


sess = tf.InteractiveSession()
train(sess, trainX, trainY, validX, validY, epochs=100)





