from __future__ import print_function

import numpy as np
from numpy.random import randn, randint

import tensorflow as tf
import h5py as h5

from data import sym, antisym

# 1. read data -> triplets
# 2. build tensorflow graph with slices
# 3. train

trainX = np.array(#+[[i,0,j] for i in range(10) for j in range(10) if sym[0][i][j]!=0]) # \
                [[i,1,j] for i in range(10) for j in range(10) if antisym[0][i][j]!=0])
trainY = np.array(#[sym[0][i][j] for i in range(10) for j in range(10) if sym[0][i][j]!=0]) # \
                [antisym[0][i][j] for i in range(10) for j in range(10) if antisym[0][i][j]!=0])

validX = np.array(#[[i,0,j] for i in range(10) for j in range(10) if sym[1][i][j]!=0]) # \
                [[i,1,j] for i in range(10) for j in range(10) if antisym[1][i][j]!=0])
validY = np.array(#+ [sym[1][i][j] for i in range(10) for j in range(10) if sym[1][i][j]!=0]) # \
                [antisym[1][i][j] for i in range(10) for j in range(10) if antisym[1][i][j]!=0])


embedding_size = 5
entity_count = trainY.shape[0]


ereals = tf.Variable(tf.random_normal([entity_count, embedding_size],
                                      stddev=1.0/embedding_size),
                    name="ereals")
eimag = tf.Variable(tf.random_normal([entity_count, embedding_size],
                                     stddev=1.0/embedding_size),
                    name="eimag")

relation_count = 2

wreals = tf.Variable(tf.random_normal([relation_count, embedding_size],
                                      stddev=1.0/embedding_size), name="wreals")
wimag = tf.Variable(tf.random_normal([relation_count, embedding_size],
                                     stddev=1.0/embedding_size), name="wimag")


def pred(esr, esc, wr, wc, eor, eoc):
    # formula 11 from paper
    return tf.reduce_sum(wr*esr*eor, 1) \
        + tf.reduce_sum(wr*esc*eoc, 1) \
        + tf.reduce_sum(wc*esr*eoc, 1) \
        - tf.reduce_sum(wc*esc*eor, 1)


def regul(esr, esc, wr, wc, eor, eoc, lmbda=0.003): # TODO from paper
    return lmbda*(tf.square(tf.reduce_mean(esr)) + tf.square(tf.reduce_mean(esc)) + \
                  tf.square(tf.reduce_mean(wr)) + tf.square(tf.reduce_mean(wc)) + \
                  tf.square(tf.reduce_mean(eor)) + tf.square(tf.reduce_mean(eoc)))


def loss(esr, esc, wr, wc, eor, eoc, Ys):
    preds = pred(esr, esc, wr, wc, eor, eoc)
    losses = tf.log(1+tf.exp(-Ys*preds))

    l = tf.reduce_sum(losses + regul(esr, esc, wr, wc, eor, eoc))

    return preds, l


def accuracy(dataX, dataY):
    trainX = dataX
    trainY = dataY
    entity_count = dataY.shape[0]
    tsub_indices = tf.placeholder(tf.int32, shape=(entity_count), name="sub_indices")
    trel_indices = tf.placeholder(tf.int32, shape=(entity_count), name="rel_indices")
    tobj_indices = tf.placeholder(tf.int32, shape=(entity_count), name="obj_indices")

    tesr = tf.gather(ereals, tsub_indices)
    tesc = tf.gather(eimag, tsub_indices)
    twr = tf.gather(wreals, trel_indices)
    twc = tf.gather(wimag, trel_indices)
    teor = tf.gather(ereals, tobj_indices)
    teoc = tf.gather(eimag, tobj_indices)

    tYs = tf.placeholder(tf.float32, shape=trainY.shape, name="tYs")
    tp, tl = loss(tesr, tesc, twr, twc, teor, teoc, tYs)

    correct_prediction = tf.cast(tp*tYs >= 0., tf.bool)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return accuracy.eval({tsub_indices: trainX.T[0],
                          trel_indices: trainX.T[1],
                          tobj_indices: trainX.T[2],
                          tYs: trainY}, sess)

def mrr(triple):

    ranks = 
    return 


def train(sess, trainX, trainY, validX, validY,
        epochs=10, learning_rate=0.001, batch_size=3, accuracy_step=10):
    train_count = trainY.shape[0]
    # prepare slicing
    sub_indices = tf.placeholder(tf.int32, shape=(batch_size), name="sub_indices")
    rel_indices = tf.placeholder(tf.int32, shape=(batch_size), name="rel_indices")
    obj_indices = tf.placeholder(tf.int32, shape=(batch_size), name="obj_indices")

    esr = tf.gather(ereals, sub_indices)
    esc = tf.gather(eimag, sub_indices)
    wr = tf.gather(wreals, rel_indices)
    wc = tf.gather(wimag, rel_indices)
    eor = tf.gather(ereals, obj_indices)
    eoc = tf.gather(eimag, obj_indices)


    Ys = tf.placeholder(tf.float32, name="Ys")
    # calculate loss
    p, l = loss(esr, esc, wr, wc, eor, eoc, Ys)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(l)
#    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(l)
    init = tf.initialize_all_variables().run()

    tacc = []
    vacc = []
    losses = []
    si, ri, oi = (0, 1, 2)
    for epoch in range(epochs):
        epoch_losses = []
        for o in range(0, train_count-1, batch_size):
            l_, opt = sess.run([l, optimizer], {sub_indices: trainX.T[si, o:o+batch_size],
                                                rel_indices: trainX.T[ri, o:o+batch_size],
                                                obj_indices: trainX.T[oi, o:o+batch_size],
                                                Ys: trainY[o:o+batch_size]})
            epoch_losses.append(l_)

        print("Epoch: ", epoch, ", mean loss:", np.mean(epoch_losses))
        losses += epoch_losses

        if epoch % accuracy_step == 0:
            ta = accuracy(trainX, trainY)
            va = accuracy(validX, validY)
            tacc.append(ta)
            vacc.append(va)
            print("Training accuracy:", ta)
            print("Validation accuracy:", va)

    return tacc, vacc, losses


sess = tf.InteractiveSession()
training_accuracy, validation_accuracy, losses = train(sess, trainX, trainY, validX, validY,
                                                       epochs=200, accuracy_step=100,
                                                       batch_size=10, negative_ratio=1.0)



def store_model(name="embeddings.h5"):
    ## store model
    import os
    os.remove(name)

    f = h5.File(name)
    f["entities_real"] = ereals.eval(sess)
    f["entities_imag"] = eimag.eval(sess)
    f["relation_real"] = wreals.eval(sess)
    f["relation_imag"] = wimag.eval(sess)
    f["losses"] = losses
    f["training_accuracy"] = training_accuracy
    f["validation_accuracy"] = validation_accuracy
    f.close()


store_model("antisym_toy.h5")
