import numpy as np
import h5py as h5
from matplotlib import pyplot as plt

from tsne import tsne

# load model
f = h5.File("embeddings.h5", "r")
ereals = np.array(f["entities_real"], dtype=np.float64)
eimag = np.array(f["entities_imag"], dtype=np.float64)
wreals = np.array(f["relation_real"], dtype=np.float64)
wimag = np.array(f["relation_imag"], dtype=np.float64)
losses = np.array(f["losses"])
training_accuracy = np.array(f["training_accuracy"])
valid_accuracy = np.array(f["validation_accuracy"])
f.close()

embeddings = [ereals, eimag, wreals, wimag]

Y2d = [tsne(np.array(vecs, dtype=np.float64))
       for vecs in embeddings]


for ys in Y2d:
    plt.scatter(ys[:,0], ys[:,1])
    plt.show()


plt.plot(training_accuracy, label="training")
plt.plot(valid_accuracy, label="validation")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()


plt.plot(losses)
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.show()
