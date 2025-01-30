'''
XOR data pipeline
'''
import numpy as np

class XORData(object):
    """a class for the generation of XOR validation and training data

    >>> d = XORData()
    >>> d.astype(int)
    array([[0, 0, 0],
           [0, 1, 1],
           [1, 0, 1],
           [1, 1, 0]])

    >>> d = XORData(batchsize=2,delta=0.5)
    >>> len(d)
    8
    >>> np.all(np.rint(d[0:4]) == XORData.TRUTHTABLE)
    True
    >>> np.all(np.rint(d[4:8]) == XORData.TRUTHTABLE)
    True
    >>> np.var(d - np.vstack([XORData.TRUTHTABLE]*2)) > 0
    True

    """

    TRUTHTABLE = np.array([
        #A,B,XOR
        [0,0,0],
        [0,1,1],
        [1,0,1],
        [1,1,0],
        ],dtype=float)

    TABLE0 = np.vstack([TRUTHTABLE,[0.5,0.5,0.0]])
    TABLE1 = np.vstack([TRUTHTABLE,[0.5,0.5,1.0]])

    def __new__(this,batchsize=1,delta=0.0,table=TRUTHTABLE):
        n = len(table)
        assert table.shape == (n,2+1)
        rands = np.random.uniform(-delta,+delta,size=(batchsize,n,2))
        zeros = np.zeros(shape=(batchsize,n,1),dtype=float)
        deltas = np.concatenate((rands,zeros),axis=2)
        assert deltas.shape == (batchsize,n,3)
        dataset = table + deltas
        dataset.shape = (batchsize*n,3)
        return dataset
