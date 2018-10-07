import numpy as np
import torch as t
import torch.nn as nn

class Pd(object):
    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def logp(self, x):
        return - self.neglogp(x)


class PdType(object):
    def pdclass(self):
        raise NotImplementedError

    def pdfromflat(self):
        raise NotImplementedError

    def param_shape(self):
        raise NotImplementedError

    def sample_shape(self):
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError


class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat

    def pdclass(self):
        return CategoricalPd

    def param_shape(self):
        return [self.ncat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return t.int32

class CategoricalPd(Pd):

    def __init__(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        np_logits = np.argmax(self.logits, axis=1)
        return t.FloatTensor(np_logits)

    def neglogp(self, x):
        return nn.CrossEntropyLoss(self.logits, x)

    def kl(self, other):
        a0 = self.logits - np.amax(self.logits, axis=1, keepdims=True)
        a1 = other.logits - np.amax(other.logits, axis=1, keepdims=True)
        ea0 = np.exp(a0)
        ea1 = np.exp(a1)
        z0 = np.sum(ea0, axis=1, keepdims=True)
        z1 = np.sum(ea1, axis=1, keepdims=True)
        p0 = ea0 / z0
        np_kl = np.sum(p0 * (a0 - np.log(z0) - a1 + np.log(z1)), axis=1)
        return t.FloatTensor(np_kl)

    def entropy(self):
        a0 = self.logits - np.amax(self.logits, axis=1, keepdims=True)
        ea0 = np.exp(a0)
        z0 = np.sum(ea0, axis=1, keepdims=True)
        p0 = ea0 / z0
        np_entropy = np.sum(p0* (np.log(z0) -a0), axis=1)
        return t.FloatTensor(np_entropy)

    def sample(self):
        u = np.random.uniform(size=np.shape(self.logits))
        np_argmax_u = np.argmax(self.logits - np.log(-np.log(u)), axis=1)
        return t.FloatTensor(np_argmax_u)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

