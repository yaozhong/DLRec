#!/usr/bin/env python

from __future__ import print_function
import numpy as np

try:
	import matplotlib
	matplotlib.use('Agg')  # this need for the linux env
	import matplotlib.pyplot as plt
except ImportError:
	pass

import chainer
import chainer.functions as F
import chainer.links as L

from chainer import reporter
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer import serializers 

import cupy

#################################################
# Baseline model
##################################################

## linear Regression Model
# linear model regresion , default is contained 
# note that try to predict from too distance away
class LM(chainer.Chain):

	def __init__(self, n_units, n_out, n_kernel):
		super(LM, self).__init__()
		with self.init_scope():
			self.l1 = L.Linear(None, n_out)
			
	def __call__(self, x):
		y = self.l1(x)
		return y


class MLP(chainer.Chain):
	def __init__(self, n_unit, n_out, n_kernel = 1):
		super(MLP, self).__init__()
		with self.init_scope():
			self.l1 = L.Linear(None, n_unit)
			self.l2 = L.Linear(None, n_out)

	def __call__(self, x):
		h1 = self.l1(x)
		y = self.l2(h1)
		return y



####################################################
# CNN models
####################################################
class CNN_dev(chainer.Chain):

	def __init__(self, n_unit, n_out, n_kernel = 1):
		super(CNN_dev, self).__init__()
		with self.init_scope():
			# middle part is the k-size
			self.conv1 = L.Convolution2D(n_kernel, 32, (1, 7))
			self.conv2 = L.Convolution2D(32, 64, (1, 3))
			self.conv3 = L.Convolution2D(64, 128, (1, 3))
			self.l1 = L.Linear(None, n_unit)
			self.l2 = L.Linear(None, n_out)

	def __call__(self, x): 
		#h = F.relu(self.conv1(x))
		h = F.max_pooling_2d(F.relu(self.conv1(x)),3)
		h = F.max_pooling_2d(F.relu(self.conv2(h)),3)
		h = F.max_pooling_2d(F.relu(self.conv3(h)),3)
		#h = F.dropout(F.relu(self.l1(h)))

		y = self.l2(h)
		return y



# two layer case (comparision), two layer cases
class CNN_2layer(chainer.Chain):

	def __init__(self, n_unit, n_out, n_kernel = 1):
		super(CNN_2layer, self).__init__()
		with self.init_scope():
			# middle part is the k-size
			self.conv1 = L.Convolution2D(n_kernel, 32, (1, 7))
			self.conv2 = L.Convolution2D(32, 64, (1, 3))
			self.l1 = L.Linear(None, n_unit)
			self.l2 = L.Linear(None, n_out)

	def __call__(self, x): 
		#h = F.relu(self.conv1(x))
		h = F.max_pooling_2d(F.relu(self.conv1(x)),3)
		h = F.max_pooling_2d(F.relu(self.conv2(h)),3)
		y = self.l2(h)
		return y


### this model is used to compare with the ISMB work, small network , no max pooling
class CNN_ismb(chainer.Chain):

	def __init__(self, n_unit, n_out, n_kernel = 1):
		super(CNN_ismb, self).__init__()
		with self.init_scope():
			# middle part is the k-size
			self.conv1 = L.Convolution2D(n_kernel, 32, (1, 7))
			self.conv2 = L.Convolution2D(32, 64, (1, 3))
			self.l1 = L.Linear(None, n_unit)
			self.l2 = L.Linear(None, n_out)

	def __call__(self, x): 
		#h = F.relu(self.conv1(x))
		h = F.relu(self.conv1(x))
		h = F.relu(self.conv2(h))
		#h = F.dropout(F.relu(self.l1(h)))
		y = self.l2(h)
		return y

class CNN_ismb_largeFilter(chainer.Chain):

	def __init__(self, n_unit, n_out, n_kernel = 1):
		super(CNN_ismb_largeFilter, self).__init__()
		with self.init_scope():
			# middle part is the k-size
			self.conv1 = L.Convolution2D(n_kernel, 32, (1, 50))
			self.conv2 = L.Convolution2D(32, 64, (1, 25))
			self.l1 = L.Linear(None, n_unit)
			self.l2 = L.Linear(None, n_out)

	def __call__(self, x): 
		#h = F.relu(self.conv1(x))
		h = F.relu(self.conv1(x))
		h = F.relu(self.conv2(h))
		#h = F.dropout(F.relu(self.l1(h)))
		y = self.l2(h)
		return y


class CNN_dev_largeFilter(chainer.Chain):

	def __init__(self, n_unit, n_out, n_kernel = 1):
		super(CNN_dev_largeFilter, self).__init__()
		with self.init_scope():
			# middle part is the k-size
			self.conv1 = L.Convolution2D(n_kernel, 32, (1, 50))
			self.conv2 = L.Convolution2D(32, 64, (1, 25))
			self.conv3 = L.Convolution2D(64, 128, (1, 10))
			self.l1 = L.Linear(None, n_unit)
			self.l2 = L.Linear(None, n_out)

	def __call__(self, x): 
		#h = F.relu(self.conv1(x))
		h = F.max_pooling_2d(F.relu(self.conv1(x)),3)
		h = F.max_pooling_2d(F.relu(self.conv2(h)),3)
		h = F.max_pooling_2d(F.relu(self.conv3(h)),3)
		#h = F.dropout(F.relu(self.l1(h)))
		y = self.l2(h)
		return y


class CNN_dev_noMaxPool(chainer.Chain):

	def __init__(self, n_unit, n_out, n_kernel = 1):
		super(CNN_dev_noMaxPool, self).__init__()
		with self.init_scope():
			# middle part is the k-size
			self.conv1 = L.Convolution2D(n_kernel, 32, (1, 7))
			self.conv2 = L.Convolution2D(32, 64, (1, 3))
			self.conv3 = L.Convolution2D(64, 128, (1, 3))
			self.l1 = L.Linear(None, n_unit)
			self.l2 = L.Linear(None, n_out)

	def __call__(self, x): 
		#h = F.relu(self.conv1(x))
		h = F.relu(self.conv1(x))
		h = F.relu(self.conv2(h))
		h = F.relu(self.conv3(h))
		#h = F.dropout(F.relu(self.l1(h)))
		y = self.l2(h)
		return y


#### Not used models ####


####################################################
# MLP models
####################################################

# 2-layer perceptron + relu
class MLP_RELU(chainer.Chain):

	def __init__(self, n_unit, n_out, n_kernel):
		super(MLP_RELU, self).__init__()
		with self.init_scope():
			self.l1 = L.Linear(None, n_unit)
			self.l2 = L.Linear(None, n_out)

	def __call__(self, x):
		h1 = F.relu(self.l1(x))
		y = self.l2(h1)
		return y


###########################################################		
#  RNN models, currently seems very slow
###########################################################		

# not determined
class LSTM(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(LSTM, self).__init__()
        with self.init_scope():
            self.l1 = L.LSTM(n_units)
            self.l2 = L.LSTM(n_units)
            self.l3 = L.Linear(u_units, n_out)

    def __call__(self, x):
        h = self.l1(x)
        h = self.l2(h)
        y = self.l3(h)
        return y


###########################################################
# CAE , 2017/11/08 to test the performance of the same data set.
# rethink why CAE results can be well generailzed?
###########################################################
class CAE(chainer.Chain):
	def __init__(self, n_unit, n_out, n_kernel = 1):
		super(CAE, self).__init__()
		with self.init_scope():
			self.conv1 = L.Convolution2D(n_kernel, 32, (1, 7))
			self.conv2 = L.Convolution2D(32, 64, (1, 3))
			self.dconv1 = L.Deconvolution2D(64, 32, (1, 3))
			self.dconv2 = L.Deconvolution2D(32, n_kernel, (1, 7))
			self.l1 = L.Linear(None, n_out)

	def __call__(self, x):
		h = F.max_pooling_2d(F.relu(self.conv1(x)),3)
		h = F.max_pooling_2d(F.relu(self.conv2(h)),3)
		h = F.relu(self.dconv1(h))
		h = F.relu(self.dconv2(h))
		y = self.l1(h)
		return y


class CAE3(chainer.Chain):
	def __init__(self, n_unit, n_out, n_kernel = 1):
		super(CAE3, self).__init__()
		with self.init_scope():
			self.conv1 = L.Convolution2D(n_kernel, 32, (1, 7))
			self.conv2 = L.Convolution2D(32, 64, (1, 3))
			self.conv3 = L.Convolution2D(64, 128, (1, 3))

			self.dconv0 = L.Deconvolution2D(128, 64, (1, 3))
			self.dconv1 = L.Deconvolution2D(64, 32, (1, 3))
			self.dconv2 = L.Deconvolution2D(32, n_kernel, (1, 7))
			self.l1 = L.Linear(None, n_out)

	def __call__(self, x):
		h = F.max_pooling_2d(F.relu(self.conv1(x)),3)
		h = F.max_pooling_2d(F.relu(self.conv2(h)),3)
		h = F.max_pooling_2d(F.relu(self.conv3(h)),3)

		h = F.relu(self.dconv0(h))
		h = F.relu(self.dconv1(h))
		h = F.relu(self.dconv2(h))
		y = self.l1(h)
		return y


###########################################################
#  Autoencoder
###########################################################
class AE(chainer.Chain):
	def __init__(self, n_unit, n_out, n_kernel = 1):
		super(AE, self).__init__()
		with self.init_scope():
			self.el1 = L.Linear(None, n_unit)
			self.el2 = L.Linear(n_unit, int(n_unit/2))
			self.el3 = L.Linear(int(n_unit/2), int(n_unit/4))
			self.dl1 = L.Linear(int(n_unit/4), int(n_unit/2))
			self.dl2 = L.Linear(int(n_unit/2), n_unit)
			self.dl3 = L.Linear(n_unit, n_out)

	def __call__(self, x):
		e = self.el1(x)
		e = self.el2(e)
		e = self.el3(e)

		d= self.dl1(e)
		d = self.dl2(d)
		d = self.dl3(d)

		return d




###########################################################
# U-net , 2017/11/06 to test.(second stage)  with data augmentation, and overlap-tiling implementation
###########################################################

class UNET(chainer.Chain):
    def __init__(self):
    	super(UNET, self).__init__(
    		c0 = L.Convolution2D(4, 32, 3, 1, 1),
    		c1 = L.Convolution2D(32, 64, 4, 2, 1),
    		c2 = L.Convolution2D(64, 64, 3, 1, 1),
    		c3 = L.Convolution2D(64, 128, 4, 2, 1),
    		c4 = L.Convolution2D(128, 128, 3, 1, 1),
    		c5 = L.Convolution2D(128, 256, 4, 2, 1),
    		c6 = L.Convolution2D(256, 256, 3, 1, 1),
    		c7 = L.Convolution2D(256, 512, 4, 2, 1),
    		c8 = L.Convolution2D(512, 512, 3, 1, 1),

    		dc8 = L.Deconvolution2D(1024, 512, 4, 2, 1),
    		dc7 = L.Convolution2D(512, 256, 3, 1, 1),
    		dc6 = L.Deconvolution2D(512, 256, 4, 2, 1),
    		dc5 = L.Convolution2D(256, 128, 3, 1, 1),
    		dc4 = L.Deconvolution2D(256, 128, 4, 2, 1),
    		dc3 = L.Convolution2D(128, 64, 3, 1, 1),
    		dc2 = L.Deconvolution2D(128, 64, 4, 2, 1),
    		dc1 = L.Convolution2D(64, 32, 3, 1, 1),
    		dc0 = L.Convolution2D(64, 3, 3, 1, 1),

    		bnc0 = L.BatchNormalization(32),
    		bnc1 = L.BatchNormalization(64),
    		bnc2 = L.BatchNormalization(64),
    		bnc3 = L.BatchNormalization(128),
    		bnc4 = L.BatchNormalization(128),
    		bnc5 = L.BatchNormalization(256),
    		bnc6 = L.BatchNormalization(256),
    		bnc7 = L.BatchNormalization(512),
    		bnc8 = L.BatchNormalization(512),

    		bnd8 = L.BatchNormalization(512),
    		bnd7 = L.BatchNormalization(256),
    		bnd6 = L.BatchNormalization(256),
    		bnd5 = L.BatchNormalization(128),
    		bnd4 = L.BatchNormalization(128),
    		bnd3 = L.BatchNormalization(64),
    		bnd2 = L.BatchNormalization(64),
    		bnd1 = L.BatchNormalization(32)
        )

    def __calc__(self,x):
    	e0 = F.relu(self.bnc0(self.c0(x)))
        e1 = F.relu(self.bnc1(self.c1(e0)))
        e2 = F.relu(self.bnc2(self.c2(e1)))
        e3 = F.relu(self.bnc3(self.c3(e2)))
        e4 = F.relu(self.bnc4(self.c4(e3)))
        e5 = F.relu(self.bnc5(self.c5(e4)))
        e6 = F.relu(self.bnc6(self.c6(e5)))
        e7 = F.relu(self.bnc7(self.c7(e6)))
        e8 = F.relu(self.bnc8(self.c8(e7)))

        d8 = F.relu(self.bnd8(self.dc8(F.concat([e7, e8]))))
        d7 = F.relu(self.bnd7(self.dc7(d8)))
        d6 = F.relu(self.bnd6(self.dc6(F.concat([e6, d7]))))
        d5 = F.relu(self.bnd5(self.dc5(d6)))
        d4 = F.relu(self.bnd4(self.dc4(F.concat([e4, d5]))))
        d3 = F.relu(self.bnd3(self.dc3(d4)))
        d2 = F.relu(self.bnd2(self.dc2(F.concat([e2, d3]))))
        d1 = F.relu(self.bnd1(self.dc1(d2)))
        d0 = self.dc0(F.concat([e0, d1]))

        return d0

###########################################################		
# loss function calucation, defined above the net work
###########################################################

"""
For the vector-to-vector case redesigned the loss function 
"""


class Augmentor(chainer.Chain):
	def __init__(self, predictor):
		super(Augmentor, self).__init__(predictor=predictor)

	def __call__(self, x, t):
		y = self.predictor(x)

		
		# y_norm = F.normalize(y)
		# t_norm = F.normalize(t)
		#cos_dist = - F.sum(y_norm*t_norm)  # keras apprximate
		cos_dist = - F.sum(y*t) /(F.sqrt(F.sum(y*y)) * F.sqrt(F.sum(t*t)))
		
		#p = 0.01
		#self.loss = F.mean_absolute_error(y, t) * p  +  cos_dist * (1-p)
		#self.accuracy = cos_dist

		self.loss = F.mean_absolute_error(y, t)
		#self.loss = F.mean_squared_error(y, t)
		self.accuracy = cos_dist

		# cosine -proximiate loss

		reporter.report({'loss': self.loss}, self)
		reporter.report({'ploss':self.accuracy}, self)
		return self.loss

	def predict(self, x):
		y = self.predictor(x)
		return y



