#!/usr/bin/env python

"""
Author: Yao-zhong Zhang @ IMSUT
Date:   2017-08-08 
version: 0.03
Revised version for the data generation process.
This include the option that incorporate different features

"""
import pyfaidx
from seqFeatures import *
from coverage import *
from accessRegions import *
import pyBigWig
from chainer.datasets import tuple_dataset
from numpy import inf
import numpy as np

class Data:

	def __init__(self, lcBamFile, hcBamFile, fastaFile, mappableFile, gcFile, binSize, sbinSize,xFeats):
		
		self.xFeats = xFeats

		self.lcBam = pysam.AlignmentFile(lcBamFile, "rb")
		self.hcBam = pysam.AlignmentFile(hcBamFile, "rb")
		self.faFile = fastaFile
		self.faIdx = pyfaidx.Fasta(fastaFile, as_raw=True)
		self.mapBW = pyBigWig.open(mappableFile)
		self.gcBW = pyBigWig.open(gcFile)
		
		self.binSize = binSize
		self.sbinSize = sbinSize

		# data results
		self.regions = None
		self.rgLists = None

	def genDLTrainData(self, min_join_gap, sampleN):

		# generate a list of accessible region list
		rgLists = list_access_regions(self.faFile)
		if(min_join_gap > 0):
			self.regions = join_small_gap(rgLists, min_join_gap)

		# generate the training and the test data set. 
		print "## Sampling regions from whole chromesone ..."	
		#self.rgLists = sampleRegions(self.faFile, self.regions, sampleN, self.binSize)
		rgLists_train, rgLists_test = sampleRegions_trainTest(self.faFile, self.regions, sampleN, self.binSize)

		print "## Prepare training data for the sampled regions..."
		train, valid = self.getData4Regions_aug(rgLists_train, True, False, True, False)
		test = self.getData4Regions_aug(rgLists_test, False, False, True, False)

		return train, valid, test

	def getData4Regions(self, rgLists):

		Xs, Ys = [], []
		nSample = 0
		binSize = self.binSize

		for rg in rgLists:
			vecLen = rg[2] - rg[1]
			brow = int(vecLen/binSize)
			bcol = binSize
			nSample += brow
			rg = (rg[0], rg[1], rg[1] + brow*bcol)

			lcVec = getARcoverage_rd(rg, self.lcBam, 10)
			hcVec = getARcoverage_rd(rg, self.hcBam, 10)

			lcVec = vecSmooth(lcVec, self.sbinSize)  
			hcVec = vecSmooth(hcVec, self.sbinSize)

			X = lcVec
			# additional features used as input
			if 'M' in self.xFeats:
				mps = getRegionMappabilit(rg, self.mapBW)
				mps = vecSmooth(mps, self.sbinSize)
				X = np.vstack((X, mps))
				
			if 'G' in self.xFeats:
				gc = getRegionGC(rg, self.gcBW)
				gc = vecSmooth(gc, self.sbinSize)
				X = np.vstack((X, gc))	

			if 'S' in self.xFeats:
				seqr = self.faIdx[rg[0]][rg[1]:rg[2]]
				seqMat = seq2onehotMat(seqr, self.sbinSize) 
				X = np.vstack((X, seqMat))

			Xs.append(X)
			Ys.append(hcVec)

		# binSize changed after smooth
		binSize = binSize/self.sbinSize

		Y = np.concatenate(tuple(Ys), 1) 
		Y  = Y.reshape(nSample, binSize) 

		X = np.concatenate(tuple(Xs), 1)

		inContent = []
		n_kernel = X.shape[0]
		for i in xrange(n_kernel):
			xt = X[i,:].reshape(nSample, binSize)
			inContent.append(xt)

		Xdata = np.stack(tuple(inContent), axis = 1)
		X = Xdata.reshape(nSample, n_kernel, 1, binSize)

		#split data for training and testing
		tidx = int(0.8*nSample)
		X_train, X_valid_test = np.split(X.astype(np.float32), [tidx])
		Y_train, Y_valid_test = np.split(Y.astype(np.float32), [tidx])

		tidx = int(0.1*nSample)
		X_test, X_valid = np.split(X_valid_test, [tidx])
		Y_test, Y_valid = np.split(Y_valid_test, [tidx])

		# display mean and variance
		# mae score for the directly mapping
		if(False):
			hc = Y_train.reshape(int(0.8*nSample)*binSize, 1)
			lc = X_train.reshape(int(0.8*nSample)*binSize, 1)
			ratio = np.ma.masked_invalid(np.divide(hc, lc)).mean()
			print("[INFO]: Ratio of %f" %ratio)
			print("[INFO]: std is %f" %(np.ma.masked_invalid(np.divide(hc, lc)).std()))
			dist = np.mean(np.apply_along_axis(sum, 1, np.abs(lc*ratio - hc)))
			print("[INFO]: Basic MAE is %f" %dist)


		train = tuple_dataset.TupleDataset(X_train, Y_train)
		test = tuple_dataset.TupleDataset(X_test, Y_test)
		valid = tuple_dataset.TupleDataset(X_valid, Y_valid)

		print("## Total generated data has [%d] bins" %(nSample))
		return train, test, valid

	def getData4Regions_aug(self, rgLists, train = True, dataAug=False, shuffle=False, verbose=False):

		Xs, Ys = [], []
		nSample = 0
		binSize = self.binSize

		for rg in rgLists:
			vecLen = rg[2] - rg[1]
			brow = int(vecLen/binSize)
			bcol = binSize
			nSample += brow
			rg = (rg[0], rg[1], rg[1] + brow*bcol)

			lcVec = getARcoverage_rd(rg, self.lcBam, 10)
			hcVec = getARcoverage_rd(rg, self.hcBam, 10)

			lcVec = vecSmooth(lcVec, self.sbinSize)  
			hcVec = vecSmooth(hcVec, self.sbinSize)

			X = lcVec
			# additional features used as input
			if 'M' in self.xFeats:
				mps = getRegionMappabilit(rg, self.mapBW)
				mps = vecSmooth(mps, self.sbinSize)
				X = np.vstack((X, mps))
				
			if 'G' in self.xFeats:
				gc = getRegionGC(rg, self.gcBW)
				gc = vecSmooth(gc, self.sbinSize)
				X = np.vstack((X, gc))	

			if 'S' in self.xFeats:
				seqr = self.faIdx[rg[0]][rg[1]:rg[2]]
				seqMat = seq2onehotMat(seqr, self.sbinSize) 
				X = np.vstack((X, seqMat))

			Xs.append(X)
			Ys.append(hcVec)

		# binSize changed after smooth
		binSize = binSize/self.sbinSize

		Y = np.concatenate(tuple(Ys), 1) 
		Y  = Y.reshape(nSample, binSize) 

		X = np.concatenate(tuple(Xs), 1)

		inContent = []
		n_kernel = X.shape[0]
		for i in xrange(n_kernel):
			xt = X[i,:].reshape(nSample, binSize)
			inContent.append(xt)

		Xdata = np.stack(tuple(inContent), axis = 1)
		X = Xdata.reshape(nSample, n_kernel, 1, binSize)

		# shuffling samples 
		if(shuffle):
			np.random.seed(100)
			ridx = np.random.permutation(nSample)
			X = X[ridx,:,:,:]
			Y = Y[ridx,:]

		
		if(verbose): 
			print("[Warning]: Note this only working for the no additional bioloical information!!")
			hc = Y.reshape(nSample*binSize, 1)
			lc = X.reshape(nSample*binSize, 1)
			ratio = np.ma.masked_invalid(np.divide(hc, lc)).mean()
			print("[INFO]: Ratio of %f" %ratio)
			print("[INFO]: std is %f" %(np.ma.masked_invalid(np.divide(hc, lc)).std()))
			dist = np.mean(np.apply_along_axis(sum, 1, np.abs(lc*ratio - hc)))
			print("[INFO]: Basic MAE is %f" %dist)

		#split data for training and testing
		if(train):
			tidx = int(0.9*nSample)
			X_train, X_valid = np.split(X.astype(np.float32), [tidx])
			Y_train, Y_valid = np.split(Y.astype(np.float32), [tidx])

			# Data augmentation, not totally tested yet.
			if(dataAug):
				# shifting the positions (guess not big improvement)
				X_train = np.vstack((X_train, X_train[:,:,:,::-1]))
				Y_train = np.vstack((Y_train, Y_train[:,::-1]))

			train = tuple_dataset.TupleDataset(X_train, Y_train)
			valid = tuple_dataset.TupleDataset(X_valid, Y_valid)

			print("## Training data has [%d] bins" %(nSample))
			return train, valid
		else:
			print("## Generating data has [%d] bins" %(nSample))
			rd_data = tuple_dataset.TupleDataset(X.astype(np.float32), Y.astype(np.float32))
			return rd_data


	# only extract input and output data for the target region
	def getData4TR(self, trg):

		Xs, Ys = [], []
		nSample = 0
		binSize = self.binSize
		rg = trg

		vecLen = rg[2]-rg[1]
		brow = int(vecLen/binSize)
		bcol = binSize
		nSample += brow
		rg = (rg[0], rg[1], rg[1]+brow*bcol)

		lcVec = getARcoverage_rd(rg, self.lcBam, 10)
		hcVec = getARcoverage_rd(rg, self.hcBam, 10)

		lcVec = vecSmooth(lcVec, self.sbinSize)  
		hcVec = vecSmooth(hcVec, self.sbinSize)

		X = lcVec
		Y = hcVec
		
		# additional features used as input
		if 'M' in self.xFeats:
			mps = getRegionMappabilit(rg, self.mapBW)
			mps = vecSmooth(mps, self.sbinSize)
			X = np.vstack((X, np.nan_to_num(mps)))
				
		if 'G' in self.xFeats:
			gc = getRegionGC(rg, self.gcBW)
			gc = vecSmooth(gc, self.sbinSize)
			X = np.vstack((X, gc))	

		if 'S' in self.xFeats:
			seqr = self.faIdx[rg[0]][rg[1]:rg[2]]
			seqMat = seq2onehotMat(seqr[:brow*bcol], self.sbinSize)
			X = np.vstack((X, seqMat))

		binSize = binSize/self.sbinSize

		Y  = Y.reshape(nSample, binSize)

		inContent = []
		n_kernel = X.shape[0]
		for i in xrange(n_kernel):
			xt = X[i,:].reshape(nSample, binSize)
			inContent.append(xt)

		Xdata = np.stack(tuple(inContent), axis = 1)
		X = Xdata.reshape(nSample, n_kernel, 1, binSize)

		return lcVec.reshape(1,brow*binSize), X.astype(np.float32), Y.astype(np.float32)

#########################################################################
