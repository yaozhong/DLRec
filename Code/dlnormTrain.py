#!/usr/bin/env python

"""
Author: Yao-zhong Zhang @ IMSUT
Date:   2017-08-17
Description: 

	dlnorm is a package used for learning the mapping from low-depth WGS data to high-depth NGS data using deep learning models.
	The algorithm is built with chainer 2.0.2 version.

MIT License

Copyright (c) [2017] [Yao-zhong Zhang]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import division
import argparse

import numpy as np
from  dl_model import *
from  dataUtil import *

import os
import timeit

"""
preData function is loading the data from bam, fasta and 
additional biological features, such as mappability and GC content information.
"""
def preData(binSize, sbinSize, sampleN, xfeats, minGap=10):

	print "\n# Preparing the data ... "

	#lcFile = "/data/Dataset/1000GP/phase3/NA12878/downSampling/25x.bam"
	lcFile = "/data/Dataset/1000GP/phase3/NA12878/NA12878.mapped.ILLUMINA.bwa.CEU.low_coverage.20121211.bam"
	hcFile = "/data/Dataset/1000GP/phase3/NA12878/NA12878.mapped.ILLUMINA.bwa.CEU.high_coverage_pcr_free.20130906.bam"
	faFile = "/data/Dataset/1000GP/reference/hs37d5.fa"
	mbFile = "/data/Dataset/1000GP/reference/mappability/wgEncodeCrgMapabilityAlign100mer.bigWig"
	gcFile = "/data/Dataset/1000GP/reference/GC/gc5Base.bigWig"

    # xfeats is the  [R, G, M, S]
	data = Data(lcFile, hcFile, faFile, mbFile, gcFile, binSize, sbinSize, xfeats)

	# revised to be Train/test/validation
	train, test, valid = data.genDLTrainData(minGap, sampleN)

	return train, test, valid


def main():

	# running deep leanring models
	parser = argparse.ArgumentParser(description='DL for RD from low to high')

	## training options
	parser.add_argument('--batchsize', '-mb', type=int, default=128, help='Number of bins in each mini-batch')
	parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of iters over the dataset for train')
	parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
	parser.add_argument('--modelPath', '-mp', default='model', help='model output path')

	parser.add_argument('--frequency', '-f', type=int, default=-1, help='Frequency of taking a snapshot')
	parser.add_argument('--resume', '-r', default='', help='Resume the raining from snapshot')

	parser.add_argument('--gpu', '-g', type=int, default=1, help='GPU ID (if none -1)')
	parser.add_argument('--binSize', '-b', type=int, default=1000, help='binSize')
	parser.add_argument('--sbinSize', '-sb', type=int, default=10, help='taking small avergy in the given range, should be divided by binSize')

	parser.add_argument('--unit', '-u', type=int, default=1024, help='Number of units')
	parser.add_argument('--model', '-m', default=LM, required=True, help='deep learning model')
	parser.add_argument('--sampleNum', '-sn', default='1e7', help='sample size scale')
	parser.add_argument('--xFeat', "-xf",  type=list, action='store', default='', help='list of extra features used for the data G/M/S')
	parser.add_argument('--addInfo', '-ai',  default='', help='Additional information used for distinguish the featurs of output')

	args = parser.parse_args()
	args.unit = int(args.binSize/args.sbinSize)

	if not os.path.exists("./" + args.out):
		os.makedirs("./" + args.out)

	sampleN = int(eval(args.sampleNum))
	outFileName = "_".join([args.model, 
		"b"+str(args.binSize), "sb"+str(args.sbinSize),
		"hu"+str(args.unit), "sn"+args.sampleNum, 
		"ep"+str(args.epoch), "xf-"+"".join(args.xFeat), args.addInfo])
	

	n_kernel = 1
	for xf in args.xFeat:
		if xf not in ['G', 'M', 'S']:
			print "[warning]: Please check the extra feature parameters, current only support G/M/S"
			return -1

		if xf == 'S':
			n_kernel +=4
		else:
			n_kernel += 1

	train, test, valid = preData(args.binSize, args.sbinSize, sampleN, args.xFeat)

	print '-----------Data Loading Done---------------'
	print('\n# [Model]:{}'.format(args.model))
	print('# [GPU]: {}'.format(args.gpu))
	print('# [unit]: {}'.format(args.unit))
	print('# [Minibatch-size]: {}'.format(args.batchsize))
	print('# [epoch]: {}'.format(args.epoch))
	print('# [Data sample scale]: {}'.format(args.sampleNum))
	print('# [Bin Size and smooth Bin size]: {}/{}'.format(args.binSize, args.sbinSize))
	print('')

	model = Augmentor(globals()[args.model](args.unit, int(args.binSize/args.sbinSize), n_kernel))

	if args.gpu >= 0:
		chainer.cuda.get_device_from_id(args.gpu).use()
		model.to_gpu()

	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)

	train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
	valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize, repeat=False, shuffle=False)
	# testing set 
	test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

	# setup a trainer
	updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
	trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
	# evluate the model with the test data set
	trainer.extend(extensions.Evaluator(valid_iter, model, device=args.gpu))
	# validation_1 is the test
	trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
	trainer.extend(extensions.dump_graph('main/loss'))

	trainer.extend(extensions.LogReport(log_name=outFileName + ".log"))

	if extensions.PlotReport.available():
		trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss', 'validation_1/main/loss'], 'epoch', file_name= outFileName+"_loss.png"))
		trainer.extend(extensions.PlotReport(['main/ploss', 'validation/main/ploss', 'validation_1/main/ploss'], 'epoch', file_name= outFileName+"_mae.png"))

	trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'validation_1/main/loss', 'main/ploss', 'validation/main/ploss','validation_1/main/ploss', 'elapsed_time']))	
	#trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/ploss', 'validation/main/ploss', 'elapsed_time']))	
	## trainer.extend(extensions.ProgressBar())

	if args.resume:
		chainer.serializers.load_npz(args.resume, trainer)

	## running the training process
	trainer.run()
	
	## saving the model 
	if not os.path.exists(args.modelPath):
		os.makedirs(args.modelPath)
	serializers.save_npz(args.modelPath +'/'+ outFileName +'.model', model)


def genRegion():

	# running deep leanring models
	parser = argparse.ArgumentParser(description='DL for RD from low to high')

	## training options
	parser.add_argument('--batchsize', '-mb', type=int, default=128, help='Number of bins in each mini-batch')
	parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of iters over the dataset for train')
	parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
	parser.add_argument('--modelPath', '-mp', default='model', help='model output path')

	parser.add_argument('--frequency', '-f', type=int, default=-1, help='Frequency of taking a snapshot')
	parser.add_argument('--resume', '-r', default='', help='Resume the raining from snapshot')

	parser.add_argument('--gpu', '-g', type=int, default=1, help='GPU ID (if none -1)')
	parser.add_argument('--binSize', '-b', type=int, default=1000, help='binSize')
	parser.add_argument('--sbinSize', '-sb', type=int, default=10, help='taking small avergy in the given range, should be divided by binSize')

	parser.add_argument('--unit', '-u', type=int, default=1024, help='Number of units')
	parser.add_argument('--model', '-m', default=LM, required=True, help='deep learning model')
	parser.add_argument('--sampleNum', '-sn', default='1e7', help='sample size scale')
	parser.add_argument('--xFeat', "-xf",  type=list, action='store', default='', help='list of extra features used for the data G/M/S')
	parser.add_argument('--addInfo', '-ai',  default='', help='Additional information used for distinguish the featurs of output')

	args = parser.parse_args()
	args.unit = int(args.binSize/args.sbinSize)

	#if not os.path.exists("./" + args.out):
	#	os.makedirs("./" + args.out)

	sampleN = int(eval(args.sampleNum))
	outFileName = "_".join([args.model, 
		"b"+str(args.binSize), "sb"+str(args.sbinSize),
		"hu"+str(args.unit), "sn"+args.sampleNum, 
		"ep"+str(args.epoch), "xf-"+"".join(args.xFeat), args.addInfo])
	

	n_kernel = 1
	for xf in args.xFeat:
		if xf not in ['G', 'M', 'S']:
			print "[warning]: Please check the extra feature parameters, current only support G/M/S"
			return -1

		if xf == 'S':
			n_kernel +=4
		else:
			n_kernel += 1

	train, test, valid = preData(args.binSize, args.sbinSize, sampleN, args.xFeat)

	
if __name__ == '__main__':

	main()
	#genRegion()



