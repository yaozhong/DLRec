from __future__ import division
import argparse

import numpy as np
from  dl_model import *
from  dataUtil import *

import matplotlib
matplotlib.use('Agg')  # this need for the linux env
import matplotlib.pyplot as plt

import os
import timeit


# one sample test
def test_region(rg=None):

	parser = argparse.ArgumentParser(description='Reconstruction testing')
	parser.add_argument('--xFeat', "-xf",  type=list, action='store', default='', help='list of extra features used for the data G/M/S')
	parser.add_argument('--unit', '-u', type=int, default=1024, help='Number of units')
	parser.add_argument('--modelPath', '-mp', default='model', help='model output path')
	parser.add_argument('--model', '-m', default=LM, help='deep learning model')
	parser.add_argument('--binSize', '-b', type=int, default=1000, help='binSize')
	parser.add_argument('--sbinSize', '-sb', type=int, default=10, help='taking small avergy in the given range')
	parser.add_argument('--sampleNum', '-sn', default='1e7', help='sample size scale')
	parser.add_argument('--out', '-o', default='figure', help='Directory to output the result')
	
	
	args = parser.parse_args()
	binSize = args.binSize
	args.unit = int(args.binSize/args.sbinSize)

	n_kernel = 1
	for xf in args.xFeat:
		if xf not in ['G', 'M', 'S']:
			print "[warning]: Please check the extra feature parameters, current only support G/M/S"
			return -1

		if xf == 'S':
			n_kernel +=4
		else:
			n_kernel += 1

	model = Augmentor(globals()[args.model](args.unit, int(args.binSize/args.sbinSize), n_kernel))
	print("loading the model from %s" %(args.modelPath))
	serializers.load_npz(args.modelPath, model)

	if rg is None:
		rg = ('1', 117199956, 117206216)

	# loading the target regions
	lcFile = "/data/Dataset/1000GP/phase3/NA12878/NA12878.mapped.ILLUMINA.bwa.CEU.low_coverage.20121211.bam"
	hcFile = "/data/Dataset/1000GP/phase3/NA12878/NA12878.mapped.ILLUMINA.bwa.CEU.high_coverage_pcr_free.20130906.bam"
	faFile = "/data/Dataset/1000GP/reference/hs37d5.fa"
	mbFile = "/data/Dataset/1000GP/reference/mappability/wgEncodeCrgMapabilityAlign100mer.bigWig"
	gcFile = "/data/Dataset/1000GP/reference/GC/gc5Base.bigWig"

	outFileName = "_".join([args.model, 
		"b"+str(args.binSize), "sb"+str(args.sbinSize),
		"hu"+str(args.unit), "xf-"+"".join(args.xFeat), "sn"+args.sampleNum])

	data = Data(lcFile, hcFile, faFile, mbFile, gcFile, binSize, args.sbinSize, args.xFeat)
	rd, X, T  = data.getData4TR(rg)
	Y = model.predictor(X).data
	
	n = rd.shape[1]
	ld = rd[0,:]
	idx = np.arange(n)
	plt.figure(figsize=(10,3), dpi=100)
	p1, = plt.plot(idx, ld.reshape(n,))
	p2, = plt.plot(idx, T.reshape(n,), color="red")
	p3, = plt.plot(idx, Y.reshape(n,), color="blue")
	plt.legend([p1,p2,p3], ["low-depth", "high-depth", "Reconstructed"])

	if not os.path.exists("./"+args.out):
		os.makedirs("./"+args.out)
	plt.savefig("./"+args.out+"/reconstruction_"+outFileName+".png")
	print "@ Error for the target region:{}".format(F.mean_absolute_error(Y, T).data)


if __name__ == '__main__':
	test_region()
	
