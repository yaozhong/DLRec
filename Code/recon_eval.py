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

# checking the output of the prediciton results
def vecSmooth(vec, sbin = 10):

	vec = vec.reshape(vec.shape[0]*vec.shape[1],)
	if sbin == 1:
		return vec

	if sbin > 1:
		vl = len(vec)
		bn = int(vl/sbin)
		svec = np.zeros(bn, dtype= vec.dtype)

		for i in xrange(bn):
			svec[i] = int(np.mean(vec[i*sbin:(i+1)*sbin]))

		return svec.reshape(bn,)

def reconstruct_signal_cmp(model_files, rg=('1', 117199956, 117206216), outSbin = 10, binSize = 1000, sbinSize=1):

	## preload information
	# loading the target regions
	lcFile = "/data/Dataset/1000GP/phase3/NA12878/NA12878.mapped.ILLUMINA.bwa.CEU.low_coverage.20121211.bam"
	hcFile = "/data/Dataset/1000GP/phase3/NA12878/NA12878.mapped.ILLUMINA.bwa.CEU.high_coverage_pcr_free.20130906.bam"
	faFile = "/data/Dataset/1000GP/reference/hs37d5.fa"
	mbFile = "/data/Dataset/1000GP/reference/mappability/wgEncodeCrgMapabilityAlign100mer.bigWig"
	gcFile = "/data/Dataset/1000GP/reference/GC/gc5Base.bigWig"
	erFile = "/data/Dataset/1000GP/phase3/p3_ref/exome_pull_down_targets/20130108.exome.targets.bed"

	unit = int(binSize/sbinSize)

	# loading realted models
	LM_model = Augmentor(LM(unit, int(binSize/sbinSize), 1))
	MLP_model = Augmentor(MLP(unit, int(binSize/sbinSize), 1))
	CNN_model = Augmentor(CNN_dev(unit, int(binSize/sbinSize), 1))
	CNN_model_gms = Augmentor(CNN_dev(unit, int(binSize/sbinSize), 7))
	
	serializers.load_npz(model_files[0], LM_model)
	serializers.load_npz(model_files[1], MLP_model)
	serializers.load_npz(model_files[2], CNN_model)
	serializers.load_npz(model_files[3], CNN_model_gms)

	# regions construct file fold
	dateTag = time.strftime("%Y%m%d")
	fig_out_fold = "./recon_figure_" + dateTag
	if not os.path.exists(fig_out_fold):
		os.makedirs(fig_out_fold)

	# data loading
	xFeat = []
	data = Data(lcFile, hcFile, faFile, mbFile, gcFile, binSize, sbinSize, xFeat)
	
	xFeat = ["G", "M", "S"]
	data2 = Data(lcFile, hcFile, faFile, mbFile, gcFile, binSize, sbinSize, xFeat)

	print rg
	rgName = "chr-"+ rg[0] + "_(" + str(rg[1]) + "," + str(rg[2]) + ")"
	fig_out_file = fig_out_fold + "/" + rgName + ".jpeg"
	# get the read depth of coverage for the target regions
	rd, X, T  = data.getData4TR(rg)

	Y_LM = LM_model.predictor(X).data
	Y_MLP = MLP_model.predictor(X).data
	Y_CNN = CNN_model.predictor(X).data

	rd2, X2, T2  = data2.getData4TR(rg)
	Y_CNN_gms = CNN_model_gms.predictor(X2).data

	n = rd.shape[1]
	if outSbin > 1:
		Y_LM =  vecSmooth(Y_LM, outSbin)
		Y_MLP = vecSmooth(Y_MLP, outSbin)
		Y_CNN = vecSmooth(Y_CNN, outSbin)
		Y_CNN_gms = vecSmooth(Y_CNN_gms, outSbin)
		rd = vecSmooth(rd, outSbin)
		T = vecSmooth(T, outSbin)
		n = len(rd)
		ld = rd
	else: 
		# smoothing all the data
		ld = rd[0,:]

	# basic plot of te graph for the reconstruction results
	idx = np.arange(n)
	
	fig = plt.figure(figsize=(10,3), dpi=110)
	p1, = plt.plot(idx, ld.reshape(n,), color= "black")
	p2, = plt.plot(idx, T.reshape(n,), color="black")

	p_LM, = plt.plot(idx, Y_LM.reshape(n,), "g--")
	p_MLP, = plt.plot(idx, Y_MLP.reshape(n,), "b--")
	p_CNN, = plt.plot(idx, Y_CNN.reshape(n,), "r--")
	p_CNN_gms, = plt.plot(idx, Y_CNN_gms.reshape(n,), "y--")
	plt.legend([p_LM,p_MLP,p_CNN, p_CNN_gms], ["LM", "MLP", "CNN", "CNN+gms"], 
		#loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
		loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4 , fancybox=True, shadow=False)
	plt.savefig(fig_out_file)

	#plt.legend([p1,p2,p3], ["low-depth", "high-depth", "Reconstructed"]
	#plt.ylim(0,100)
	#plt.xlim([0,6000])
	#print "@ Error for the target region:{}".format(F.mean_absolute_error(Y, T).data)


if __name__ == '__main__':
 	
 	
 	path = "/data/exp-archieve/dlnorm_201708_/model_cmp_reconstruct/"
 	path = path + "model_20170921/"

	model_files =[path+"LM_b1000_sb1_hu1000_sn1e8_ep97_xf-_.model",
	path+"MLP_b1000_sb1_hu1000_sn1e8_ep37_xf-_.model",
	path+"CNN_dev_b1000_sb1_hu1000_sn1e8_ep22_xf-_.model", 
	path+"CNN_dev_b1000_sb1_hu1000_sn1e8_ep7_xf-GMS_.model"]

	rgs_test =[("8",145432588,145977588),('11',51090853,51590853),('7',142098195,142273195),('21',44888050,45018050),('10', 51448845,51873845)]
	rgs_train = [('1',10000,175000), ('7',61510465,61675465)]
	#rgs = [("10",51538845,51543845), ("10",51539345,51544345)]
	rgs = rgs_test
	# zoom the pictures
	zoom = 0
	for rg in rgs:	
		if zoom > 0:
			rgLen = rg[2] - rg[1]
			if rgLen > zoom:
				rgLen = rgLen - rgLen%zoom

			n = int(rgLen/zoom)

			for i in range(n):
				reconstruct_signal_cmp(model_files, (rg[0], rg[1]+i*zoom, rg[1]+(i+1)*zoom),10)
		else:
			reconstruct_signal_cmp(model_files, rg, 1000)
	