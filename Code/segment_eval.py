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

import cnv_cbs

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
	CNN_model_ms = Augmentor(CNN_dev(unit, int(binSize/sbinSize), 6))
	
	serializers.load_npz(model_files[0], LM_model)
	serializers.load_npz(model_files[1], MLP_model)
	serializers.load_npz(model_files[2], CNN_model)
	serializers.load_npz(model_files[3], CNN_model_gms)
	serializers.load_npz(model_files[4], CNN_model_ms)

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

	xFeat = ["M", "S"]
	data3 = Data(lcFile, hcFile, faFile, mbFile, gcFile, binSize, sbinSize, xFeat)

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

	rd3, X3, T3  = data3.getData4TR(rg)
	Y_CNN_ms = CNN_model_gms.predictor(X3).data

	n = rd.shape[1]
	if outSbin > 1:
		Y_LM =  vecSmooth(Y_LM, outSbin)
		Y_MLP = vecSmooth(Y_MLP, outSbin)
		Y_CNN = vecSmooth(Y_CNN, outSbin)
		Y_CNN_gms = vecSmooth(Y_CNN_gms, outSbin)
		Y_CNN_ms = vecSmooth(Y_CNN_ms, outSbin)
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
	p_CNN_ms, = plt.plot(idx, Y_CNN_ms.reshape(n,), "p--")
	plt.legend([p_LM,p_MLP,p_CNN, p_CNN_gms], ["LM", "MLP", "CNN", "CNN+gms", "CNN+ms"], 
		#loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
		loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4 , fancybox=True, shadow=False)
	plt.savefig(fig_out_file)

	#plt.legend([p1,p2,p3], ["low-depth", "high-depth", "Reconstructed"]
	#plt.ylim(0,100)
	#plt.xlim([0,6000])
	#print "@ Error for the target region:{}".format(F.mean_absolute_error(Y, T).data)


def segmentRegion2CSV(outPath, model_files, rgs, outSbin = 10, binSize = 1000, sbinSize=1):

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
	CNN_model_ms = Augmentor(CNN_dev(unit, int(binSize/sbinSize), 6))
	
	serializers.load_npz(model_files[0], LM_model)
	serializers.load_npz(model_files[1], MLP_model)
	serializers.load_npz(model_files[2], CNN_model)
	serializers.load_npz(model_files[3], CNN_model_gms)
	serializers.load_npz(model_files[4], CNN_model_ms)


	# data loading
	xFeat = []
	data = Data(lcFile, hcFile, faFile, mbFile, gcFile, binSize, sbinSize, xFeat)
	
	xFeat = ["G", "M", "S"]
	data2 = Data(lcFile, hcFile, faFile, mbFile, gcFile, binSize, sbinSize, xFeat)

	xFeat = ["M", "S"]
	data3 = Data(lcFile, hcFile, faFile, mbFile, gcFile, binSize, sbinSize, xFeat)

	for rg in rgs:
		rgName = rg[0] + "_" + str(rg[1]) + "_" + str(rg[2])
		out_file = outPath + "/" + rgName + ".csv"
		
		# get the read depth of coverage for the target regions
		rd, X, T  = data.getData4TR(rg)

		Y_LM = LM_model.predictor(X).data
		Y_MLP = MLP_model.predictor(X).data
		Y_CNN = CNN_model.predictor(X).data

		rd2, X2, T2  = data2.getData4TR(rg)
		Y_CNN_gms = CNN_model_gms.predictor(X2).data

		rd3, X3, T3  = data3.getData4TR(rg)
		Y_CNN_ms = CNN_model_ms.predictor(X3).data

		n = rd.shape[1]
		if outSbin > 1:
			Y_LM =  vecSmooth(Y_LM, outSbin)
			Y_MLP = vecSmooth(Y_MLP, outSbin)
			Y_CNN = vecSmooth(Y_CNN, outSbin)
			Y_CNN_gms = vecSmooth(Y_CNN_gms, outSbin)
			Y_CNN_ms = vecSmooth(Y_CNN_ms, outSbin)
			rd = vecSmooth(rd, outSbin)
			T = vecSmooth(T, outSbin)
			n = len(rd)
		else: 
			# formatting 
			Y_LM =  Y_LM.reshape(n,)
			Y_MLP = Y_MLP.reshape(n,)
			Y_CNN = Y_CNN.reshape(n,)
			Y_CNN_gms = Y_CNN_gms.reshape(n,)
			Y_CNN_ms = Y_CNN_ms.reshape(n,)
			rd = rd.reshape(n,)
			T = T.reshape(n,)

		fout = open(out_file, "w+")
		fout.write("ld,hd,LM,MLP,CNN,CNN_gms,CNN_ms\n")
		# combine the table
		for i in range(n):
			fout.write(",".join([str(rd[i]), str(T[i]), str(Y_LM[i]), str(Y_MLP[i]), str(Y_CNN[i]), str(Y_CNN_gms[i]), str(Y_CNN_ms[i]) ]))
			fout.write("\n")
		fout.close()

def getRegionsFromFile(filePath):

	rgs = []
	with open(filePath) as f:
		for line in f:
			elems = line.split("\t")
			rgs.append((elems[0], int(elems[1]), int(elems[2])))
	return rgs


def genCNVData():

	path = "/data/exp-archieve/dlnorm_201708_/model_cmp_reconstruct/"
 	path = path + "model_20170921/"

	model_files =[path+"LM_b1000_sb1_hu1000_sn1e8_ep97_xf-_.model",
	path+"MLP_b1000_sb1_hu1000_sn1e8_ep37_xf-_.model",
	path+"CNN_dev_b1000_sb1_hu1000_sn1e8_ep22_xf-_.model", 
	path+"CNN_dev_b1000_sb1_hu1000_sn1e8_ep7_xf-GMS_.model",
	path+"CNN_dev_b1000_sb1_hu1000_sn1e8_ep14_xf-MS_.model"]


	### loading the testRegion Lists tables
	rgs = getRegionsFromFile("rgInfo/test_regions.bed")

	dateTag = time.strftime("%Y%m%d")
	smooth = 500
	fig_out_fold = "./" + dateTag + "_test_cnv_data_s" + str(smooth) + "/" 
	if not os.path.exists(fig_out_fold):
		os.makedirs(fig_out_fold)

	segmentRegion2CSV(fig_out_fold, model_files, rgs, smooth, 1000, 1)

if __name__ == '__main__':

	genCNVData()
 	
	## saving the csv file of the read depth information

	## segment the target regionss: (ld, hd, rc) three different performance.
