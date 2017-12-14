#!/sur/bin/env python
"""
Date: 2017-06-30
Author: Yao-zhong Zhang
Description:  fasta file processing, get the accessible regions (non 'N' sequences)
the implementaiton referred to CNVkit access.py function
#!/usr/bin/python

*log: 2017-8-25, changed the sampleRegion to 0-based distance calucating
*log: 2017-9-20, fix the testing set for evaluation.
"""
from __future__ import division
import logging
import numpy as np
import pyfaidx
from seqFeatures import *
import time
import random
import os

# sub main function
def get_access_regions(fastaFile, min_join_gap=10, min_region=1000):

	regions = list_access_regions(fastaFile)
	if(min_join_gap > 0):
		regions = join_small_gap(regions, min_join_gap)
	
	return regions

def get_access_regions_withSeq(fastaFile, min_join_gap=10, min_region=1000):

	regions = list_access_regions(fastaFile)
	if(min_join_gap > 0):
		regions = join_small_gap(regions, min_join_gap)
	
	# filter out 
	# tgt_chrs = genTargetChrNameFromFasta(fastaFile)
	# return(filterRegions(tgt_chrs, regions, min_region))
	return(filterRegions_Seq(fastaFile, regions, min_region))

# 2017/07/06 revised version for the fastq analysis
def list_access_regions(fastaFile):

	with open(fastaFile, 'r') as infile:
		chrom = cursor = run_start = None

		for line in infile:
			# processing the chrome name line
			if line.startswith('>'):
				# output the previous accesible region
				if run_start is not None:
					yield log_this(chrom, run_start, cursor)

				# start new 
				chrom = line.split(None, 1)[0][1:]
				run_start = None
				cursor = 0
				logging.info("%s: Scanning for accessbile regions", chrom)

			else:
				# processing the sequence line
				line = line.rstrip()

				# for non-'N' line
				if 'N' not in line:
					if run_start is None:
						run_start = cursor
				# 'N' is in the line
				else:
					if all( c == 'N' for c  in line): 
						if run_start is not None:
							yield log_this(chrom, run_start, cursor)
							run_start = None
					else:
						
						# non full 'N' situation, at least one N, checking the speed?
						idxs = np.array([idx for (idx, val) in enumerate(line) if val == 'N'])
						if run_start is not None:
							yield log_this(chrom, run_start, cursor + idxs[0])
						elif idxs[0] != 0:
							yield log_this(chrom, cursor, cursor + idxs[0])

						# gap case: (revise)
						gap_mask = np.diff(idxs) > 1 
						if gap_mask.any():
							ok_starts = cursor + idxs[:-1][gap_mask] + 1 
							ok_ends = cursor + idxs[1:][gap_mask]
							for start, end in zip(ok_starts, ok_ends):
								yield log_this(chrom, start, end)
						# for the end cases
						if idxs[-1] + 1 < len(line):
							run_start = cursor + idxs[-1] + 1
						else:
							run_start = None
				# moving to the next line
				cursor += len(line)
		# final line
		if run_start is not None:
			yield log_this(chrom, run_start, cursor)


def log_this(chrom, run_start, run_end):

	# print("\t Accessible region %s:%d-%d (size %d)" %(chrom, run_start, run_end, run_end-run_start))
	logging.info("\t Accessible region %s:%d-%d (size %d)", chrom, run_start, run_end, run_end-run_start)
	return (chrom, run_start, run_end)


# considering the effect of incorporating additiona N
def join_small_gap(regions, min_gap_size=10):

	ri = iter(regions)
	prev_chr, prev_start, prev_end = next(ri)

	for region in ri:
		cur_chr, cur_start, cur_end = region
		if cur_chr != prev_chr:
                        # print("%s finsihed" %prev_chr)
		        yield(prev_chr, prev_start, prev_end)
                        prev_chr = cur_chr
			prev_start = cur_start
			prev_end = cur_end
		else:
			dist = cur_start - prev_end
			if dist < min_gap_size:
                                # print("join the gap size with %d" %dist)
				pre_end = cur_end
			else:
				yield(prev_chr, prev_start, prev_end)
                                prev_chr, prev_start, prev_end = cur_chr, cur_start,cur_end

        yield(prev_chr, prev_start, prev_end)

"""
Filtering regions and assocate nucleotide sequences with each region
"""
def filterRegions_Seq(fastaFile, regions, minLen=1000):
	filtered_num = 0
	dic = {}
	lenDic = {}
	chroms = genTargetChrNameFromFasta(fastaFile)

	with pyfaidx.Fasta(fastaFile, as_raw=True) as fa_file:

		for region in iter(regions):
			if region[0] in chroms:
				rLen = region[2] - region[1]
				seqs = fa_file[region[0]][region[1]:(region[2]+1)]

				if rLen >= minLen:
					if dic.has_key(region[0]):
						dic[region[0]].append( (region,seqs) )
						lenDic[region[0]] += rLen
					else:
						dic[region[0]] = [ (region,seqs) ]
						lenDic[region[0]] = rLen							
				else:
					filtered_num += 1 
	logging.info("** Total filtered out accessible ranges less than [%d]: %d", minLen, filtered_num)
	return (dic, lenDic)

"""
generate chromesome name list from fasta file
"""

def genTargetChrNameFromFasta(fastaFile, XY=False):

	with open(fastaFile, 'r') as infile:
		line = infile.readline()
		chrom = line.split(None, 1)[0][1:]

		if "chr" not in chrom:
			chroms = [str(i) for i in xrange(1,23)]
			if(XY):
				chroms.extend(["X","Y"])
		else:
			chroms = ["chr"+str(i) for i in xrange(1,23)]
			if(XY):
				chroms.extend(["chrX","chrY"])

		return chroms

# get regions, and print the extra regions used for visualization
def sampleRegions(fastaFile, regions, nSize, minLen=1000, rgOut=True):

	rgDic, lenDic = {}, {}
	chroms = genTargetChrNameFromFasta(fastaFile)
	count = 0
	len_total = 0
	rgLists = []

	if(rgOut):
		dateTag = time.strftime("%y%m%d_%H%M")
		outFold = "./rgInfo_" + dateTag
		if not os.path.exists(outFold):
			os.makedirs(outFold)
		f_train = open(outFold+'/train_regions.bed', 'w+')
		f_test = open(outFold+'/noTrain_regions.bed', 'w+')
	
	# get the whole statistics of the whole accessible region
	for region in iter(regions):
		if region[0] in chroms:
			rLen = region[2] - region[1]
			if rLen >= minLen:
				count += 1
				len_total += rLen
				if lenDic.has_key(region[0]):
					lenDic[region[0]] += rLen
					rgDic[region[0]].append(region)
				else:
					lenDic[region[0]] = rLen
					rgDic[region[0]] = [region]
	
	# accordingt to chromesome AR proportions, prepare accessible region

	for chr in lenDic.keys():
		n = int(nSize * lenDic[chr]/float(len_total))

		# random.shuffle(rgDic[chr])
		covbase = 0
		idx = 0
		
		while(covbase < n and idx < len(rgDic[chr])):
			rgLen = rgDic[chr][idx][2] - rgDic[chr][idx][1]
			if covbase + rgLen < n :
				rgLists.append(rgDic[chr][idx])
				covbase += rgLen

				if(rgOut):
					tmp_start = rgDic[chr][idx][1]
					tmp_end = rgDic[chr][idx][2]
					f_train.write(chr + "\t" + str(tmp_start) + "\t" + str(tmp_end)
					+ "\t" + str(tmp_end-tmp_start) + "\n")
			else:
				extraLen = n-covbase
				if extraLen < minLen:
					extraLen = minLen
				rgLists.append( (chr, rgDic[chr][idx][1], rgDic[chr][idx][1]+ extraLen) )
				covbase += extraLen

				if(rgOut):
					tmp_start = rgDic[chr][idx][1]
					tmp_end = rgDic[chr][idx][1] + extraLen
					f_train.write(chr + "\t" + str(tmp_start) + "\t" + str(tmp_end) 
						+"\t" + str(extraLen) + "\n")

					tmp_start = rgDic[chr][idx][1]+ extraLen + 1
					tmp_end = rgDic[chr][idx][2]
					f_test.write(chr + "\t" + str(tmp_start) + "\t" + str(tmp_end) 
						+ "\t" + str(tmp_end - tmp_start)+ "\t(cut RG)\n")

			idx += 1

		# print the not used accessible regions for testing

		if(rgOut and idx < len(rgDic[chr])):
			for rest in range(idx, len(rgDic[chr])):
				tmp_start = rgDic[chr][rest][1]
				tmp_end = rgDic[chr][rest][2]
				f_test.write(chr + "\t" + str(tmp_start) + "\t" + str(tmp_end) 
					+ "\t" + str(tmp_end - tmp_start) + "\n")

	if(rgOut):
		f_train.close()
		f_test.close()

	return rgLists

### 2017-09-20, be very carefull about the cutting region, minLen is setted as the 
def sampleRegions_trainTest(fastaFile, regions, nSize, minLen=5000, rgOut=True, testProportion=0.1, adj=5000):

	# this is used to avoid the effect of selecting miniSize
	if adj > 0:
		minLen = adj

	rgDic, lenDic = {}, {}
	chroms = genTargetChrNameFromFasta(fastaFile)
	count = 0
	len_total = 0
	rgLists, rgLists_test = [], []

	if(rgOut):
		dateTag = time.strftime("%y%m%d_%H%M")
		outFold = "./rgInfo_" + dateTag
		if not os.path.exists(outFold):
			os.makedirs(outFold)
		f_train = open(outFold+'/train_regions.bed', 'w+')
		f_ftest = open(outFold+'/test_regions.bed', 'w+')
		f_test = open(outFold+'/noTrain_regions.bed', 'w+')
	
	# regions by chromsome,  sample size is porportion to the base pairs in chromesome 
	for region in iter(regions):
		if region[0] in chroms:
			rLen = region[2] - region[1]
			if rLen >= minLen:
				count += 1
				len_total += rLen
				if lenDic.has_key(region[0]):
					lenDic[region[0]] += rLen
					rgDic[region[0]].append(region)
				else:
					lenDic[region[0]] = rLen
					rgDic[region[0]] = [region]
	

	# iter over chromesome, and get the training and testing regions
	for chr in lenDic.keys():
		n = int(nSize * lenDic[chr]/float(len_total))
		n_t = int(n*testProportion)

		random.seed(100)
		random.shuffle(rgDic[chr])
		
		## the target base pair to find
		covbase = 0; covbase_t = 0
		idx = 0;  # idx is the rg index in the chromesome

		### Training part preparion
		while(covbase < n and idx < len(rgDic[chr])):

			rgLen = rgDic[chr][idx][2] - rgDic[chr][idx][1]

			if( adj > 0 ):
				rgLen = rgLen - rgLen%adj
				# too short region filter out
				if rgLen == 0:
					if(rgOut):
						tmp_start = rgDic[chr][idx][1]
						tmp_end = rgDic[chr][idx][2]
						f_test.write(chr + "\t" + str(tmp_start) + "\t" + str(tmp_end)
							+ "\t" + str(tmp_end - tmp_start) + "\n")	
					idx += 1
					continue

			if covbase + rgLen < n :
				# rgLists.append(rgDic[chr][idx])
				rgLists.append( (chr, rgDic[chr][idx][1], rgDic[chr][idx][1]+ rgLen) )
				covbase += rgLen

				if(rgOut):
					tmp_start = rgDic[chr][idx][1]
					tmp_end = rgDic[chr][idx][1] + rgLen 
					f_train.write(chr + "\t" + str(tmp_start) + "\t" + str(tmp_end)
					+ "\t" + str(rgLen) + "\n")

			else:
				# last number of regions, too small discard
				extraLen = n-covbase

				if extraLen < adj:
					if(rgOut):
						tmp_start = rgDic[chr][idx][1]
						tmp_end = rgDic[chr][idx][2]
						f_test.write(chr + "\t" + str(tmp_start) + "\t" + str(tmp_end) 
							+ "\t" + str(tmp_end - tmp_start)+ "\t(too short than 5000)\n")
					idx += 1
					break
				else:
					extraLen = extraLen - extraLen%adj
							
				rgLists.append( (chr, rgDic[chr][idx][1], rgDic[chr][idx][1]+ extraLen) )
				covbase += extraLen

				if(rgOut):	
					tmp_start = rgDic[chr][idx][1]
					tmp_end = rgDic[chr][idx][1] + extraLen
					f_train.write(chr + "\t" + str(tmp_start) + "\t" + str(tmp_end) 
							+"\t" + str(extraLen) + "\n")

					tmp_start = rgDic[chr][idx][1] + extraLen + 1
					tmp_end = rgDic[chr][idx][2]
					f_test.write(chr + "\t" + str(tmp_start) + "\t" + str(tmp_end) 
						+ "\t" + str(tmp_end - tmp_start)+ "\t(cut RG)\n")

			idx += 1

		# the rest part are used for testing (this may be changed to avoid no testing)

		if( idx < len(rgDic[chr]) ):
			# all output non-used regions
			for rest in range(idx, len(rgDic[chr])):
				tmp_start = rgDic[chr][rest][1]
				tmp_end = rgDic[chr][rest][2]
				
				if(rgOut):
					f_test.write(chr + "\t" + str(tmp_start) + "\t" + str(tmp_end) 
						+ "\t" + str(tmp_end - tmp_start) + "\n")

				## adding the testing regions, adding the adjustment.
				if(covbase_t < n_t):
					rgLen_t = rgDic[chr][rest][2] - rgDic[chr][rest][1]

					if( adj > 0 ):
						rgLen_t = rgLen_t - rgLen_t%adj

					# too small to be used in the test
					if rgLen_t == 0:
						continue

					if covbase_t + rgLen_t < n_t :
						rgLists_test.append( (chr, rgDic[chr][rest][1], rgDic[chr][rest][1]+ rgLen_t) )
						covbase += rgLen_t

						if(rgOut):
							tmp_start = rgDic[chr][rest][1]
							tmp_end = rgDic[chr][rest][1] + rgLen_t
							f_ftest.write(chr + "\t" + str(tmp_start) + "\t" + str(tmp_end)
							+ "\t" + str(rgLen_t) + "\n")

					else:
						# last extra number of reads
						extraLen_t = n_t-covbase_t

						if( adj > 0 ):
							extraLen_t = extraLen_t - extraLen_t%adj
							if(extraLen_t == 0):
								# just using the continue to keep all the nonTrain regions, but actually discarded
								continue

						if extraLen_t < minLen:
							continue
						##	extraLen_t = minLen

						rgLists_test.append( (chr, rgDic[chr][rest][1], rgDic[chr][rest][1]+ extraLen_t) )
						covbase_t += extraLen_t

						if(rgOut):
							tmp_start = rgDic[chr][rest][1]
							tmp_end = rgDic[chr][rest][1]+ extraLen_t
							f_ftest.write(chr + "\t" + str(tmp_start) + "\t" + str(tmp_end)
							+ "\t" + str(extraLen_t) + "\n")

	if(rgOut):
		f_train.close()
		f_test.close()
		f_ftest.close()

	return rgLists, rgLists_test







#########################################################################
## Developmnet testing files
#########################################################################

def filterRegions(chroms, regions, minLen=1000):
	filtered_num = 0
	dic = {}
	lenDic = {}

	for region in iter(regions):
		if region[0] in chroms:
			rLen = region[2] - region[1]
			if rLen >= minLen:
				if dic.has_key(region[0]):
					dic[region[0]].append(region)
					lenDic[region[0]] += rLen
				else:
					dic[region[0]] = [region]
					lenDic[region[0]] = rLen

			else:
				filtered_num += 1 

	logging.info("** Total filtered out accessible ranges less than [%d]: %d", minLen, filtered_num)
	return (dic, lenDic)


def checkRegionDistance(regions):
	gap_dist_dic = {}
	count = 1
	ri = iter(regions)
	prev_chr, prev_start, prev_end = next(ri)
	for region in ri:
		count += 1
		cur_chr, cur_start, cur_end = region
		if cur_chr != prev_chr:
			prev_chr = cur_chr
			prev_start = cur_start
			prev_end = cur_end
		else:
			dist = cur_start - prev_end
			if dist in gap_dist_dic.keys():
				gap_dist_dic[dist] += 1
			else:
				gap_dist_dic[dist] = 1
		pre_chr = cur_chr
		prev_start = cur_start
		prev_end = cur_end
	print("Total accessible regions %d" %count)
	return gap_dist_dic
