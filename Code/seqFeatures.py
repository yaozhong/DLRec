#!/usr/bin/env python

"""
Author: Yao-zhong Zhang
Date:   2017-08-04
Description:
    This file contains functions which are used to correct GC and repeakMask in a given region
    repeat mask are low cases in the fasta file

"""
from __future__ import division
import numpy as np
import pyBigWig
from coverage import vecSmooth

def calcualte_gc_low(subseq):

	gc_lo = subseq.count('g') + subseq.count('c')
	gc_up = subseq.count('G') + subseq.count('C')
	at_lo = subseq.count('a') + subseq.count('t')
	at_up = subseq.count('A') + subseq.count('T')
	total = float(gc_lo + gc_up + at_lo + at_up)
	if not total:
		return 0.0, 0.0
	gc = (gc_lo + gc_up)/total
	low = (at_lo + gc_lo)/total
	return gc, low

def calcualte_gc(subseq):

	gc_lo = subseq.count('g') + subseq.count('c')
	gc_up = subseq.count('G') + subseq.count('C')
	gc = (gc_lo + gc_up)
	return gc


def calcuate_seq_gc(seq, a=0, l=10):
	eLen = len(seq)-l-a
	gcVec = np.zeros(eLen)

	for i in range(0, eLen):
		gcVec[i] = calcualte_gc(seq[(i+a):(i+a+l)])

	return gcVec


# to be checked
def calcuate_seq_gc_ip(seq, a=0, l=10):

	eLen = len(seq)-l-a
	gcVec = np.zeros(eLen)

	count = 0
	for i in range(a, a+l):
		if seq[i] in ['G', 'g', 'C', 'c']:
			count += 1
	gcVec[0] = count


	for i in range(1, eLen):
		
		if seq[i+(a+l)] in ['G', 'g', 'C', 'c']:
			gcVec[i] = gcVec[i-1] + 1

		if seq[i+a-1] in ['G', 'g', 'C', 'c']:
			gcVec[i] -= 1

	return gcVec


# mappable regions loading 

def getRegionMappabilit(rg, bw):

	chrom = rg[0]
	if "chr" not in chrom:
		chrom = "chr" + rg[0]

	mp = bw.values(chrom, rg[1], rg[2])
	mp = np.nan_to_num(mp)
	return mp

## note the gc is one based!! pay attention to this
def getRegionGC(rg, gcbw):

	chrom = rg[0]
	if "chr" not in chrom:
		chrom = "chr" + rg[0]

	gc = np.array(gcbw.values(chrom, rg[1], rg[2]), dtype=np.float32)
	return gc


def seq2onehotMat_sb1(seq):

	# note in the fasta file M, R is also contained
	CHARS = 'ACGT'
	dic = {'A':0, 'a':0, 'C':1, 'c':1, 'G':2, 'g':2, 'T':3, 't':3}
	CHARS_COUNT = len(CHARS)
	seqLen = len(seq)

	res = np.zeros((CHARS_COUNT, seqLen), dtype=np.uint8)

	for i in xrange(seqLen):
		if seq[i] in dic:
			res[dic[seq[i]], i] = 1
	return res

def seq2onehotMat(seq, sbin):

	if sbin == 1:
		return seq2onehotMat_sb1(seq)
	else:
	# note in the fasta file M, R is also contained
		CHARS = 'ACGT'
		dic = {'A':0, 'a':0, 'C':1, 'c':1, 'G':2, 'g':2, 'T':3, 't':3}
		CHARS_COUNT = len(CHARS)
		seqLen = len(seq)

		res = np.zeros((CHARS_COUNT, seqLen), dtype=np.uint8)

		for i in xrange(seqLen):
			if seq[i] in dic:
				res[dic[seq[i]], i] = 1

		sr = np.concatenate((vecSmooth(res[0,:], sbin), vecSmooth(res[1,:], sbin), 
			vecSmooth(res[2,:],sbin), vecSmooth(res[3,:], sbin)))
		return sr


