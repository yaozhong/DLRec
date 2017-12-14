#!/usr/bin/env python

"""
Date:       2017-07-08
Author:     Yao-zhong Zhang@imsut
Descrption:
Given accessible regions of chrome AR(chr, start, end)
calculate the the coverage for the target AR based on bam file 
"""

import logging
import pysam
import time
import numpy as np


def getReadDepth_test(region, bamfile, min_mapq=20):

	rc = np.zeros(region[2]-region[1]+1)
	pileup = bamfile.pileup(region[0], region[1], region[2])

	for pColumn in pileup:
		if pColumn.pos >= region[1] and pColumn.pos <= region[2]:
			rc[pColumn.pos - region[1]] = pColumn.nsegments 

def getARcoverage_rc(region, samfile, min_mapq=20):

	rlen = region[2]-region[1]+1

	rc = np.zeros((1, rlen), dtype=np.uint8)
	reads = samfile.fetch(region[0], region[1], region[2] +1)

	for r in reads:
		if r.mapq >= min_mapq:
			left = r.pos - region[1]
			# Note this has the problem, the CIAGAR Information need to be considered
			right = left + r.rlen
			rc[left:(right+1)] = rc[left:(right+1)] + 1	

	return rc

#################################################################################
# Count based approaches
#################################################################################

## this function do not distinguish left or right reads
def getARcoverage_rc_start(region, samfile, min_mapq=20):

	rc = np.zeros(region[2]-region[1] + 1)
	reads = samfile.fetch(region[0], region[1], region[2])

	for r in reads:
		if r.mapq >= min_mapq:
			left = r.pos - region[1]
			rc[left] += 1	
	return rc

def getARcoverage_fc_start(region, samfile, min_mapq= 20):

	fc = np.zeros(region[2] - region[1] + 1)

	reads = samfile.fetch(region[0], region[1], region[2])

	for r in reads:
		
		if r.is_proper_pair and r.is_read1 and r.mapq >= min_mapq:
			
			left = r.pos
			right = r.mpos

			if r.is_reverse:
				f_left 	= right
			else:
				f_left = left

			# avoid the left read is not proper position of right 
			if(f_left <= region[2] and f_left >= region[1]):
				f_left = f_left - region[1]
				fc[f_left] += 1	

	return fc


#################################################################################
# development part
#################################################################################

# only work for the fetched reads
def filter_read(read, min_mapq=20):
	return (read.is_unmapped or read.is_sceondary or read.is_qcfail or read.mapq < min_mapq)

"""
the positional based count is slow This implemenation is slow 
if no filtering is used, directly add pp.n
position based 
more time consuming, be careful about this function
"""
def getARcoverage_rd_filter(region, samfile, min_mapq=20):

	rc = np.zeros(region[2]-region[1] +1)

	for pColumn in samfile.pileup(region[0], region[1], region[2]):	

		if pColumn.pos >= region[1] and pColumn.pos <= region[2]:
			count = 0
			for r in pColumn.pileups:
				if r.alignment.mapping_quality >= min_mapq:
					count += 1
			rc[pColumn.pos - region[1]] = count
			
	return rc


"""
Using the bedtools to detect
note this version will longer than the assigned region with extra read length
"""
def getARcoverage_rd(region, bamfile, min_mapq=20):

	# rlen = region[2] - region[1] + 1
	rlen = region[2] - region[1] 
	rc = np.zeros(rlen, dtype=np.uint8)
	pileup = bamfile.pileup(region[0], region[1], region[2])

	for pColumn in pileup:	
		if pColumn.pos >= region[1] and pColumn.pos < region[2]:
			rc[pColumn.pos - region[1]] = pColumn.nsegments

	return rc

"""
Input is a vector 
"""
def vecSmooth(vec, sbin = 10):

	if sbin == 1:
		return vec.reshape(1, len(vec))

	if sbin > 1:
		vl = len(vec)
		bn = int(vl/sbin)
		svec = np.zeros(bn, dtype= vec.dtype)

		for i in xrange(bn):
			svec[i] = int(np.mean(vec[i*sbin:(i+1)*sbin]))

		return svec.reshape(1,bn)

"""
Other different ways to calcuate the read coverge of the target positions
"""
def get_coverage(self, bam_file):

        alignment = pybedtools.BedTool(bam_file)
        regions = pybedtools.BedTool(self.region_file)
        print 'Calculating coverage over regions ...'
        sys.stdout.flush()
        t0 = time.time()
        coverage_result = alignment.coverage(regions).sort()
        coverage_array = np.array([i[-1] for i in coverage_result], dtype=int)

        t1 = time.time()
        print 'completed in %.2fs' % (t1 - t0)
        sys.stdout.flush()
        return coverage_array




