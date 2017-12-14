# File Name : visPlot.py
# Description: this function is used to draw the picture of the target regions
# Author: Yao-zhong Zhang @ IMSUT
# Date: 2017-09-18

## Input given a region: (chr, start, end),
## Plot: Annotation, nocoding and nocoding, depth-coverage

"""
Reference: http://fullstackdatascientist.io/15/03/2016/genomic-data-visualization-using-python/

2017-9-20 clean the code and fix the test data set.
"""
import sys, os
import pybedtools
import numpy as np
import time

bam_file = "/data/Dataset/1000GP/phase3/NA12878/NA12878.mapped.ILLUMINA.bwa.CEU.low_coverage.20121211.bam"
#rg = "chr1 117199956 117206216"
rg = "1 117199956 117200000"


def get_coverage(bam_file, rg):

        alignment = pybedtools.BedTool(bam_file)
        regions = pybedtools.BedTool(rg, from_string=True)
        print 'Calculating coverage over regions ...'
        sys.stdout.flush()
        t0 = time.time()
        coverage_result = alignment.coverage(regions).sort()
        coverage_array = np.array([i[-1] for i in coverage_result], dtype=int)

        t1 = time.time()
        print 'completed in %.2fs' % (t1 - t0)
        sys.stdout.flush()
        return coverage_array

def get_coverage2(bam_file, rg):

	alignment = pybedtools.BedTool(bam_file)
	regions = pybedtools.BedTool(rg, from_string=True)
	t0 = time.time()
	coverage = alignment.coverage(regions, hist=True)
	t1 = time.time()
	print 'completed in %.2fs' % (t1 - t0)



get_coverage2(bam_file, rg)


