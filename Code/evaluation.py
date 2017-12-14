# this function is used to test various experiment settings

## Training part
### evluaiton of different models CNN, reference reconstruction，call CNV-detector part
"""
(0). Data re-checking 
(1). Target data prepare and basic statistic analysis

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

	# revised to be Train/test/validation, add data shuffling?
	train, test, valid = data.genDLTrainData(minGap, sampleN)

	return train, test, valid




## Testing part
### what is the region features, GC, mappability and sequence features affect for the results. ?? more detailed analysis.


"""
Evluation function:
(1).Reconstruction performance 
(2).Training on exome data and reconstruction for both coding and non-coding region. 
"""