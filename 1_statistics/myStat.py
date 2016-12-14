# -*- coding: utf-8 -*-
"""
Created on Tue Dec 06 11:32:27 2016

@author: Ilho
"""



from __future__ import division
import scipy.stats as stats
import math

critical95 = 1.96
critical98 = 2.66

def rev_ztable(propotion):
    return stats.norm.ppf(propotion)

def ztable(zscore):
    return stats.norm.cdf(zscore)

def zscore(mean, stddev, value):
    deviation = value - mean
    return deviation / stddev
    
def zscoreOnSampleDistribution(popMean, popSigma, sampleSize, value):
    se = findStandardError(popSigma, sampleSize)
    return zscore(popMean, se, value)

def probabilityOnSampleDistributionZscore_TwoTail(zscore):
    return 1-ztable(zscore)

def sampleStd(popStd, sampleSize):
    return popStd / math.sqrt(sampleSize)

def findCriticalValue(propotion):
    lowZValue = rev_ztable((1-propotion) / 2)
    return abs(lowZValue)
    
def findZCriticalValueOneTail(alphaLevel):
    return rev_ztable(1-alphaLevel)
    
def findZCriticalValueTwoTail(alphaLevel):
    return rev_ztable(1-(alphaLevel/2))
    
def findStandardError(popDeviation, sampleSize):
    return sampleStd(popDeviation, sampleSize)
    
def findMarginOfError(stdError, confidenceLevel):
    zvalue = findCriticalValue(confidenceLevel)
    return zvalue * stdError
    
def findConfidenceInterval(marginOfError, sampleMean):
    return [sampleMean - marginOfError, sampleMean + marginOfError]

def findConfidenceIntervalWithAllParams(popDeviation, sampleSize, confidenceLevel, sampleMean):
    return findConfidenceInterval(findMarginOfError(findStandardError(popDeviation, sampleSize), confidenceLevel), sampleMean)
    
def isNullHypothesisRejected_zTest(popMean, popSigma, sampleMean, sampleSize, alphaLevel, isTwoTail=True):
    criticalValue = 0.0
    if isTwoTail == True:
        criticalValue = findZCriticalValueTwoTail(alphaLevel)
    else:
        criticalValue = findZCriticalValueOneTail(alphaLevel)
    z_scoreOfSampleDistribution = zscoreOnSampleDistribution(popMean, popSigma, sampleSize, sampleMean)
    reject = False
    
    if z_scoreOfSampleDistribution >= 0:
        if z_scoreOfSampleDistribution > criticalValue:
            reject = True
    else:
        if z_scoreOfSampleDistribution < -1 * criticalValue:
            reject = True
            
    if reject == True:
        print "The null hypothesis (H_0) is REJECTED at p<{0}".format(alphaLevel)
    else:
        print "The null hypothesis (H_0) is FAILED to be rejected with p<" + str(alphaLevel)
    p_detail = probabilityOnSampleDistributionZscore_TwoTail(z_scoreOfSampleDistribution)
    print "\t- standard error of the sample size {0}: {1:.2f}".format(sampleSize, findStandardError(popSigma, sampleSize))
    print "\t- z-score on sample distribution: {0:.2f}".format(z_scoreOfSampleDistribution)
    print "\t- critical value for a={0}: ±{1:.2f}".format(alphaLevel, criticalValue)
    print "\t- detailed probability: {0:.2f} or {1:.2f}%".format(p_detail, p_detail * 100)
        
#======================== T-test====================   
conventional_alpha = 0.05


def findBesselsSampleDeviation(sampleList):
    sampleSize = len(sampleList)
    sampleMean = sum(sampleList) / sampleSize
    sqOfDev = []
    for element in sampleList:
        dev = element - sampleMean
        sqOfDev.append(dev * dev)
    sampleAverageOfDev = sum(sqOfDev) / (sampleSize-1)
    return math.sqrt(sampleAverageOfDev)
    
def tvalueFromList(sampleList, nullAverage):
    S = findBesselsSampleDeviation(sampleList)
    sampleMean = sum(sampleList) / len(sampleList)
    return tvalue(nullAverage, sampleMean, len(sampleList), S)
    
def tvalue(nullAverage, sampleMean, sampleSize, estimatedPopDeviation):
    diffMean = sampleMean - nullAverage
    se = estimatedPopDeviation / math.sqrt(sampleSize)
    return diffMean / se

def pvalue(tvalue, sampleSize, isTwoTail):
    if isTwoTail == True:
        return 2 * pvalue_oneTail(tvalue, sampleSize)
    return abs(stats.t.sf(abs(tvalue), sampleSize-1))
    
def pvalue_oneTail(tvalue, sampleSize):
    return abs(stats.t.sf(abs(tvalue), sampleSize-1))
    
def pvalue_twoTail(tvalue, sampleSize):
    return 2 * pvalue_oneTail(tvalue, sampleSize)
    
def pvalue_fromSample_twoTail(sampleList, nullAverage):
    tvalue = tvalueFromList(sampleList, nullAverage)
    return pvalue_twoTail(tvalue, len(sampleList))

def findTCriticalValue(alphaLevel, sampleSize, isTwoTail=True):
    if isTwoTail == True:
        return abs(stats.t.ppf(alphaLevel/2, sampleSize-1))
    return abs(stats.t.ppf(alphaLevel, sampleSize-1))
    

def findAlphaValueFromTStatistic_oneTail(sampleSize, tStatistic):
    return 1-stats.t.cdf(tStatistic, sampleSize-1)
    
def getCohensD(sampleMean, popMean, estimatedDeviation):
     return (sampleMean - popMean) / estimatedDeviation

def isNullHypothesisRejected_tTest(popMean, sampleSize, sampleMean, estimatedDeviation, alphaLevel, isTwoTail=True):   
    criticalValue = findTCriticalValue(alphaLevel, sampleSize, isTwoTail)
    se = estimatedDeviation / math.sqrt(sampleSize)
    marginOfError = se * criticalValue
    criticalValueTwoTail = findTCriticalValue(alphaLevel, sampleSize, True)
    marginOfErrorTwoTail = se * criticalValueTwoTail
    
    t_value = tvalue(popMean, sampleMean, sampleSize, estimatedDeviation)
    reject = False
    if t_value >= 0:        
        if t_value > criticalValue:
            reject = True
    else:
        if t_value < -1 * criticalValue:
            reject = True
            
    if reject == True:
        print "The null hypothesis (H_0) is REJECTED at p<{0}".format(("%.2f"%alphaLevel).lstrip('0'))
    else:
        print "The null hypothesis (H_0) is FAILED to be rejected with p<{0}".format(alphaLevel)
    
    print "\t- t-critical value for a={0}: ±{1:.3f}".format(alphaLevel, criticalValue)

    print "\t- margin of error for a={0}: {1:.3f}".format(alphaLevel, marginOfError)
    print "\t- confidence interval for a={0}: {1:.3f} ~ {2:.3f} ({3:.2f}% CI)".format(alphaLevel, popMean-marginOfError, popMean+marginOfError, 100*(1-alphaLevel))
    print "\t- t-statistic for {1} with {2} samples: {0:.3f}".format(t_value, sampleMean, sampleSize)
    p_value = pvalue(t_value, sampleSize, isTwoTail)
    print "\t- p-value: {0:.5f}".format(p_value)
    cohenD = getCohensD(sampleMean, popMean, estimatedDeviation)
    print "\t- Cohen's d: {0}".format(cohenD)
    print "\t- Standard Error: {0:.2f}".format(se)
    direction = "two-tailed"
    if isTwoTail == False:
        direction = "one-tailed"
    print "\t- APA style:"
    print "\t\t M={0:.2f}, SD={1:.2f}".format(sampleMean, estimatedDeviation)
    print "\t\t t({0})={1:1.2f}, p={2} ({3})".format(sampleSize-1, t_value, ("%.5f"%p_value).lstrip('0'), direction)
    print "\t\t Confidence interval (two-tailed) on sample mean; {0:.0f}%CI=({1:.2f}, {2:.2f})".format(100*(1-alphaLevel), sampleMean-marginOfErrorTwoTail, sampleMean+marginOfErrorTwoTail)
    print "\t\t Effect size measures; d={0:.2f}, r^2={1}".format(cohenD, ("%.2f"%rsq(t_value, sampleSize-1)).lstrip('0'))
    
def ttest_1Sample_twoTail(sampleList, nullAverage, alphaLevel=conventional_alpha):
    tvalue = tvalueFromList(sampleList, nullAverage)
    pvalue = pvalue_twoTail(tvalue, len(sampleList))
    if pvalue >= alphaLevel:
        print "By the criteria (alpha={0}), this difference is considered to be NOT statistically significant.".format(alphaLevel)
    else:
        print "By the criteria (alpha={0}), this difference is considered to be statistically significant.".format(alphaLevel)
    print "\t- t-statistic = {0:.3f}".format(tvalue)
    print "\t- pvalue = {0:.3f}".format(pvalue)

    
def ttest_2Sample(sampleSize, sampleMean1, sampleDev1, sampleMean2, sampleDev2, alphaLevel, isTwoTail=True):
    pointEstimation = sampleMean2 - sampleMean1
    deviationEstimation = math.sqrt(sampleDev1*sampleDev1 + sampleDev2*sampleDev2)
    nullMean = 0
    print "\t- pointEstimation Mean = {0:.3f}".format(pointEstimation)
    print "\t- pointEstimation Deviation = {0:.3f}".format(deviationEstimation)
    isNullHypothesisRejected_tTest(nullMean, sampleSize, pointEstimation, deviationEstimation, alphaLevel, isTwoTail)
    
def rsq(t_value, dof):
    tsq = t_value * t_value
    return tsq / (tsq + dof)
    
    
    
import numpy

def getStandardErrorOfMean_2sample_independent(std1, std2, n1, n2):
    firstTerm = (std1 * std1) / n1
    secondTerm = (std2 * std2) / n2
    return math.sqrt(firstTerm + secondTerm)  
    
def getStandardErrorOfMean_2sample_independent_pooledVariace(pooledVar, n1, n2):
    firstTerm = pooledVar / n1
    secondTerm = pooledVar / n2
    return math.sqrt(firstTerm + secondTerm)

def getTStatistic(mean1, mean2, sem, expectedDiff=0):
    observedDiff = mean1 - mean2
    return (observedDiff-expectedDiff) / sem

    
def ttest_2sample_independent(sample1, sample2, alphaLevel, isTwoTail=True, usePooledVar=False, expectedDiff=0):
    n1 = len(sample1)
    n2 = len(sample2)    
    mean1 = numpy.mean(sample1)
    mean2 = numpy.mean(sample2)
    std1 = numpy.std(sample1, ddof=1)
    std2 = numpy.std(sample2, ddof=1)
    if usePooledVar == False:
        return ttest_2sample_independent_withStat(n1, n2, mean1, mean2, std1, std2, alphaLevel, isTwoTail, expectedDiff)
    else:
        return ttest_2sample_independent_pooledVar(sample1, sample2, n1, n2, mean1, mean2, std1, std2, alphaLevel, isTwoTail, expectedDiff)
    
def ttest_2sample_independent_pooledVar(sample1, sample2, n1, n2, mean1, mean2, std1, std2, alphaLevel, isTwoTail=True, expectedDiff=0):
    pooledVar = getPooledVariance(sample1, sample2)
    sem = getStandardErrorOfMean_2sample_independent_pooledVariace(pooledVar, n1, n2)
    print "\t- Pooled variace: {0}".format(pooledVar)
    return doTtest_2sample_independent(n1, n2, mean1, mean2, alphaLevel, sem, isTwoTail, expectedDiff)

def ttest_2sample_independent_withStat(n1, n2, mean1, mean2, std1, std2, alphaLevel, isTwoTail=True, expectedDiff=0):    
    sem = getStandardErrorOfMean_2sample_independent(std1, std2, n1, n2)
    return doTtest_2sample_independent(n1, n2, mean1, mean2, alphaLevel, sem, isTwoTail, expectedDiff)

    
def ttest_2sample_independent_withStat_pooledVar(n1, n2, mean1, mean2, pooledVar, alphaLevel, isTwoTail=True, expectedDiff=0):    
    sem = getStandardErrorOfMean_2sample_independent_pooledVariace(pooledVar, n1, n2)
    print "\t- Pooled variace: {0}".format(pooledVar)
    return doTtest_2sample_independent(n1, n2, mean1, mean2, alphaLevel, sem, isTwoTail, expectedDiff)

    
def ttest_2sample_independent_withStat_sumOfSqDev(n1, n2, mean1, mean2, ss1, ss2, alphaLevel, isTwoTail=True, expectedDiff=0):    
    pooledVar = getPooledVarianceFromSumOfSquareDeviations(ss1, ss2, n1, n2)
    sem = getStandardErrorOfMean_2sample_independent_pooledVariace(pooledVar, n1, n2)
    print "\t- Pooled variace: {0}".format(pooledVar)
    return doTtest_2sample_independent(n1, n2, mean1, mean2, alphaLevel, sem, isTwoTail, expectedDiff)

def getTCriticalValue(alphaLevel, dof, isTwoTail):
    if isTwoTail == True:
        tcritical = abs(stats.t.ppf(alphaLevel/2, dof))
    else:
        tcritical = abs(stats.t.ppf(alphaLevel, dof))    
    return tcritical

def doTtest_2sample_independent(n1, n2, mean1, mean2, alphaLevel, sem, isTwoTail, expectedDiff): 
    dof = n1 + n2 -2
    tstatistic = getTStatistic(mean1, mean2, sem, expectedDiff)
    tcritical = getTCriticalValue(alphaLevel, dof, isTwoTail)    
    isNullRejected = False
    if tstatistic >= 0:
        if tstatistic > tcritical:
            isNullRejected = True
    else:
        if tstatistic < -1*tcritical:
            isNullRejected = True           
    p_value = pvalue(tstatistic, dof, isTwoTail)
    marginOfError = tcritical * sem
    meanDiff = mean1 - mean2
    ci = [meanDiff - marginOfError, meanDiff + marginOfError]
    r_sq = rsq(tstatistic, dof)
    
    print "\t- Degrees of Freedom: {0}".format(dof)
    print "\t- Sample mean [1]: {0:.2f}".format(mean1)
    print "\t- Sample mean [2]: {0:.2f}".format(mean2)
    print "\t- Observed difference: {0}".format(mean1-mean2)
    print "\t- Expected difference: {0}".format(expectedDiff)
    print "\t- Standard Error: {0:.2f}".format(sem)
    print "\t- t-statistic (x_1 - x_2): {0:.2f}".format(tstatistic)
    print "\t- t-critical value (a={0}): ±{1:.3f}".format(alphaLevel, tcritical)
    print "\t- pvalue = {0:.3f}".format(p_value)    
    print "\t- {0:.2f}% CI: ({1:.2f}, {2:.2f})".format((1-alphaLevel)*100, min(ci), max(ci))
    print "\t- r_square: {0:.2f}".format(r_sq)
    print "\t" + "-" * 40
    if isNullRejected == True:
        print "\t The null hypothesis H_0 is REJECTED at p<{0} or p={1:.3f}".format(alphaLevel, p_value)
        print "\t (which means two samples are statistically significant)"
    else:
        print "\t The null hypothesis H_0 is ACCEPTED at p<{0} or p={1:.3f}.".format(alphaLevel, p_value)
        print "\t (which means two samples are statistically similar)"
    
def getPooledVarianceFromSumOfSquareDeviations(ss1, ss2, n1, n2):
    return (ss1 + ss2) / (n1-1 + n2-1)

def getPooledVariance(sample_x, sample_y):
    meanx = numpy.mean(sample_x)
    meany = numpy.mean(sample_y)
    dxdx = []
    dydy = []
    for x in sample_x:
        dx = x - meanx
        dxdx.append(dx*dx)
    for y in sample_y:
        dy = y-meany
        dydy.append(dy*dy)
    return (sum(dxdx) + sum(dydy)) / (len(sample_x)-1 + len(sample_y)-1)