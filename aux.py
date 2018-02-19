#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Auxiliary file for GibbsMotion inference algorithm
# This code is available under the MIT License.
# (c)2017 Daniel Rugeles 
# daniel007[at]e.ntu.edu.sg / Nanyang Technological University

import math
import numpy as np
from random import randint,seed


__author__ = "Daniel Rugeles"
__copyright__ = "Copyright 2017, Auxiliary"
__credits__ = ["Daniel Rugeles"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Daniel Rugeles"
__email__ = "daniel007@e.ntu.edu.sg"
__status__ = "Released"

# June 5, 2017
# Added cols as parameters for dictionate()

#--combine------------------------------------------------------
"""Combines two dictionaries
Input: dict,dict
Output: dict """
#-------------------------------------------------------------------
def combine(dict1,dict2):
	dict3={}
	for k1 in dict1:
		dict3[k1]=dict2[dict1[k1]]
	return dict3
	
	
#--dictionate------------------------------------------------------
"""Maps every column so that its elements go from 0 to len(column)-1
Input: 2darray
cols: list identifying the columns that will be map onto a dictionary. 
Note that idx2vals will still contain the dictionate for all columns
Output: 2darray,List of dict,list of dict, list"""
#-------------------------------------------------------------------
def dictionate(alldata,cols=xrange(1000000)):
	alldata=np.array(alldata)
	alldata_T=alldata.transpose()	
	val2idxs=[]
	idx2vals=[]
	counts=[]
	
	
	for row in alldata_T:
		
		sortedvalues= np.sort(row)
		idx=-1
		prevval=-1
		val2idx={}
		idx2val={}
		count=[]
		c=0
		for value in sortedvalues:
			c+=1
			if value!=prevval:
				count.append(c)
				c=0
				prevval=value
				idx+=1
				val2idx[value]=idx
				idx2val[idx]=value
		count.append(c+1)
		counts.append(count[1:])

		val2idxs.append(val2idx)
		idx2vals.append(idx2val)
		
	alldata_T_hashed=alldata_T.copy()


	for idx,col in enumerate(alldata_T):
		if idx in cols:
			alldata_T_hashed[idx]=map(lambda x: val2idxs[idx][x],alldata_T[idx]) 
	
	return alldata_T_hashed.transpose(),idx2vals,val2idxs,counts

#--join2------------------------------------------------------
"""Maps every column so that its elements go from 0 to len(column)-1
Input: 2darray
Output: 2darray
Requirement: data must be dictionated with dictionate(data)"""
#-------------------------------------------------------------------
def join2(data):
	
	max_=np.max(data,axis=0)
	D=np.zeros(max_+1)	

	for row in data:
		D[tuple(row)]+=1

	return D

#--addrandomtopic------------------------------------------------------
"""Add random values from 0 to k-1 to the column index by col
Input: 2darray
Output: 2darray"""
#-------------------------------------------------------------------	
def addrandomtopic(data,k,col):

	k=round(k)
	for idx in range(len(data)):
		data[idx][col]=float(randint(0,k-1))
	return data



#--addrandomcol------------------------------------------------------
"""Add random values from 0 to k-1 to the column index by col
Input: 2darray
Output: 2darray

Randomness means how often we change the samples.
 0 means no random values added 
 1 means 100% random values added
 2 means 50% random values added
"""
#-------------------------------------------------------------------
def addrandomcol(data,k,col,randomness=0):

	#data=np.array(data,dtype=np.int32)
	if randomness>0:
		k=round(k)
		order=np.random.permutation(len(data))
		print "Random Initialization",randomness,"Number of Topics:",k
		thresh=int(round((1.0-1.0/randomness)*len(data)))
		for idx in order[thresh:]:
			data[idx][col]=randint(0,k-1)
		del order
			
	return data
	
#--addrandomcol------------------------------------------------------
"""Normalizes every row so that it sums to one
Input: 2darray
Output: 2darray
"""
#-------------------------------------------------------------------
def normalize_rows(a):
	row_sums = a.sum(axis=1)
	return a / row_sums[:, np.newaxis]
	
#--duplicates------------------------------------------------------
"""Check for duplicate row elements in two matrices
Input: 2darray, 2darray
Output: set of 1darray
"""
#-------------------------------------------------------------------
def duplicates(a,b):
	train=set()
	for e in a:
		train.add(tuple(e))

	test=set()
	for e in b:
		test.add(tuple(e))
		
	return train.intersection(test)
	
#--toDistribution--------------------------------------------------------------
"""one-hot encoding of position in a K-dimensional vector
Input: int,int
Output: 1darray"""
#-------------------------------------------------------------------
def toDistribution(position,K):
	distribution=np.zeros(K)
	distribution[int(position)]=1
	return np.array(distribution,dtype=np.int32)
	

