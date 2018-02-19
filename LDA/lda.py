#!/usr/bin/python2.7
# prepared invocations and structures -----------------------------------------
from __future__ import absolute_import
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Auxiliary functions
from aux import dictionate,join2,addrandomcol,normalize_rows,duplicates,toDistribution,normalize_rows
np.set_printoptions(precision=3)
from deco import *
from getData import *

#** Algorithms
from LDA.heron import *
from LDA.cgs import *
from LDA.herongpu import *
from LDA.cool import *


__author__ = "Anonymous"
__copyright__ = "Copyright 2017, Heron LDA"
__credits__ = ["Anonymous"]
__license__ = "GPL"
__version__ = "2.1.1"
__maintainer__ = "Anonymous"
__email__ = "Anonymous"
__status__ = "Released"

# Version two released Heron GPU
# 2.1 Added CGS GPU
# Version one released Heron LDA


#--lda--------------------------------------------------------------
"""Runs LDA over the given data
data: 2darray shape (n,3) where n is the number of records. each record is a tuple (Document,word,topic)
K: number of topics
it: number of iterations
alpha,beta: hyperparameters
dict_: True if data must be passed through dictionate()
verbose: True if it will output the extracted topics - not implemented yet
randomness= 1 if we want to intialize the topics, 0 if we dont want to
PATH: Path to save results
algo: 'cgs' for collapsed gibbs sampling, 'motion' for Gibbs Motion
compressed= Only for Heron model
"""
#-------------------------------------------------------------------
#TODO: Make sure that data is np.int32
def lda(data,K,it,alpha,beta,dict_=True,verbose=True,randomness=1,compressed=False,batch=0,PATH="",algo='cgs'):

	#** 1. Random topics and dictionate
	if randomness>0:
		data=addrandomcol(data,K,-1,randomness)#K
	
	if dict_:
		if compressed:
			data,idx2vals,vals2idx,_=dictionate(data,cols=[0,1])
		else:
			data,idx2vals,vals2idx,_=dictionate(data)
	else:
		idx2vals=None
		vals2idx=None

	
	if algo=="cgs" :
		z_d=join2(data[:][:,[0,2]])
		w_z=join2(data[:][:,[2,1]])
		z_=join2(data[:][:,[2]])

	#** 2. Inference
	if algo=="cgs":
		afterdata,D,W=cgs(data,z_d,w_z,z_,K,it,alpha,beta,PATH)
		
	elif algo=="heron":
		herondata,D,W,Z=preprocessData_old(data,K,compressed)
		herondata,D,W,Z=fixedp(g,herondata,D,W,Z,K,it,alpha,beta,PATH,maxiter=it)
	
	elif algo=="herongpu":	
		
		if batch>0 : # and compressed
			
			data,pz,D,W,Z=preprocessData(data,K,compressed)
			
			if batch>len(data):
				print "Batch size=",batch,"> len(data)=",len(data)
				batch=len(data)
			
			from_=list(xrange(0,len(data),batch))
			to_=from_[1:]+[from_[-1]+batch]
		
			Z = np.array(Z[:,np.newaxis],dtype=np.float32,order='C')
			
			
			for i in range(it):
				print "Iteration",i,"------------------------------------------"

				fullD=np.zeros(np.shape(D),dtype=np.float32)
				fullW=np.zeros(np.shape(W),dtype=np.float32)
				fullZ=np.zeros(np.shape(Z),dtype=np.float32)
			
				for f,t in zip(from_,to_):
					data_batch=data[f:t]
					pz_batch=pz[f:t]

					_,partD,partW,partZ=LDAHERONGPU(data_batch[:,[0,1,2]],W, D, Z,pz_batch,K,1,alpha,beta,PATH="")

					fullD+=partD
					fullW+=partW	
					fullZ+=partZ
					
					del _,data_batch
			
				D=fullD
				W=fullW
				Z=fullZ	
				
				
				if PATH!="":
					if (i+1)%5==0:
						np.save(PATH+"wz_lda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),fullW)
						np.save(PATH+"dz_lda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),fullD)
				

		else:
			data,pz,D,W,Z=preprocessData(data,K,compressed)
			Z = np.array(Z[:,np.newaxis],dtype=np.float32,order='C')
			afterdata,D,W,Z=LDAHERONGPU(data[:][:,[0,1,2]],W, D, Z,pz,K,it,alpha,beta,PATH)

	elif algo=="cool":

		if batch>0 : # and compressed
			
			data,pz,D,W,Z=preprocessData(data,K,compressed)
			
			if batch>len(data):
				print "Batch size=",batch,"> len(data)=",len(data)
				batch=len(data)
			
			from_=list(xrange(0,len(data),batch))
			to_=from_[1:]+[from_[-1]+batch]
			
		
			Z = np.array(Z[:,np.newaxis],dtype=np.float32,order='C')
			del pz 
			
			for i in range(it):
				print "Iteration",i,"------------------------------------------"

				fullD=np.zeros(np.shape(D),dtype=np.float32)
				fullW=np.zeros(np.shape(W),dtype=np.float32)
				fullZ=np.zeros(np.shape(Z),dtype=np.float32)
			
				for f,t in zip(from_,to_):
					data_batch=data[f:t].copy()
					_,partD,partW,partZ=LDACOOLGPU(data_batch[:,[0,1,2]],W, D, Z,K,1,alpha,beta,PATH="")
					fullD+=partD
					fullW+=partW	
					fullZ+=partZ
					
					del _,data_batch
			
				D=fullD
				W=fullW
				Z=fullZ	
				
				
				
				if PATH!="":
					if (i+1)%5==0:
						np.save(PATH+"wz_lda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),fullW)
						np.save(PATH+"dz_lda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),fullD)

		else:
			data,pz,D,W,Z=preprocessData(data,K,compressed)
			del pz
			Z = np.array(Z[:,np.newaxis],dtype=np.float32,order='C')
			afterdata,D,W,Z=LDACOOLGPU(data[:][:,[0,1,2]],W, D, Z,K,it,alpha,beta,PATH)
	else:
		print "Inference method not supported"
		assert(0)
	
			
	return data,D,W,idx2vals,vals2idx


