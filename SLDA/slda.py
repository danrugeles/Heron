#!/usr/bin/python2.7
# prepared invocations and structures -----------------------------------------
from __future__ import absolute_import
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Auxiliary functions
from aux import dictionate,join2,addrandomcol,normalize_rows,duplicates,toDistribution
np.set_printoptions(precision=3)
from deco import *
from getData import *

#** Algorithms
from SLDA.heron import *
from SLDA.cgs import *
from SLDA.herongpu import *
from SLDA.cool import *


__author__ = "Anonymous"
__copyright__ = "Copyright 2017, Heron SLDA"
__credits__ = ["Anonymous"]
__license__ = "GPL"
__version__ = "2.1.1"
__maintainer__ = "Anonymous"
__email__ = "Anonymous@Anonymous"
__status__ = "Released"

# Version two released Heron GPU
# 2.1 Added CGS GPU
# Version one released Heron SLDA


@cache
def slda(data,K,it,a,alpha,beta,eta,dict_=True,verbose=True,randomness=1,compressed=False,batch=0,PATH="",form='standard',algo='cgs'):

	#** 1. Random topics and dictionate
	if randomness>0:
		data=addrandomcol(data,K,3,randomness) 

	if dict_:
		data,idx2vals,vals2idx,_=dictionate(data,cols=[0,1]) 
	else:
		idx2vals=None
		vals2idx=None

	if algo=="cgs" or algo=="cgsgpu":
		dz=join2(data[:][:,[0,3]]) 
		wz=join2(data[:][:,[1,3]]) 
		z=join2(data[:][:,[3]])  
	
	
	#** TODO: UPDATE rating range from 0-5 for all methods except cool and herongpu
	
	#** 2. Inference
	if algo=="cgs":
		print "cgs ---------------------------------------------"
		print data[:10]
		
		afterdata,D,W=cgs(data,dz,wz,z,K,it,a,alpha,beta,eta,PATH)
		
	elif algo=="heron":
		print "heron ----------------------------------------------"
		herondata,D,W,Z=preprocessData_old(data,K,compressed)
		herondata,D,W,Z=fixedp(g,herondata,D,W,Z,K,a,alpha,beta,eta,PATH,maxiter=it)
	
	elif algo=="cgsgpu":
		print "cgs gpu ------------------------------------------"
		afterdata,D,W,Z=SLDACGSGPU(data,wz,dz,z,K,it,a,alpha,beta,eta,PATH)
		
	elif algo=="herongpu":	
		print "heron gpu ----------------------------------------"

		if batch>0 : # and compressed
			
			data,pz,D,W,Z=preprocessData(data,K,compressed)
			
			if batch>len(data):
				print "Batch size=",batch,"> len(data)=",len(data)
				batch=len(data)
			
			from_=list(xrange(0,len(data),batch))
			to_=from_[1:]+[from_[-1]+batch]
		
			Z = np.array(Z[:,np.newaxis],dtype=np.float32,order='C')
			
			print data
		
			
			for i in range(it):
				print "Iteration",i,"------------------------------------------"

				fullD=np.zeros(np.shape(D),dtype=np.float32)
				fullW=np.zeros(np.shape(W),dtype=np.float32)
				fullZ=np.zeros(np.shape(Z),dtype=np.float32)
			
				for f,t in zip(from_,to_):
					data_batch=data[f:t]
					pz_batch=pz[f:t]

					_,partD,partW,partZ=SLDAHERONGPU(data_batch[:,[0,1,2,3]],W, D, Z,pz_batch,K,1,a,alpha,beta,eta,PATH="")

					fullD+=partD
					fullW+=partW	
					fullZ+=partZ
					
					del _,data_batch
			
				D=fullD
				W=fullW
				Z=fullZ	
				
				
				if PATH!="":
					if (i+1)%5==0:
						np.save(PATH+"wz_slda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),fullW)
						np.save(PATH+"dz_slda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),fullD)
							
		else:	
		
			herondata,pz,D,W,Z=preprocessData(data,K,compressed)
			Z = np.array(Z[:,np.newaxis],dtype=np.float32,order='C')		
			afterdata,D,W,Z=SLDAHERONGPU(herondata[:][:,[0,1,2,3]],W, D, Z,pz,K,it,a,alpha,beta,eta,PATH)
			
	elif algo=="cool":
		print "cool ----------------------------------------"
		
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
					_,partD,partW,partZ=SLDACOOLGPU(data_batch[:,[0,1,2,3]],W, D, Z,K,1,a,alpha,beta,eta,PATH="")
					fullD+=partD
					fullW+=partW	
					fullZ+=partZ
					
					del _,data_batch
			
				D=fullD
				W=fullW
				Z=fullZ	
				
				
				if PATH!="":
					if (i+1)%5==0:
						np.save(PATH+"wz_slda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),fullW)
						np.save(PATH+"dz_slda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),fullD)
				
		
		else:
			herondata,pz,D,W,Z=preprocessData_old(data,K,compressed)
			del pz
			Z = np.array(Z[:,np.newaxis],dtype=np.float32,order='C')
			afterdata,D,W,Z=SLDACOOLGPU(herondata[:][:,[0,1,2,3]],W, D, Z,pz,K,it,a,alpha,beta,eta,PATH)
	
	
	else:
		print "Inference method not supported"
		assert(0)
	
			
	return data,D,W,idx2vals,vals2idx
