import numpy as np
import time
#-----------------------------------------------------------------------------------------
#------------------------------------ CGS -------------------------------------
#-----------------------------------------------------------------------------------------

	
#--computePosterior-------------------------------------------------
"""Computes the posterior p(z_i|z,alpha,beta)
Input: 2darray,2darray,1darray,float,float,int
Output: 1darray"""
#-------------------------------------------------------------------
def computePosterior(z_d,w_z,z_,row,alpha,beta,rowid):
	d,w,z=row
	
	#print "f1",z_d[d]+alpha
	#print "f2",(w_z[:][:,w]+beta)/(z_+beta*len(w_z.T))
	#print "z",z_
	p_z=np.multiply(z_d[d]+alpha,(w_z[:][:,w]+beta)/(z_+beta*len(w_z.T)))
	#print "p_z",p_z
	#print ""
	p_z=p_z/p_z.sum()
	
	return p_z
	
#--sampling---------------------------------------------------------
"""Performs Collapsed Gibbs Sampling
Input: 2darray,2darray,2darray,1darray,float,float
Output: 1darray"""
#-------------------------------------------------------------------
def sampling(data,z_d,w_z,z_,alpha,beta):
	for rowid,row in enumerate(data):
		d,w,z=row
		z_d[d][z]-=1
		w_z[z][w]-=1
		z_[z]-=1
		p_z=computePosterior(z_d,w_z,z_,row,alpha,beta,rowid)
		newk=np.random.multinomial(1, p_z).argmax()
		#print "newk",newk
		z_d[d][newk]+=1
		w_z[newk][w]+=1
		z_[newk]+=1
		
		data[rowid][2]=newk
	return data,z_d,w_z,z_
	

	
#--motion---------------------------------------------------------
"""Performs Gibbs Motion
Input: 2darray,2darray,2darray,1darray,float,float
Output: 1darray"""
#-------------------------------------------------------------------
#elif algo=="motion":
#	data=map(lambda row: [row[0],row[1],toDistribution(row[2],K)],data)
#data,z_d,w_z,z_=motion(data,z_d,w_z,z_,alpha,beta)

def motion(data,z_d,w_z,z_,alpha,beta):
	for rowid,row in enumerate(data):
		d,w,p_z=row
		z_d[d]-=p_z
		w_z[:][:,w]-=p_z
		z_-=p_z
		#assert(np.sum(z_d<0)==0)
		#assert(np.sum(w_z<0)==0)
		p_z=computePosterior(z_d,w_z,z_,row,alpha,beta,rowid)
		
		z_d[d]+=p_z
		w_z[:][:,w]+=p_z
		z_+=p_z
		
		data[rowid][2]=p_z
	return data,z_d,w_z,z_



def cgs(data,z_d,w_z,z_,K,it,alpha,beta,PATH=""):

	if PATH!="":
		np.save(PATH+"wz_lda"+"_".join(map(str,[algo,K,alpha,beta,0])),w_z.T)
		np.save(PATH+"dz_lda"+"_".join(map(str,[algo,K,alpha,beta,0])),z_d)
		
	for i in range(it):
		start=time.time()
		data,z_d,w_z,z_=sampling(data,z_d,w_z,z_,alpha,beta)
		if PATH!="":
			if (i+1)%5==0:
				np.save(PATH+"wz_lda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),w_z.T)
				np.save(PATH+"dz_lda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),z_d)
		print("Iteration %d took %f",i,time.time()-start)
			
	return data,z_d,w_z.T

