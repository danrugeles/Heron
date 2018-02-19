import numpy as np
import time
#-----------------------------------------------------------------------------------------
#------------------------------------ CGS -------------------------------------
#-----------------------------------------------------------------------------------------

def computef3(cited,dz,d,eta):

	f3=np.zeros(np.shape(dz)[1])
	
	for citeddoc in cited:
		f3+=dz[citeddoc]
		
	if len(cited)>0:
		f3/=f3.sum()
		Nd=dz[d].sum()
		return np.exp((eta/Nd)*f3)
	
	# No cited documents
	else:
		return np.ones(np.shape(dz)[1])
	
	
	
#--computePosterior-------------------------------------------------
"""Computes the posterior p(z_i|z,alpha,beta)
Input: 2darray,2darray,1darray,float,float,int
Output: 1darray"""
#-------------------------------------------------------------------
def computePosterior(Factor3,dz,wz,z,row,alpha,beta,eta,rowid):

	d,w,k=row
	Factor1=dz[d]+alpha
	Factor2=(wz[w]+beta)/(z+beta*len(wz))
	
	
	p_z=np.multiply(np.multiply(Factor1,Factor2),Factor3)
	p_z=p_z/p_z.sum()
	
	return p_z
	
#--sampling---------------------------------------------------------
"""Performs Collapsed Gibbs Sampling
Input: 2darray,2darray,2darray,1darray,float,float
Output: 1darray"""
#-------------------------------------------------------------------
def sampling(data,dd,dz,wz,z,alpha,beta,eta):

	prevd=-1
	f3=0
	for rowid,row in enumerate(data):
		d,w,k=row
		
		if d != prevd:
			f3=computef3(dd[d],dz,d,eta)
		
		dz[d][k]-=1
		wz[w][k]-=1
		z[k]-=1
					
		p_z=computePosterior(f3,dz,wz,z,row,alpha,beta,eta,rowid)
		newk=np.random.multinomial(1, p_z).argmax()
		dz[d][newk]+=1
		wz[w][newk]+=1
		z[newk]+=1
		
		data[rowid][2]=newk
		prevd=d

	return data,dz,wz,z


def cgs(data,dd,dz,wz,z,K,it,alpha,beta,eta,PATH=""):

	if PATH!="":
		np.save(PATH+"wz_lda"+"_".join(map(str,[algo,K,alpha,beta,0])),w_z.T)
		np.save(PATH+"dz_lda"+"_".join(map(str,[algo,K,alpha,beta,0])),z_d)
		
	for i in range(it):
		start=time.time()
		data,dz,wz,z=sampling(data,dd,dz,wz,z,alpha,beta,eta)
		
		if PATH!="":
			if (i+1)%5==0:
				np.save(PATH+"wz_lda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),w_z.T)
				np.save(PATH+"dz_lda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),z_d)
		print("Iteration %d took %f",i,time.time()-start)
			
	return data,dz,wz

