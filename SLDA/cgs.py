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
def computePosterior(dz,wz,z,row,a,alpha,beta,eta,rowid):

	d,w,y,k=row
	Factor1=dz[d]+alpha
	Factor2=(wz[w]+beta)/(z+beta*len(wz))
	
	Nd=dz[d].sum()
	dzd_=dz[d]/Nd
	Factor3=np.exp(2*(eta/Nd)*(y-a-eta*dzd_)-(eta/Nd)**2)
	
	#print "row",row
	#print "f1",Factor1
	#print "f2",Factor2
	#print "f3",Factor3,"\n"
	
	p_z=np.multiply(np.multiply(Factor1,Factor2),Factor3)

	p_z=p_z/p_z.sum()
	
	return p_z
	
#--sampling---------------------------------------------------------
"""Performs Collapsed Gibbs Sampling
Input: 2darray,2darray,2darray,1darray,float,float
Output: 1darray"""
#-------------------------------------------------------------------
def sampling(data,dz,wz,z,a,alpha,beta,eta):

	for rowid,row in enumerate(data):
		d,w,y,k=row
		
		dz[d][k]-=1
		wz[w][k]-=1
		z[k]-=1
					
		p_z=computePosterior(dz,wz,z,row,a,alpha,beta,eta,rowid)
		newk=np.random.multinomial(1, p_z).argmax()
		
		
		dz[d][newk]+=1
		wz[w][newk]+=1
		z[newk]+=1
		
		data[rowid][3]=newk


	return data,dz,wz,z


def cgs(data,dz,wz,z,K,it,a,alpha,beta,eta,PATH=""):

	if PATH!="":
		np.save(PATH+"wz_lda"+"_".join(map(str,[algo,K,alpha,beta,0])),w_z.T)
		np.save(PATH+"dz_lda"+"_".join(map(str,[algo,K,alpha,beta,0])),z_d)
		
	for i in range(it):
		start=time.time()
		data,dz,wz,z=sampling(data,dz,wz,z,a,alpha,beta,eta)
		
		if PATH!="":
			if (i+1)%5==0:
				np.save(PATH+"wz_lda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),w_z.T)
				np.save(PATH+"dz_lda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),z_d)
		print("Iteration %d took %f",i,time.time()-start)
			
	return data,dz,wz

