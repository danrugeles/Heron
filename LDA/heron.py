import numpy as np
from aux import toDistribution
import time
#-----------------------------------------------------------------------------------------
#    ------------------------------------ HERON -------------------------------------
#-----------------------------------------------------------------------------------------
def preprocessData(data,K,compressed):

	if compressed:
		data=np.sort(data.copy().view('i8,i8,i8,i8'), order=['f0','f1'], axis=0).view(np.int)
		data=map(lambda row: [row[0],row[1],row[2],toDistribution(row[3],K)],data)
		data=np.array(data,dtype=object)
		
		
		#** 2. Find model Statistics: D,W,Z,newdata
		pz=0
		c=0
		prevd=data[0][0]
		
		Nd=max(data.T[0])+1
		Nw=max(data.T[1])+1
		
		D=np.zeros((Nd,K))
		W=np.zeros((Nw,K))
		Z=0
		for row in data:
			d,w,c,z=row
	
			W[w]+=z
			D[d]+=z
			Z+=z
			
		return data,D,W,Z
			
	else:
	
		data=np.sort(data.copy().view('i8,i8,i8'), order=['f0','f1'], axis=0).view(np.int)
		data=map(lambda row: [row[0],row[1],toDistribution(row[2],K)],data)
		data=np.array(data,dtype=object)

		#** 2. Find model Statistics: D,W,Z,newdata
		newdata=[]
		pz=0
		c=0
		prevd=data[0][0]
		prevw=data[0][1]
	
		Nd=max(data.T[0])+1
		Nw=max(data.T[1])+1
	
		D=np.zeros((Nd,K))
		W=np.zeros((Nw,K))
		Z=0
		for row in data:
			d,w,z=row
	
			W[w]+=z
			D[d]+=z
			Z+=z
		
			if d!=prevd or w!=prevw:
				newdata.append([prevd,prevw,c,pz/pz.sum()])
				pz=z
				c=1
				prevd=d
				prevw=w
			else:
				pz+=z
				c+=1
		newdata.append([prevd,prevw,c,pz/pz.sum()])
	
		print("len(herondata) %d",len(newdata))
	
		return np.array(newdata,dtype=object),D,W,Z


def fixedp(f,data,D,W,Z,K,it,alpha,beta,PATH="",tol=10e-5,maxiter=300):
	""" Fixed point algorithm """
	e = 1
	itr = 0

	Nd=len(D)
	Nw=len(W)
	algo="heron"
	
	if PATH!="":
			np.save(PATH+"wz_lda"+"_".join(map(str,[algo,K,alpha,beta,0])),W)
			np.save(PATH+"dz_lda"+"_".join(map(str,[algo,K,alpha,beta,0])),D)
		
	#while(e > tol and itr < maxiter):
	while( itr < maxiter):
		start=time.time()
			
		data,D,W,Z = f(data,D,W,Z,K,alpha,beta)      # fixed point equation
		
		#print "new Heron D",D
		#print "new Heron W",W
		#print "new Heron Z",Z
		
		#e = norm(x0-x) # error at the current step
		
		print "Iteration",itr,"took",time.time()-start
		
		itr = itr + 1
		
		if PATH!="":
			if (itr+1)%5==0:
				np.save(PATH+"wz_lda"+"_".join(map(str,[algo,K,alpha,beta,(itr+1)])),W)
				np.save(PATH+"dz_lda"+"_".join(map(str,[algo,K,alpha,beta,(itr+1)])),D)
		
		
	return data,D,W,Z
 

def g(newdata,D,W,Z,K,alpha,beta):
	
	Nw=len(W)
	Nd=len(D)
	newD=np.zeros((Nd,K))
	newW=np.zeros((Nw,K))
	newZ=0
	
	#print "heron"
	#print D
	#print W
	#print Z
	#print newdata
	#print "-----"
	
	for i,row in enumerate(newdata):
	
		d,w,c,z=row
		
		newz=np.multiply((D[d]-z+alpha),(W[w]-z+beta))/(Z-z+Nw*beta)
		
		#print i,newz,"stat:",D[d],W[w],Z,z
		newz=newz/newz.sum()
		
		newdata[i][-1]=newz
		newD[d]+=c*newz
		newW[w]+=c*newz
		newZ+=c*newz

	return newdata,newD,newW,newZ
