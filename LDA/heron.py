import numpy as np
from aux import toDistribution
import time
from itertools import izip
#-----------------------------------------------------------------------------------------
#    ------------------------------------ HERON -------------------------------------
#-----------------------------------------------------------------------------------------
def preprocessData(data,K,compressed):

	if compressed:
	
		pz=np.zeros((len(data),K),dtype=np.float32)
		for id_,row in enumerate(data):
			pz[id_][row[3]]+=1
	
		
		print "Finding statistics ..."
		
		#** 2. Find model Statistics: D,W,Z,newdata
		c=0
		prevd=data[0][0]
		
		Nd=max(data.T[0])+1
		Nw=max(data.T[1])+1
		print "Nd,Nw",Nd,Nw
		
		if 1:
			D=np.zeros((Nd,K),dtype=np.float32)
			W=np.zeros((Nw,K),dtype=np.float32)
			Z=0
			for row,p in izip(data,pz):
				d,w,c,_=row
			
				W[w]+=p*c
				D[d]+=p*c
				Z+=p*c
				
			np.save("D_"+str(K),D)
			np.save("W_"+str(K),W)
			np.save("Z_"+str(K),Z)	
				
		else:
			D=np.load("D.npz")
			W=np.load("W.npz")
			Z=np.load("Z.npz")
		
		return data,pz,D,W,Z
			
	else:
	
		data=np.sort(data.copy().view('i8,i8,i8'), order=['f0','f1'], axis=0).view(np.int)

		#** 2. Find model Statistics: D,W,Z,newdata
		newdata=[]
		p_z=[]
		pz=np.zeros((K))
		c=0
		prevd=data[0][0]
		prevw=data[0][1]
	
		Nd=max(data.T[0])+1
		Nw=max(data.T[1])+1
	
		D=np.zeros((Nd,K),dtype=np.float32)
		W=np.zeros((Nw,K),dtype=np.float32)
		Z=np.zeros((K))
		for row in data:
			d,w,z=row
	
			W[w][z]+=1
			D[d][z]+=1
			Z[z]+=1
		
			if d!=prevd or w!=prevw:
				newdata.append([prevd,prevw,c])
				p_z.append(pz/pz.sum())
				pz[z]=1
				c=1
				prevd=d
				prevw=w
			else:
				pz[z]+=1
				c+=1
		newdata.append([prevd,prevw,c])
		p_z.append(pz/pz.sum())
		
		
		print "len(herondata)",len(newdata),"tuples"
		
		return np.array(newdata,dtype=np.float32),np.array(p_z,dtype=np.float32),D,W,Z
	
	
	
	
#-----------------OLD (used by heron)
def preprocessData_old(data,K,compressed):

	data.view('i8,i8,i8').sort(order=['f0','f1'], axis=0)
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

	print "len(herondata)",len(newdata),"tuples"
	
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
	
	
	for i,row in enumerate(newdata):
	
		d,w,c,z=row
		
		newz=np.multiply((D[d]-z+alpha),(W[w]-z+beta))/(Z-z+Nw*beta)
	
		newz=newz/newz.sum()
		
		newdata[i][-1]=newz
		newD[d]+=c*newz
		newW[w]+=c*newz
		newZ+=c*newz

	return newdata,newD,newW,newZ
