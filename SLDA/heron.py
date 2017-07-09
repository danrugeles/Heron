import numpy as np
from aux import toDistribution
import time
#-----------------------------------------------------------------------------------------
#    ------------------------------------ HERON -------------------------------------
#-----------------------------------------------------------------------------------------
def preprocessData(data,K):

	data=np.sort(data.copy().view('i8,i8,i8,i8'), order=['f0','f1'], axis=0).view(np.int)
	data=map(lambda row: [row[0],row[1],row[2],toDistribution(row[3],K)],data)
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
		d,w,y,z=row
	
		W[w]+=z
		D[d]+=z
		Z+=z
		
		if d!=prevd or w!=prevw:
			newdata.append([prevd,prevw,c,y,pz/pz.sum()])
			pz=z
			c=1
			prevd=d
			prevw=w
		else:
			pz+=z
			c+=1
	newdata.append([prevd,prevw,c,y,pz/pz.sum()])
	
	print("len(herondata) %d",len(newdata))
	
	return np.array(newdata,dtype=object),D,W,Z

#Finds the fixed point given a set of equations
def fixedp(f,data,D,W,Z,K,a,alpha,beta,eta,PATH="",tol=10e-5,maxiter=300):
	""" Fixed point algorithm """
	e = 1
	itr = 0

	Nd=len(D)
	Nw=len(W)
	
	algo="heron"
	if PATH!="":
		np.save(PATH+"wz_slda"+"_".join(map(str,[algo,K,alpha,beta,eta,0])),W) 
		np.save(PATH+"dz_slda"+"_".join(map(str,[algo,K,alpha,beta,eta,0])),D)
	
				
	#while(e > tol and itr < maxiter):
	while( itr < maxiter):
		start=time.time()
		
		data,D,W,Z = f(data,D,W,Z,K,a,alpha,beta,eta)      # fixed point equation
		#e = norm(x0-x) # error at the current step
		
		print "Iteration",itr,"took",time.time()-start
		

		if PATH!="":
			if (itr+1)%5==0:
				np.save(PATH+"wz_slda"+"_".join(map(str,[algo,K,alpha,beta,eta,(itr+1)])),W)
				np.save(PATH+"dz_slda"+"_".join(map(str,[algo,K,alpha,beta,eta,(itr+1)])),D)
		
		itr = itr + 1	
 	return data,D,W,Z


#Runs one pass throught all equations
def g(newdata,D,W,Z,K,a,alpha,beta,eta):

	Nw=len(W)
	Nd=len(D)
	newD=np.zeros((Nd,K))
	newW=np.zeros((Nw,K))
	newZ=0
	Factor3=0
	

	for i,row in enumerate(newdata):
	
		d,w,c,y,z=row

		Factor1=(D[d]-z+alpha)
		Factor2=(W[w]-z+beta)/(Z-z+Nw*beta)
		
		#Nd=(D[d]-z).sum()
		#Dd_=(D[d]-z)/Nd
		#Factor3=np.exp(2*(eta/Nd)*(y-a-eta*Dd_)-(eta/Nd)**2)
		
		Nd=D[d].sum()
		Dd_=(D[d])/Nd
		Factor3=np.exp(-((y-a-eta*Dd_)**2))
	
		newz=np.multiply(np.multiply(Factor1,Factor2),Factor3)
		newz=newz/newz.sum()
		
		newdata[i][-1]=newz
		newD[d]+=c*newz
		newW[w]+=c*newz
		newZ+=c*newz

	return newdata,newD,newW,newZ
