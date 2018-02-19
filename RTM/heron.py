import numpy as np
from aux import toDistribution
import time
#-----------------------------------------------------------------------------------------
#    ------------------------------------ HERON -------------------------------------
#-----------------------------------------------------------------------------------------


def computef3(cited,dz,d,eta):

	f3=np.zeros(np.shape(dz)[1])
	
	if len(cited)>0:
		for citeddoc in cited:
			f3+=dz[citeddoc]
		
		Nd=dz[d].sum()
		f3/=f3.sum()
		
		return np.exp((eta/Nd)*f3)
	
	# No cited documents
	else:
		return np.ones(np.shape(dz)[1])
		

def preprocessData(data,K):

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
	newdata=np.array(newdata,dtype=object)
		

	return np.array(newdata,dtype=object),D,W,Z



#Finds the fixed point given a set of equations
def fixedp(f,data,dd,D,W,Z,K,alpha,beta,eta,PATH,tol=10e-5,maxiter=300):
	""" Fixed point algorithm """
	e = 1
	itr = 0

	Nd=len(D)
	Nw=len(W)
	
	#while(e > tol and itr < maxiter):
	while( itr < maxiter):
		
	
		data,D,W,Z = f(data,dd,D,W,Z,K,alpha,beta,eta)      # fixed point equation
		#e = norm(x0-x) # error at the current step
		
		print "heron\n",D,W
	
		itr = itr + 1
		
		if PATH!="":
			if itr%5==0:
				algo="heron"
				np.save(PATH+"wz_rtm"+"_".join(map(str,[algo,K,alpha,beta,eta,itr])),W)
				np.save(PATH+"dz_rtm"+"_".join(map(str,[algo,K,alpha,beta,eta,itr])),D)
			
 	return data,D,W,Z


#Runs one pass throught all equations
def g(newdata,dd,D,W,Z,K,alpha,beta,eta):

	Nw=len(W)
	Nd=len(D)
	newD=np.zeros((Nd,K))
	newW=np.zeros((Nw,K))
	newZ=0
	prevd=-1
	Factor3=0
	
	print len(newdata)
	start=time.time()

	for i,row in enumerate(newdata):
	
		d,w,c,z=row
		
		if prevd!=d:
			Factor3=computef3(dd[d],D,d,eta)
			
		Factor1=(D[d]-z+alpha)
		Factor2=(W[w]-z+beta)/(Z-z+Nw*beta)
		
		newz=np.multiply(np.multiply(Factor1,Factor2),Factor3)
		newz=newz/newz.sum()
		
		newdata[i][-1]=newz
		newD[d]+=c*newz
		newW[w]+=c*newz
		newZ+=c*newz
		
	print "Iteration took",time.time()-start
		
	return newdata,newD,newW,newZ

