import numpy as np
from aux import toDistribution
import time
#-----------------------------------------------------------------------------------------
#    ------------------------------------ HERON -------------------------------------
#-----------------------------------------------------------------------------------------
def preprocessData_old(data,K,compressed):

	data=np.sort(data.copy().view('i8,i8,i8,i8'), order=['f0','f1'], axis=0).view(np.int)
	
	data=map(lambda row: [row[0],row[1],row[2],toDistribution(row[3],K)],data)
	data=np.array(data,dtype=object)

	#** Only for SLDA
	maxrating=np.max(data[:][:,2])	
	data[:][:,2]=(data[:][:,2]*5.0)/maxrating #idx 2 holds the rating at this stage
	
	
	#** 2. Find model Statistics: D,W,Z,newdata
	newdata=[]
	pz=0
	c=0
	prevd=data[0][0]
	prevw=data[0][1]
	prevy=data[0][2]

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
			newdata.append([prevd,prevw,c,prevy,pz/pz.sum()])
			pz=z
			c=1
			prevd=d
			prevw=w
			prevy=y
		else:
			pz+=z
			c+=1
			
	newdata.append([prevd,prevw,c,prevy,pz/pz.sum()])
	
	#** Aggregate results per doc
	doc2rat={}
	doc2doclen={}
	for row in newdata:	
		d,w,c,y,z=row
		try:
			doc2rat[d]+=y
			doc2doclen[d]+=1.0
		except:
			doc2rat[d]=y
			doc2doclen[d]=1.0
	
	lastdata=[]
	for row in newdata:	
		d,w,c,y,z=row
		row[3]=doc2rat[d]/doc2doclen[d]
		lastdata.append(row)

	return np.array(lastdata,dtype=object),D,W,Z

# d,w,y,z input format
def preprocessData(data,K,compressed):

	data=np.asarray(data,dtype=np.float32)
	

	#TODO: HAS not been tested since translated from lda - prevy and y is surely missing 
	if compressed:
		print "preprocess has not been implemented for compressed data"
		assert(0)
		# No need for sorting
		#data=np.sort(data.copy().view('i8,i8,i8,i8'), order=['f0','f1'], axis=0)
		#data.view('int8,int8,int8,int8').sort(order=['f0','f1'], axis=0)
		#data=data.astype(object,copy=False)
		#data=map(lambda row: [row[0],row[1],row[2],toDistribution(row[3],K)],data)
	
		pz=np.zeros((len(data),K),dtype=np.float32)
		for id_,row in enumerate(data):
			#pzlist.append(list(toDistribution(row[-1],K)))
			pz[id_][row[-1]]+=1
	
		
		#** 2. Find model Statistics: D,W,Z,newdata
		c=0
		prevd=data[0][0]
		newdata=[]
		Nd=max(data.T[0])+1
		Nw=max(data.T[1])+1
		print "Nd,Nw",Nd,Nw
		
	
		D=np.zeros((Nd,K),dtype=np.float32)
		W=np.zeros((Nw,K),dtype=np.float32)
		Z=0
		for row,p in izip(data,pz):
			d,w,c,_=row
		
			W[w]+=p*c
			D[d]+=p*c
			Z+=p*c
			

		#** Aggregate results per doc
		doc2rat={}
		doc2doclen={}
		for row in newdata:	
			d,w,c,y=row
			try:
				doc2rat[d]+=y
				doc2doclen[d]+=1.0
			except:
				doc2rat[d]=y
				doc2doclen[d]=1.0

		lastdata=[]
		for row in newdata:	
			d,w,c,y=row
			row[3]=doc2rat[d]/doc2doclen[d]
			lastdata.append(row)

		del newdata

		return np.array(lastdata,dtype=np.float32),np.array(p_z,dtype=np.float32),D,W,Z
		
			
	else:
		#TODO: Update so that it returns pz
		data=np.array(data,dtype=np.int)
		#data=np.sort(data.copy().view('i8,i8,i8,i8'), order=['f0','f1'], axis=0).view(np.int)
		data.view('i8,i8,i8,i8').sort(order=['f0','f1'], axis=0)
	
		#** 2. Find model Statistics: D,W,Z,newdata
		newdata=[]
		p_z=[]
		pz=np.zeros((K))
		c=0
		prevd=data[0][0]
		prevw=data[0][1]
		prevy=data[0][2]
		Nd=max(data.T[0])+1
		Nw=max(data.T[1])+1
	
		D=np.zeros((Nd,K),dtype=np.float32)
		W=np.zeros((Nw,K),dtype=np.float32)
		Z=np.zeros((K))
		for row in data:
			d,w,y,z=row
	
			W[w][z]+=1
			D[d][z]+=1
			Z[z]+=1
		
			if d!=prevd or w!=prevw:
				newdata.append([prevd,prevw,c,prevy])
				p_z.append(pz/pz.sum())
				pz[z]=1
				c=1
				prevd=d
				prevw=w
				prevy=y
			else:
				pz[z]+=1
				c+=1
		newdata.append([prevd,prevw,c,prevy])
		p_z.append(pz/pz.sum())
		

		print newdata[:10]
		#** Aggregate results per doc
		doc2rat={}
		doc2doclen={}
		for row in newdata:	
			d,w,c,y=row
			try:
				doc2rat[d]+=y
				doc2doclen[d]+=1.0
			except:
				doc2rat[d]=y
				doc2doclen[d]=1.0

		lastdata=[]
		for row in newdata:	
			d,w,c,y=row
			row[3]=doc2rat[d]/doc2doclen[d]
			lastdata.append(row)

		del newdata

		lastdata=np.array(lastdata,dtype=np.float32)
		
		#** Only for SLDA - note that 2nd row shifted to 3 row after added counts
		maxrating=np.max(lastdata[:][:,3])	
		lastdata[:][:,3]=(lastdata[:][:,3]*5.0)/(1.0*maxrating) #idx 2 holds the rating at this stage
	
		return lastdata,np.array(p_z,dtype=np.float32),D,W,Z
	
	
	

	


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
		
		print D
		print W
		
		print "Iteration",itr,"took",time.time()-start
		
	
		if PATH!="":
			if (itr+1)%20==0:
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
		
		Nd=(D[d]-z).sum()
		Dd_=(D[d]-z)/Nd
		Factor3=np.exp(2*(eta/Nd)*(y-a-eta*Dd_)-(eta/Nd)**2)
		
		#** TODO: Whats this below??
		#Nd=D[d].sum()
		#Dd_=(D[d])/Nd
		#Factor3=np.exp(-((y-a-eta*Dd_)**2))
	
		newz=np.multiply(np.multiply(Factor1,Factor2),Factor3)
		newz=newz/newz.sum()
		
		newdata[i][-1]=newz
		newD[d]+=c*newz
		newW[w]+=c*newz
		newZ+=c*newz

	return newdata,newD,newW,newZ
