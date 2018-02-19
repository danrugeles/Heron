import numpy as np
import pycuda.driver as cuda
from  auxgpu import *
from pycuda.compiler import SourceModule
import math 
from pycuda.curandom import rand as curand
import time
from itertools import izip
#--toDistribution--------------------------------------------------------------
"""one-hot encoding of position in a K-dimensional vector
Input: int,int
Output: 1darray"""
#-------------------------------------------------------------------
def toDistribution(position,K):
	distribution=np.zeros(K)
	distribution[position]=1
	return np.array(distribution)
	
# removes duplicates
# Adds counts in third column
# Adds dd_id on fourth column
# Adds distribution in last column
def preprocessDataHeronGPU_old(data,dd,K,blocksize):

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
			newdata.append([prevd,prevw,c,0,pz/pz.sum()])
			pz=z
			c=1
			prevd=d
			prevw=w
		else:
			pz+=z
			c+=1
	newdata.append([prevd,prevw,c,0,pz/pz.sum()])
	newdata=np.array(newdata,dtype=object)
	
	#** Add dd_id
	
	i=0
	prevd=-1
	for rowid,row in enumerate(newdata):
		d=row[0]
		if i==blocksize or d!=prevd:
			i=0		
		try:
			dd_id=dd[d][i]
		except IndexError:
			dd_id=-1
		newdata[rowid][3]=dd_id	
		i+=1
		prevd=d
	
	return np.array(newdata,dtype=object),D,W,Z


def preprocessDataHeronGPU(data,dd,K,blocksize,compressed):

	if compressed:
	
		assert 0
	
	else:
		#TODO: Update so that it returns pz
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
				newdata.append([prevd,prevw,c,0])
				p_z.append(pz/pz.sum())
				pz[z]=1
				c=1
				prevd=d
				prevw=w
			else:
				pz[z]+=1
				c+=1
		newdata.append([prevd,prevw,c,0])
		p_z.append(pz/pz.sum())
		
			#** Add dd_id
	
		i=0
		prevd=-1
		for rowid,row in enumerate(newdata):
			d=row[0]
			if i==blocksize or d!=prevd:
				i=0		
			try:
				dd_id=dd[d][i]
			except IndexError:
				dd_id=-1
			newdata[rowid][3]=dd_id	
			i+=1
			prevd=d
		
		return np.array(newdata,dtype=np.float32),np.array(p_z,dtype=np.float32),D,W,Z
	



def RTMHeronKernel(K,alpha,beta,eta,nwords):

	mod=SourceModule("""	
			#include <stdio.h>
			
			#define K """+str(K)+"""
			#define alpha """+str(alpha)+"""
			#define beta """+str(beta)+"""
			#define eta """+str(eta)+"""
			#define nwords """+str(nwords)+"""
			
		  struct Matrix{
		  	int N,M,stride, __padding;
		  	float* data;
		  };
		
			__device__ float getValue(Matrix* data,int x,int y){
	     	float* p =(float*)((char*)data->data + x*data->M*data->stride+y*data->stride );
	     	return *p;
	    }
	  
	    __device__ void assignValue(Matrix* data,int x,int y, float value){
	    	 	float* p = (float*)((char*)data->data + x*data->M*data->stride+y*data->stride  );
	    	 	*p=value;
	    }
	    
	    __device__ void addValue(Matrix* data,int x,int y, float value){
	    	 	float* p = (float*)((char*)data->data + x*data->M*data->stride+y*data->stride  );
	    	 	atomicAdd(p,value);
	    }
	    
	    
		  __global__ void computePosterior(Matrix *data,Matrix *dz,Matrix *ddz,Matrix *wz,Matrix *z,Matrix *p_z)
		  {
		  	 const int idx = threadIdx.x + blockDim.x*blockIdx.x;

		     if (idx < data->N){
		     
				   	float d=getValue(data,idx,0);
				   	float w=getValue(data,idx,1);
				   	float dd_id=getValue(data,idx,3);
				   	
				   	//printf("idx:%d,dd_id:%d\\n",int(idx),int(dd_id));
				   	
				   	// Compute ddz
	 	   			if (dd_id>=0){
				   		for(int k=0;k<K;k++){
				   			float dzk= getValue(dz,dd_id,k);
			   				addValue(ddz,d,k,dzk);
				   		}	
				   	}

				   	__syncthreads();
				   				   			
				   	float Nd=0;
				   	float Ndd=0;
				   	for(int k=0;k<K;k++){
				   			float ddzk=getValue(ddz,d,k);
				   			float dzk=getValue(dz,d,k);
				   			Nd+=dzk; 
				   			Ndd+=ddzk;
				   	}		
				   	
				   	// Compute Posterior
				   	float total=0;
				   	float temp_pk [K];
				   	if (Ndd>0){
							for(int k=0;k<K;k++){
				   			float dzk=getValue(dz,d,k);
				   			float ddzk=getValue(ddz,d,k);
				   			
				   			float statD = dzk-getValue(p_z,idx,k)+alpha;
				   			float statW = getValue(wz,w,k)-getValue(p_z,idx,k)+beta;
				   			float statZ = getValue(z,k,0)-getValue(p_z,idx,k)+nwords*beta;
				   			
				   			//printf("%d->eta(%0.2f)/Nd(%d)*ddzk(%0.2f)/Ndd(%d)\\n",idx,eta,int(Nd),ddzk,int(Ndd),(eta/Nd)*(ddzk/Ndd));
				   			float link = __expf((eta/Nd)*(ddzk/Ndd));
				   			
				   			//printf("%d link at %d: %f\\n",idx,int(k),link);
				   			temp_pk[k]=statD*statW*link/statZ;
				   			total+=temp_pk[k];

				   			//printf("Assign %f to %d,%d -- %f,%f,%f,%f\\n",pk,idx,k,getValue(dz,d,k),getValue(wz,w,k),getValue(z,k,0),getValue(p_z,idx,k));	
				   		}
						}else{
							for(int k=0;k<K;k++){
						 			float dzk=getValue(dz,d,k);
						 			float ddzk=getValue(ddz,d,k);
						 			float statD = dzk-getValue(p_z,idx,k)+alpha;
						 			float statW = getValue(wz,w,k)-getValue(p_z,idx,k)+beta;
						 			float statZ = getValue(z,k,0)-getValue(p_z,idx,k)+nwords*beta;
						 			temp_pk[k]=statD*statW/statZ;
						 			total+=temp_pk[k];
						 			//printf("Assign %f to %d,%d -- %f,%f,%f,%f\\n",pk,idx,k,getValue(dz,d,k),getValue(wz,w,k),getValue(z,k,0),getValue(p_z,idx,k));
						 		}
						}
						
						//Normalize  	   	 	
				   	for(int k=0;k<K;k++){
				   		//printf("temp,total,idx: %f,%f,%d, ",temp_pk[k],total,idx);
				   		float pzk = temp_pk[k]/total;
				   		
				   		assignValue(p_z,idx,k,pzk);
				   	}
	  	   }
		  } 
		  """)
	#** Compile Kernel
	return mod.get_function("computePosterior")
	
	
def updateStatsKernel(K):

	mod=SourceModule("""	
			#include <stdio.h>
			
			#define K """+str(K)+"""
			
		  struct Matrix{
		  	int N,M,stride, __padding;
		  	float* data;
		  };
		
			__device__ float getValue(Matrix* data,int x,int y){
	     	float* p =(float*)((char*)data->data + x*data->M*data->stride+y*data->stride );
	     	return *p;
	    }
	  
	    __device__ void addValue(Matrix* data,int x,int y, float value){
	    	 	float* p = (float*)((char*)data->data + x*data->M*data->stride+y*data->stride  );
	    	 	atomicAdd(p,value);
	    }

		  __global__ void updateStats(Matrix *data,Matrix *dz,Matrix *wz,Matrix *z,Matrix *p_z)
		  {
		  	 const int idx = threadIdx.x + blockDim.x*blockIdx.x;

		     if (idx < data->N){

	  	   	float d=getValue(data,idx,0);
	  	   	float w=getValue(data,idx,1);
	  	   	float c=getValue(data,idx,2);
	  	   	
	  	   	//Update Stats
	  	   	for(int k=0;k<K;k++){
	  	   		float pzk=getValue(p_z,idx,k);
	  	   		addValue(dz,d,k,pzk*c);
	  	   		addValue(wz,w,k,pzk*c);
	  	   		addValue(z,k,0,pzk*c);  	
	  	  
	  	   }
		  } 
		 }
		  """)
	
	#** Compile Kernel
	return mod.get_function("updateStats")

def RTMHERONGPU(data, wz, dz, z, p_z,K,it,alpha,beta,eta,PATH=""):
	
	algo="heron"
	tau = 64.0
	gamma = 0.5
	
	ddz = np.array(np.zeros(np.shape(dz)),dtype=np.float32,order='C')

	data=data.astype(np.float32,copy=False,order='C')		
	assert(str(dz.dtype)=="float32") #order must be 'C'
	assert(str(wz.dtype)=="float32") #order must be 'C'
	assert(str(z.dtype)=="float32") #order must be 'C'
	assert(str(p_z.dtype)=="float32") #order must be 'C'
	dzn = np.zeros(np.shape(dz),dtype=np.float32,order='C')
	wzn = np.zeros(np.shape(wz),dtype=np.float32,order='C')
	zn = np.zeros(np.shape(z),dtype=np.float32,order='C')

	nwords=np.shape(wz)[0]
	nwords,K = map(np.int32,[nwords,K])
	alpha,beta,eta = map(np.float32,[alpha,beta,eta])
	N=len(data)	
	blocksize=32 # Its also used on the preprocessing!
	
	#** Allocate structures in GPU arch
	struct_data_ptr = cuda.mem_alloc(Matrix.mem_size)
	struct_wz_ptr = cuda.mem_alloc(Matrix.mem_size)
	struct_dz_ptr = cuda.mem_alloc(Matrix.mem_size)
	struct_ddz_ptr = cuda.mem_alloc(Matrix.mem_size)
	struct_z_ptr = cuda.mem_alloc(Matrix.mem_size)
	struct_p_z_ptr = cuda.mem_alloc(Matrix.mem_size)
	
	#** Prepare the auxiliary structures
	data_mat= Matrix(data,struct_data_ptr)
	wz_mat= Matrix(wz,struct_wz_ptr)
	dz_mat= Matrix(dz,struct_dz_ptr)
	ddz_mat= Matrix(ddz,struct_ddz_ptr)
	z_mat= Matrix(z,struct_z_ptr)
	p_z_mat= Matrix(p_z,struct_p_z_ptr)

	#** Create Kernel
	func = RTMHeronKernel(K,alpha,beta,eta,nwords)
	func2 = updateStatsKernel(K)
	
	#** Call Kernel
	grid=(int(math.ceil(N/(blocksize*1.))), 1)
	block=(blocksize, 1, 1)
	
	if PATH!="":
		np.save(PATH+"wz_slda"+"_".join(map(str,[algo,K,alpha,beta,eta,0])),wz)
		np.save(PATH+"dz_slda"+"_".join(map(str,[algo,K,alpha,beta,eta,0])),dz)

	prevwz=wz
	for i in range(it):
		start=time.time()
		
		# For batch
		ro =(tau+i)**(-gamma)
		
		func(struct_data_ptr,struct_dz_ptr,struct_ddz_ptr,struct_wz_ptr,struct_z_ptr,struct_p_z_ptr,grid=grid, block=block, shared=0)
	
		dz_mat=Matrix(np.zeros(np.shape(dz),dtype=np.float32,order='C'),struct_dz_ptr)
		ddz_mat=Matrix(np.zeros(np.shape(ddz),dtype=np.float32,order='C'),struct_ddz_ptr) 
		wz_mat=Matrix(np.zeros(np.shape(wz),dtype=np.float32,order='C'),struct_wz_ptr)
		z_mat=Matrix(np.zeros(np.shape(z),dtype=np.float32,order='C'),struct_z_ptr)

		func2(struct_data_ptr,struct_dz_ptr,struct_wz_ptr,struct_z_ptr,struct_p_z_ptr,grid=grid,block=block, shared=0)
		
		# For batch
		wz=bringToCpu(wz_mat)
		assert(ro<=1)
		wz=wz*(1-ro)+prevwz*(ro)
		prevwz=wz
		wz_mat= Matrix(wz,struct_wz_ptr)

		
		print "Iteration took",time.time()-start
		
		if PATH!="":
			if (i+1)%20==0:
				D,W=bringToCpu(dz_mat),bringToCpu(wz_mat)
				
				np.save(PATH+"wz_rtm"+"_".join(map(str,[algo,K,alpha,beta,eta,(i+1)])),W)
				np.save(PATH+"dz_rtm"+"_".join(map(str,[algo,K,alpha,beta,eta,(i+1)])),D)
	
	return data,bringToCpu(dz_mat),bringToCpu(wz_mat),bringToCpu(z_mat)
	
