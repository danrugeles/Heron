import numpy as np
import pycuda.driver as cuda
from  auxgpu import *
from pycuda.compiler import SourceModule
import math 
from pycuda.curandom import rand as curand
import time 

def SLDAHeronKernel(K,a,alpha,beta,eta,nwords):

	mod=SourceModule("""	
			#include <stdio.h>
			
			
			#define a """+str(a)+"""
			#define alpha """+str(alpha)+"""
			#define beta """+str(beta)+"""
			#define eta """+str(eta)+"""
			#define nwords """+str(nwords)+"""
			
			
			const int K = """+str(K)+""";
			
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
	    
		  __global__ void computePosterior(Matrix *data,Matrix *dz,Matrix *wz,Matrix *z,Matrix *p_z)
		  {
		  	 const int idx = threadIdx.x + blockDim.x*blockIdx.x;

		     if (idx < data->N){

	  	   	float d=getValue(data,idx,0);
	  	   	float w=getValue(data,idx,1);
	  	   	float y=getValue(data,idx,3);
	  	   	
	  	   	
	  	   	float Nd=0;
	  	   	for(int k=0;k<K;k++){
	  	   			float dzk=getValue(dz,d,k);
	  	   			float pzk=getValue(p_z,idx,k);
	  	   			Nd+=dzk;
	  	   			Nd-=pzk;
	  	   	}		
	  	   		
	  	   	
	  	   	
	  	   	float total=0;
	  	   	float temp_pk [K];
	  	   	for(int k=0;k<K;k++){
	  	   			float dzk=getValue(dz,d,k);
	  	   			dzk-=getValue(p_z,idx,k);
	  	   			
	  	   			//printf("SECOND: %f at %d,%d at %d\\n",dzk,d,k,idx);
	  	   			
	  	   			float statD = dzk+alpha;
	  	   			float statW = getValue(wz,w,k)-getValue(p_z,idx,k)+beta;
	  	   			float statZ = getValue(z,k,0)-getValue(p_z,idx,k)+nwords*beta;	  	   
	  	   			float link = __expf(2*(eta/Nd)*(y-a-eta*(dzk/Nd))-((eta/Nd)*(eta/Nd)));
	  	   		
	  	   			//printf("%f=%f,%f,%f,%f,%f at %d\\n",link,y,a,eta,dzk,Nd,idx);
	  	   			temp_pk[k]=statD*statW*link/statZ;
	  	   			total+=temp_pk[k];
	  	   	
	  	   			//printf("Assign %f to %d,%d -- %f,%f,%f,%f\\n",pk,idx,k,getValue(dz,d,k),getValue(wz,w,k),getValue(z,k,0),getValue(p_z,idx,k));
	  	   			
	  	   	}
	  	   	
	  	   	for(int k=0;k<K;k++){
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

def SLDAHERONGPU(data, wz, dz, z, p_z,K,it,a,alpha,beta,eta,PATH=""):
	
	tau = 64.0
	gamma = 0.5
	

	algo="heron"
	data=np.array(data,dtype=np.float32,order='C')
	#dz = np.array(dz,dtype=np.float32,order='C')
	#wz = np.array(wz,dtype=np.float32,order='C')
	#z = np.array(z[:,np.newaxis],dtype=np.float32,order='C')
	#p_z=np.concatenate(p_z).reshape((len(data),K),order='c').astype(np.float32)
	assert(str(dz.dtype)=="float32") #order must be 'C'
	assert(str(wz.dtype)=="float32") #order must be 'C'
	assert(str(p_z.dtype)=="float32") #order must be 'C'
	assert(str(z.dtype)=="float32") #order must be 'C'


	nwords=np.shape(wz)[0]
	nwords,K = map(np.int32,[nwords,K])
	alpha,beta = map(np.float32,[alpha,beta])
	N=len(data)	
	blocksize=32
	
	#** Allocate structures in GPU arch
	struct_data_ptr = cuda.mem_alloc(Matrix.mem_size)
	struct_wz_ptr = cuda.mem_alloc(Matrix.mem_size)
	struct_dz_ptr = cuda.mem_alloc(Matrix.mem_size)
	struct_z_ptr = cuda.mem_alloc(Matrix.mem_size)
	struct_p_z_ptr = cuda.mem_alloc(Matrix.mem_size)
	
	#** Prepare the auxiliary structures
	data_mat= Matrix(data,struct_data_ptr)
	wz_mat= Matrix(wz,struct_wz_ptr)
	dz_mat= Matrix(dz,struct_dz_ptr)
	z_mat= Matrix(z,struct_z_ptr)
	p_z_mat= Matrix(p_z,struct_p_z_ptr)

	#** Create Kernel
	func = SLDAHeronKernel(K,a,alpha,beta,eta,nwords)
	func2 = updateStatsKernel(K)
	
	#** Call Kernel
	grid=(int(math.ceil(N/(blocksize*1.))), 1)
	block=(blocksize, 1, 1)
	
	if PATH!="":
		np.save(PATH+"wz_slda"+"_".join(map(str,[algo,K,alpha,beta,0])),wz)
		np.save(PATH+"dz_slda"+"_".join(map(str,[algo,K,alpha,beta,0])),dz)
		
	prevwz=wz
	
	for i in range(it):
		start=time.time()

		ro =(tau+i)**(-gamma)
		
		func(struct_data_ptr,struct_dz_ptr,struct_wz_ptr,struct_z_ptr,struct_p_z_ptr,grid=grid, block=block, shared=0)
	
		# Result holder # Cant be dz_mat2 because dz_mat2 holds current result
		dz_mat=Matrix(np.zeros(np.shape(dz),dtype=np.float32,order='C'),struct_dz_ptr)
		wz_mat=Matrix(np.zeros(np.shape(wz),dtype=np.float32,order='C'),struct_wz_ptr)
		z_mat=Matrix(np.zeros(np.shape(z),dtype=np.float32,order='C'),struct_z_ptr)

		func2(struct_data_ptr,struct_dz_ptr,struct_wz_ptr,struct_z_ptr,struct_p_z_ptr,grid=grid,block=block, shared=0)
	
		#----------Added for batch ----------------
		wz=bringToCpu(wz_mat)
		assert(ro<=1)
		wz=wz*(1-ro)+prevwz*(ro)
		prevwz=wz
	
		wz_mat= Matrix(wz,struct_wz_ptr)
		
		if PATH!="":
			if (i+1)%20==0:
				D,W=bringToCpu(dz_mat),bringToCpu(wz_mat)
				
				np.save(PATH+"wz_slda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),W)
				np.save(PATH+"dz_slda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),D)
		
		print "Iteration took ",time.time()-start
		
	return data,bringToCpu(dz_mat),bringToCpu(wz_mat),bringToCpu(z_mat)
	
