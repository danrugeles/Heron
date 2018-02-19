import numpy as np
import pycuda.driver as cuda
from  auxgpu import *
from pycuda.compiler import SourceModule
import math 
from pycuda.curandom import rand as curand
from pycuda import characterize
import time 

def SLDACoolKernel(K,a,alpha,beta,eta,nwords):

	mod=SourceModule("""	
			#include <stdio.h>
			#include <curand_kernel.h>
			
			#define K """+str(K)+"""
			#define a """+str(a)+"""
			#define alpha """+str(alpha)+"""
			#define beta """+str(beta)+"""
			#define eta """+str(eta)+"""
			#define nwords """+str(nwords)+"""
			#define nsamps 100
			
			
		extern "C"{
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
			 
		
				//p_z not used!
			  __global__ void computePosterior(Matrix *data,Matrix *dz,Matrix *wz,Matrix *z,Matrix *dzn,Matrix *wzn,Matrix *zn,curandState* rstates)
			  {
			  
			  
			  const int idx = threadIdx.x + blockDim.x*blockIdx.x;

			  if (idx < data->N){
			  
				  curandState rstate = rstates[idx];

		  	   	float d=getValue(data,idx,0);
		  	   	float w=getValue(data,idx,1);
		  	   	float c=getValue(data,idx,2);
		  	   	float y=getValue(data,idx,3);
	  	   	
		  	   	
		  	   	float Nd=0;
		  	   	for(int k=0;k<K;k++){
		  	   			float dzk=getValue(dz,d,k);
		  	   			//float pzk=getValue(p_z,idx,k);
		  	   			Nd+=dzk;
		  	   			
		  	   	}		
		  	   	   	
		  	   	float total=0;
		  	   	float temp_pk [K];
		  	   	for(int k=0;k<K;k++){
		  	   		float dzk=getValue(dz,d,k);
	  	   			
	  	   			float statD = getValue(dz,d,k)+alpha;
	  	   			float statW = getValue(wz,w,k)+beta;
	  	   			float statZ = getValue(z,k,0)+nwords*beta;
	  	   			
	  	   			float link = __expf(2*(eta/Nd)*(y-a-eta*(dzk/Nd))-((eta/Nd)*(eta/Nd)));
	  	   			//printf("%f at %d\\n",link,idx);
	  	   			
	  	   			temp_pk[k]=statD*statW*link/statZ;
	  	   			total+=temp_pk[k];
	  	   			
	  	   			//printf("Assign %f to %d,%d -- %f,%f,%f,%f\\n",pk,idx,k,getValue(dz,d,k),getValue(wz,w,k),getValue(z,k,0),getValue(p_z,idx,k));
	  	   			
		  	   	}
		  	   	
		  	   	float pval=nsamps*c;
		  	   	
		  	   	for(int k=0;k<K;k++){
		  	   		float pzk = temp_pk[k]/total;
		  	   		float cr = curand_poisson(&rstate, pzk * pval);
		  	   		//assignValue(p_z,idx,k,cr);
		  	   		addValue(dzn,d,k,cr/nsamps); //nsamps seems to be optional
		  	   		addValue(wzn,w,k,cr/nsamps); //nsamps seems to be optional
		  	   		addValue(zn,k,0,cr/nsamps);  //nsamps seems to be optional
		  	   	}

		  	   }
		  	   
		  	  
			  }
		  } 
		  """,no_extern_c=True)
	#** Compile Kernel
	return mod.get_function("computePosterior")
	
	

def get_rng_states(size,block,grid):

	rng_states = cuda.mem_alloc(size*characterize.sizeof('curandState', "#include <curand_kernel.h>"))

	mod = SourceModule("""
	#include <curand_kernel.h>

	extern "C"
	{

		__global__ void init_rng(int nthreads, curandState *s )
		{
				int idx = blockIdx.x*blockDim.x + threadIdx.x;

				if (idx >= nthreads)
				        return;

				curand_init(1234, idx, 0, &s[idx]);
		}

	} // extern "C"
	""", no_extern_c=True)
	
	init_rng = mod.get_function('init_rng')

	init_rng(np.int32(size), rng_states, block=block, grid=grid)

 	return rng_states


def SLDACOOLGPU(data, wz, dz, z,K,it,a,alpha,beta,eta,PATH=""):
	
	algo="cool"
	
	tau = 64.0
	gamma = 0.5
	
	#data=np.array(data,dtype=np.float32,order='C')
	#dz = np.array(dz,dtype=np.float32,order='C')
	#wz = np.array(wz,dtype=np.float32,order='C')
	#z = np.array(z[:,np.newaxis],dtype=np.float32,order='C')
	data=data.astype(np.float32,copy=False,order='C')			
	assert(str(dz.dtype)=="float32") #order must be 'C'
	assert(str(wz.dtype)=="float32") #order must be 'C'
	assert(str(z.dtype)=="float32") #order must be 'C'
	dzn = np.zeros(np.shape(dz),dtype=np.float32,order='C')
	wzn = np.zeros(np.shape(wz),dtype=np.float32,order='C')
	zn = np.zeros(np.shape(z),dtype=np.float32,order='C')


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
	struct_wzn_ptr = cuda.mem_alloc(Matrix.mem_size)
	struct_dzn_ptr = cuda.mem_alloc(Matrix.mem_size)
	struct_zn_ptr = cuda.mem_alloc(Matrix.mem_size)
	
	#** Prepare the auxiliary structures
	data_mat= Matrix(data,struct_data_ptr)
	wz_mat= Matrix(wz,struct_wz_ptr)
	dz_mat= Matrix(dz,struct_dz_ptr)
	z_mat= Matrix(z,struct_z_ptr)
	wzn_mat = Matrix(wzn,struct_wzn_ptr)
	dzn_mat = Matrix(dzn,struct_dzn_ptr)
	zn_mat = Matrix(zn,struct_zn_ptr)
	
	#** Create Kernel
	func = SLDACoolKernel(K,a,alpha,beta,eta,nwords)

	
	#** Call Kernel
	grid=(int(math.ceil(N/(blocksize*1.))), 1)
	block=(blocksize, 1, 1)
	
	prevwz=wz
	if PATH!="":
		np.save(PATH+"wz_slda"+"_".join(map(str,[algo,K,alpha,beta,0])),wz)
		np.save(PATH+"dz_slda"+"_".join(map(str,[algo,K,alpha,beta,0])),dz)
	
	for i in range(it):
		
		start=time.time()
		ro =(tau+i)**(-gamma)

		
		rng_states_ptr=get_rng_states(N,block,grid)
		func(struct_data_ptr,struct_dz_ptr,struct_wz_ptr,struct_z_ptr,struct_dzn_ptr,struct_wzn_ptr,struct_zn_ptr,rng_states_ptr,grid=grid, block=block, shared=0)
	

		#print("Partial - Iteration %d took %f",i,time.time()-start)
		wz=bringToCpu(wzn_mat)

		assert(ro<=1)
		wz=wz*(1-ro)+prevwz*(ro)
		prevwz=wz
		


		dz=bringToCpu(dzn_mat)
		z=bringToCpu(zn_mat)
		
		
		dz_mat= Matrix(dz,struct_dz_ptr)
		wz_mat= Matrix(wz,struct_wz_ptr)
		z_mat= Matrix(z,struct_z_ptr)

		
		dzn_mat=Matrix(np.zeros(np.shape(dz),dtype=np.float32,order='C'),struct_dzn_ptr)
		wzn_mat=Matrix(np.zeros(np.shape(wz),dtype=np.float32,order='C'),struct_wzn_ptr)
		zn_mat=Matrix(np.zeros(np.shape(z),dtype=np.float32,order='C'),struct_zn_ptr)

		if PATH!="":
			if (i+1)%20==0:
				D,W=bringToCpu(dz_mat),bringToCpu(wz_mat)
				
				np.save(PATH+"wz_slda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),W)
				np.save(PATH+"dz_slda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),D)
	
		print "Iteration took ",time.time()-start
		
	return data,dz,wz,z
	
