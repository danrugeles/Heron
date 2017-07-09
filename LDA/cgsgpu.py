import numpy as np
import pycuda.driver as cuda
from  auxgpu import *
from pycuda.compiler import SourceModule
import math 
from pycuda.curandom import rand as curand

	
def LDACGSGPU(data,dz,wz,z,K,it,alpha,beta,PATH=""):
	
	alpha,beta,K = map(np.float32,[alpha,beta,K])
	N=len(data)	
	blocksize=32
	
	data=np.array(data,dtype=np.float32,order='C')
	dz = np.array(dz,dtype=np.float32,order='C')
	wz = np.array(wz,dtype=np.float32,order='C')
	z = np.array(z[:,np.newaxis],dtype=np.float32,order='C')
	

	nwords=np.shape(wz)[0]
	nwords = np.int32(nwords)
	
	#** Allocate structures in GPU arch
	struct_data_ptr = cuda.mem_alloc(Matrix.mem_size)
	struct_wz_ptr = cuda.mem_alloc(Matrix.mem_size)
	struct_dz_ptr = cuda.mem_alloc(Matrix.mem_size)
	struct_z_ptr = cuda.mem_alloc(Matrix.mem_size)
	
	#** Prepare the auxiliary structures
	data_mat= Matrix(data,struct_data_ptr)
	wz_mat= Matrix(wz,struct_wz_ptr)
	dz_mat= Matrix(dz,struct_dz_ptr)
	z_mat= Matrix(z,struct_z_ptr)

	#** Create Kernel
	func = LDACGSKernel(K,alpha,beta,nwords)
	
	#** Call Kernel
	grid=(int(math.ceil(N/(blocksize*1.))), 1)
	block=(blocksize, 1, 1)
	
	for i in range(it):
		topic_gpu = curand((N,))
		func(struct_data_ptr,struct_dz_ptr,struct_wz_ptr,struct_z_ptr,topic_gpu,grid=grid, block=block, shared=0)
		print bringToCpu(dz_mat),bringToCpu(wz_mat),bringToCpu(z_mat)
		if PATH!="":
				if (i+1)%5==0:
					D,W=bringToCpu(dz_mat),bringToCpu(wz_mat)
					np.save(PATH+"wz_lda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),W)
					np.save(PATH+"dz_lda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),D)
	
	return data,bringToCpu(dz_mat),bringToCpu(wz_mat),bringToCpu(z_mat)
	
	

def LDACGSKernel(K,alpha,beta,nwords):

	mod=SourceModule("""	
			#include <stdio.h>
			
			#define alpha """+str(alpha)+"""
			#define beta """+str(beta)+"""
			#define nwords """+str(nwords)+"""		
			const int K="""+str(K)+"""; 
						
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
	    	 	*p=*p+value;
	    	 	__syncthreads();
	    	 	//atomicAdd(p,value);
	    }
	    
	
		  __global__ void computePosterior(Matrix *data,Matrix *dz,Matrix *wz,Matrix *z,float* topics)
		  {
		  	
		  	 const int idx = threadIdx.x + blockDim.x*blockIdx.x;

		     if (idx < data->N){

	  	   	float d=getValue(data,idx,0);
	  	   	float w=getValue(data,idx,1);  
	  	   	float oldk=getValue(data,idx,2);  
	  	   		   		   	
	  	   	float total=0;
	  	   	float temp_pk [K];

	  	   	//Substract statistics	
	  	    addValue(dz,d,oldk,-1.0f);
	  	   	addValue(wz,w,oldk,-1.0f);
	  	   	addValue(z,oldk,0,-1.0f);
	  	   	
	  	   	//Compute Posterior   	
	  	   	for(int k=0;k<K;k++){
			 			float statD = getValue(dz,d,k)+alpha;
			 			float statW = getValue(wz,w,k)+beta;
			 			float statZ = getValue(z,k,0)+nwords*beta;
			 			temp_pk[k] = statD*statW/statZ;
			 			//printf("%d ->  %f,%f,%f-%f,%f,%f = %f\\n",idx,d,w,k,getValue(dz,d,k),getValue(wz,w,k),statZ,temp_pk[k]);
			 			//printf("Posterior %d,%d: %f\\n",idx,int(k),statD);
			 			total+=temp_pk[k];
			 		}		
		 			for(int k=0;k<K;k++){
		 				temp_pk[k] = temp_pk[k]/total;
	  	   	}
	  	   	
	  	   	
		 			//printf("Assign %f to %d,%d -- %f,%f,%f,%f\\n",pk,idx,k,getValue(dz,d,k),getValue(wz,w,k),getValue(z,k,0),getValue(p_z,idx,k));
		 			
		 			//Find new topic from CDF
		 			float cdf=0;
		 			int newk;
		 			for(int k=0;k<K;k++){
		 				cdf+=temp_pk[k];
		 				if(topics[idx]<cdf){
		 					newk=k;
		 					break;
		 				}
	  	   	}
	  	   		
	  	   	//update statistics
	  	   	addValue(dz,d,newk,1);
	  	   	addValue(wz,w,newk,1);
	  	   	addValue(z,newk,0,1);
	  	   	assignValue(data,idx,2,newk);
	  	   }
		  } 
		  """)
	#mod=SourceModule("")
	
	#** Compile Kernel
	return mod.get_function("computePosterior")
