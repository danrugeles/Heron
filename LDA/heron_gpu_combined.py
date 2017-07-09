# prepared invocations and structures -----------------------------------------
from __future__ import print_function
from __future__ import absolute_import
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import math
import numpy as np

from aux import dictionate,join2,addrandomcol,normalize_rows,duplicates,toDistribution
np.set_printoptions(precision=3)
from deco import *
import time
from random import randint,seed
seed(1)
np.random.seed(0)
from getData import *


from LDA.heron import *
from LDA.cgs import *


#TODO: Don't send zeros after every iteration. Keep the update as a new kernel and save memory too
#TODO: Use Share memory


__author__ = "Daniel Rugeles"
__copyright__ = "Copyright 2017, Heron LDA"
__credits__ = ["Daniel Rugeles"]
__license__ = "GPL"
__version__ = "2.0.1"
__maintainer__ = "Daniel Rugeles"
__email__ = "daniel007@e.ntu.edu.sg"
__status__ = "Released"

# Version two released Heron GPU
# Version one released Heron LDA



#--lda--------------------------------------------------------------
"""Runs LDA over the given data
data: 2darray shape (n,3) where n is the number of records. each record is a tuple (Document,word,topic)
K: number of topics
it: number of iterations
alpha,beta: hyperparameters
dict_: True if data must be passed through dictionate()
verbose: True if it will output the extracted topics - not implemented yet
randomness= 1 if we want to intialize the topics, 0 if we dont want to
PATH: Path to save results
algo: 'cgs' for collapsed gibbs sampling, 'motion' for Gibbs Motion
"""
#-------------------------------------------------------------------
@cache
def lda(data,K,it,alpha,beta,dict_=True,verbose=True,randomness=1,PATH="",algo='cgs'):

	#** 1. Random topics and dictionate
	data=np.asarray(data)
	if randomness>0:
		data=addrandomcol(data,K,-1,randomness)#K

	if dict_:
		data,idx2vals,vals2idx,_=dictionate(data)
	else:
		idx2vals=None
		vals2idx=None
	
	data=data.astype(float)		
	data=data.astype(np.int)
	
	z_d=join2(data[:][:,[0,2]])
	w_z=join2(data[:][:,[2,1]])
	z_=join2(data[:][:,[2]])

	
	#** 2. Inference
	if algo=="heron":
		herondata,D,W,Z=preprocessData(data,K)
		print("len(herondata) %d",len(herondata))
		herondata,D,W,Z=fixedp(g,herondata,D,W,Z,K,alpha,beta,PATH,maxiter=it)
		return herondata,D,W,idx2vals,vals2idx
	
	elif algo=="motion":
		data=map(lambda row: [row[0],row[1],toDistribution(row[2],K)],data)
		assert(False)

	elif algo=="herongpu":	
		herondata,D,W,Z=preprocessData(data,K)
		#p_z=map(lambda row: list(toDistribution(row[2],K)),data)
		#data=np.array(data,dtype=np.object)
		herondata,D,W,Z=LDAGPU(herondata[:][:,[0,1,2]],W, D, Z, herondata[:][:,-1],K,alpha,beta)
		return herondata,D,W,idx2vals,vals2idx

	elif algo=="cgs":
	
		if PATH!="":
			np.save(PATH+"wz_lda"+"_".join(map(str,[algo,K,alpha,beta,0])),w_z.T)
			np.save(PATH+"dz_lda"+"_".join(map(str,[algo,K,alpha,beta,0])),z_d)
		for i in range(it):
			start=time.time()
			if algo=="cgs":
				data,z_d,w_z,z_=sampling(data,z_d,w_z,z_,alpha,beta)
			elif algo=="motion":
				data,z_d,w_z,z_=motion(data,z_d,w_z,z_,alpha,beta)
			else:
				print("Only cgs and motion are implemented")
				assert(False)
		
			if PATH!="":
				if (i+1)%5==0:
					np.save(PATH+"wz_lda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),w_z.T)
					np.save(PATH+"dz_lda"+"_".join(map(str,[algo,K,alpha,beta,(i+1)])),z_d)
			print("Iteration %d took %f",i,time.time()-start)
				

	return data,w_z,z_d,idx2vals,vals2idx

# Helper class
class Matrix:
	
		# Float* + 4 integers
		mem_size = 8 + 2*np.uintp(0).nbytes
		
		def __init__(self,matrix,struct_ptr):
		
			self.data = cuda.to_device(matrix) 
			self.shape, self.dtype = matrix.shape, matrix.dtype
			
			cuda.memcpy_htod(int(struct_ptr), np.getbuffer(np.int32(np.shape(matrix)[0])))
			cuda.memcpy_htod(int(struct_ptr) + 4, np.getbuffer(np.int32(np.shape(matrix)[1])))
			cuda.memcpy_htod(int(struct_ptr) + 8, np.getbuffer(np.int32(matrix.strides[1])))               
			cuda.memcpy_htod(int(struct_ptr) + 16, np.getbuffer(np.uintp(int(self.data))))
		
		def __str__(self):
			return "Matrix: "+str(cuda.from_device(self.data,self.shape,self.dtype))

def bringToCpu(*args):
	arrays=[]
	for arg in args:
		arrays.append(cuda.from_device(arg.data,arg.shape,arg.dtype))
	
	return arrays
		


def LDAHeronKernel(K,alpha,beta,nwords):

	mod=SourceModule("""	
			#include <stdio.h>
			
			#define K """+str(K)+"""
			#define alpha """+str(alpha)+"""
			#define beta """+str(beta)+"""
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

		  __global__ void computePosterior(Matrix *data,Matrix *dz,Matrix *wz,Matrix *z,
		                                   Matrix *dz2,Matrix *wz2,Matrix *z2,Matrix *p_z)
		  {
		  	 const int idx = threadIdx.x + blockDim.x*blockIdx.x;

		     if (idx < data->N){

	  	   	float d=getValue(data,idx,0);
	  	   	float w=getValue(data,idx,1);
	  	   	float c=getValue(data,idx,2);
	  	   	
	  	   	
	  	   	float total=0;
	  	   	float temp_pk [K];
	  	   	for(int k=0;k<K;k++){
	  	   			
	  	   			float statD = getValue(dz,d,k)-getValue(p_z,idx,k)+alpha;
	  	   			float statW = getValue(wz,w,k)-getValue(p_z,idx,k)+beta;
	  	   			float statZ = getValue(z,k,0)-getValue(p_z,idx,k)+nwords*beta;
	  	   			temp_pk[k]=statD*statW/statZ;
	  	   			total+=temp_pk[k];
	  	   			
	  	   			//printf("Assign %f to %d,%d -- %f,%f,%f,%f\\n",pk,idx,k,getValue(dz,d,k),getValue(wz,w,k),getValue(z,k,0),getValue(p_z,idx,k));
	  	   			
	  	   	}
	  	   	
	  	   	//Update Stats
	  	   	for(int k=0;k<K;k++){
	  	   		float pzk = temp_pk[k]/total;
	  	   		assignValue(p_z,idx,k,pzk);
	  	   		addValue(dz2,d,k,pzk*c);
	  	   		addValue(wz2,w,k,pzk*c);
	  	   		addValue(z2,k,0,pzk*c);  	
	  	   	}
	  	   
	  	   }
	  	   
	  	  
	  	
		  } 
		  """)
	#mod=SourceModule("")
	
	#** Compile Kernel
	return mod.get_function("computePosterior")

def execute(func,struct_data_ptr,struct_dz_ptr,struct_wz_ptr,struct_z_ptr,struct_dz_ptr2,struct_wz_ptr2,struct_z_ptr2,struct_p_z_ptr,N,blocksize):
	func(struct_data_ptr,struct_dz_ptr,struct_wz_ptr,struct_z_ptr,struct_dz_ptr2,struct_wz_ptr2,struct_z_ptr2,struct_p_z_ptr,grid=(int(math.ceil(N/(blocksize*1.))), 1), block=(blocksize, 1, 1), shared=0)
	return struct_dz_ptr,struct_wz_ptr,struct_z_ptr,struct_dz_ptr2,struct_wz_ptr2,struct_z_ptr2

def LDAGPU(data, wz, dz, z, p_z,K,alpha,beta):
	

	data=np.array(data,dtype=np.float32,order='C')
	dz = np.array(dz,dtype=np.float32,order='C')
	wz = np.array(wz,dtype=np.float32,order='C')
	z = np.array(z[:,np.newaxis],dtype=np.float32,order='C')
	dz2=np.zeros(np.shape(dz),dtype=np.float32,order='C')
	wz2=np.zeros(np.shape(wz),dtype=np.float32,order='C')
	z2=np.zeros(np.shape(z),dtype=np.float32,order='C')
	p_z=np.concatenate(p_z).reshape((len(data),K),order='c').astype(np.float32)
	
	#print("gpui")
	#print(wz)
	#print(dz)
	#print(z)
	#print(p_z)
	#print("-----")
	

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
	struct_wz_ptr2 = cuda.mem_alloc(Matrix.mem_size)
	struct_dz_ptr2 = cuda.mem_alloc(Matrix.mem_size)
	struct_z_ptr2 = cuda.mem_alloc(Matrix.mem_size)
	struct_p_z_ptr = cuda.mem_alloc(Matrix.mem_size)
	
	#** Prepare the auxiliary structures
	# In disorder to adjust the function execute
	data_mat= Matrix(data,struct_data_ptr)
	wz_mat= Matrix(wz,struct_wz_ptr2)
	dz_mat= Matrix(dz,struct_dz_ptr2)
	z_mat= Matrix(z,struct_z_ptr2)
	wz_mat2= Matrix(wz2,struct_wz_ptr)
	dz_mat2= Matrix(dz2,struct_dz_ptr)
	z_mat2= Matrix(z2,struct_z_ptr)
	p_z_mat= Matrix(p_z,struct_p_z_ptr)

	#** Create Kernel
	func = LDAHeronKernel(K,alpha,beta,nwords)
	
	#** Call Kernel
	for it in range(50):
		struct_dz_ptr,struct_wz_ptr,struct_z_ptr,struct_dz_ptr2,struct_wz_ptr2,struct_z_ptr2=execute(func,struct_data_ptr,struct_dz_ptr2,struct_wz_ptr2,struct_z_ptr2,struct_dz_ptr,struct_wz_ptr,struct_z_ptr,struct_p_z_ptr,N,blocksize)
	
		
		#print(bringToCpu(dz_mat2))
	
		# Result holder # Cant be dz_mat2 because dz_mat2 holds current result
		dz_mat=Matrix(np.zeros(np.shape(dz),dtype=np.float32,order='C'),struct_dz_ptr)
		wz_mat=Matrix(np.zeros(np.shape(wz),dtype=np.float32,order='C'),struct_wz_ptr)
		z_mat=Matrix(np.zeros(np.shape(z),dtype=np.float32,order='C'),struct_z_ptr)
	
		struct_dz_ptr,struct_wz_ptr,struct_z_ptr,struct_dz_ptr2,struct_wz_ptr2,struct_z_ptr2=execute(func,struct_data_ptr,struct_dz_ptr2,struct_wz_ptr2,struct_z_ptr2,struct_dz_ptr,struct_wz_ptr,struct_z_ptr,struct_p_z_ptr,N,blocksize)
	
		#if it == 99:
			#print(bringToCpu(dz_mat))
	
		# Result holder # Cant be dz_mat because dz_mat holds current result
		dz_mat2=Matrix(np.zeros(np.shape(dz),dtype=np.float32,order='C'),struct_dz_ptr)
		wz_mat2=Matrix(np.zeros(np.shape(wz),dtype=np.float32,order='C'),struct_wz_ptr)
		z_mat2=Matrix(np.zeros(np.shape(z),dtype=np.float32,order='C'),struct_z_ptr)

	return data,dz,wz,z
	
	
		
"""----------------------------*
*                              *
*   |\  /|   /\    |  |\  |    * 
*   | \/ |  /__\   |  | \ |    *
*   |    | /    \  |  |  \|    *
*                              *
*----------------------------"""
if __name__=="__main__":

	#Data
	K=3
	alpha=0.01
	beta=0.01
	it=10

	data=np.array([[0, 0, 2],
								 [0, 1, 1],
								 [0, 1, 0],
								 [0, 1, 2],
								 [0, 1, 0],
								 [0, 1, 2],
								 [0, 1, 0],
								 [1, 0, 0],
								 [1, 0, 0],
								 [1, 0, 1],
								 [1, 1, 1],
								 [1, 1, 2],
								 [1, 1, 0],
								 [1, 1 ,1],
								 [1, 2 ,1],
								 [1, 2 ,0],
								 [1, 4 ,2],
								 [2, 2 ,1],
								 [2, 1 ,2],
								 [2, 1 ,1],
								 [2, 2 ,1],
								 [2, 3, 2],
								 [2, 4, 2],
								 [3, 2, 0],
								 [3, 2, 1],
								 [3, 2, 1],
								 [3, 3, 0],
								 [3, 3, 2],
								 [3, 3, 1],
								 [3, 3, 2],
								 [3, 4, 0],
								 [3, 5, 0],
								 [4, 3, 0],
								 [4, 3, 0],
								 [4, 3, 1],
								 [4, 3 ,0],
								 [4, 3 ,2],
								 [4, 3 ,0],
								 [4, 3 ,0],
								 [4, 3 ,2],
								 [4, 4 ,0],
								 [4, 4 ,1]])
	# Data
	herondata=data.copy()	
	
	if 0:
		data_after_lda,w_z,z_d,idx2vals,vals2idx=lda(data,K,it,alpha,beta,randomness=0,algo="herongpu")
		data_after_lda,w_z,z_d,idx2vals,vals2idx=lda(herondata,K,it,alpha,beta,randomness=0,algo="heron")

	if 1:
		data=np.load("Datasets/movielens/dictionateddata.npy")
		data,_,_,_=dictionate(data,cols=[0,1])
		train,test=splitTrainTestRepeated(data,0.7)		
	
		herontrain=train.copy()
		
		it=200
		path="Save/movielens/"
		path=""
		
		for alpha,beta in [(1.2,1.2)]:#(0.2,0.2),(0.5,0.5),(0.9,0.9),(1.2,1.2)
			for K in [100]:
				#print alpha,beta,K
			
					
				#print "\nGpu ...\n"
				import time
				start= time.time()
				data,dz,wz,idx2vals,vals2idx=lda(train,K,it,alpha,beta,randomness=1,dict_=False,PATH=path,algo="herongpu")
				print(time.time()-start)
				#print "\nFixed Point ...\n"	
				#data,dz,wz,idx2vals,vals2idx=lda(herontrain,K,it,alpha,beta,randomness=1,dict_=False,PATH=path,algo="heron")




