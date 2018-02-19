import numpy as np
import pycuda.driver as cuda

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



def bringToCpu(arg):
	return cuda.from_device(arg.data,arg.shape,arg.dtype)
	
	
