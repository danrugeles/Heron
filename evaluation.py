import numpy as np
import optparse
from aux import dictionate,join2,normalize_rows,addrandomcol
#from settings import *

from getData import getData,splitTrainTestRepeated,splitTrainTest#preprocessdiggs

def precrecperp(testdata,x_y_):
	loglik=0
	for row in testdata:
		x,y=row
		try:
			loglik+=np.log(x_y_[x][y])
		except IndexError:
			#print x,y,"out of bounds in",np.shape(x_y_)
			pass
	return loglik
	
	
if __name__=="__main__":

	#**1. Parse Settings
	parser = optparse.OptionParser()
	parser.add_option("-d", dest="dataset", help="name of the dataset, either cora,movielens or digg")
	parser.add_option("-m", dest="model", type="string", help="RTM, LDA or SLDA [default: %default]", default="LDA")
	parser.add_option("--inference", dest="inference", type="string", help="cgs, heron, cool [default: %default]", default="heron")
	parser.add_option("--alpha", dest="alpha", type="float", help="Hyper-parameter alpha [default: %default]", default=0.75)
	parser.add_option("--beta", dest="beta", type="float", help="Hyper-parameter beta [default: %default]", default=0.75)
	parser.add_option("--eta", dest="eta", type="float", help="Hyper-parameter eta [default: %default]", default=1.0)
	parser.add_option("-k", dest="K", type="int", help="Number of topics [default: %default]", default=20)
	parser.add_option("-i", dest="iterations", type="int", help="Number of iterations [default: %default]", default=500)
	parser.add_option("--path", dest="path", type="string", help="path for saving results. if given an empty string \"\", the parameters will not be saved. [default: %default]", default="Save/")
	(options, args) = parser.parse_args()
	
	dataset=options.dataset
	method=options.model.lower()+options.inference
	alpha=options.alpha
	beta=options.beta
	eta=options.eta
	iterations=options.iterations
	K=options.K
	path=options.path	
	
	results=[]
	


	#** Split data into training and testing set
	SPLIT=0.7
	
	if "cora" in dataset:
		testdata=np.load(dataset+"/test.npy")
		testdata=testdata[:][:,[0,1]]

	if "movielens" in dataset:
		data=np.load(dataset+"/dictionateddata.npy")
		data,_,_,_=dictionate(data,cols=[0,1])
		if method[:3]=="lda" or method[:3]=="sld" : 
			train,testdata=splitTrainTest(data,0.7)
			
		elif method[:3]=="rtm": 
			train,testdata=splitTrainTestRepeated(data,0.7)
		else:
			print "Unsupported Method"
			assert(0)

	print dataset

	print method,dataset,",".join(map(str,[K,alpha,beta,eta]))+"----------------------------------------"
	for it in xrange(20,iterations+1,20):

		if method[:3]=="lda" or method[:3]=="sld":
			file_w_z="_".join(map(str,["wz",method,K,alpha,beta,it]))
			file_z_d="_".join(map(str,["dz",method,K,alpha,beta,it]))
		elif  method[:3]=="rtm" :
			file_w_z="_".join(map(str,["wz",method,K,alpha,beta,eta,it]))
			file_z_d="_".join(map(str,["dz",method,K,alpha,beta,eta,it]))
		else:
			print "Unsupported Method",method
			assert(0)
			

		try:

			z_x=np.load(path+file_w_z+".npy")								
			
			x_z=z_x.T											
			z_y=np.load(path+file_z_d+".npy")	


		except IOError:
			print path+file_w_z+".npy","cannot be found or"
			print path+file_z_d+".npy","cannot be found"
			continue


		x_z_=normalize_rows(x_z+beta)
		z_y_=normalize_rows(z_y+beta)	
		x_y_=np.dot(z_y_,x_z_)	
		x_y_=normalize_rows(x_y_)		

		p=np.exp(-precrecperp(testdata[:][:,[0,1]],x_y_)/len(testdata))

		print K,it,p

			
	print "*********************************************************************\n\n"


