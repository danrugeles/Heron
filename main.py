import numpy as np
import optparse
import csv
import os

# Auxiliary functions
from aux import dictionate,addrandomcol
from deco import *
from random import seed
from getData import *

#**1. Parse Settings  
parser = optparse.OptionParser()
parser.add_option("-f", dest="filename", help="Path to the filename of the dataset")
parser.add_option("-m", dest="model", type="string", help="RTM, LDA or SLDA [default: %default]", default="LDA")
parser.add_option("--inference", dest="inference", type="string", help="cgs, heron, cool or herongpu [default: %default]", default="heron")
parser.add_option("--batch", dest="batch", type="int", help="Number of tuples in a batch. 0 indicates that no batches are performed which is the fastest version if the RAM can cope. Not that the CPU versions, namely cgs and heron do not support batches. [default: %default]", default=0)
parser.add_option("--alpha", dest="alpha", type="float", help="Hyper-parameter alpha [default: %default]", default=0.75)
parser.add_option("--beta", dest="beta", type="float", help="Hyper-parameter beta [default: %default]", default=0.75)
parser.add_option("--eta", dest="eta", type="float", help="Hyper-parameter eta [default: %default]", default=1.0)
parser.add_option("-a", dest="a", type="float", help="Hyper-parameter a [default: %default]", default=0.5)
parser.add_option("-k", dest="K", type="int", help="Number of topics [default: %default]", default=20)
parser.add_option("-i", dest="iteration", type="int", help="Number of iterations [default: %default]", default=500)
parser.add_option("-r", dest="randomness", type="int", help="Random initialization: 0 for using a given initialization, or 1 for uniform initialization of the topic assignments [default: %default]", default=0)
parser.add_option("--path", dest="path", type="string", help="path for saving results. if given an empty string \"\", the parameters will not be saved. [default: %default]", default="Save/")
parser.add_option("--seed", dest="seed", type="int", help="random seed")
parser.add_option("--compression", action="store_true", dest="compression", help="Experimental support for alternative reading of a different data structure. Please refer to Readme.")
(options, args) = parser.parse_args()


#**2. Error Handling
if not options.filename:
	parser.error("filename(-f) is required. Try main.py -h for help")

if options.seed != None:
	np.random.seed(options.seed)
	seed(options.seed)

if options.iteration<1 or options.K<1:
	parser.error("Number of iterations and number of topics must be greater than 0. Try main.py -h for help")
	

if options.batch<0 or options.alpha<0 or options.beta<0 or options.eta<0 or options.a<0:
	print "Batch size, alpha, beta, eta and a must be positive values. Try main.py -h for help"

if (options.model=="cgs" and options.batch>0) or (options.model=="heron" and options.batch>0):
	print "CGS, neither heron support batches please try cool or herongpu."

#**3. Run Inference Algorithm
if options.randomness:
	print "\nRunning ",options.inference,options.model,"["+str(options.K)+" topics] - Uniformly initialized\nHyperparameters: alpha:",options.alpha,"beta:",options.beta,"eta:",options.eta,"a:",options.a,"\nNumber of iterations:",options.iteration,"; number of tuples in a batch is ",options.batch,"\n"
else:
	print "\nRunning ",options.inference,options.model,"["+str(options.K)+" topics] - NO initialization\nHyperparameters: alpha:",options.alpha,"beta:",options.beta,"eta:",options.eta,"a:",options.a,"\nNumber of iterations:",options.iteration,"; number of tuples in a batch is ",options.batch,"\n"

print "The parameters are being saved at:",options.path,"\n"
	
if options.model == "LDA":
	import LDA.lda as lda
	data=np.load(options.filename)
	data,_,_,_=dictionate(data,cols=[0,1])
	train,test=splitTrainTestRepeated(data,0.7)		
	
	it=options.iteration
	path=options.path
	
	#** Initialize outside the method call for fair comparison between models 
	train=addrandomcol(train,options.K,-1,1)

	data,dz,wz,idx2vals,vals2idx=lda.lda(train,options.K,options.iteration,options.alpha,options.beta,batch=options.batch,randomness=options.randomness,dict_=False,PATH=options.path,algo=options.inference,compressed=options.compression)

elif options.model == "RTM":
	import RTM.rtm as rtm

	path,_=os.path.split(options.filename)
	dd=np.load(path+"/dd.npy")
	train=np.load(options.filename)

	train=addrandomcol(train,options.K,-1,1)#K

	data,dz,wz,idx2vals,vals2idx=rtm.rtm(train,dd,options.K,options.iteration,options.alpha,options.beta,options.eta,randomness=options.randomness,dict_=False,batch=options.batch,PATH=options.path,algo=options.inference,compressed=options.compression)


elif options.model == "SLDA":
	import SLDA.slda as slda

	data=np.load("Datasets/movielens/dictionateddata.npy")
	data,_,_,_=dictionate(data,cols=[0,1])
	train,test=splitTrainTest(data,0.7)		

	train=addrandomcol(train,options.K,-1,1)

	data,dz,wz,idx2vals,vals2idx=slda.slda(train,options.K,options.iteration,options.a,options.alpha,options.beta,options.eta,batch=options.batch,randomness=options.randomness,dict_=False,PATH=options.path,algo=options.inference,compressed=options.compression)


else:
	parser.error("model "+options.model+" is not supported please try one of the following: RTM, LDA or SLDA. Try main.py -h for help")


