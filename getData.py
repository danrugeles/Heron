#import rpy2.robjects as robjects
import numpy as np
from aux import dictionate
import mycsv

#2410 docs
#4356 links
#2960 vocab
#136394 number of tokens


#-----------------------------------------------------------------------------------------
#------------------------------------ Cora -------------------------------------
#-----------------------------------------------------------------------------------------

# data is the matrix with columns doc,word,count 
# dd is a list of numpy int arrays
def getData(path):

	def rdatopy(path,filename):
		# load your file
		robjects.r['load'](path+filename+'.rda')
		# retrieve the matrix that was loaded from the file
		return robjects.r[filename]

	#**1. Citation matrix
	cites=rdatopy(path,'cora.cites')
	dd=[]
	for row in cites:
		dd.append(np.array(row,dtype=np.int))
	
	#**2. Documents
	docs=rdatopy(path,'cora.documents')
	
	data=[]
	for docid,doc in enumerate(docs):
		words,counts=np.array(doc,dtype=np.int)
		for w,c in zip(words,counts):
			data.append([docid,w,c])
			
			
	return np.array(data,dtype=np.int),dd


#-----------------------------------------------------------------------------------------
#------------------------------------ ALL -------------------------------------
#-----------------------------------------------------------------------------------------
#Doc,word,others
def splitTrainTest(data,per):
	
	data=np.array(data,dtype=np.int)
	data=data[data[:,0].argsort()]
	
	# Index into lists for each Doc 
	indexeddata={}
	for row in data:
		try:		
			indexeddata[row[0]].append(row[1:])
		except KeyError:
			indexeddata[row[0]]=[row[1:]]

	# Split each Doc into train and test
	traindata=[]
	testdata=[]
	for k,v in indexeddata.items():
		trainlen=len(v)*per
		if trainlen<5:
			print "Warning: Document "+str(k)+" has only "+str(len(v))+" words in it"
			
		for idx,elem in enumerate(v):
				if idx<trainlen:
					traindata.append([k]+list(elem)+[0])
				else:
					testdata.append([k]+list(elem))
					
	return np.array(traindata,dtype=np.int),np.array(testdata,dtype=np.int)

# Doc,word,count
def splitTrainTestRepeated(data,per):
	
	data=np.array(data,dtype=np.int)
	data=data[data[:,0].argsort()]
	
	# Index into lists for each Doc 
	indexeddata={}
	for row in data:
		try:		
			indexeddata[row[0]].append(row[1:])
		except KeyError:
			indexeddata[row[0]]=[row[1:]]

	# Split each Doc into train and test
	traindata=[]
	testdata=[]
	for k,v in indexeddata.items():
		trainlen=len(v)*per
		if trainlen<5:
			print "Warning: Document "+str(k)+" has only "+str(len(v))+" words in it"
			
		for idx,elem in enumerate(v):
		
			for i in range(elem[1]):
			#for i in range(1):
			
				if idx<trainlen:
					traindata.append([k,elem[0],0])
				else:
					testdata.append([k,elem[0]])
					
	return np.array(traindata,dtype=np.int),np.array(testdata,dtype=np.int)


#-----------------------------------------------------------------------------------------
#    ------------------------------------ DIGGS -------------------------------------
#-----------------------------------------------------------------------------------------
"""import nltk
from nltk.stem.porter import *

allwords=set()
alldocs=[]
def preprocessdiggs(path):
	with open(path+"docs.txt") as f:
		for doc in f.readlines():
			doc=doc.lower()
			words=nltk.word_tokenize(doc)
			stemmer = PorterStemmer()
			finaldoc=map(stemmer.stem,words)
			for word in finaldoc:
				allwords.add(word)
			alldocs.append(finaldoc)
			
	_,_,val2idxs,_=dictionate(np.array(list(allwords))[:,np.newaxis])

	val2idxs=val2idxs[0]
	
	with open(path+"diggs.dat") as f:
		diggs=f.readlines()
		
	finaldata=[]
	for docid,(doc,count) in enumerate(zip(alldocs,diggs)):
		for word in doc:
			finaldata.append([docid,val2idxs[word],count])
	
	return np.array(finaldata,dtype=np.int)
"""

#-----------------------------------------------------------------------------------------
#    ---------------------------------- MOVIELENS -------------------------------------
#-----------------------------------------------------------------------------------------	
def preprocessMovielens(path):
	
	#** Split data into training and testing set
	data=np.load(path+"/dictionateddata.npy")
	return splitTrainTest2(data,0.7)
	
	

#-----------------------------------------------------------------------------------------
#    ------------------------------------ UCI -------------------------------------
#-----------------------------------------------------------------------------------------	

#** Takes a doc,word,count table and transforms it into repeated doc,word,topic
#def extend

def preprocessUCI(path):
	
	#** Split data into training and testing set
	data=mycsv.getCol(path,[0,1,2,2],delimiter=" ")
	data=np.array(data,np.int)
	return data-1
	


	
if __name__=="__main__":

	#** CORA
	if 0:
		path="Datasets/cora/"
		data,dd=getData(path)
		#print data
		#print dd
		#print len(data)
	
		data=data[data[:,0].argsort()]
		train,test=splitTrainTest2(data,0.7)
		print train
		print test
	
	#** Diggs
	if 0:
		path="Datasets/diggs/"
		data=preprocessdiggs(path)
		print data[:100]
		print np.max(data[:][:,1])
	
	#** UCI
	if 0:
		path="Datasets/uci/docword.nytimes1m.txt"
		data=preprocessUCI(path)
		print data[:10]
	
	#** DBLP
	if 1:
		path="Datasets/dblp/"
		data=np.load(path+"data.npy")
		train,test=splitTrainTest(data,0.7)
	
		print train[:10]
		print test[:10]
		np.save("train",train)
		np.save("test",test)
	
	

		

	


