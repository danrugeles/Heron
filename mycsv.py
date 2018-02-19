#!/usr/bin/env python
import csv

__author__ = "Dan Rugeles"
__copyright__ = "Copyright 2013, MySystem"
__credits__ = ["Dan Rugeles"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Dan Rugeles"
__email__ = "danrugeles@gmail.com"
__status__ = "Production"

#11/26/2014 Added Header option
#03/15/2015 Fixed Bug: line 27 "if header" instead of "if no header"
#04/25/2015 Added toCsv
#02/06/2015 toCsv improved

#--getCol------------------------------------------------------
"""Gets the data of a set of columns of a csv file

filename: String representing path to file
col: integer representing the column of the csv file"""
#-------------------------------------------------------------------
def getCol(filename,col,delimiter=",",header=False):
	result=[]
	with open(filename,"rb") as f:
		alldata=f.readlines()
		#alldata=csv.reader(csvfile,delimiter=delimiter)
		for idx,row in enumerate(alldata):
			row=row.rstrip('\n').split(delimiter)
			if header and idx==0:
				continue
			else:
				if type(col)==list:
					if len(col)>1:					
						
						result.append([row[x] for x in col])
					else:
						result.append(row[col[0]])
				elif type(col)==int:
					result.append(row[col])
				else:
					print "Warning: Incorrect type of argument in getCol() in mycsv.py"
					break
	return result
	

#--getCol------------------------------------------------------
"""Exports a matrix to a csv file

filename: String representing path to file
csv: if false export as tsv"""
#-------------------------------------------------------------------

def toCsv(filename,data,header=False,headertext="",mix=","):
	with open(filename,"w") as w:
		w.write(headertext)
		for idr,row in enumerate(data):
			w.write(str(idr)+mix+mix.join(row.astype('|S10'))+"\n")
		

"""----------------------------*
*                              *
*   |\  /|   /\    |  |\  |    * 
*   | \/ |  /__\   |  | \ |    *
*   |    | /    \  |  |  \|    *
*                              *
*----------------------------"""
if __name__=="__main__":
	print "First\n"
	print getCol("User/73.csv",[0,1])
	print "\nSecond\n"
	print getCol("User/73.csv",0.9)
	print "\nThird\n"
	print getCol("User/73.csv",[0])
	
