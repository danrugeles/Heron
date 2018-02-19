#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Auxiliary Decorators
# This code is available under the MIT License.
# (c)2017 Daniel Rugeles 
# daniel007[at]e.ntu.edu.sg / Nanyang Technological University
import time 

#--Varname-----------------------------------------------------------
""" Gets the str representation of variable 'var'.

var: the variable
"""
#-------------------------------------------------------------------
def varname(var):
	return [x for x in dir() if a is eval(x)][0]


#--timewatch-----------------------------------------------------------
""" Decorator counts how much time the function lasts.

func: function to be decorated
"""
#-------------------------------------------------------------------
def timewatch(func):

  def wrapper(*arg,**kwargs):
      t = time.clock()
      res = func(*arg,**kwargs)
      print func.func_name+":", "Time Spent:", time.clock()-t
      return res

  return wrapper


import numpy as np

#--CACHE-----------------------------------------------------------
""" Loads the results of the function cache'd on memory.

if the function is asked to load and it has never been saved,
then saveas will be used to save the function. if saveas is not 
provided in this case, the function will return an error.

f: function to be decorated
"""
#-------------------------------------------------------------------
def cache(f):
	
	def wrapper(*args,**kwargs):
		
		LOAD=kwargs.pop("LOAD",None)
		saveas=kwargs.pop("saveas",None)
			
		if LOAD is None or saveas is None:
			return f(*args,**kwargs)
			
			
		if not LOAD:
			res = f(*args,**kwargs)
		 	np.save(saveas,res)	# Used to prepend Save/
		else:
			try:
				res=np.load(saveas+".npy")# Used to prepend Save/
			except IOError:
				res = f(*args,**kwargs)
		 		np.save(saveas,res)	# Used to prepend Save/
			
		return res
	
	return wrapper

"""----------------------------*
*                              *
*   |\  /|   /\    |  |\  |    * 
*   | \/ |  /__\   |  | \ |    *
*   |    | /    \  |  |  \|    *
*                              *
*----------------------------"""
if __name__=="__main__":
	
	@timewatch
	def myFunction(n):
		for i in range(n):
			pass
		
	myFunction(100000)
	myFunction(1000000)
	
	@timewatch
	@cache
	def lol(x,joe=None):
		myFunction(100000000)
		return x,x
	
	print lol('2',LOAD=False,saveas="first_lol")

	print lol('3',LOAD=True,saveas="second_lol")

	print lol('5')
	

	
	
	
