import os

for file_ in ["movielens"]:
	for alpha in [0.01,0.05,0.1,0.25,0.5]:
		for beta in [0.01,0.05,0.1,0.25,0.5]:
			command=map(str,["python evaluation.py -d",file_,"--alpha",alpha,"--beta",beta,'-k',25,'-i',200,'--path','Save/movielens/','--inference','heron'])
			command=" ".join(command)
			print "\n",command,"-----------------"
			os.system(command)

