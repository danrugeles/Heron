import os

for file_ in ["Datasets/movielens/dictionateddata.npy"]:
	for alpha in [0.01,0.05,0.1,0.25,0.5]:
		for beta in [0.01,0.05,0.1,0.25,0.5]:
			command=map(str,["python main.py -f",file_,"--alpha",alpha,"--beta",beta,'-k',25,'-i',200,'--path','Save/movielens/','--inference','cgs','--seed',10])
			command=" ".join(command)
			print command,"-----------------------"
			os.system(command)

