from functions import run_neumf

results =[]
for i in range(10):
	results.append( run_neumf.run_neumf(n_layers=i+1) )
	