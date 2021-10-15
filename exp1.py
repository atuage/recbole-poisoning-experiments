from functions import run_neumf


#TODO: ファイルの書き出しなどをモジュール化
#多分data_path+datasetの場所を調べる挙動になってる
parameter_dict = {
	'use_gpu':True,
	'epochs':30,
	'stopping_step':30,
	'seed':2020,
	'data_path':"",
	'dataset':'modded-ml-100k'
}
results =[]

master_inter_data =[]
with open("dataset/ml-100k/ml-100k.inter",mode = 'r') as f:
	master_inter_data = f.readlines()
try:
	with open("modded-ml-100k/ml-100k.inter", mode='x') as f:
		f.writelines(master_inter_data)
		f.write("195	242	3	881250949\n")
except FileExistsError:
	with open("modded-ml-100k/ml-100k.inter", mode = 'a' ) as f:
		f.write("195	242	3	881250949\n")
	
	

for i in range(10):
	results.append( run_neumf.run_neumf(n_layers=i+1, parameter_dict=parameter_dict) )
df1 = pd.DataFrame(results)
df1.to_csv()

