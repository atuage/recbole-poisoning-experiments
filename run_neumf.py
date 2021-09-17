from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from recbole.utils.utils import set_color

import logging
from logging import getLogger

saved=True

n_layers =20
mlp_list=[]

for i in range(n_layers):
	mlp_list.append(128)

#List of all config params
#https://recbole.io/docs/v0.1.2/user_guide/config_settings.html
parameter_dict = {
	'mlp_hidden_size': mlp_list,
	'use_gpu':False,
	'epochs':30,
	'stopping_step':0
}

#config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)

config = Config(model="NeuMF", dataset="ml-100k", config_file_list=None, config_dict=parameter_dict)
init_seed(config['seed'], config['reproducibility'])

init_logger(config)
logger = getLogger()

dataset = create_dataset(config)

train_data, valid_data, test_data = data_preparation(config, dataset)

model = get_model(config['model'])(config, train_data).to(config['device'])

logger.info(model)

trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

best_valid_score, best_valid_result = trainer.fit(
	train_data, valid_data, saved=saved, show_progress=config['show_progress']
)

# model evaluation
test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
logger.info(set_color('test result', 'yellow') + f': {test_result}')



