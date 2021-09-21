import logging
from logging import getLogger

import torch
import pickle

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color

def run_neumf(n_layers = 3, saved = True ):

	mlp_list=[]
	
	for i in range(n_layers):
		mlp_list.append(128)
		
	parameter_dict = {
		'mlp_hidden_size': mlp_list,
		'use_gpu':True,
		'epochs':30,
		'stopping_step':30,
		'seed':2020
	}
	config = Config(model="NeuMF", dataset="ml-100k", config_file_list=None, config_dict=parameter_dict)
	
	init_seed(config['seed'], config['reproducibility'])

	init_logger(config)
	logger = getLogger()

	logger.info(config)

	# dataset filtering
	dataset = create_dataset(config)
	if config['save_dataset']:
		dataset.save()
	logger.info(dataset)

	# dataset splitting
	train_data, valid_data, test_data = data_preparation(config, dataset)
	if config['save_dataloaders']:
		save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

	# model loading and initialization
	model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
	logger.info(model)

	# trainer loading and initialization
	trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

	# model training
	best_valid_score, best_valid_result = trainer.fit(
		train_data, valid_data, saved=saved, show_progress=config['show_progress']
	)

	# model evaluation
	test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

	logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
	logger.info(set_color('test result', 'yellow') + f': {test_result}')

	return {
		'best_valid_score': best_valid_score,
		'valid_score_bigger': config['valid_metric_bigger'],
		'best_valid_result': best_valid_result,
		'test_result': test_result
	}
	
	
if __name__ == "__main__":
	run_neumf(n_layers = 3)