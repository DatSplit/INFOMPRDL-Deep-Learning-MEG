# INFOMPRDL_Project

## File Structure
- data: Training data with the folders 'Cross' and 'Intra' inside
- checkpoints: Folder for the saved model states
- test.py: Script to test model
- train.py: Script to train the model w/ hyperparameters listed at the top
- model.py: Outline of the model along with its train and test functions
- utils.py: Utility functions mostly for loading and preprocessing data

## Preprocess the data
1. Download the data and add it to the 'data' folder
1. Install requirements `pip install -r requirements.txt` (Optionally do this in a python environment)
1. Run Preprocessing.py
1. After preprocessing the data can be found in the Preprocessed_scale folder.

## How to run connors model
1. Download the data and add it to the 'data' folder
1. Install requirements `pip install -r requirements.txt` (Optionally do this in a python environment)
1. Train the model `python model_connor.py --train_set [TRAIN_SET]`
    - TRAIN_SET can either be 'cross' or 'train'
1. Test model `python model_connor.py --test_set [TEST_SET]`
    - TEST_SET can either be 'intra', 'cross1', 'cross2' or 'cross3'
1. Hyperparameter tuning `python model_connor.py --hyperparameter_tuning [TRAIN_SET] [TEST_SET]`
