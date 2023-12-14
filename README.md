# INFOMPRDL_Project

## File Structure
- data: Training data with the folders 'Cross' and 'Intra' inside
- checkpoints: Folder for the saved model states
- test.py: Script to test model
- train.py: Script to train the model w/ hyperparameters listed at the top
- model.py: Outline of the model along with its train and test functions
- utils.py: Utility functions mostly for loading and preprocessing data

## How to run
1. Download the data and add it to the 'data' folder
1. Install requirements `pip install -r requirements.txt` (Optionally do this in a python environment)
1. Train the model `python train.py --dataset [DATASET]`
    - DATASET can either be 'cross' or 'train'
1. Test model `python test.py --testset [TESTSET]`
    - TESTSET can either be 'intra', 'cross1', 'cross2' or 'cross3'