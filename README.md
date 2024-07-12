# HRI Error Detection: STAI Team Contribution

## Reference
Forthcoming

## Usage
To get the test predictions, use time_series_classifiers.py and specify the model you want to get predictions from. 

Moreover the files can be used for: 
- time_series_classifiers.py: Optuna search for models and save the best one; use models for inference
- data_loader.py: preprocessing & loading of datasets
- get_metrics.py: official comeptition metrics script 

## Dependencies
We used Python 3.11.9 All the python packages we used can be installed from the requirements.txt. 
The easiest way is to create a virtual environment like this: 
```
python3.11.9 -m venv venv_hri_err
source venv_hri_err/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

