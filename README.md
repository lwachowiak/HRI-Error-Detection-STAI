# HRI Error Detection: STAI Team Contribution

**TL;DR:** Careful feature selection and models that utilize convolutions are key to successful interaction rupture predictions!

**Abstract:** To be able to react to interaction ruptures such as errors, a robot needs a way of realizing such a rupture occurred. We test whether it is possible to detect interaction ruptures from the user's anonymized speech, posture, and facial features. We showcase how to approach this task, presenting a time series classification pipeline that works well with various machine learning models. A sliding window is applied to the data and the continuously updated predictions make it suitable for detecting ruptures in real-time.
Our best model, an ensemble of MiniRocket classifiers, is the winning approach to the ICMI ERR@HRI challenge. A feature importance analysis shows that the model heavily relies on speaker diarization data that indicates who spoke when. Posture data, on the other hand, impedes performance.

![alt text](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/readme_image.png)
<center>*Change in model accuracy compared to baseline for different feature combinations*</center>

## Reference

If you use our research in your work, please consider citing our paper:
Bibtex forthcoming!

## Environment Setup

We used Python 3.11.9. All required Python packages can be installed from the requirements.txt. 
The easiest way is to create a virtual environment like this: 
```
python3.11.9 -m venv venv_hri_err
source venv_hri_err/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, if you have [Apptainer](https://apptainer.org/) installed, we provide our container definition file [here](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/hri_cont.def).
To build the container, run the following command in your terminal:
```
sudo apptainer build <CONTAINER_NAME>.sif hri_cont.def
```
Once built, you can run the container, which will start a new terminal shell from which you can then run the scripts. To train the deep learning models, make sure to run the container with the ```--nv``` flag.
```
apptainer run --nv <CONTAINER_NAME>.sif
```
## Usage

As we are not allowed to re-publish the dataset, you need to request it from the competition organizers in case you want to reproduce our work. To facilitate reproduction, [this](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/data/tree.txt) text file indicates the necessary data folder structure to run the code as is.

We offer several scripts which perform different steps of our pipeline:
- [time_series_classifiers.py](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/time_series_classifiers.py): run Optuna search / train single model / evaluate on competition test set / create learing curve
- [data_loader.py](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/data_loader.py): preprocessing & loading of datasets
- [get_metrics.py](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/get_metrics.py): official competition metrics script
- [visualizations.py](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/visualizations.py): plotting script used to generate all plots contained in the main paper and the appendix
- [evaluate_best_model.py](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/evaluate_best_model.py): helper script to evaluate one of the saved best model configs

### Model Search

Model searches can be easily specified via json files. We provide many examples for the [genetic searches](https://github.com/lwachowiak/HRI-Error-Detection-STAI/tree/main/code/search_configs) and [grid searches](https://github.com/lwachowiak/HRI-Error-Detection-STAI/tree/main/code/grid_search_configs).

Both genetic and grid searches use the same config structure, but the grid search configs must contain ```grid``` in their name.

To run a (grid) search, use the following command:
```
python HRI-Error-Detection-STAI/code/time_series_classifiers.py --config grid_search_configs/config_minirocket_grid.json --njobs -1 --type search
```
### Train Best Model

To the best MiniRocket model we found (as specified in the table below), set the ```--type``` argument to be ```train_single```.
```
python HRI-Error-Detection-STAI/code/time_series_classifiers.py  --njobs -1 --type train_single
```

### Get Predictions on Test Data

To get the test predictions for the hidden competition test sets, use [time_series_classifiers.py](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/time_series_classifiers.py) and specify the ```type``` flag to be ```competition_eval```. This will generate predictions from MiniRocket models that were submitted to the competition:
```
python HRI-Error-Detection-STAI/code/time_series_classifiers.py --njobs -1 --type competition_eval
```

### Get Learning Curves

To run the learning curve experiment, use [time_series_classifiers.py](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/time_series_classifiers.py) and specify the ```type``` flag to be ```learning_curve```. This will produce a learning curve for each of the 4 models considered in the paper.
```
python HRI-Error-Detection-STAI/code/time_series_classifiers.py --njobs -1 --type learning_curve
```

### Evaluate Your Model

If you wish to evaluate one of the models you trained and saved in the ```best_model_configs/``` folder, use the [evaluate_best_model.py](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/evaluate_best_model.py) script. Specify the name of the config using the ```--file``` flag.
```
python HRI-Error-Detection-STAI/code/evaluate_best_model.py --file <YOUR_MODEL_CONFIG>.json
```

### Reproduce Plots

If you would like to reproduce all plots in the paper and the appendix, simply run the [visualization.py](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/visualization.py) script which will create the plots in approximate order of appearance and store PDFs into the ```plots/``` folder. All data used to generate plots is available in ```plots/run_histories/```.
```
python HRI-Error-Detection-STAI/code/visualizations.py
```

## Configs of Submitted and Best Models
Submitted models still missed the zero padding. The last column on the right shows our final best model, trained after the competition ended:

| **Category**          | **Parameter**              | **Interaction Rupture (submitted)**                   | **Robot Error (submitted)**                   | **User Awkwardness (submitted)**                   | **Interaction Rupture (best MiniRocket)**                            |
|-----------------------|----------------------------|-------------------------------|-------------------------------|-------------------------------|------------------------------------------|
| **Task**              | Task                       | 2                             | 1                             | 0                             | 2                                        |
| **Model**              | Model Type                 | MiniRocket                    | MiniRocket                    | MiniRocket                    | MiniRocket                               |
| **Data Param.**   | Interval Length            | 1500                          | 1600                          | 1500                          | 2500                                     |
| **Data Param.**   | Stride Train               | 400                           | 400                           | 400                           | 600                                      |
| **Data Param.**   | Stride Eval                | 225                           | 225                           | 225                           | 300                                      |
| **Data Param.**   | FPS                        | 25                            | 25                            | 25                            | 25                                       |
| **Data Param.**   | Columns to Remove          | vel_dist, c_openface          | openpose, c_openface          | vel_dist, c_openface          | openpose, c_openface                     |
| **Data Param.**   | Label Creation             | stride_eval                   | stride_eval                   | stride_eval                   | stride_eval                              |
| **Data Param.**   | NaN Handling               | avg                      | avg                           | avg                           | avg                                 |
| **Data Param.**   | Oversampling Rate          | 0.15                          | 0.2                           | 0.1                           | 0.1                                     |
| **Data Param.**   | Undersampling Rate         | 0.05                          | 0.0                           | 0.05                          | 0.1                                      |
| **Data Param.**   | Rescaling                  | normalization                 | none                          | none                          | none                            |
| **Data Param.**   | Zero Padding                  | False                 | False                          | False                          | True                                     |
| **Model Param.**  | Number of Estimators       | 25                            | 20                            | 20                            | 10                                        |
| **Model Param.**  | Max Dilations per Kernel   | 64                            | 32                            | 64                            | 32                                       |
| **Model Param.**  | Class Weight               | None                          | None                          | None                          | None                                     |
| **Model Param.**  | Random State               | 42                            | 42                            | 42                            | 42                                       |
| **Performance** | Accuracy (Cross-Val.)                  | 0.82           | 0.89           | 0.84          | 0.84                  |
| **Performance** | Macro F1  (Cross-Val.)                 | 0.74           | 0.77            | 0.55          | 0.76                      |
| **Performance** | Accuracy (Test)                  | 0.80            | 0.87           | 0.76           | N/A                                      |
| **Performance** | Macro F1  (Test)                 | 0.75           | 0.73          | 0.55           | N/A                                      |

Best configs of other models:
- [ConvTran](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/best_model_configs/ConvTranPlus_2024-07-13-14.json)
- [TST](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/best_model_configs/TST_2024-07-16-10.json)
- [RandomForest](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/best_model_configs/RandomForest_2024-06-15-11.json)
- [MiniRocket](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/best_model_configs/MiniRocket_2024-07-18-06.json)

