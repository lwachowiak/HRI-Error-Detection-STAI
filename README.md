# HRI Error Detection: STAI Team Contribution

## Reference
Forthcoming

## Usage
To get the test predictions, use [time_series_classifiers.py](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/time_series_classifiers.py) and specify the model you want to get predictions from. 

Moreover, the files can be used for: 
- time_series_classifiers.py: run Optuna search for models and save the best one; use models for inference
- data_loader.py: preprocessing & loading of datasets
- get_metrics.py: official competition metrics script

Model searches can be easily specified via json files. We provide many examples for the [genetic searches](https://github.com/lwachowiak/HRI-Error-Detection-STAI/tree/main/code/search_configs) and [grid searches](https://github.com/lwachowiak/HRI-Error-Detection-STAI/tree/main/code/grid_search_configs) we used ourselves.

## Dependencies
We used Python 3.11.9. All the Python packages we used can be installed from the requirements.txt. 
The easiest way is to create a virtual environment like this: 
```
python3.11.9 -m venv venv_hri_err
source venv_hri_err/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

As we are not allowed to re-publish the dataset, you need to request it from the competition organizers in case you want to reproduce our work. To facilitate reproduction, [this](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/data/tree.txt) text file indicates the necessary data folder structure to run the code as is. 

## Configs of Submitted and Best Models
Submitted models still missed the zero padding. The last column on the right shows our final best model, trained after the competition ended:

| **Category**          | **Parameter**              | **Interaction Rupture (submitted)**                   | **Robot Error (submitted)**                   | **User Awkwardness (submitted)**                   | **Interaction Rupture (best MiniRocket)**                            |
|-----------------------|----------------------------|-------------------------------|-------------------------------|-------------------------------|------------------------------------------|
| **Task**              | Task                       | 2                             | 1                             | 0                             | 2                                        |
| **Model**              | Model Type                 | MiniRocket                    | MiniRocket                    | MiniRocket                    | MiniRocket                               |
| **Data Param.**   | Interval Length            | 1500                          | 1600                          | 1500                          | 3900                                     |
| **Data Param.**   | Stride Train               | 400                           | 400                           | 400                           | 500                                      |
| **Data Param.**   | Stride Eval                | 225                           | 225                           | 225                           | 200                                      |
| **Data Param.**   | FPS                        | 25                            | 25                            | 25                            | 25                                       |
| **Data Param.**   | Columns to Remove          | vel_dist, c_openface          | openpose, c_openface          | vel_dist, c_openface          | openpose, c_openface                     |
| **Data Param.**   | Label Creation             | stride_eval                   | stride_eval                   | stride_eval                   | stride_eval                              |
| **Data Param.**   | NaN Handling               | standard                      | avg                           | avg                           | standard                                 |
| **Data Param.**   | Oversampling Rate          | 0.15                          | 0.2                           | 0.1                           | 0.05                                     |
| **Data Param.**   | Undersampling Rate         | 0.05                          | 0.0                           | 0.05                          | 0.0                                      |
| **Data Param.**   | Rescaling                  | normalization                 | none                          | none                          | normalization                            |
| **Data Param.**   | Zero Padding                  | False                 | False                          | False                          | True                                     |
| **Model Param.**  | Number of Estimators       | 25                            | 20                            | 20                            | 4                                        |
| **Model Param.**  | Max Dilations per Kernel   | 64                            | 32                            | 64                            | 32                                       |
| **Model Param.**  | Class Weight               | None                          | None                          | None                          | None                                     |
| **Model Param.**  | Random State               | 42                            | 42                            | 42                            | 42                                       |
| **Performance** | Accuracy (Cross-Val.)                  | 0.82           | 0.89           | 0.84          | 0.83                  |
| **Performance** | Macro F1  (Cross-Val.)                 | 0.74           | 0.77            | 0.55          | 0.75                      |
| **Performance** | Accuracy (Test)                  | 0.80            | 0.87           | 0.76           | N/A                                      |
| **Performance** | Macro F1  (Test)                 | 0.75           | 0.73          | 0.55           | N/A                                      |

Best configs of other models:
- [ConvTran](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/best_model_configs/ConvTranPlus_2024-07-13-14.json)
- [TST](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/best_model_configs/TST_2024-07-16-10.json)
- [RandomForest](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/best_model_configs/RandomForest_2024-06-15-11.json)
- [MiniRocket](https://github.com/lwachowiak/HRI-Error-Detection-STAI/blob/main/code/best_model_configs/MiniRocket_2024-07-18-06.json)

