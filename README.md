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

## Configs of Submitted Models
Submitted models still missed the zero padding. The last column on the right shows our final best model, trained after the competition ended:

| **Category**          | **Parameter**              | **Interaction Rupture**                   | **Robot Error**                   | **User Awkwardness**                   |
|-----------------------|----------------------------|-------------------------------|-------------------------------|-------------------------------|
| **Task**              | Task                       | 2                             | 1                             | 0                             |
| **Model**              | Model Type                 | MiniRocket                    | MiniRocket                    | MiniRocket                    |
| **Data Param.**   | Interval Length            | 1500                          | 1600                          | 1500                          |
| **Data Param.**   | Stride Train               | 400                           | 400                           | 400                           |
| **Data Param.**   | Stride Eval                | 225                           | 225                           | 225                           |
| **Data Param.**   | FPS                        | 25                            | 25                            | 25                            |
| **Data Param.**   | Columns to Remove          | vel_dist, c_openface          | openpose, c_openface          | vel_dist, c_openface          |
| **Data Param.**   | Label Creation             | stride_eval                   | stride_eval                   | stride_eval                   |
| **Data Param.**   | NaN Handling               | standard                      | avg                           | avg                           |
| **Data Param.**   | Oversampling Rate          | 0.15                          | 0.2                           | 0.1                           |
| **Data Param.**   | Undersampling Rate         | 0.05                          | 0.0                           | 0.05                          |
| **Data Param.**   | Rescaling                  | normalization                 | none                          | none                          |
| **Model Param.**  | Number of Estimators       | 25                            | 20                            | 20                            |
| **Model Param.**  | Max Dilations per Kernel   | 64                            | 32                            | 64                            |
| **Model Param.**  | Class Weight               | None                          | None                          | None                          |
| **Model Param.**  | Random State               | 42                            | 42                            | 42                            |
| **Performance** | Accuracy (Cross-Val.)                  | 0.82           | 0.89           | 0.84          |
| **Performance** | Macro F1  (Cross-Val.)                 | 0.74           | 0.77            | 0.55          |
| **Performance** | Accuracy (Test)                  | 0.80            | 0.87           | 0.76           |
| **Performance** | Macro F1  (Test)                 | 0.75           | 0.73          | 0.55           |


