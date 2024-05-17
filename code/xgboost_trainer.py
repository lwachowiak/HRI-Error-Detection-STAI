from xgboost import XGBClassifier
from data_loader import DataLoader_HRI as DL
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import optuna
from optuna.integration import WeightsAndBiasesCallback
import json
from datetime import datetime
import wandb

class XGBTrainer:
    def __init__(self, task: int):
        self.data_loader = DL()
        self.verbose = True
        data_params = {'interval_length': 100, 
                       'stride_train': 100, 
                       'stride_eval': 100, 
                       'fps': 100, 
                       'label_creation': "full", 
                       'summary': 'mean'}
        self.X_val, self.Y_val, self.X_train, self.Y_train = self.data_loader.get_summary_format(**data_params)
        self.task = task
        self.Y_train = self.Y_train[:, self.task]

    def train(self, model_params: dict):
        self.model = XGBClassifier(**model_params)
        self.model.fit(self.X_train, self.Y_train)

    def objective(self, trial: optuna.Trial):
        model_params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'n_jobs': 6,
        }
        
        self.train(model_params)
       
        y_pred = []
        y_true = []
        for batch in range(len(self.X_val)):
            for row in range(len(self.X_val[batch])):
                y_pred.append(self.model.predict([self.X_val[batch][row]]))
                y_true.append(self.Y_val[batch][row, self.task])
        
        return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')
    
    def read_config(self, file_path):
        """Reads a JSON configuration file and returns the configuration as a dictionary."""
        try:
            with open(file_path, 'r') as file:
                config = json.load(file)
                # print config
                print("\nConfiguration loaded successfully.")
                for key, value in config.items():
                    print(f"{key}: {value}")
                
                return config
        except FileNotFoundError:
            print("\nError: The configuration file was not found.")
        except json.JSONDecodeError:
            print("\nError: The configuration file is not in proper JSON format.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def hyperparam_search(self, config_file, n_trials=100):

        config = self.read_config(config_file)
        
        wandb_kwargs = {"project": "HRI-Errors"}
        #wandbc = WeightsAndBiasesCallback(
            #metric_name=["accuracy", "macro f1"], 
            #wandb_kwargs=wandb_kwargs)

        date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        study = optuna.create_study(
            directions=["maximize", "maximize"], study_name=date + "_" + "RF")
        print(f"Sampler is {study.sampler.__class__.__name__}")

        study.optimize(self.objective, n_trials=n_trials)

        trial_with_highest_accuracy = max(
            study.best_trials, key=lambda t: t.values[0])
        
        if self.verbose:
            print("Best trial Accuracy:")
            print(
                "Values: ", trial_with_highest_accuracy.values, "Params: ", trial_with_highest_accuracy.params)
            for key, value in trial_with_highest_accuracy.params.items():
                print(f"    {key}: {value}")

        
        f = "best_{}".format
        #for param_name, param_value in trial_with_highest_accuracy.params.items():
            #wandb.run.summary[f(param_name)] = param_value

        #wandb.run.summary["best accuracy"] = trial_with_highest_accuracy.values[0]
        #wandb.run.summary["best macro f1"] = trial_with_highest_accuracy.values[1]

        #wandb.finish()

        return study

if __name__ == '__main__':
    trainer = XGBTrainer(0)
    trainer.hyperparam_search('config_rf.json', 100)