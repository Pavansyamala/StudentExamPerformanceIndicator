import os 
import sys
from dataclasses import dataclass 

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor , GradientBoostingRegressor , RandomForestRegressor 

from sklearn.linear_model import LinearRegression ,Lasso , Ridge
from sklearn.metrics import r2_score 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

from src.exception import CustomException 
from src.logger import logging  

from src.utils import save_object
from src.utils import evaluvate_model


@dataclass 
class ModelTrainingConfig :
    trained_model_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer :
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try :
            logging.info("Split Training and Testing Data")
            x_train,y_train,x_test,y_test = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor()
                  }
            model_report:dict = evaluvate_model(x_train=x_train, y_train=y_train, x_test = x_test, y_test = y_test ,models=models) 

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name] 
            if best_model_score < 0.6 :
                raise CustomException("No Best Model Found")
            logging.info("Best found model on training and testing dataset")

            save_object(
                file_path= self.model_trainer_config.trained_model_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)
            r2_square =r2_score(y_test,predicted)

            return r2_square

        except Exception as e :
            raise CustomException(e,sys) 
        
# if __name__ == '__main__':
#     obj = ModelTrainer()