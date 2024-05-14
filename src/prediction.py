from src.inference import Inference
from src.logger import logging
from src.exception import CustomException
import sys

class Prediction:

    def prediction(self, df):
        logging.info('prediction has started')
        try:
                
            ml_model_path = r"E:\Seoul Bike Sharing Demand Prediction\models\xgboost_regressor_r2_0_949_v1.pkl"
            standard_scaler_path = r"E:\Seoul Bike Sharing Demand Prediction\models\sc.pkl"
            infrence_obj=Inference(ml_model_path,standard_scaler_path)
            scaled_data = infrence_obj.sc_obj.transform(df)
            prediction_ = infrence_obj.model_obj.predict(scaled_data)
            logging.info(f'Number of bikes required {int(prediction_)}')
            return prediction_
            
        except Exception as e:
            logging.info('error occured at prediction stage')
            raise CustomException(e,sys)
        