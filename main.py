from src.inference import Inference
from src.prediction import Prediction
from src.logger import logging
from src.exception import CustomException


if __name__ == "__main__":
    
    ml_model_path = r"E:\Seoul Bike Sharing Demand Prediction\models\xgboost_regressor_r2_0_949_v1.pkl"
    standard_scaler_path = r"E:\Seoul Bike Sharing Demand Prediction\models\sc.pkl"
    infrence_obj=Inference(ml_model_path,standard_scaler_path)
    dataframe=infrence_obj.users_input()
    pred_class_obj=Prediction()
    prediction_=pred_class_obj.prediction(dataframe)

    print(f"Rented Bike Demand prdiction for a day with respect to Time is : {round(prediction_.tolist()[0])}")