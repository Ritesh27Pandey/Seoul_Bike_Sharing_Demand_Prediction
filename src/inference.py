import pickle
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException


class Inference:
    def __init__(self, model_path, sc_path):
        self.model_path = model_path
        self.sc_path = sc_path
        logging.info('loading model and scaling object ')
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.sc_path):
                self.model_obj = pickle.load(open(self.model_path, "rb"))
                self.sc_obj = pickle.load(open(self.sc_path, "rb"))
        except Exception as e:
            logging.info('error occured at loading object')
            raise CustomException(e,sys)

    def get_string_to_datetime(self, date):
        try:
            logging.info('converting string to date ')
            dt = datetime.strptime(date, "%d/%m/%Y")
            return {"day": dt.day, "month": dt.month, "year": dt.year, "week_day": dt.strftime("%A")}

        except Exception as e:
            logging.info('exception occured at get_string_to_datetime')
            raise CustomException(e,sys)
        




    def season_to_df(self , seasons):
        logging.info('converting seasons to dataframe')
        try:
            seasons_cols = ['Spring', 'Summer', 'Winter']
            seasons_data = np.zeros((1,len(seasons_cols)), dtype = "int")

            df_seasons =  pd.DataFrame(seasons_data, columns = seasons_cols)
            if seasons in seasons_cols:
                df_seasons[seasons] = 1
            return df_seasons
        except Exception as e:
            logging.info('exception occured at get_string_to_datetime')
            raise CustomException(e,sys)
    def days_df(self, week_day):
        logging.info('converting day to df')
        try:
            days_names = ['Monday', 'Saturday', 'Sunday', 
                  'Thursday', 'Tuesday', 'Wednesday']
            days_name_data = np.zeros((1, len(days_names)), dtype = "int")
    
            df_days = pd.DataFrame(days_name_data, columns = days_names)
    
            if week_day in days_names:
                df_days[week_day] = 1
            return df_days
        except Exception as e:
            logging.info('exception occurs at converting day to df')
            raise CustomException (e,sys)
    def users_input(self):
        logging.info('converting user input to dataframe')
        try:
            print("Enter correct information to predict Rented Bike Count for a day with respect to time.")
            
            date = input("Date (dd/mm/yyyy): ")
            hour = int(input("Hours (0-23): "))
            temperature = float(input("Temperature in °C: "))
            humidity = float(input("Humidity: "))
            wind_speed = float(input("Wind Speed: "))
            visibility = float(input("Visibility: "))
            solar_Radiation = float(input("Solar Radiation: "))
            rainfall = float(input("Rainfall: "))
            snowfall = float(input("Snowfall :"))
            seasons = input("Season (Autumn, Spring, Summer, Winter): ")
            holiday = input("Holiday (Holiday/No Holiday): ")
            functioning_Day = input("Functioning Day (Yes/No): ")

            holiday_dic = {"No Holiday":0, "Holiday":1}
            functioning_day = {"No":0, "Yes":1}

            str_to_date = self.get_string_to_datetime(date)
            
            u_input_list = [hour, temperature, humidity, wind_speed, visibility,
                            solar_Radiation, rainfall, snowfall,
                            holiday_dic[holiday], functioning_day[functioning_Day],
                            str_to_date["day"], str_to_date["month"], str_to_date["year"]]

            features_name = ['Hour', 'Temperature(°C)', 'Humidity(%)',
                            'Wind speed (m/s)', 'Visibility (10m)', 'Solar Radiation (MJ/m2)',
                            'Rainfall(mm)', 'Snowfall (cm)', 'Holiday', 'Functioning Day', 'Day',
                            'Month', 'Year']
            
            df_u_input = pd.DataFrame([u_input_list], columns = features_name)
            df_seasons = self.season_to_df(seasons)
            df_days = self.days_df(str_to_date["week_day"])

            df_for_pred = pd.concat([df_u_input, df_seasons, df_days], axis = 1)

            return df_for_pred
        except Exception as e:
            logging.info('error occurs at user input')
            raise CustomException (e,sys)    
    

