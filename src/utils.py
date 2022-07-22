
import pandas as pd


class Helpers():
    @classmethod
    def set_value (self,data_value, outcome_value,mean_nodiab,mean_diab):
        if (outcome_value == 0 and data_value==0):
            return mean_nodiab
        elif (outcome_value ==1 and data_value ==0 ):
            return mean_diab
        else:
            return data_value
  

