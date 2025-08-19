
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import shap
import warnings
import logging
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from utils import Utils

def main():
    ut = Utils(dataset_path="'/Users/ihteshamshah/Desktop/Postdoc/Dataset/NKR_IKNL_breast_syntheticdata.csv'")
    warnings.filterwarnings('ignore')
    data, X, Y, classes_names = ut.data_read_function( )
    Plot_Data={}
    Treatments=['chemo','target','hormonal','radio','surgery']
    for treatment_name in Treatments :
        Plot_Data= ut.main_function_guidlineComparison(data, X, Y, classes_names, Plot_Data, treatment_name)

    colors = ['skyblue', 'orange'] #plot colors
    Plot_the_data(Plot_Data, colors)







if __name__ == "__main__":
    # code to run when the file is executed directly
    main()