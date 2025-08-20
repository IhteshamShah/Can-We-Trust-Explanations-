
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
    ut = Utils(dataset_path="./data/NKR_IKNL_breast_syntheticdata.csv")
    warnings.filterwarnings('ignore')
    data, X, Y, classes_names = ut.data_read_function( )
    Treatments=['chemo','target','hormonal','radio','surgery']

    guidline_Plot_Data={}
    fidelity_Plot_Data={}

    for treatment_name in Treatments :
        guidline_Plot_Data= ut.main_function_guidlineComparison(X, Y, classes_names, guidline_Plot_Data)
        fidelity_Plot_Data= ut.Lime_Shap_fidelity(data, X, Y, classes_names, fidelity_Plot_Data)

    colors = ['skyblue', 'orange'] #plot colors for shap and Lime
    ut.Plot_the_data(guidline_Plot_Data, colors, filename = 'guidline_comparison_plot')







    
    

    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(Plot_Data)
    results_df.to_csv("GuidlineComparison_plotData.csv", index=False)







if __name__ == "__main__":
    # code to run when the file is executed directly
    main()