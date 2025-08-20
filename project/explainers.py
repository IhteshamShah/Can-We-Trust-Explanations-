import numpy as np
import shap
import warnings
from lime.lime_tabular import LimeTabularExplainer
import logging
import itertools

# Configure logger
from logging_config import get_logger
logger = get_logger(__name__)

logging.getLogger("shap").setLevel(logging.WARNING)
class Explainers:
    def __init__(self):
        logger.info("Explainers initialized.")

    def shap_explain(self, model, x_train, y_test, data_row, Predicted_class):
        """
        Explain model predictions using SHAP KernelExplainer.
        """
        try:
            warnings.filterwarnings('ignore')

            # Background sample for SHAP
            background_sample = shap.sample(x_train, 200)

            # Create SHAP explainer
            shap_explainer = shap.KernelExplainer(model, data=background_sample)

            # Compute SHAP values
            shap_values = shap_explainer.shap_values(
                data_row, nsamples=2000, l1_reg="num_features(46)"
            )

            # Extract for predicted class
            shap_values_class = shap_values[Predicted_class][0]
            column_names = x_train.columns

            # Get positive SHAP values
            positive_indices = np.where(shap_values_class > 0)[0]
            positive_values = shap_values_class[positive_indices]
            positive_columns = np.array(column_names)[positive_indices]

            # Create sorted list
            positive_list = list(zip(positive_columns, positive_values))
            positive_list.sort(key=lambda x: x[1], reverse=False)
            top_features = positive_list[:25]

            return top_features

        except Exception as e:
            logger.error(f"Error in shap_explain: {e}")
            raise

    def lime_explain(self, model, x_train, y_test, sample, Predicted_class, classes_names):
        """
        Explain model predictions using LIME TabularExplainer.
        """
        try:
            warnings.filterwarnings('ignore')

            # Create LIME explainer
            lime_explainer = LimeTabularExplainer(
                x_train.values,
                feature_names=x_train.columns,
                class_names=classes_names[y_test.name],
                discretize_continuous=True,
                verbose=True
            )

            # Explain instance
            LIME_explanation = lime_explainer.explain_instance(sample[0], model, num_features=46, num_samples=2000, top_labels=20)          

            # Extract positive contributions
            lime_values = LIME_explanation.local_exp[Predicted_class]
            column_names = x_train.columns
            positive_list = [(x, y) for x, y in lime_values if y > 0]

            x_indices = [x for x, _ in positive_list]
            x_names = [column_names[idx] for idx in x_indices]
            y_values = [y for _, y in positive_list]

            # Sort and take top 25
            sorted_list = sorted(zip(x_names, y_values), key=lambda pair: pair[1], reverse=False)[:25]
            
            return sorted_list

        except Exception as e:
            logger.error(f"Error in lime_explain: {e}")
            raise


    def concordance(self, pair):
        """Function to calculate the concordance of a pair of explanations."""
        try:
            g1, g2 = pair
            return len(set(g1).intersection(set(g2)))
        except Exception as e:
            logger.error(f"Error in calculation concordance: {e}")
            raise

    def calculate_vsi(self, explanations):
        """Calculate the Variance Score Index (VSI) from LIME explanations."""
        try:
            m = len(explanations)
            n = 0
            p = len(explanations[0])  # Assuming all explanations have the same length

            # Generate all pairs of explanations
            pairs = list(itertools.combinations(explanations, 2))

            # Calculate the sum of concordances
            for pair in pairs:
                n += self.concordance(pair) / p

            # Calculate the total number of pairs
            total_pairs = len(pairs)

            # Calculate VSI
            vsi = n / total_pairs
            return vsi
        
        except Exception as e:
            logger.error(f"Error in calculate_vsi: {e}")
            raise


    def lime_vsi_fucntion(self, rf, x_train, y_test, sample, classes_names):

        
        Predicted_class= rf.predict(sample)[0]
        
            #Lime Explainer
        LIME_explainer = LimeTabularExplainer(x_train.values, 
                                    feature_names=x_train.columns, 
                                    class_names=classes_names[y_test.name], 
                                    discretize_continuous=True, verbose=True)


        Lime_Explanations= []


        for _ in range(3): #generate 3 time explaination of same instance and compare them

            ##############. Lime  #################
            LIME_explanation = LIME_explainer.explain_instance(sample[0], rf .predict_proba, num_features=46, num_samples=2000, top_labels=5)
            lime_values_of_Prid_cls= LIME_explanation.local_exp[Predicted_class]
            column_names=x_train.columns
            positive_list = [(x, y) for x, y in lime_values_of_Prid_cls if y > 0]
            # Separate the x and y values
            x_values = [x for x, y in positive_list]
            x_names = [column_names[t] for t in x_values]
            y_values = [y for x, y in positive_list]

            # Combine x_names and y_values into a list of tuples and sort by y_values
            sorted_list = sorted(zip(x_names, y_values), key=lambda pair: pair[1], reverse=True)

            # Unzip the sorted list
            x_names_sorted, y_values_sorted = zip(*sorted_list)
            top_five_features = x_names_sorted[:10] #selected top 10
            Lime_Explanations.append(top_five_features)

        Lime_vsi = self.calculate_vsi(Lime_Explanations)
        
        return Lime_vsi




    def shap_vsi_fucntion(self, rf, x_train, data_row, sample):
        #Instances =np.random.randint(100, size=(10)) #randomly pick 10 instances from data (in the range of 100 intances)

        Predicted_class= rf.predict(sample)[0]
        

        Shap_Explanations= []
        
                #Shap Explainer 

        # Define the number of samples to summarize the background data
        K = 100  # Choose an appropriate value for K

        # Summarize the background data using shap.sample()
        background_sample = shap.sample(x_train, K)

        # Use the summarized background sample in your SHAP model
        #Shap_explainer = shap.KernelExplainer(model=rf.predict_proba, data=background_sample, link = 'logit')
        Shap_explainer = shap.KernelExplainer(model=rf.predict_proba, data=background_sample)


        for _ in range(3): #generate 3 time explaination of same instance and compare them

            #############. Shap  #####################
            shap_values_single_instance = Shap_explainer.shap_values(data_row, nsamples=2000, l1_reg="num_features(46)")

            # Provided SHAP values and column names
            #shap_valuesss = shap_values_single_instance[Predicted_class][0]
            shap_valuesss = np.abs(shap_values_single_instance).mean(axis=0).mean(axis=0)
            column_namesss = x_train.columns
            # Flatten shap_values array
            #shap_values = shap_values.flatten()

            # Filter positive SHAP values
            positive_shap_indices = np.where(shap_valuesss > 0)[0]
            positive_shap_values = shap_valuesss[positive_shap_indices]
            positive_shap_columns = np.array(column_namesss)[positive_shap_indices]

            # Combine and sort by SHAP values
            positive_shap_list = list(zip(positive_shap_columns, positive_shap_values))
            positive_shap_list.sort(key=lambda x: x[1], reverse=True)

            # Select top 20
            top_5_shap = positive_shap_list[:10] #top 10
            # Unzip to get column names and values
            top_5_columns, top_5_values = zip(*top_5_shap)
            Shap_Explanations.append(top_5_columns)

        Shap_vsi= self.calculate_vsi(Shap_Explanations)
        
        return Shap_vsi
