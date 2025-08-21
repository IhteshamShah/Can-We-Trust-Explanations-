import numpy as np
import shap
import warnings
from lime.lime_tabular import LimeTabularExplainer
import logging
import itertools
import yaml
import os

# Load config.yaml
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

explainer_config = config["explainer"]
shap_params = explainer_config["shap"]
lime_params = explainer_config["lime"]
vsi_params = explainer_config["vsi"]

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
            background_sample = shap.sample(x_train, shap_params["background_sample_size"])

            # Create SHAP explainer
            shap_explainer = shap.KernelExplainer(model, data=background_sample)

            # Compute SHAP values
            shap_values = shap_explainer.shap_values(
                data_row, 
                nsamples=shap_params["nsamples"], 
                l1_reg=shap_params["l1_reg"]
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
            top_features = positive_list[:shap_params["top_features"]]

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
                discretize_continuous=lime_params["discretize_continuous"],
                verbose=lime_params["verbose"]
            )

            # Explain instance
            LIME_explanation = lime_explainer.explain_instance(
                sample[0], 
                model, 
                num_features=lime_params["num_features"], 
                num_samples=lime_params["num_samples"], 
                top_labels=lime_params["top_labels"]
            )          

            # Extract positive contributions
            lime_values = LIME_explanation.local_exp[Predicted_class]
            column_names = x_train.columns
            positive_list = [(x, y) for x, y in lime_values if y > 0]

            x_indices = [x for x, _ in positive_list]
            x_names = [column_names[idx] for idx in x_indices]
            y_values = [y for _, y in positive_list]

            # Sort and take top N
            sorted_list = sorted(zip(x_names, y_values), key=lambda pair: pair[1], reverse=False)[:lime_params["top_features"]]
            
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
        """Calculate the Variance Score Index (VSI) from explanations."""
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
        Predicted_class = rf.predict(sample)[0]
        Lime_Explanations = []

        lime_explainer = LimeTabularExplainer(
            x_train.values, 
            feature_names=x_train.columns, 
            class_names=classes_names[y_test.name], 
            discretize_continuous=lime_params["discretize_continuous"], 
            verbose=lime_params["verbose"]
        )

        for _ in range(vsi_params["explanation_runs"]): #generate explanations and compare
            LIME_explanation = lime_explainer.explain_instance(
                sample[0], 
                rf.predict_proba, 
                num_features=lime_params["num_features"], 
                num_samples=lime_params["num_samples"], 
                top_labels=lime_params["top_labels"]
            )
            lime_values_of_Prid_cls = LIME_explanation.local_exp[Predicted_class]
            column_names = x_train.columns
            positive_list = [(x, y) for x, y in lime_values_of_Prid_cls if y > 0]
            x_values = [x for x, y in positive_list]
            x_names = [column_names[t] for t in x_values]
            y_values = [y for x, y in positive_list]
            sorted_list = sorted(zip(x_names, y_values), key=lambda pair: pair[1], reverse=True)
            x_names_sorted, y_values_sorted = zip(*sorted_list)
            top_features = x_names_sorted[:vsi_params["vsi_top_features"]]
            Lime_Explanations.append(top_features)

        Lime_vsi = self.calculate_vsi(Lime_Explanations)
        return Lime_vsi

    def shap_vsi_fucntion(self, rf, x_train, data_row, sample):
        Predicted_class = rf.predict(sample)[0]
        Shap_Explanations = []

        K = shap_params["background_sample_size"]
        background_sample = shap.sample(x_train, K)
        Shap_explainer = shap.KernelExplainer(model=rf.predict_proba, data=background_sample)

        for _ in range(vsi_params["explanation_runs"]): #generate explanations and compare
            shap_values_single_instance = Shap_explainer.shap_values(
                data_row, 
                nsamples=shap_params["nsamples"], 
                l1_reg=shap_params["l1_reg"]
            )
            shap_valuesss = np.abs(shap_values_single_instance).mean(axis=0).mean(axis=0)
            column_namesss = x_train.columns
            positive_shap_indices = np.where(shap_valuesss > 0)[0]
            positive_shap_values = shap_valuesss[positive_shap_indices]
            positive_shap_columns = np.array(column_namesss)[positive_shap_indices]
            positive_shap_list = list(zip(positive_shap_columns, positive_shap_values))
            positive_shap_list.sort(key=lambda x: x[1], reverse=True)
            top_shap = positive_shap_list[:vsi_params["vsi_top_features"]]
            top_columns, top_values = zip(*top_shap)
            Shap_Explanations.append(top_columns)

        Shap_vsi = self.calculate_vsi(Shap_Explanations)
        return Shap_vsi
