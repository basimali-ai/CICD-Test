import gradio as gr
import skops.io as sio
import numpy as np  # Often needed for sklearn pipelines
import pandas as pd # Often needed for sklearn pipelines

# --- Model Loading ---
# Use a try-except block for robust loading
try:
    # Ensure the path "./Model/drug_pipeline.skops" is correct relative to where you run the script.
    trusted_types = sio.get_untrusted_types(file="Model/drug_pipeline.skops")
    pipe = sio.load("Model/drug_pipeline.skops", trusted=trusted_types)
# Add specific exception for file not found
except FileNotFoundError:
    print("Error: Model file './Model/drug_pipeline.skops' not found.")
    print("Please ensure the file exists in the 'Model' directory relative to the script.")
    # Provide a dummy pipeline to allow the Gradio app to launch for UI testing
    class DummyPipeline:
        def predict(self, data):
            print(f"Warning: Using dummy pipeline. Prediction called with: {data}")
            # Return a plausible default prediction based on input structure
            return ["DrugY"] * len(data) # Return a list of predictions
        def predict_proba(self, data): # Add if your UI expects probabilities
             print(f"Warning: Using dummy pipeline. predict_proba called with: {data}")
             # Example: return dummy probabilities for 5 classes
             return np.array([[0.1, 0.1, 0.6, 0.1, 0.1]] * len(data))

    pipe = DummyPipeline()
    print("Warning: Using a dummy pipeline because the model file was not found.")
# Catch other potential loading errors
except Exception as e:
    print(f"An error occurred while loading the pipeline: {e}")
    # You might want to exit or use a dummy pipeline here as well
    raise  # Re-raise the exception if you want the script to stop

# --- Prediction Function ---
def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    """
    Predict drugs based on patient features.

    Args:
        age (int): Age of patient
        sex (str): Sex of patient ('M' or 'F')
        blood_pressure (str): Blood pressure level ('HIGH', 'LOW', 'NORMAL')
        cholesterol (str): Cholesterol level ('HIGH', 'NORMAL')
        na_to_k_ratio (float): Ratio of sodium to potassium in blood

    Returns:
        str: Formatted string with the predicted drug label (e.g., "Predicted Drug: DrugY")
           or an error message.
    """
    try:
        # Create the feature list in the exact order the pipeline expects.
        features = [age, sex, blood_pressure, cholesterol, na_to_k_ratio]

        # Scikit-learn's predict method typically expects a 2D array-like input
        # (e.g., a list of lists, a NumPy array, or a Pandas DataFrame).
        # Since we are predicting for a single instance, wrap 'features' in another list.
        input_data = [features]

        # Make the prediction
        predicted_drug_array = pipe.predict(input_data)

        # Check if prediction returned a result (it should return a list/array)
        if len(predicted_drug_array) > 0:
            predicted_drug = predicted_drug_array[0] # Get the first (and only) prediction
            label = f"Predicted Drug: {predicted_drug}"
            return label
        else:
            return "Error: Prediction returned an empty result."

    # Catch potential errors during the prediction phase
    except AttributeError as e:
         # Handle cases where 'pipe' might be None or lacks 'predict' (e.g., loading failed silently)
         print(f"Prediction Error: 'pipe' object doesn't have 'predict' method or is None. {e}")
         return "Error: Model pipeline not loaded correctly."
    except Exception as e:
        # Catch any other unexpected errors during prediction
        print(f"An unexpected error occurred during prediction: {e}")
        # Return a user-friendly error message
        return f"Prediction Error: {e}. Please check input values."


# --- Gradio Interface Components ---

# Define Inputs
# Use explicit minimum/maximum for sliders for clarity
inputs = [
    gr.Slider(minimum=15, maximum=74, step=1, label="Age", info="Enter patient's age (15-74)"),
    gr.Radio(["M", "F"], label="Sex", info="Select patient's sex"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure", info="Select blood pressure level"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol", info="Select cholesterol level"),
    gr.Slider(minimum=6.2, maximum=38.2, step=0.1, label="Na_to_K Ratio", info="Enter Sodium to Potassium ratio (6.2-38.2)"),
]

# Define Outputs
# Use gr.Label for displaying the single text prediction result.
# num_top_classes is for classification tasks where you return probabilities/confidences.
outputs = [
    gr.Label(label="Prediction Result") # Add a label for clarity
]

# Define Examples
# Ensure numeric values for sliders are floats if they have decimal steps.
examples = [
    [47, "F", "LOW", "HIGH", 14.2],
    [30, "M", "HIGH", "NORMAL", 15.4],
    [35, "F", "LOW", "NORMAL", 8.0], # Use float for slider value
    [50, "M", "HIGH", "HIGH", 34.0], # Use float for slider value
    [68, "M", "LOW", "HIGH", 11.0], # Use float for slider value
    [22, "F", "NORMAL", "NORMAL", 28.1]
]


# --- Interface Metadata ---
title = "Drug Classification Predictor"
description = "Enter patient details (Age, Sex, Blood Pressure, Cholesterol, Na/K Ratio) to predict the most suitable drug type using a pre-trained machine learning model."
article = """
<div style='text-align: center; margin-top: 20px;'>
<p>This application demonstrates the use of a scikit-learn pipeline, saved with Skops, to perform drug classification.</p>
<p>It can be used as part of a CI/CD workflow for machine learning models, automating training, evaluation, and deployment.</p>
</div>
""" # Used simple HTML for better formatting


# --- Create and Launch the Gradio Interface ---
# Use if __name__ == "__main__": block for standard Python practice
if __name__ == "__main__":
    iface = gr.Interface(
        fn=predict_drug,         # The prediction function
        inputs=inputs,           # List of input components
        outputs=outputs,         # List of output components
        examples=examples,       # List of example inputs
        title=title,             # Interface title
        description=description, # Interface description
        article=article,         # Explanatory text/article
        theme=gr.themes.Soft(),  # Apply a visual theme
        allow_flagging="never"   # Disable flagging if not needed
    )

    # Launch the interface
    iface.launch()