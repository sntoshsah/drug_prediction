import gradio as gr
import skops.io as sio
import numpy as np
import pandas as pd
import pickle

pipe = sio.load("./model/drug_pipeline.skops", trusted=[np.dtype, pd.DataFrame])

print("Model loaded successfully")


def predict_drug(age, sex, blood_pressure, cholestrol, na_to_k_ratio):
    """
    Predict the Drug based on Patient Information


    Args:
    age(int): Age of Patient
    sex(str): Gender of Patient
    blood_pressure(str): Blood Pressure of Patient
    cholestrol(str): Cholestrol Level of Patient
    na_to_k_ratio(float): Sodium to Potassium Ratio of Patient

    Returns:
    str: Predicted Drug
    """

    features = [age, sex, blood_pressure, cholestrol, na_to_k_ratio]
    predicted_drug = pipe.predict([features])[0]

    label = f"Predicted Drug: {predicted_drug}"

    return label


inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Gender"),
    gr.Radio(["HIGH", "NORMAL", "LOW"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholestrol"),
    gr.Slider(6.0, 40.0, step=0.1, label="Na to K Ratio"),
]

outputs = [gr.Label(num_top_classes=5)]

examples = [
    [30, "M", "HIGH", "HIGH", 15.0],
    [35, "F", "NORMAL", "NORMAL", 10.0],
    [50, "M", "LOW", "NORMAL", 20.0],
]

title = "Drug Prediction System"
description = "Predict the Drug based on Patient Information"
article = "<p>Drug Prediction System</p>"
gr.Interface(
    fn=predict_drug,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
).launch(share=True)
