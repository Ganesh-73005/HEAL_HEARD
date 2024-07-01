from flask import Flask, redirect, render_template, request, jsonify, Response
import os
import cv2
import numpy as np
import textwrap
from roboflow import Roboflow
from ultralytics import YOLO
import openai
import googlemaps
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import time
import anthropic
import json

rf = Roboflow(api_key="AXwpMdB3TsNEHit6rI2Z")
roboflow_project = rf.workspace("ganesh73005").project("cat-rvm7j")
roboflow_model = roboflow_project.version(1).model

yolo_model = YOLO('best.pt')

client = anthropic.Anthropic(
    api_key="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"
)

def predict(image,language):
    roboflow_output = predict_disease_with_roboflow(image,language)
    yolo_output = predict_disease_with_yolo(image,language)
    
    json_data = {
        'roboflow_output': roboflow_output,
        'yolo_output': yolo_output,
    }
    
    response = jsonify(json_data)
    return response

def fetch_ai_recommendations(disease_classes, target_language='en'):
    recommendations = []
    for disease_class in disease_classes:
        if disease_class == 'dermatophilus':
            prompt = f"What are the medicines name in English and recommended healing activities for cattle's dermatophilosis with supplements? Give medicine names in English and healing activities in {target_language}"
        elif disease_class == 'pediculosis':
            prompt = f"What are the medicines name in English and recommended healing activities for cattle's pediculosis with supplements? Give medicine names in English and healing activities in {target_language}"
        elif disease_class == 'ringworm':
            prompt = f"What are the medicines name in English and recommended healing activities for cattle's ringworm with supplements? Give medicine names in English and healing activities in {target_language}"

        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        recommendations_text = response.content[0].text.strip()
        recommendations.append(recommendations_text)
    
    return recommendations

def fetch_ai_2_recommendations(disease_class, target_language='en'):
    if disease_class == 'FMD':
        prompt = f"What are the medicines name in English and recommended healing activities for cattle's FMD Disease with supplements? Give medicine names in English and healing activities in {target_language}"
    elif disease_class == 'IBK':
        prompt = f"What are the medicines name in English and recommended healing activities for cattle's Eye infectious keratoconjunctivitis with supplements? Give medicine names in English and healing activities in {target_language}"
    elif disease_class == 'LSD':
        prompt = f"What are the medicines name in English and recommended healing activities for cattle's lumpy skin with supplements? Give medicine names in English and healing activities in {target_language}"
    elif disease_class == 'NOR':
        prompt = f"What are the recommended activities for healthy cattle maintenance with supplements? Give in {target_language}"

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    recommendations_text = response.content[0].text.strip()
    return recommendations_text

def predict_disease_with_roboflow(img_path, language='en'):
    response = roboflow_model.predict(img_path, confidence=40, overlap=30).json()
    predictions = response['predictions']
    selected_classes = [prediction['class'] for prediction in predictions if prediction['class'].lower() not in ['foot infected', 'mouth infected', 'lumpy skin', 'healthy cow', 'healthy_cow_mouth']]
    recommendations = fetch_ai_recommendations(selected_classes, language)
    output = ""
    for i, disease_class in enumerate(selected_classes):
        output += f"Disease {i + 1}: {disease_class.capitalize()}\n\nRecommended Activities: {recommendations[i]}\n\n"
    return output.strip()

def predict_disease_with_yolo(img_path, language='en'):
    results = yolo_model.predict(source=img_path)
    probs = results[0].probs
    class_names = yolo_model.names

    if probs is not None:
        max_prob_index = probs.top1
        max_class_name = class_names[max_prob_index]
        recommendations = fetch_ai_2_recommendations(max_class_name, language)
        output = f"Disease: {max_class_name.capitalize()}\n\nRecommended Activities: {recommendations}\n\n"
    else:
        output = "No predictions found."

    return output

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/s')
def s():
    return render_template('static/uploads')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/map')
def map_page():
    return render_template('map.html') 

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        language = request.form.get("language")
        filename = image.filename
        
        save_path = f'C:/Users/ganes/Documents/LiveStock/App/static/uploads/{filename}'
        image.save(save_path)
       
        pred = predict(save_path,language)  # Call the predict function here
        title = "HERE YOU GO"
        img_url = f'../static/uploads/{filename}'
        
        print(pred)

        prevent = pred.get_json()  # Extract the JSON data
        roboflow_output_html = prevent['roboflow_output'].replace('\n', '<br>')
        yolo_output_html = prevent['yolo_output'].replace('\n', '<br>')

        return render_template('submit.html', title=title, 
                               roboflow_output=roboflow_output_html, 
                               yolo_output=yolo_output_html,
                               img_url=img_url)

if __name__ == '__main__':
    app.run(debug=True)
