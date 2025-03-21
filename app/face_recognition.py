import numpy as np
import pandas as pd
import sklearn
import pickle
import cv2
from datetime import datetime
import os

# load all model
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
model_svm = pickle.load(open('./model/model_svm.pickle','rb'))
pca_models = pickle.load(open('./model/cpa_dict_face.pickle','rb'))
model_pca = pca_models['pca']
mean_face_arr = pca_models['mean_face']

def save_attendance(name):
    """Save attendance record to CSV file"""
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    
    attendance_file = 'attendance.csv'
    
    # Create file if it doesn't exist
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write('Name,Date,Time\n')
    
    # Append attendance record
    with open(attendance_file, 'a') as f:
        f.write(f'{name},{date},{time}\n')

def face_recognitionPipeline(frame, path=False):
    if path:
        img = cv2.imread(frame)
    else:
        img = frame
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.5, 3)
    predictions = []
    
    for x,y,w,h in faces:
        roi = gray[y:y+h, x:x+w]
        roi = roi/255.0
        
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_CUBIC)
            
        roi_reshape = roi_resize.reshape(1,10000)
        roi_mean = roi_reshape - mean_face_arr
        eigen_image = model_pca.transform(roi_mean)
        eig_img = model_pca.inverse_transform(eigen_image)
        results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()

        # Only mark attendance if confidence is above 70%
        if prob_score_max >= 0.7:
            text = f"{results[0]}: {prob_score_max * 100:.1f}%"
            color = (0, 255, 0)  # Green for success
            save_attendance(results[0])
        else:
            text = "Unknown"
            color = (0, 0, 255)  # Red for low confidence
            
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        output = {
            'roi': roi,
            'eig_img': eig_img,
            'prediction_name': results[0],
            'score': prob_score_max
        }
        predictions.append(output)
        
    return img, predictions