from flask import Flask,render_template, Response, redirect, url_for, request
import os
from keras.models import load_model
import tensorflow as tf
from keras.preprocessing import image
import cv2
import numpy as np
import os

app = Flask(__name__)

path = 'model'
label = ""
face_classifier = cv2.CascadeClassifier(r'static/model/haarcascade_frontalface_default.xml')
classifier = load_model(r'static/model/model.h5')
emotion_labels =  { 0: "happy", 1: "sad", 2: "neutral",}

cap = cv2.VideoCapture(0)
def gen_frames():
    global label
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation = cv2.INTER_AREA)

            if np.sum([roi_gray]!=0):
                roi = roi_gray.astype('float')/255.0
                roi = tf.keras.preprocessing.image.img_to_array(roi)
                roi = np.expand_dims(roi, axis = 0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        #return render_template('index.html', label = label)        



@app.route('/')
def home():
    global cap
    cap = cv2.VideoCapture(0)
    return render_template('home.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),mimetype = 'multipart/x-mixed-replace; boundary=frame')

song_count = 0

@app.route('/songs', methods=['GET','POST'])
def play():
    cap.release()
    global song_count
    global label
    if request.method == 'POST':
        if "next" in request.form:
            song_count = song_count+1
        elif "previous" in request.form:
            song_count = song_count-1
        if "happy" in request.form:
            label = 'happy'
        elif "neutral" in request.form:
            label = 'neutral'
        elif "sad" in request.form:
            label = 'sad'
    files = []
    path = 'static/songs/'+label
    css_path = 'style/'+label+'.css'
    mysongs = os.listdir(path)
    for name in mysongs:
        files.append(name)
    if(song_count >= len(files) or song_count < 0):
        song_count = 0
    curr_song = files[song_count]
    return render_template('player.html', song = curr_song, dir = path, emotion = css_path, type = label)

"""@app.context_processor
def context_processor():
    return dict(key=label)"""

if __name__ == '__main__':
    app.run(debug = True)