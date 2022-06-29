# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 22:16:58 2022

@author: 65889
"""
import nemo
import nemo.collections.asr as nemo_asr
import requests 
from pydub import AudioSegment
import speech_recognition as sr
r = sr.Recognizer()

from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/audio", methods = ["POST", "GET"])
def audio(): 
    url = request.json.get('url')
    audio = requests.get(url)
    with open("audio_file.wav", "wb") as file:
        file.write(audio.content)
    quartznet = nemo_asr.models.EncDecCTCModel.restore_from('QuartzNet15x5Base-En.nemo')
    transcript = quartznet.transcribe(paths2audio_files=["audio_file.wav"])
    t = {} 
    t['transcript'] = transcript[0]
    return jsonify(t)

@app.route("/text", methods = ["POST", "GET"])
def text(): 
    text = request.json.get('text')
    with open("text.txt", "a") as file:
        file.write("\n")
        file.write(text)
    return 'done'

if __name__ == "__main__": 
	app.run(debug=True, port=2000)