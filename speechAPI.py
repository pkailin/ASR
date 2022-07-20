# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:01:00 2022

@author: coco
"""
import nemo
import nemo.collections.asr as nemo_asr
import base64
import numpy as np
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/data", methods = ["POST", "GET"])
def data(): 
    buffer = request.json.get('buffer')
    wav_file = open('audio.wav', 'wb')
    decode_string = base64.b64decode(buffer)
    wav_file.write(decode_string)
    
    quartznet = nemo_asr.models.EncDecCTCModel.restore_from('QuartzNet15x5Base-En.nemo')
    transcript = quartznet.transcribe(paths2audio_files=["audio.wav"])[0]
    print(transcript)
    return transcript

if __name__ == "__main__": 
	app.run(debug=True, port=2000)