# StyleTTS 2 HTTP Streaming API by @fakerybakery - Copyright (c) 2023 mrfakename. All rights reserved.
# Docs: API_DOCS.md
# To-Do:
# * Support voice cloning
# * Implement authentication, user "credits" system w/ SQLite3
import io
import os
import hashlib
import threading
import markdown
import re
import json
from tortoise.utils.text import split_and_recombine_text
from flask import Flask, Response, request, jsonify, send_file
from scipy.io.wavfile import write
import numpy as np
import ljinference
import msinference
import torch
import yaml
from flask_cors import CORS
from decimal import Decimal
voice_path = "voices/"

# Load GPU config from file
with open('gpu_config.yml', 'r') as file:
    gpu_config = yaml.safe_load(file)

# Extract GPU device ID from config
gpu_device_id = gpu_config.get('gpu_device_id', 0)

# Check if CUDA is available
if torch.cuda.is_available() and gpu_device_id != 999:
    # Set the device to the specified GPU
    torch.cuda.set_device(gpu_device_id)
    device = torch.device('cuda')
else:
    # If CUDA is not available or GPU ID is 999, use CPU
    device = torch.device('cpu')

#print(f"Selected device: {device}")

def genHeader(sampleRate, bitsPerSample, channels):
    datasize = 2000 * 10**6
    o = bytes("RIFF", "ascii")
    o += (datasize + 36).to_bytes(4, "little")
    o += bytes("WAVE", "ascii")
    o += bytes("fmt ", "ascii")
    o += (16).to_bytes(4, "little")
    o += (1).to_bytes(2, "little")
    o += (channels).to_bytes(2, "little")
    o += (sampleRate).to_bytes(4, "little")
    o += (sampleRate * channels * bitsPerSample // 8).to_bytes(4, "little")
    o += (channels * bitsPerSample // 8).to_bytes(2, "little")
    o += (bitsPerSample).to_bytes(2, "little")
    o += bytes("data", "ascii")
    o += (datasize).to_bytes(4, "little")
    return o
    
def find_wav_files(directory):
    wav_files = []
    
    # List all files in the directory
    files = os.listdir(directory)
    
    for file in files:
        # Check if the file has a .wav extension
        if file.lower().endswith(".wav"):
            # Remove the file extension and add to the wav_files list
            file_name_without_extension = os.path.splitext(file)[0]
            wav_files.append(file_name_without_extension)
            print (file_name_without_extension)
            wav_files.sort()
    
    return wav_files    

voicelist = find_wav_files(voice_path)#['f-us-1', 'f-us-2', 'f-us-3', 'f-us-4', 'm-us-1', 'm-us-2', 'm-us-3', 'm-us-4', 'untitled' , 'ironmouse', 'snuffy', 'silver', 'cotton', 'female_03', '1', '2', '3', '4']
voices = {}
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
print("Computing voices")
for v in voicelist:
    voices[v] = msinference.compute_style(f'voices/{v}.wav')
print("Starting Flask app")

app = Flask(__name__)
cors = CORS(app)

@app.route("/")
def index():
    with open('API_DOCS.md', 'r') as f:
        return markdown.markdown(f.read())

def synthesize(text, voice, steps):
    v = voice.lower()
    return msinference.inference(t, voices[v], alpha=0.3, beta=0.7, diffusion_steps=lngsteps, embedding_scale=1)
def ljsynthesize(text, steps):
    return ljinference.inference(text, torch.randn(1,1,256).to(device), diffusion_steps=7, embedding_scale=1)
# def ljsynthesize(text):
#     texts = split_and_recombine_text(text)
#     v = voice.lower()
#     audios = []
#     noise = torch.randn(1,1,256).to('cuda' if torch.cuda.is_available() else 'cpu')
#     for t in texts:
#         audios.append(ljinference.inference(text, noise, diffusion_steps=7, embedding_scale=1))
#     return np.concatenate(audios)

@app.get("/speakers")
def get_speakers():
    #speakers = XTTS.get_speakers_special()
    speakers_special = []
    for speaker in voicelist:
        preview_url = f"http://192.168.0.121:8001/sample/{speaker}.wav"

        speaker_special = {
                'name': speaker,
                'voice_id': speaker,
                'preview_url': preview_url
        }
        speakers_special.append(speaker_special)

    return speakers_special
    
@app.get('/sample/<filename>')
def get_sample(filename: str):
    file_path = os.path.join(voice_path, filename)
    if os.path.isfile(file_path):
        return send_file(file_path, mimetype='audio/wav', as_attachment=True)
    else:
        logger.error("File not found")
        raise HTTPException(status_code=404, detail="File not found")

@app.route("/api/v1/static2", methods=['POST'])
def serve_wav_stream():
    if 'text' not in request.form or 'voice' not in request.form:
        error_response = {'error': 'Missing required fields. Please include "text" and "voice" in your request.'}
        return jsonify(error_response), 400

    text = request.form['text'].strip()
    voice = request.form['voice'].strip().lower()

    if not voice in voices:
        error_response = {'error': 'Invalid voice selected'}
        return jsonify(error_response), 400

    v = voices[voice]
    texts = split_and_recombine_text(text)

    def generate():
        for t in texts:
            audio_chunk = msinference.inference(t, v, alpha=0.3, beta=0.7, diffusion_steps=25, embedding_scale=1)
            yield (b'--frame\r\n'
                   b'Content-Type: audio/wav\r\n\r\n' + audio_chunk.tobytes() + b'\r\n')

    return Response(generate(), content_type='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/v1/static", methods=['POST'])
def serve_wav():
    if 'text' not in request.form or 'voice' not in request.form:
        error_response = {'error': 'Missing required fields. Please include "text" and "voice" in your request.'}
        return jsonify(error_response), 400
    text = request.form['text'].strip()
    voice = request.form['voice'].strip().lower()
    # Extract additional variables with default values if not present
    alpha_form = request.form.get('alpha', '.3')
    alpha_float = float(alpha_form)
    beta_form = request.form.get('beta', '.1')
    beta_float = float(beta_form)
    diffusion_steps_form = request.form.get('diffusion_steps', '5')
    diffusion_steps_int = int(diffusion_steps_form)
    embedding_scale_form = request.form.get('embedding_scale', '1')
    embedding_scale_float = float(embedding_scale_form)

    if not voice in voices:
        error_response = {'error': 'Invalid voice selected'}
        return jsonify(error_response), 400
    v = voices[voice]
    texts = split_and_recombine_text(text, 25, 225)
    audios = []
    for t in texts:
        audios.append(msinference.inference(t, v, alpha_float, beta_float, diffusion_steps_int, embedding_scale_float))
    output_buffer = io.BytesIO()
    write(output_buffer, 24000, np.concatenate(audios))
    response = Response(output_buffer.getvalue())
    response.headers["Content-Type"] = "audio/wav"
    return response
if __name__ == "__main__":
    app.run("0.0.0.0", port=8001)
