# config.py

import os
from google.colab import userdata

# --- Data Paths ---
SONG_DATA_PATH = "" 
MODEL_PATH = "" 
VEC_PLAYID_PATH = ""
VEC_PLAY_PATH = ""
VEC_GENRECHART_PATH = "" 

# --- API Configuration ---
def get_openai_api_key():
    return userdata.get('OPENAI_API_KEY')