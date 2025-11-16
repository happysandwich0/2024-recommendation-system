# config.py

import os

USE_FAISS = True 
# True: FAISS를 사용한 고속 Top K 검색 
# False: NumPy 기반의 Brute-force Top K 검색

# --- Data Paths ---
SONG_DATA_PATH = "" 
MODEL_PATH = "" 
VEC_PLAYID_PATH = ""
VEC_PLAY_PATH = ""
VEC_GENRECHART_PATH = "" 

# --- API Configuration ---
def get_openai_api_key():
    return userdata.get('OPENAI_API_KEY')
