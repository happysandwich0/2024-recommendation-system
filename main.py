import pandas as pd
import numpy as np
import pickle
from IPython.display import clear_output
from config import (
    get_openai_api_key, 
    SONG_DATA_PATH, 
    MODEL_PATH, 
    VEC_PLAYID_PATH, 
    VEC_PLAY_PATH, 
    VEC_GENRECHART_PATH
)
from utils import (
    load_data, 
    load_vector_models, 
    get_keywords_from_diary, 
    build_and_transform_vectors,
    calculate_playlist_similarity,
    filter_genre_chart,
    generate_song_vectors_and_filter,
    recommend_final_songs
)

clear_output()

playid_df, play_df, genrechart_df = load_data(SONG_DATA_PATH)
if playid_df is None:
    exit()

fasttext_model = load_vector_models(MODEL_PATH)
if fasttext_model is None:
    exit()

api_key = get_openai_api_key()
if not api_key:
    print("Error: OPENAI_API_KEY not found.")
    exit()

# --- 1. 일기 태깅 --- (예시)
diary_input = """
드디어 한 학기가 끝나고 종강이야! 시험도 과제도 전부 끝나서 완전 후련하고 뿌듯해.
어제는 여행도 다녀왔구, 편안하고 몽글몽글한 시간들이었어.
오늘은 크리스마스 이브라서, 연말이 왔다는 거에 설레고 두근두근해!!
"""

diary_tag = get_keywords_from_diary(api_key, diary_input)
print(f"--- 일기 태그 (감정 및 키워드) ---")
print(diary_tag)
print("---------------------------------")

# --- 2. 플레이리스트 벡터화 및 저장 ---

try:
    with open(VEC_PLAYID_PATH, 'rb') as f:
        vec_playid = pickle.load(f)
except FileNotFoundError:
    print("Warning: vec_playid.pkl not found. Building and saving new vector data...")
    playid_df['tag'] = playid_df['tag'].apply(lambda x: x.split(', ') if isinstance(x, str) else x)
    
    _, _ = build_and_transform_vectors(playid_df.copy(), MODEL_PATH, VEC_PLAYID_PATH)
    with open(VEC_PLAYID_PATH, 'rb') as f:
        vec_playid = pickle.load(f)

# --- 3. 유사 플레이리스트 검색 및 1차 후보 곡 선정 ---

playid_df['tag'] = playid_df['tag'].apply(lambda x: x.split(', ') if isinstance(x, str) else x)

similar_playlist_names = calculate_playlist_similarity(
    playid_df.copy(), 
    diary_tag, 
    fasttext_model, 
    vec_playid, 
    pca_components=30, 
    top_n=5
)
print(f"--- 유사 플레이리스트 Top 5 ---")
print(similar_playlist_names)
print("---------------------------------")

filtered_play_df = play_df[play_df['플리'].isin(similar_playlist_names)].drop_duplicates(subset=['제목', '가수']).copy()

# --- 4. 취향 조건 필터링 및 2차 후보 곡 선정 (장르/차트) ---
user_genre = input("나의 취향 장르는? (예: 힙합, 발라드 등): ")
user_ox = input("나는 홍대병이 있다? (O: 상위 75%, X: 하위 75% 이하): ")
user_release = input("나는 옛날 노래가 좋다? (O, X): ")

filtered_genrechart_df = filter_genre_chart(genrechart_df.copy(), user_genre, user_ox, user_release)

# --- 5. 최종 후보 곡 벡터화 (Mean 적용) ---

try:
    with open(VEC_PLAY_PATH, 'rb') as f:
        vec_play = pickle.load(f)
    with open(VEC_GENRECHART_PATH, 'rb') as f:
        vec_genrechart = pickle.load(f)
except FileNotFoundError:
    print("Vector files not found. Please run the preprocessing steps to create vec_play.pkl and vec_genrechart.pkl.")
    vec_play = vec_playid
    vec_genrechart = vec_playid


filtered_play_df = generate_song_vectors_and_filter(filtered_play_df, fasttext_model, vec_play)

if not filtered_genrechart_df.empty:
    filtered_genrechart_df = generate_song_vectors_and_filter(filtered_genrechart_df, fasttext_model, vec_genrechart)

# --- 6. 최종 후보군 통합 및 추천 ---

final_df = pd.concat([
    filtered_genrechart_df[['장르', '제목', '가수', '가사', 'word_vecs']].copy(), 
    filtered_play_df[['장르', '제목', '가수', '가사', 'word_vecs']].copy()
], axis=0, ignore_index=True)

final_df = final_df.drop_duplicates(subset = ['제목', '가수'], keep='first').reset_index(drop=True)

if final_df.empty:
    print("조건에 맞는 최종 후보 곡이 없습니다.")
else:
    top_10_recommendations = recommend_final_songs(final_df, diary_tag, fasttext_model, top_n=10)
    
    print(f"\n--- 🌟 일기 기반 최종 추천 Top 10 곡 🌟 ---")
    print(top_10_recommendations)