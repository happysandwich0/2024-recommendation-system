import pandas as pd
import numpy as np
import pickle
import itertools
import ast
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from konlpy.tag import Okt 

def load_data(song_data_path):
    try:
        song_data = pd.read_pickle(song_data_path)
        playid = song_data['playid']
        play = song_data['play']
        genrechart = song_data['genrechart']
        return playid, play, genrechart
    except FileNotFoundError:
        print(f"Error: Data file not found at {song_data_path}")
        return None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def load_vector_models(model_path):
    """FastText 모델을 로드합니다."""
    try:
        fasttext = KeyedVectors.load_word2vec_format(
            model_path,
            binary=False,
            encoding='utf-8',
            unicode_errors='ignore'
        )
        return fasttext
    except Exception as e:
        print(f"Error loading FastText model: {e}")
        return None

def get_keywords_from_diary(api_key, input_text):
    """GPT API를 사용하여 일기에서 감정 및 키워드를 추출합니다."""
    client = OpenAI(api_key=api_key)
    prompt = f"""
    You are a helpful assistant.
    Extract **three emotions** and **three keyword** in **Korean** from the following text : {input_text}.
    Make sure all the answers are **nouns**.
    Follow the given format :
    '{[emotion1, emotion2, emotion3, keyword1, keyword2, keyword3]}' """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        keywords = response.choices[0].message.content
        diary_tag = ast.literal_eval(keywords)
        return diary_tag
    except Exception as e:
        print(f"Error getting keywords from GPT: {e}")
        return 

def build_and_transform_vectors(df, model_path, vector_path, pca_components=30):
    """
    DataFrame의 태그를 벡터화하고 PCA를 적용합니다.
    """
    fasttext = load_vector_models(model_path)
    if fasttext is None:
        return None, None

    df['tag'] = df['tag'].apply(lambda x: x.split(', ') if isinstance(x, str) else x)
    
    # 태그 통합 및 유사 단어 추가 과정 (기존 코드 유지)
    all_tags = list(itertools.chain.from_iterable(df['tag']))
    all_tags = list(set(all_tags))

    target_words = all_tags.copy()
    all_words = all_tags.copy()
    missing_keys = []

    for word in target_words:
        if word in fasttext:
            sim_words = [x[0] for x in fasttext.most_similar(word, topn=100)]
            all_words += sim_words
        else:
            missing_keys.append(word)

    all_words = [i for i in all_words if i not in missing_keys]
    target_words = [i for i in target_words if i not in missing_keys]
    all_words = list(set(all_words))
    target_words = list(set(target_words))

    word_vecs = np.array([fasttext.get_vector(w) for w in all_words])
    
    # 1. 태그 벡터 저장 
    vector_data = {'all_words': all_words, 'target_words': target_words, 'word_vecs': word_vecs}
    with open(vector_path, 'wb') as f:
        pickle.dump(vector_data, f)
        
    # 2. PCA 적용
    pca = PCA(n_components=pca_components)
    comps = pca.fit_transform(word_vecs)
    df_comps = pd.DataFrame(comps, 
                            columns=[f'comp_{i+1}' for i in range(pca_components)], 
                            index=all_words)
    
    # 3. 플레이리스트 벡터화
    # 기존 매칭 행렬 (Tag Presence Matrix)
    matrix = np.array([
        [(1 if tag in tags else 0) for tag in target_words]
        for tags in df['tag']
    ])
    matrix_df = pd.DataFrame(matrix, columns=target_words, index=df.index)

    df_targets = df_comps.loc[target_words]
    
    # 행렬 곱 (Summation) 후 태그 개수로 나누어 평균을 구함 (Mean)
    df_doc_trans_sum = matrix_df.dot(df_targets)
    
    # 각 플레이리스트의 태그 개수
    tag_counts = matrix_df.sum(axis=1)
    
    # 태그 개수가 0이 아닌 경우에만 나눗셈 수행
    df_doc_trans_mean = df_doc_trans_sum.div(tag_counts, axis=0)
    df_doc_trans_mean = df_doc_trans_mean.fillna(0) # 태그 0개인 경우 0으로 채움
    
    return df_doc_trans_mean, df_comps

def calculate_playlist_similarity(playid_df, diary_tag, fasttext, playid_vec_data, pca_components=30, top_n=5):
    """일기 태그와 기존 플레이리스트 간의 코사인 유사도를 계산하고 Top N 플레이리스트를 반환합니다."""
    
    # 기존 벡터 데이터 로드
    all_words = playid_vec_data['all_words']
    target_words = playid_vec_data['target_words']
    word_vecs = playid_vec_data['word_vecs']
    
    # 1. 새 태그에 대한 벡터 공간 확장 및 PCA 
    new_word_list = diary_tag
    new_all_words = new_word_list.copy()
    new_missing_keys = []

    for word in new_word_list:
        if word in fasttext:
            sim_words = [x[0] for x in fasttext.most_similar(word, topn=20)]
            new_all_words += sim_words
        else:
            new_missing_keys.append(word)

    new_all_words = [i for i in new_all_words if i not in new_missing_keys]
    new_target_words = [i for i in new_word_list if i not in new_missing_keys]
    
    new_all_words = list(set(new_all_words))
    new_target_words = list(set(new_target_words))

    new_all_words_to_add = [i for i in new_all_words if i not in all_words]
    new_target_words_to_add = [i for i in new_target_words if i not in target_words]
    
    if new_all_words_to_add:
        new_word_vecs = np.array([fasttext.get_vector(w) for w in new_all_words_to_add])
        new_vecs_300 = np.concatenate((word_vecs, new_word_vecs), axis=0)
    else:
        new_vecs_300 = word_vecs.copy()
        
    new_all_words2 = list(set(all_words + new_all_words_to_add))
    new_target_words2 = list(set(target_words + new_target_words_to_add))
    
    new_pca = PCA(n_components=pca_components)
    new_comps = new_pca.fit_transform(new_vecs_300)
    new_df_comps = pd.DataFrame(new_comps, index=new_all_words2)

    # 2. 플레이리스트 벡터 재계산 및 일기 벡터 생성
    
    # 일기 태그 포함 매칭 행렬
    new_matrix = np.array([
        [(1 if tag in tags else 0) for tag in new_target_words2]
        for tags in playid_df['tag'].tolist() + [diary_tag]
    ])
    col = playid_df['name'].tolist() + ['new']
    new_matrix_df = pd.DataFrame(new_matrix, columns=new_target_words2, index=col)

    new_df_targets = new_df_comps.loc[new_target_words2]
    new_df_doc_trans_sum = new_matrix_df.dot(new_df_targets)

    tag_counts = new_matrix_df.sum(axis=1)
    new_df_doc_trans_mean = new_df_doc_trans_sum.div(tag_counts, axis=0)
    new_df_doc_trans_mean = new_df_doc_trans_mean.fillna(0) # 태그 0개인 경우 0으로 채움
    
    # 3. 코사인 유사도 계산
    new_vectors = new_df_doc_trans_mean.loc[col].values
    new_similarity_matrix = cosine_similarity(new_vectors)
    new_similarity_df = pd.DataFrame(new_similarity_matrix, index=col, columns=col)

    # 4. Top N 유사 플레이리스트 찾기
    person_idx = len(new_similarity_df) - 1 # 'new' 입력 태그의 인덱스
    similarity_scores = new_similarity_matrix[person_idx]
    
    # 자기 자신('new')을 제외하고 Top N 선택
    most_similar_indices = np.argsort(similarity_scores)[::-1][1:top_n + 1] 
    
    selected_names = [playid_df.iloc[idx]['name'] for idx in most_similar_indices]
    
    return selected_names

def filter_genre_chart(genrechart_df, genre, ox_input, release_input):
    """
    사용자 취향에 따라 장르/차트 곡을 1차 필터링합니다.
    1) 어떤 장르의 곡을 좋아하나요?
    2) 옛날 곡을 자주 듣나요?
    3) 유명한 곡을 자주 듣나요?
    """
    genre_df = genrechart_df[genrechart_df['장르'].str.contains(genre, case=False, na=False)]

    if release_input == 'O':
        release_df = genrechart_df[genrechart_df['발매년도'] < 2010]
    elif release_input == 'X':
        release_df = genrechart_df[genrechart_df['발매년도'] >= 2010]
    else:
        return pd.DataFrame() 

    if release_df is not None:
        genre_df = genre_df[genre_df['발매년도'].isin(release_df['발매년도'])]

    # 좋아요 수 필터링 
    if not genre_df.empty:
        upper_75 = genre_df['좋아요 수'].quantile(0.75)
        
        ox_input = ox_input.lower()

        if ox_input == 'o':
            genre_df = genre_df[genre_df['좋아요 수'] >= upper_75]
        elif ox_input == 'x':
            genre_df = genre_df[genre_df['좋아요 수'] < upper_75]
        else:
            return pd.DataFrame()

    return genre_df if not genre_df.empty else pd.DataFrame()

def generate_song_vectors_and_filter(df, fasttext, vec_data):
    """
    준비된 곡의 태그를 벡터화하고 word_vecs 열을 추가합니다.
    """
    
    all_new_word_vecs_list = []

    for _, row in df.iterrows():
        keyword_list = row['키워드']
        emotion_list = row['감정']
        all_words_for_row = keyword_list + emotion_list
        
        # 유사 단어 20개 추가 및 missing key 제거
        new_all_words = all_words_for_row.copy()
        new_missing_keys = []
        for word in all_words_for_row:
            if word in fasttext:
                sim_words = [x[0] for x in fasttext.most_similar(word, topn=20)]
                new_all_words += sim_words
            else:
                new_missing_keys.append(word)
        
        new_all_words = [i for i in new_all_words if i not in new_missing_keys]
        
        # 태그 벡터들을 모으고 평균을 계산 (Mean)
        word_vectors = []
        for word in new_all_words:
            if word in vec_data['all_words']:
                word_idx = vec_data['all_words'].index(word)
                word_vectors.append(vec_data['word_vecs'][word_idx])
        
        if word_vectors:
            # 모든 태그 벡터의 평균(Mean)을 곡의 대표 벡터로 사용 (300차원)
            song_vector = np.mean(word_vectors, axis=0)
        else:
            song_vector = np.zeros(300) # 벡터가 없는 경우 0으로 채움
            
        all_new_word_vecs_list.append(song_vector)

    df['word_vecs'] = all_new_word_vecs_list
    df['tag'] = df['키워드'] + df['감정']
    return df

def recommend_final_songs(final_df, diary_tag, fasttext, top_n=10):
    """
    최종 후보 곡들과 일기 태그 벡터 간의 유클리드 거리를 계산하여 Top 10을 추천합니다.
    """
    def compute_euclidean_distance(vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    # 일기 태그의 300차원 벡터 (Mean 적용하지 않음: 단일 태그)
    final_word_vecs = []
    for tag in diary_tag:
        if tag in fasttext:
            final_word_vecs.append(fasttext.get_vector(tag))
    
    if not final_word_vecs:
        return pd.DataFrame()

    final_word_vec = np.array(final_word_vecs)

    distances = []
    for idx, row in final_df.iterrows():
        song_vec = row['word_vecs']
        
        # 곡 벡터와 일기 태그 3개 각각의 거리 계산
        min_distance = float('inf')
        for i in range(final_word_vec.shape[0]):
            distance = compute_euclidean_distance(song_vec, final_word_vec[i])
            if distance < min_distance:
                min_distance = distance
                
        # 최소 거리가 가장 가까운 유사도로 채택됨
        distances.append((idx, min_distance))

    distances_sorted_df = pd.DataFrame(distances, columns=['index', 'distance'])

    top_10_similar = distances_sorted_df.sort_values(by='distance', ascending=True).head(top_n)
    top_10_indices = top_10_similar['index']
    
    top_10_rows = final_df.iloc[top_10_indices]
    
    return top_10_rows[['장르', '제목', '가수', '가사']]