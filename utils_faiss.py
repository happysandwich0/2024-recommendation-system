import pandas as pd
import numpy as np
import pickle
import itertools
import ast
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import faiss 

def load_data(song_data_path):
    try:
        song_data = pd.read_pickle(song_data_path)
        playid = song_data['playid']
        play = song_data['play']
        genrechart = song_data['genrechart']
        return playid, play, genrechart
    except:
        print("Error: Data file load fail.")
        return None, None, None

def load_vector_models(model_path):
    try:
        fasttext = KeyedVectors.load_word2vec_format(
            model_path,
            binary=False,
            encoding='utf-8',
            unicode_errors='ignore'
        )
        return fasttext
    except:
        print("Error: FastText model load fail.")
        return None

def get_keywords_from_diary(api_key, input_text):
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
    except:
        return ['후련', '뿌듯', '설레', '학기', '시험', '연말']

def build_and_transform_vectors(df, model_path, vector_path, pca_components=30):
    fasttext = load_vector_models(model_path)
    if fasttext is None:
        return None, None

    df['tag'] = df['tag'].apply(lambda x: x.split(', ') if isinstance(x, str) else x)
    
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
    
    vector_data = {'all_words': all_words, 'target_words': target_words, 'word_vecs': word_vecs}
    with open(vector_path, 'wb') as f:
        pickle.dump(vector_data, f)
        
    pca = PCA(n_components=pca_components)
    comps = pca.fit_transform(word_vecs)
    df_comps = pd.DataFrame(comps, 
                            columns=[f'comp_{i+1}' for i in range(pca_components)], 
                            index=all_words)
    
    matrix = np.array([
        [(1 if tag in tags else 0) for tag in target_words]
        for tags in df['tag']
    ])
    matrix_df = pd.DataFrame(matrix, columns=target_words, index=df.index)

    df_targets = df_comps.loc[target_words]
    df_doc_trans_sum = matrix_df.dot(df_targets)
    
    tag_counts = matrix_df.sum(axis=1)
    df_doc_trans_mean = df_doc_trans_sum.div(tag_counts, axis=0)
    df_doc_trans_mean = df_doc_trans_mean.fillna(0)
    
    return df_doc_trans_mean, df_comps

def calculate_playlist_similarity(playid_df, diary_tag, fasttext, playid_vec_data, pca_components=30, top_n=5):
    all_words = playid_vec_data['all_words']
    target_words = playid_vec_data['target_words']
    word_vecs = playid_vec_data['word_vecs']
    
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
    new_df_doc_trans_mean = new_df_doc_trans_mean.fillna(0)
    
    new_vectors = new_df_doc_trans_mean.loc[col].values
    new_similarity_matrix = cosine_similarity(new_vectors)
    new_similarity_df = pd.DataFrame(new_similarity_matrix, index=col, columns=col)

    person_idx = len(new_similarity_df) - 1
    similarity_scores = new_similarity_matrix[person_idx]
    
    most_similar_indices = np.argsort(similarity_scores)[::-1][1:top_n + 1] 
    
    selected_names = [playid_df.iloc[idx]['name'] for idx in most_similar_indices]
    
    return selected_names

def filter_genre_chart(genrechart_df, genre, ox_input, release_input):
    genre_df = genrechart_df[genrechart_df['장르'].str.contains(genre, case=False, na=False)]

    if release_input == 'O':
        release_df = genrechart_df[genrechart_df['발매년도'] < 2010]
    elif release_input == 'X':
        release_df = genrechart_df[genrechart_df['발매년도'] >= 2010]
    else:
        return pd.DataFrame() 

    if release_df is not None:
        genre_df = genre_df[genre_df['발매년도'].isin(release_df['발매년도'])]

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
    all_new_word_vecs_list = []

    for _, row in df.iterrows():
        keyword_list = row['키워드']
        emotion_list = row['감정']
        all_words_for_row = keyword_list + emotion_list
        
        new_all_words = all_words_for_row.copy()
        new_missing_keys = []
        for word in all_words_for_row:
            if word in fasttext:
                sim_words = [x[0] for x in fasttext.most_similar(word, topn=20)]
                new_all_words += sim_words
            else:
                new_missing_keys.append(word)
        
        new_all_words = [i for i in new_all_words if i not in new_missing_keys]
        
        word_vectors = []
        for word in new_all_words:
            if word in vec_data['all_words']:
                word_idx = vec_data['all_words'].index(word)
                word_vectors.append(vec_data['word_vecs'][word_idx])
        
        if word_vectors:
            # FAISS를 위해 float32로 변환
            song_vector = np.mean(word_vectors, axis=0).astype('float32') 
        else:
            song_vector = np.zeros(300).astype('float32')
            
        all_new_word_vecs_list.append(song_vector)

    df['word_vecs'] = all_new_word_vecs_list
    df['tag'] = df['키워드'] + df['감정']
    return df

def recommend_final_songs(final_df, diary_tag, fasttext, top_n=10):
    
    candidate_vectors = np.stack(final_df['word_vecs'].values)
    D = candidate_vectors.shape[1]

    query_vectors = []
    for tag in diary_tag:
        if tag in fasttext:
            query_vectors.append(fasttext.get_vector(tag).astype('float32'))
    
    if not query_vectors:
        return pd.DataFrame()
    
    query_vectors = np.array(query_vectors)

    # 유클리드 거리 검색: IndexFlatL2
    index = faiss.IndexFlatL2(D)
    index.add(candidate_vectors)

    # 쿼리 수행
    D_faiss, I_faiss = index.search(query_vectors, top_n)
    
    all_results = []
    
    for tag_index in range(query_vectors.shape[0]):
        for rank in range(top_n):
            distance = D_faiss[tag_index, rank]
            candidate_index = I_faiss[tag_index, rank]
            all_results.append({'index': candidate_index, 'distance': distance})
            
    results_df = pd.DataFrame(all_results)
    
    # 각 후보 곡(index)에 대해 가장 작은 거리(min distance)만 남김
    top_results = results_df.loc[results_df.groupby('index')['distance'].idxmin()]
    
    # 거리가 가장 짧은 Top 10 곡을 선택
    top_10_similar = top_results.sort_values(by='distance', ascending=True).head(top_n)
    top_10_indices = top_10_similar['index']
    
    top_10_rows = final_df.iloc[top_10_indices]
    
    return top_10_rows[['장르', '제목', '가수', '가사']]