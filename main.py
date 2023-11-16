pip install surprise

import pandas as pd
from surprise import Dataset, Reader, KNNBasic, KNNBaseline
from surprise.model_selection import train_test_split

# Загрузка данных
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')\

#  Объединение данных
movie_ratings = pd.merge(ratings, movies, on='movieId')

# Создание объекта Surprise Dataset
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(movie_ratings[['userId', 'movieId', 'rating']], reader)

# Разделение данных на обучающий и тестовый наборы
trainset, testset = train_test_split(data, test_size=0.5, random_state=42)

# Обучение модели (в данном случае используется алгоритм KNNBasic)
sim_options = {'name': 'cosine', 'user_based': False}
model = KNNBaseline(sim_options=sim_options)
model.fit(trainset)

# Получение рекомендаций для конкретного фильма
movie_id_to_name = movies.set_index('movieId')['title'].to_dict()
movie_name = 'Man About Town (2006)'  # Замените на конкретный фильм, для которого хотите получить рекомендации
movie_id = movies[movies['title'] == movie_name]['movieId'].values[0]
movie_inner_id = model.trainset.to_inner_iid(movie_id)

# Получение похожих фильмов
similarity_row = model.get_neighbors(movie_inner_id, k=10)  # k - количество похожих фильмов

print(f"Рекомендации для фильма '{movie_name}':")
for inner_id in similarity_row:
    movie_id = model.trainset.to_raw_iid(inner_id)
    print(f"{movie_id_to_name[movie_id]}")
