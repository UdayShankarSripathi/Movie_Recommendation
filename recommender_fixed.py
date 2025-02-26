import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load datasets
movies = pd.read_csv("/mnt/data/movies_1.csv")
ratings = pd.read_csv("/mnt/data/rating_1.csv")

# Pivot ratings into a user-item matrix
final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)

# Convert to a sparse matrix
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

# Train KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
knn.fit(csr_data)

# Function to get recommendations
def get_recommendation(movie_name):
    movies_to_recommend = 10
    movie_list = movies[movies['title'].str.contains(movie_name, case=False, regex=False)]

    if not movie_list.empty:
        movie_idx = movie_list.iloc[0]['movieId']
        
        # Check if the movieId exists in final_dataset
        if movie_idx not in final_dataset['movieId'].values:
            return "Movie not found in the rating dataset."
        
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        
        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=movies_to_recommend + 1)
        rec_movie_indices = sorted(
            list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), 
            key=lambda x: x[1]
        )[1:]  # Exclude the first item (itself)
        
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            
            if not idx.empty:
                recommend_frame.append({'Title': movies.iloc[idx[0]]['title'], 'Distance': val[1]})
        
        df = pd.DataFrame(recommend_frame, index=range(1, len(recommend_frame) + 1))
        return df
    else:
        return "No movies found. Please check your input."

# Example test case
print(get_recommendation("Toy Story"))
