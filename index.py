"""
Flask API for our Artist Recognition Game
Exposes Top 3 predictions given a song title

NOTE: I know Flask is EXTREMELY unsafe to use in a production setting, so if I were to be making this for a real job (or have more time), I would use Java SpringBoot or C#'s Minimal API
"""

from flask import Flask, request, jsonify
import sqlite3
import numpy as np
import tables
import os
from sklearn.neighbors import NearestNeighbors
import ArtistRecognition.hdf5_getters as GETTERS
import ArtistRecognition.process_train_set as TRAIN

app = Flask(__name__)

MODEL_PATH = 'trained_knn_unbalanced.h5'
TMDB_PATH = 'track_metadata.db'
MSD_DIR = 'D:\\MillionSongSubset'

nn_model = None
h5model = None
training_feats = None

def fullpath_from_trackid(trackid):
    """Pretty please serialize the trackid before going through here, it literally could expose the host's file system :skull:"""
    p = os.path.join(MSD_DIR, trackid[2])
    p = os.path.join(p, trackid[3])
    p = os.path.join(p, trackid[4])
    p = os.path.join(p, trackid + '.h5')
    return str(p)

def load_model():
    """Load the trained model and build k-d tree once at startup"""
    global nn_model, h5model, training_feats

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found")
    
    print("Loading model...")
    h5model = tables.open_file(MODEL_PATH, mode='r')

    # Build k-d tree with K=3 (for top 3 nearest predictions)
    nn_model = NearestNeighbors(n_neighbors=3, algorithm='kd_tree')
    training_feats = h5model.root.data.feats.read()

    # Handle 3D arrays if needed
    if training_feats.ndim == 3:
        dims = sorted(training_feats.shape, reverse=True)
        training_feats = training_feats.reshape(dims[0], dims[1])

    nn_model.fit(training_feats)
    print("Model loaded successfully!")

def get_artist_name_from_id(artist_id):
    """Look up artist name from artist_id in track_metadata.db"""
    try:
        conn = sqlite3.connect(TMDB_PATH)
        cursor = conn.cursor()

        # artist_id might be bytes (i really don't know, but it was with python 2), convert it to string if it is
        if isinstance(artist_id, bytes):
            artist_id = artist_id.decode('utf-8')

        cursor.execute("SELECT artist_name FROM songs WHERE artist_id = ? LIMIT 1", (artist_id,))
        result = cursor.fetchone()
        conn.close()

        return result[0] if result else "Unknown Artist"
    except Exception as e:
        print(f"Error looking up artist name: {e}")
        return "Unknown Artist"

def get_top_predictions(track_id, k=3):
    """Get top-k artist predictions for track_id"""
    try:
        filepath = fullpath_from_trackid(track_id)

        if not os.path.isfile(filepath):
            return None, f"Song file not found: {filepath}"
        
        h5 = GETTERS.open_h5_file_read(filepath)
        processed_feats = TRAIN.compute_features(h5)
        actual_artist_id = GETTERS.get_artist_id(h5)
        h5.close()

        if processed_feats is None:
            return None, "Could not extract features from song"
        
        # Make sure the shape is the correct one for prediction
        feats_to_predict = np.asarray(processed_feats)
        if feats_to_predict.ndim == 1:
            feats_to_predict = feats_to_predict.reshape(1, -1)

        distances, indices = nn_model.kneighbors(feats_to_predict)

        # Get artist IDs and their frequencies
        indices_flat = indices.flatten()
        artist_votes = {}

        for pos, idx in enumerate(indices_flat):
            artist_id = h5model.root.data.artist_id[idx]
            if artist_id not in artist_votes:
                artist_votes[artist_id] = { 
                    'count' : 1,
                    'best_rank' : pos,
                    'total_rank' : pos
                }
            else:
                artist_votes[artist_id]['count'] += 1
                artist_votes[artist_id]['total_rank'] += pos
            
        # Sort the artists by count (descending), then by rank (ascending)
        sorted_artists = sorted(
            artist_votes.items(),
            key = lambda x: (-x[1]['count'], x[1]['best_rank'])
        )

        top_artists = []
        for artist_id, stats in sorted_artists[:4]:
            artist_name = get_artist_name_from_id(artist_id)
            top_artists.append({
                'artist_id' : artist_id.decode('utf-8') if isinstance(artist_id, bytes) else artist_id,
                'artist_name' : artist_name,
                'confidence' : stats['count'] / k # Proportion of votes
            })

        actual_artist_name = get_artist_name_from_id(actual_artist_id)
        actual_artist_id_str = actual_artist_id.decode('utf-8') if isinstance(actual_artist_id, bytes) else actual_artist_id

        return {
            'predictions' : top_artists,
            'actual' : {
                'artist_id' : actual_artist_id_str,
                'artist_name' : actual_artist_name
            }
        }, None
    
    except Exception as e:
        return None, f"Error making predictions: {str(e)}"

@app.route('/random-song', methods=['GET'])
def random_song():
    """Gets a random song from the test set (Returns a lot of metadata)"""
    try:
        conn = sqlite3.connect(TMDB_PATH)
        cursor = conn.cursor()

        # Get a random song with all available metadata
        cursor.execute("""
            SELECT track_id, title, artist_name, year, duration
            FROM songs
            WHERE track_id IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 1
        """)
        result = cursor.fetchone()
        conn.close()

        if not result:
            return jsonify({'error' : 'No songs found'}), 404
        
        track_id, title, artist, year, duration = result

        # Get top 3 predictions for the song
        predictions_result, error = get_top_predictions(track_id, k=3)

        if error:
            return jsonify({'error' : error }), 500
        
        # Get the audio features from the HDF5 file
        filepath = fullpath_from_trackid(track_id)
        audio_features = None

        if os.path.isfile(filepath):
            try:
                h5 = GETTERS.open_h5_file_read(filepath)
                audio_features = {
                    'tempo': float(GETTERS.get_tempo(h5)) if GETTERS.get_tempo(h5) else None,
                    'loudness': float(GETTERS.get_loudness(h5)) if GETTERS.get_loudness(h5) else None,
                    'key': int(GETTERS.get_key(h5)) if GETTERS.get_key(h5) is not None else None,
                    'mode': int(GETTERS.get_mode(h5)) if GETTERS.get_mode(h5) is not None else None,
                    'time_signature': int(GETTERS.get_time_signature(h5)) if GETTERS.get_time_signature(h5) else None,
                    'energy': float(GETTERS.get_energy(h5)) if GETTERS.get_energy(h5) else None,
                    'danceability': float(GETTERS.get_danceability(h5)) if GETTERS.get_danceability(h5) else None,
                }
                h5.close()
            except Exception as e:
                print(f"Error extracing audio features: {e}")
        
        return jsonify({
            'track_id' : track_id,
            'song_title' : title,
            'year' : int(year) if year and year > 0 else None,
            'duration' : float(duration) if duration else None,
            'predictions' : predictions_result['predictions'],
            'actual' : predictions_result['actual'],
            'audio_features' : audio_features
        })
    
    except Exception as e:
        return jsonify({'error' : f'Error getting random song: {str(e)}'}), 500



@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': nn_model is not None
    })

if __name__ == '__main__':
    try:
        load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Starting server anyway, but predictions will fail :shrug:")

    app.run(debug=True, host='0.0.0.0', port=5000)