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



@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': nn_model is not None
    })

@app.route('/debug-song', methods=['POST'])
def debug_song():
    """
    Debug endpoint to analyze why predictions might be failing
    
    Request body: {"song_title": "Song Name"}
    """
    try:
        data = request.get_json()
        if not data or 'song_title' not in data:
            return jsonify({'error': 'Missing song_title in request body'}), 400
        
        song_title = data['song_title']
        song_info = find_track_by_title(song_title)
        
        if not song_info:
            return jsonify({'error': f'Song "{song_title}" not found'}), 404
        
        track_id, actual_title, actual_artist, artist_id = song_info
        
        # Check if this artist exists in training data
        conn = sqlite3.connect(TMDB_PATH)
        cursor = conn.cursor()
        
        # Count how many songs by this artist are in the dataset
        cursor.execute("SELECT COUNT(*) FROM songs WHERE artist_id = ?", (artist_id,))
        artist_song_count = cursor.fetchone()[0]
        
        # Check unique artists in training data
        if isinstance(artist_id, bytes):
            artist_id_str = artist_id.decode('utf-8')
        else:
            artist_id_str = artist_id
        
        # Check if artist is in training model
        all_artist_ids = h5model.root.data.artist_id.read()
        artist_ids_decoded = [aid.decode('utf-8') if isinstance(aid, bytes) else aid for aid in all_artist_ids]
        artist_in_training = artist_id_str in artist_ids_decoded
        training_examples_count = artist_ids_decoded.count(artist_id_str) if artist_in_training else 0
        
        cursor.execute("SELECT COUNT(DISTINCT artist_id) FROM songs")
        total_artists_in_db = cursor.fetchone()[0]
        
        conn.close()
        
        # Get predictions
        result, error = get_top_predictions(track_id, k=10)  # Get top 10 for debugging
        
        if error:
            return jsonify({'error': error}), 500
        
        # Find rank of correct artist
        correct_rank = None
        for i, pred in enumerate(result['predictions']):
            if pred['artist_id'] == artist_id_str:
                correct_rank = i + 1
                break
        
        return jsonify({
            'song_title': actual_title,
            'actual_artist': actual_artist,
            'actual_artist_id': artist_id_str,
            'diagnostics': {
                'artist_in_training_data': artist_in_training,
                'training_examples_for_this_artist': training_examples_count,
                'total_songs_by_artist_in_db': artist_song_count,
                'total_artists_in_database': total_artists_in_db,
                'total_training_examples': len(all_artist_ids),
                'correct_artist_rank': correct_rank if correct_rank else 'Not in top 10',
            },
            'top_10_predictions': result['predictions']
        })
        
    except Exception as e:
        return jsonify({'error': f'Debug error: {str(e)}'}), 500

if __name__ == '__main__':
    try:
        load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Starting server anyway, but predictions will fail :shrug:")

    app.run(debug=True, host='0.0.0.0', port=5000)