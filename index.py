"""
Flash API for our Artist Recognition Game
Exposes Top 4 predictions given a song title

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

    # Build k-d tree with K=4 (for top 4 nearest predictions)
    nn_model = NearestNeighbors(n_neighbors=4, algorithm='kd_tree')
    training_feats = h5model.root.data.feats.read()

    # Handle 3D arrays if needed
    if training_feats.ndim == 3:
        dims = sorted(training_feats.shape, reverse=True)
        training_feats = training_feats.reshape(dims[0], dims[1])

    nn_model.fit(training_feats)
    print("Model loaded successfully!")

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