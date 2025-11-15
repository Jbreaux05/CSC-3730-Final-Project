@echo off
python ArtistRecognition/process_test_set.py -nthreads 5 -K 3 D:\MillionSongSubset trained_knn_unbalanced.h5 Data/songs_test_unbalanced.txt track_metadata.db
PAUSE