@echo off
python ArtistRecognition/process_train_set.py -onlytesta -nthreads 5 D:\MillionSongSubset Data/songs_train_unbalanced.txt track_metadata.db trained_knn_unbalanced.h5
PAUSE