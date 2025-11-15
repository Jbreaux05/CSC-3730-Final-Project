"""
Thierry Bertin-Mahieux (2011) Columbia University
tb2332@columbia.edu
MODIFIED FOR PYTHON 3 with scikit-learn (2025)

Code to parse the whole testing set using a trained KNN
and predict an artist.

This is part of the Million Song Dataset project from
LabROSA (Columbia University) and The Echo Nest.

Copyright (c) 2011, Thierry Bertin-Mahieux, All Rights Reserved

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import sys
import time
import glob
import copy
import tables
import sqlite3
import datetime
import multiprocessing
import numpy as np
from operator import itemgetter
import hdf5_getters as GETTERS
import process_train_set as TRAIN # for the features
try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    print('You need scikit-learn. Install it with:')
    print('pip install scikit-learn')
    sys.exit(0)
    
# error passing problems, useful for multiprocessing
class KeyboardInterruptError(Exception):pass

def fullpath_from_trackid(maindir,trackid):
    """ Creates proper file paths for song files """
    p = os.path.join(maindir,trackid[2])
    p = os.path.join(p,trackid[3])
    p = os.path.join(p,trackid[4])
    p = os.path.join(p,trackid+'.h5')
    return str(p)

def get_all_files(basedir,ext='.h5'):
    """
    From a root directory, go through all subdirectories
    and find all files with the given extension.
    Return all absolute paths in a list.
    """
    allfiles = []
    apply_to_all_files(basedir,func=lambda x: allfiles.append(x),ext=ext)
    return allfiles


def apply_to_all_files(basedir,func=lambda x: x,ext='.h5'):
    """
    From a root directory, go through all subdirectories
    and find all files with the given extension.
    Apply the given function func
    If no function passed, does nothing and counts file
    Return number of files
    """
    cnt = 0
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files :
            func(f)
            cnt += 1
    return cnt


def compute_features(h5):
    """
    Get the same features than during training
    """
    return TRAIN.compute_features(h5)
    

def do_prediction(processed_feats, nn_model, h5model, K=1):
    """
    Receive processed features from test set, apply KNN,
    return an actual predicted artist ID.
    INPUT
       processed_feats - extracted from a test song
                nn_model - A *fitted* scikit-learn NearestNeighbors model
               h5model - open h5 file with data.artist_id
                     K - K-nn parameter
    """
    # --- FIX for 3D ValueError ---
    # The .kneighbors() method expects a 2D array of shape (n_samples, n_features).
    # 'processed_feats' is likely already a 2D array (e.g., shape (1, 90)).
    # Wrapping it in [processed_feats] makes it 3D (e.g., shape (1, 1, 90)), causing the crash.
    
    # We must ensure the array is 2D.
    feats_to_predict = np.asarray(processed_feats)
    if feats_to_predict.ndim == 1:
        # It was 1D (e.g., shape (90,)), reshape to (1, 90)
        feats_to_predict = feats_to_predict.reshape(1, -1)
    
    # If it was already 2D (e.g., shape (1, 90)), it remains (1, 90).
    # Now we pass the correctly-shaped 2D array.
    distances, indices = nn_model.kneighbors(feats_to_predict)
    # --- End Fix ---

    if K == 1:
        index = indices[0][0]
        pred_artist_id = h5model.root.data.artist_id[index]
    else:
        # find artist with most results
        # if tie, the one that was the highest ranking wins
        indices_flat = indices.flatten()
        artists = {}
        for pos, i in enumerate(indices_flat):
            artist_id = h5model.root.data.artist_id[i]
            if artist_id not in artists:
                artists[artist_id] = [1, -pos]
            else:
                artists[artist_id][0] += 1
        
        tuples = zip(artists.keys(), artists.values())
        res = sorted(tuples, key=itemgetter(1), reverse=True)
        pred_artist_id = res[0][0]
    # done
    return pred_artist_id
    

def process_filelist_test(filelist=None, model=None, tmpfilename=None, K=1, testsongs=None):
    """
    Main function, process all files in the list (as long as their track_id
    is not in testsongs)
    INPUT
       filelist     - a list of song files
       model        - h5 file containing feats and artist_id for all train songs
       tmpfilename  - where to save our processed features
       K            - K-nn parameter (default=1)
       testsongs    - set of test song track_ids (as strings)
    """
    # sanity check
    for arg in [filelist, model, tmpfilename, K, testsongs]:
        assert arg is not None,'process_filelist_test, missing an argument, something still None'
    if os.path.isfile(tmpfilename):
        print('ERROR: file',tmpfilename,'already exists.')
        return
    if not os.path.isfile(model):
        print('ERROR: model',model,'does not exist.')
        return
    # dimension fixed (12-dimensional timbre vector)
    ndim = 12
    finaldim = 90
    
    h5model = tables.open_file(model, mode='r')
    assert h5model.root.data.feats.shape[1]==finaldim,'inconsistency in final dim'
    
    print(f"Building k-d tree for K={K}...")
    nn_model = NearestNeighbors(n_neighbors=K, algorithm='kd_tree')
    
    training_feats = h5model.root.data.feats.read() 
    
    # Check dimensions and reshape if necessary
    if training_feats.ndim == 3:
        print(f"Note: Training data is 3D {training_feats.shape}. Reshaping to 2D.")
        dims = sorted(training_feats.shape, reverse=True)
        training_feats = training_feats.reshape(dims[0], dims[1])
        print(f"New shape is: {training_feats.shape}")
        
    nn_model.fit(training_feats)
    print("k-d tree is built.")
    
    # create outputfile
    output = tables.open_file(tmpfilename, mode='a')
    group = output.create_group("/",'data','TMP FILE FOR ARTIST RECOGNITION')
    # StringAtom(18) expects bytes, which is what GETTERS and the model provide.
    output.create_earray(group,'artist_id_real',tables.StringAtom(18,shape=()),(0,),'',
                        expectedrows=len(filelist))
    output.create_earray(group,'artist_id_pred',tables.StringAtom(18,shape=()),(0,),'',
                        expectedrows=len(filelist))
    # iterate over files
    cnt_f = 0
    for f in filelist:
        cnt_f += 1
        # verbose
        if cnt_f % 1000 == 0: # Lowered for more frequent feedback
            print(f'testing... checking file #{cnt_f} / {len(filelist)}')
            
        # check what file/song is this
        h5 = GETTERS.open_h5_file_read(f)
        artist_id = GETTERS.get_artist_id(h5)
        track_id = GETTERS.get_track_id(h5)
        
        # if track_id.decode('utf-8') in testsongs: # just in case, but should not be necessary
        #    print('Found test track_id during training? weird.',track_id)
        #    h5.close()
        #    continue
            
        # extract features, then close file
        processed_feats = compute_features(h5)
        h5.close()
        if processed_feats is None:
            continue
            
        # do prediction
        # artist_id_pred will be bytes (from h5model), which is correct
        artist_id_pred = do_prediction(processed_feats, nn_model, h5model, K)
        
        # save features to tmp file
        # artist_id is bytes (from GETTERS)
        # artist_id_pred is bytes (from h5model)
        # This matches the StringAtom(18) which expects bytes.
        output.root.data.artist_id_real.append( np.array( [artist_id] ) )
        output.root.data.artist_id_pred.append( np.array( [artist_id_pred] ) )
    
    # we're done, close output
    h5model.close()
    output.close()
    return

            
def process_filelist_test_wrapper(args):
    """ wrapper function for multiprocessor, calls process_filelist_test """
    try:
        process_filelist_test(**args)
    except KeyboardInterrupt:
        raise KeyboardInterruptError()


def process_filelist_test_main_pass(nthreads,model,testsongs_set,testsongs_list,K):
    """
    Do the main walk through the data, deals with the threads,
    creates the tmpfiles.
    INPUT
      - nthreads        - number of threads to use
      - model           - h5 files containing feats and artist_id for all train songs
      - testsongs_set   - set of test song track_ids (as str)
      - testsongs_list  - list of full paths to test song files
      - K               - K-nn parameter
    RETURN
      - tmpfiles     - list of tmpfiles that were created
                       or None if something went wrong
    """
    # sanity checks
    assert nthreads >= 0,'Come on, give me at least one thread!'
    # prepare params for each thread
    params_list = []
    default_params = {'model':model, 'K':K, 'testsongs': testsongs_set}
    tmpfiles_stub = 'mainpasstest_artistrec_tmp_output_'
    tmpfiles = list(map(lambda x: os.path.join(os.path.abspath('.'),tmpfiles_stub+str(x)+'.h5'),range(nthreads)))
    nfiles_per_thread = int(np.ceil(len(testsongs_list) / nthreads))
    for k in range(nthreads):
        # params for one specific thread
        p = copy.deepcopy(default_params)
        p['tmpfilename'] = tmpfiles[k]
        p['filelist'] = testsongs_list[k*nfiles_per_thread:(k+1)*nfiles_per_thread]
        params_list.append(p)
    # launch, run all the jobs
    pool = multiprocessing.Pool(processes=nthreads)
    try:
        pool.map(process_filelist_test_wrapper, params_list)
        pool.close()
        pool.join()
    except KeyboardInterruptError:
        print('MULTIPROCESSING')
        print('stopping multiprocessing due to a keyboard interrupt')
        pool.terminate()
        pool.join()
        return None
    except Exception as e:
        print('MULTIPROCESSING')
        print(f'got exception: {e!r}, terminating the pool')
        pool.terminate()
        pool.join()
        return None
    # all done!
    return tmpfiles


def test(nthreads,model,testsongs_set,testsongs_list,K):
    """
    Main function to do the training
    Do the main pass with the number of given threads.
    Then, reads the tmp files, creates the main output, delete the tmpfiles.
    INPUT
      - nthreads        - number of threads to use
      - model           - h5 files containing feats and artist_id for all train songs
      - testsongs_set   - set of test song track_ids (as str)
      - testsongs_list  - list of full paths to test song files
      - K               - K-nn parameter
    RETURN
       - nothing :)
    """
    # initial time
    t1 = time.time()
    # do main pass
    tmpfiles = process_filelist_test_main_pass(nthreads,model,testsongs_set,testsongs_list,K)
    if tmpfiles is None:
        print('Something went wrong, tmpfiles are None')
        return
    # intermediate time
    t2 = time.time()
    stimelen = str(datetime.timedelta(seconds=t2-t1))
    print('Main pass done after',stimelen, flush=True)
    # aggregate temp files
    artist_id_found = 0
    total_predictions = 0
    for tmpf in tmpfiles:
        if not os.path.isfile(tmpf):
            print(f"Warning: tmp file {tmpf} not found. Skipping.")
            continue
        h5 = tables.open_file(tmpf)
        for k in range( h5.root.data.artist_id_real.shape[0] ):
            total_predictions += 1
            # This comparison is bytes == bytes, which is correct.
            if h5.root.data.artist_id_real[k] == h5.root.data.artist_id_pred[k]:
                artist_id_found += 1
        h5.close()
        # delete tmp file
        os.remove(tmpf)
    # final time
    t3 = time.time()
    stimelen = str(datetime.timedelta(seconds=t3-t1))
    print('Whole testing done after',stimelen)
    # results
    print('We found the right artist_id',artist_id_found,'times out of',total_predictions,'predictions.')
    if total_predictions > 0:
        print('e.g., accuracy is:',artist_id_found / total_predictions)
    else:
        print('e.g., accuracy is: 0.0 (no predictions made)')
    # done
    return


def die_with_usage():
    """ HELP MENU """
    print('process_test_set.py')
    print('   by T. Bertin-Mahieux (2011) Columbia University')
    print('      tb2332@columbia.edu')
    print('Code to perform artist recognition on the Million Song Dataset.')
    print('This performs the evaluation of a trained KNN model.')
    print('MODIFIED to use scikit-learn (requires: pip install scikit-learn tables numpy)')
    print('USAGE:')
    print('  python process_test_set.py [FLAGS] <MSD_DIR> <model> <testsongs> <tmdb>')
    print('PARAMS:')
    print('        MSD_DIR  - main directory of the MSD dataset')
    print('          model  - h5 file where the training is saved')
    print('      testsongs  - file containing test songs (to ignore)')
    print('           tmdb  - path to track_metadata.db')
    print('FLAGS:')
    print('           -K n  - K-nn parameter (default=1)')
    print('    -nthreads n  - number of threads to use (default: 1)')
    sys.exit(0)


if __name__ == '__main__':

    # help menu
    if len(sys.argv) < 5:
        die_with_usage()

    # flags
    nthreads = 1
    K = 1
    while True:
        if sys.argv[1] == '-nthreads':
            nthreads = int(sys.argv[2])
            sys.argv.pop(1)
        elif sys.argv[1] == '-K':
            K = int(sys.argv[2])
            sys.argv.pop(1)
        else:
            break
        sys.argv.pop(1)

    # params
    msd_dir = sys.argv[1]
    model = sys.argv[2]
    testsongs = sys.argv[3]
    tmdb = sys.argv[4]

    # sanity check
    assert os.path.isdir(msd_dir),'ERROR: dir '+msd_dir+' does not exist.'
    assert os.path.isfile(testsongs),'ERROR: file '+testsongs+' does not exist.'
    assert os.path.isfile(model),'ERROR: file '+model+' does not exist.'
    assert os.path.isfile(tmdb),'ERROR: file '+tmdb+' does not exist.'

    # read test artists
    if not os.path.isfile(testsongs):
        print('ERROR:',testsongs,'does not exist.')
        sys.exit(0)
    testsongs_set = set()
    with open(testsongs,'r') as f:
        for line in f:
            if line == '' or line.strip() == '':
                continue
            testsongs_set.add( line.strip().split('<SEP>')[0] )
    
    testsongs_list = list(map(lambda x: fullpath_from_trackid(msd_dir,x), testsongs_set))
    
    # Filter out files that don't exist
    testsongs_list_exists = [f for f in testsongs_list if os.path.isfile(f)]
    if len(testsongs_list_exists) != len(testsongs_list):
        print(f"Warning: {len(testsongs_list) - len(testsongs_list_exists)} test song files not found in MSD_DIR. Using {len(testsongs_list_exists)} files.")
    
    if len(testsongs_list_exists) == 0:
        print("Error: No test song files were found. Check your MSD_DIR and testsongs file paths.")
        sys.exit(1)


    # settings
    print('msd dir:',msd_dir)
    print('testsongs:',testsongs,'('+str(len(testsongs_set))+' songs, '+str(len(testsongs_list_exists))+' found)')
    print('tmdb:',tmdb)
    print('nthreads:',nthreads)
    print('K:',K)

    # launch testing
    test(nthreads,model,testsongs_set,testsongs_list_exists,K)

    # done
    print('DONE!')