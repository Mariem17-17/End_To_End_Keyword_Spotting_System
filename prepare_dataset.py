# Feature to extract :  MFCC (Mel-Frequency Cepstral Coefficients): tell us a lot about the timbra of an audio signal 
# we take a snapshot at diffretnt segments of our file
# and these snapshots are the mfccs coefficients
# mfcc ( number of time steps,number of coefficients)

import librosa
import os
import json

DATASET_PATH = "data"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050 # worth of 1 sec of sound given the default settings in librosa used for sounds( 22050 samples per second)
# n_mfcc: number of coefficients that we want to extract
# hop_length: number of frames, how big the segment should be in number of frames
# n_fft : how big the window for the fft 
def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    # data dictionary
    data = {
        "mappings":[],
        "labels":[],
        "MFCCs":[],
        "files":[]
    }

    # loop through all the sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # we need to ensure we rae not at root level
        if dirpath is not dataset_path:
            # update the mappings
            category = dirpath.split("/")[-1] # was : data/go -> become : [data, go]
            data["mappings"].append(category)
            print(f"Processing {category}")
            # loop through all the filenames and extract mfccs
            for f in filenames:
                # get the file path 
                file_path = os.path.join(dirpath, f)

                # load audio file
                signal, sr = librosa.load(file_path)

                # ensure the audio file is at least 1s
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    # enforce 1 seconde long signal
                    signal = signal[:SAMPLES_TO_CONSIDER]
                    # extract mfccs
                    MFCCs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
                    # store data
                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCCs.T.tolist()) # array to list
                    data["files"].append(file_path)
                    print(f"{file_path} {i-1}")

    # store data in teh json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH, n_mfcc=13, hop_length=512, n_fft=2048)