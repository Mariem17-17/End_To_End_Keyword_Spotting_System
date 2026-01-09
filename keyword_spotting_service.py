import tensorflow.keras as keras
import numpy as np
import librosa

MODEL_PATH = "model.h5"
NUM_SAMPLES_TO_CONSIDER=22050  # 1 sec

# to ensure we only have one instance when using the flask api service to make the predictions

class _Keyword_Spotting_Service:

    model = None
    _mappings = ["cat",
        "dog",
        "down",
        "happy",
        "left",
        "right",
        "stop",
        "up",
        "wow",
        "yes"]
    
    _instance = None

    def predict(self, file_path):
        # extract MFCCs
        MFCCs = self.preprocess(file_path)  # (num segments, num coefficients)

        # convert 2d mfccs arrays into 4d arrays (num samples that we want to predict,num segments, num coefficients, channels=1 )
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make predictions
        predictions = self.model.predict(MFCCs) # --> [ [10 different valuesof probablities] ] 
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword


    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER] #slice it and resize

        # extract mfccs
        MFCCs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T



def Keyword_Spotting_Service():
    # ensure that we only have one instance of kss
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    
    return _Keyword_Spotting_Service._instance




if __name__ =="__main__":
    kss = Keyword_Spotting_Service()

    keyword1 = kss.predict("test/down.wav")
    keyword2 = kss.predict("test/left.wav")

    print(f"Predicted words:\n - Down : {keyword1} \n - left : {keyword2}")