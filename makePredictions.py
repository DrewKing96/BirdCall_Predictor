import os
import librosa
import math
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

#wget -O <path_to_save> <url>

prediction_categories = [
		"Marsh Wren",
                "American Robin",
                "American Yellow Warbler",
                "Bewick's Wren",
                "Black-Headed Grosbeak",
                "Carolina Wren",
                "Common Yellowthroat",
                "Curve-Billed Thrasher",
                "Dark-Eyed Junco",
                "Eastern Towhee",
                "Green-Tailed Towhee",
                "Hermit Thrush",
                "House Wren",
                "Indigo Bunting",
		"Northern Cardinal",
                "Northern Mockingbird",
		"Orange-Crowned Warbler",
                "Red-Eyed Vireo",
                "Red-Winged Blackbird",
                "Song Sparrow",
		"Spotted Towhee",
                "Swainson's Thrush",
                "Warbling Vireo",
                "Western Meadowlark",
                "White-Crowned Sparrow",
                "White-Eyed Vireo",
                "Wilson's Warbler",
                "Wood Thrush",
                "Yellow-Breasted Chat"
	]

def grabAudioFile(path_to_save, url, filename, extension):
	try:
		print("scp" + " " + url + filename + extension + " " + path_to_save + filename + extension)
		os.system("scp" + " " + url + filename + extension + " " + str(path_to_save))
	except OSError:
		print('download failed')


def predict(url,filename):
	url = url
	path_to_save = '/root/audioFiles/'
	AUDIOFILEPATH = '/root/audioFiles/'
	#MODELPATH =
	SAMPLE_RATE = 22050
	DURATION = 5 
	N_FFT = 2048
	N_MFCC = 32
	HOPLENGTH = 512
	#NUM_SEGMENTS = 10

	#SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

	#num_samples_per_segment = int(SAMPLES_PER_TRACK/NUM_SEGMENTS)
	#segment_number = 5
	#start_sample = num_samples_per_segment * segment_number
	#end_sample = start_sample + num_samples_per_segment
	file = filename.split(".")
	nameFile = file[0]
	extension = "." +  file[1]

	grabAudioFile(path_to_save, url, nameFile, extension)
	modelname = 'Diesel_Model_29.h5'
	#Load Saved Model
	model = keras.models.load_model(modelname)

	#model.summary()
	#Load audio file
	signal, sr = librosa.load(AUDIOFILEPATH+filename, sr = SAMPLE_RATE, duration = DURATION)

	#create MFCC from first 5 seconds of audio
	mfcc = librosa.feature.mfcc(signal, 
							sr = sr, 
							n_fft = N_FFT, 
							n_mfcc = N_MFCC, 
							hop_length = HOPLENGTH)
	mfcc = mfcc.T

	#create MFCC from loaded audiofile, specify audio placement
	'''mfcc = librosa.feature.mfcc(signal[start_sample:end_sample], 
							sr = sr, 
							n_fft = N_FFT, 
							n_mfcc = N_MFCC, 
							hop_length = HOPLENGTH)
	mfcc = mfcc.T
	'''
	#convert mfcc to numpy array for keras model
	audioData = np.array(mfcc)

	audioData = audioData[np.newaxis, ..., np.newaxis]

	prediction = model.predict_proba(audioData)
	#print(prediction)
	#print('max: ', max(prediction[0]))
	result =  np.where(prediction[0] == max(prediction[0]))
	#print('max index: ', result[0][0])
	return str(prediction_categories[result[0][0]])
