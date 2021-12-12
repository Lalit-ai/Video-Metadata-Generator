import numpy as np
import os
from pickle import load
import keras
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from sklearn.feature_extraction.text import TfidfVectorizer
import load_data as ld
import psutil
import generate_model as gen
from clustering import *
from time import sleep
#Configurable Parameters
# load the tokenizer
model_filename = 'models/model_weight.h5'
#global graph
#graph = tf.get_default_graph() 
#model = load_model(model_filename)
#print(model.summary())
#tokenizer = load(open('models/tokenizer.pkl', 'rb'))
#index_word = load(open('models/index_word.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34

# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	#convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature

# generate a description for an image
def generate_desc( photo, max_length, beam_size=1):
	captions = [['startseq', 0.0]]
	keras.backend.clear_session()
	model = load_model(model_filename)
	tokenizer = load(open('models/tokenizer.pkl', 'rb'))
	index_word = load(open('models/index_word.pkl', 'rb'))
	#seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		all_caps = []
		# expand each current candidate
		for cap in captions:
			sentence, score = cap
			# if final word is 'end' token, just add the current caption
			if sentence.split()[-1] == 'endseq':
				all_caps.append(cap)
				continue
			# integer encode input sequence
			sequence = tokenizer.texts_to_sequences([sentence])[0]
			# pad input
			sequence = pad_sequences([sequence], maxlen=max_length)
			# predict next words
			y_pred = model.predict([photo,sequence], verbose=0)[0]
			# convert probability to integer
			yhats = np.argsort(y_pred)[-beam_size:]
			for j in yhats:
				# map integer to word
				word = index_word.get(j)
				# stop if we cannot map the word
				if word is None:
					continue
				# Add word to caption, and generate log prob
				caption = [sentence + ' ' + word, score + np.log(y_pred[j])]
				all_caps.append(caption)
		# order all candidates by score
		ordered = sorted(all_caps, key=lambda tup:tup[1], reverse=True)
		captions = ordered[:beam_size]
	return captions

def generate_caption(index):
	list_idx = list(index.keys())[1:]
	csv_caption = []
	height, width = 480,640
	if(len(list_idx)==0):
		return csv_caption
	video_name = index[0][4].split("/")[-1]
	video_id = video_name.split(".")[0]
	global vidcap
	global image
	print("Internal Video Name : ", "/video_db/"+video_name)
	vidcap = cv2.VideoCapture(os.path.abspath(os.getcwd())+"/video_db/"+video_name)
	vectorizer = TfidfVectorizer()
	for caption_id in list_idx:
		caption = []
		count = 0
		cluster_list = index[caption_id]
		cap_vid_name = video_id+"_"+str(caption_id)+".mp4"
		caption.append(caption_id)
		caption.append(video_name)
		caption.append(video_id+"_"+str(caption_id)+".mp4")
		video_store_path = "caption_db/"+video_id
		if not os.path.exists(video_store_path):
			os.makedirs(video_store_path)
		video = cv2.VideoWriter("caption_db/"+video_id+"/"+cap_vid_name,0x7634706d,20, (width,height))
		try:
			cap_img_list = random.sample(range(cluster_list[0],cluster_list[1],100),2)
		except ValueError:
			cap_img_list = [cluster_list[0]]
		corpus = []
		for i in range(cluster_list[0],cluster_list[1],50):
			vidcap.set(cv2.CAP_PROP_POS_MSEC,i)
			success,image = vidcap.read()
			if success:
				image = cv2.resize(image,(640,480),interpolation=cv2.INTER_AREA)
				video.write(image)
			else:
				break
			if i in cap_img_list:
				cv2.imwrite("temp" + "/frame%d.jpg" % count, image)
				image_path = "temp" + "/frame%d.jpg"%count
				photo = extract_features(image_path)
				caption_gen = generate_desc(photo, max_length)
				for cap in caption_gen:
						# remove start and end tokens
						seq = cap[0].split()[1:-1]
						desc = ' '.join(seq)
						print("Caption: "+str(caption_id))
						print('{} [log prob: {:1.2f}]'.format(desc,cap[1]))
						if(cap[1]>=-18.0):
							corpus.append(desc)
		temp_caption_string = ""
		for alpha in corpus:
			temp_caption_string = temp_caption_string + "$" + alpha
		caption.append(temp_caption_string)
		X = vectorizer.fit_transform(corpus)
		temp_caption_string = ""
		for alpha in vectorizer.get_feature_names():
			temp_caption_string = temp_caption_string + "$" + alpha
		caption.append(temp_caption_string)
		csv_caption.append(caption)
		video.release()
		if psutil.virtual_memory()[2]>60:
			break
	vidcap.release()
	cv2.destroyAllWindows()
	return csv_caption

