from flask import Flask,render_template,request,url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from video_capgen import *
from keras.models import load_model
from pickle import load
import os
from time import sleep
from threading import Thread
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

UPLOAD_FOLDER = os.path.abspath(os.getcwd())+"/video_db"
file_path = os.path.abspath(os.getcwd())+"/database_files/filestorage.db"

app=Flask(__name__,template_folder='template')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'+file_path
app.config['SQLAlCHEMY_TRACK_MODIFICATIONS'] = False 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
db = SQLAlchemy(app)

class videofiles(db.Model):
	id = db.Column(db.Integer, primary_key = True)
	cap_id = db.Column(db.Integer)
	vid_name = db.Column(db.String(100), nullable = False)
	cap_vid_name = db.Column(db.String(100), nullable = False)
	caption = db.Column(db.String(500))
	caption_tag = db.Column(db.String(500))

def slow_function(some_object):
	sleep(2)

def async_slow_function(some_object):
	thr = Thread(target=slow_function, args=[some_object])
	thr.start()
	return thr

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/capgenhome')
def capgen():
	return render_template('capgenhome.html')

@app.route('/searchqueryhome')
def searchqueryhome():
	return render_template('searchqueryhome.html')

@app.route('/upload', methods = ['POST'])
def upload():
	if(request.method == 'POST'):
		f = open("database_files/video_tracker.txt","r")
		video_number = int(f.read()[0]) + 1
		f.close()
		file = request.files['inputFile']
		file_extension = file.filename.split(".")[-1]
		file.filename = "video_"+str(video_number)+"."+file_extension
		print("File Name : ",file.filename)
		video_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
		file.save(video_location)
		index = video_clustering(video_location)
		async_slow_function("DELAY_1")
		global caption_result
		caption_result = generate_caption(index)
		async_slow_function("DELAY_2")
		print("Internal Result : ",caption_result)
		return render_template('capgen_submit.html', foobar=caption_result)

@app.route('/upload_db',methods=['POST'])
def upload_db():
	async_slow_function("DELAY_3")
	print("External Result : ",caption_result)
	async_slow_function("DELAY_4")
	for cap in caption_result:
		newFile = videofiles(cap_id = cap[0],cap_vid_name = cap[1],vid_name = cap[2],caption = cap[3],caption_tag = cap[4])
		async_slow_function("DELAY_5")
		db.session.add(newFile)
		db.session.commit()
	f = open("database_files/video_tracker.txt","r")
	vid_count = int(f.read()[0])
	f.close()
	f = open("database_files/video_tracker.txt","w")
	f.write(str(int(vid_count)+1))
	f.close()
	return render_template('index.html')

@app.route('/exit',methods = ['GET','POST'])
def exit():
	return render_template('index.html')

@app.route('/search_query',methods = ['GET','POST'])
def search_query():
	if (request.method == "POST"):
		query_search = request.form.get("query")
		file_data = videofiles.query.all()
		video_result_list = []
		for i in file_data:
			corpus = []
			temp_query = i.caption
			temp_vid_name = i.vid_name
			query_search_list = [i for i in temp_query.split("$")]
			for Y_list in query_search_list:
				if(len(Y_list)>2):
					#print(Y_list)
					X_list = word_tokenize(query_search) 
					Y_list = word_tokenize(Y_list)
					sw = stopwords.words('english') 
					l1 =[];l2 =[]
					X_set = {w for w in X_list if not w in sw} 
					Y_set = {w for w in Y_list if not w in sw}
					rvector = X_set.union(Y_set) 
					for w in rvector:
						if w in X_set: l1.append(1) # create a vector
						else: l1.append(0)
						if w in Y_set: l2.append(1)
						else: l2.append(0)
					c = 0
					# cosine formula 
					for i in range(len(rvector)):
						c+= l1[i]*l2[i]
					cosine = c / float((sum(l1)*sum(l2))**0.5)
					if(cosine>=0.20):
						if temp_vid_name not in video_result_list:
							video_result_list.append(temp_vid_name)
							#print(temp_vid_name , "similarity: ", cosine)
		print(video_result_list)
		return render_template('video_search_result.html',video_result = video_result_list)

if __name__ == '__main__':
    app.debug = True
    app.run()

