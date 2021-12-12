#Import Packages'
import time
import os
import glob
import cv2
import random
import numpy as np
from scipy.spatial import distance as dist

#Function to find similarity of the images
def compare_images(image_1,image_2):
	#image_1 = cv2.cvtColor(image_1,cv2.COLOR_BGR2RGB)
	#image_2 = cv2.cvtColor(image_2,cv2.COLOR_BGR2RGB)
	hist_1 = cv2.calcHist([image_1], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
	hist_1 = cv2.normalize(hist_1, hist_1).flatten()
	hist_2 = cv2.calcHist([image_2], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
	hist_2 = cv2.normalize(hist_2, hist_2).flatten()
	dist = cv2.compareHist(hist_1,hist_2,cv2.HISTCMP_CORREL)
	return dist

def video_clustering(video_name):
	cap_id = 1
	count = 0
	clip_len = 0
	flag = False
	video_cluster_dict = {}
	prev_img = 255*np.ones((480,320,30), np.uint8)
	vidcap = cv2.VideoCapture(video_name)
	while(True):
		vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*50))
		success, image = vidcap.read()
		if(not success):
			video_cluster_dict[cap_id] = [start_time,int(milliseconds),start_frame,frame_count,video_name,clip_len]
			break
		frame_count = "frame"+ "%d"%count
		image = cv2.resize(image,(480,320),interpolation=cv2.INTER_AREA)
		milliseconds = vidcap.get(cv2.CAP_PROP_POS_MSEC)
		if(count == 0):
			video_cluster_dict[0] = ["Start Time","End Time","Start Frame","End Frame",video_name,"Video Length"]
			start_frame = frame_count
			start_time = int(milliseconds)
		dist = compare_images(prev_img,image)
		if(dist<0.8):
			end_time = int(milliseconds)
			if(clip_len>10):
				video_cluster_dict[cap_id] = [start_time,end_time,start_frame,frame_count,video_name,clip_len]
				cap_id += 1
			start_frame = frame_count
			start_time = int(milliseconds)
			flag = True
			clip_len = 0
		else:
			clip_len += 1
		prev_img = image
		count += 1
	vidcap.release()
	return video_cluster_dict