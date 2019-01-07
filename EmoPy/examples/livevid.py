import os
import cv2
import numpy as np 

import keras
from glob import glob
from keras import applications
from keras.models import Model 
from keras.layers import Dropout, Flatten, Dense

from EmoPy.src.fermodel import FERModel
from pkg_resources import resource_filename
from EmoPy.library.align import AlignDlib

def setROI(inputImg_size, roiwidth, roiheight):
	w = inputImg_size[0]
	h = inputImg_size[1]
	xo = 0
	yo = 0
	xb = w
	yb = h
	if w>roiwidth:
		xo = int(w/2 - roiwidth/2)
		xb = int(xo + roiwidth)
	if h>roiheight:
		yo = int(h/2 - roiheight/2)
		yb = int(yo + roiheight)

	return xo, xb, yo, yb

def main():

	print("\n\n")
	print("Press c to identify emotion or Esc to exit.")
	print("The object should cover at least 80 percent of the captured image")

	target_emotions = ['anger','happiness', 'calm']
	model_emotions = FERModel(target_emotions, verbose=True)
	model = keras.models.load_model('weights-improvement-07-0.97.hdf5')
	
	video = cv2.VideoCapture(0)
	ret, inputImg = video.read()
	xo, xb, yo, yb = setROI(inputImg.shape, 900, 720)
	inputImg = inputImg[xo:xb, yo:yb]
	alignment = AlignDlib('landmarks.dat')
	boxColor = (0,0,255)
	boxThickness = 3
	text = ''

	key = 0
	while key!=27:
		ret, inputImg = video.read()
		inputImg = inputImg[xo:xb, yo:yb]
		bb = alignment.getLargestFaceBoundingBox(inputImg)
		
		
		if bb is not None:
			pt1 = (bb.left(), bb.top())
			pt2 = (bb.left()+bb.width(), bb.height()+bb.top())
			cv2.rectangle(inputImg, pt1, pt2, boxColor, boxThickness)
		
		if key == 99:
			if bb is not None:
				imgAligned = alignment.align(96, inputImg, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
				# imgAligned = tensor_img[:, :, 0]
				text = model_emotions.predict_image_array(imgAligned)

		
		###########################################################################
		#Insert Text
		inputImg = cv2.putText(inputImg, text, (50, 50), 0, 1.0, (0, 255, 0), 2) 
		#Title
		cv2.imshow("Face Analysis", inputImg)
		#Key check
		key = cv2.waitKey(1)

	video.release()

if __name__ == '__main__':
	main()