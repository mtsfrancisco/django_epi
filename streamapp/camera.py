import cv2,os,urllib.request
import numpy as np
from django.conf import settings
from ultralytics import YOLO

yolo = YOLO('yolo_model/best.pt')

def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
		self.video.release()

	def get_frame(self):
		success, image = self.video.read()
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.
		
		results = yolo.track(image)

		for result in results:
			# get the classes names
			classes_names = result.names

			for box in result.boxes:
				if box.conf[0] > 0.4:
					[x1,y1,x2,y2] = box.xyxy[0]
					x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

					cls = int(box.cls[0])
					class_name = classes_names[0]

					colour = getColours(cls)

					cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
					cv2.putText(image, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

		ret, jpeg = cv2.imencode('.jpg', image)
		return jpeg.tobytes()
