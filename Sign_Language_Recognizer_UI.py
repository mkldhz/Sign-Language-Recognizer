import tkinter
from tkinter import ttk
from tkinter import *
from ttkbootstrap import Style
import cv2
import numpy as np
from PIL import Image, ImageTk

import os
import six.moves.urllib as urllib
import sys
import tarfile
import sys
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt

from IPython.display import display

from tkinter.filedialog import askopenfilename, asksaveasfilename
import pathlib
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



class Application(tkinter.Tk):

	def __init__(self):
		super().__init__()
		self.title('Sign Language Recognizer')
		self.geometry('1280x600')
		self.style = Style('superhero')
		self.home_screen = HomeScreen(self)
		self.home_screen.pack(fill='both', expand='yes')


class HomeScreen(ttk.Frame):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


		ttk.Button(self, text='Open Image', command=self.open_image).place(x=450,y=40)
		ttk.Button(self, text='Predict Hand Sign', command=self.single_image_pred).place(x=750,y=40)
		ttk.Button(self, text='Predict Live Feed', style='danger.TButton', command=self.live_pred).place(x=1100,y=540)


		self.image_view = ttk.Label(self, image='')
		self.image_view.place(x=100, y=150)

		
		self.image_view_2 = ttk.Label(self, image='')
		self.image_view_2.place(x=670, y=150)	


	def show_inference(self,model, image_path):

		image_np = np.array(Image.open(image_path))
		output_dict = self.run_inference_for_single_image(model, image_np)
		vis_util.visualize_boxes_and_labels_on_image_array(image_np,output_dict['detection_boxes'],output_dict['detection_classes'],output_dict['detection_scores'],self.category_index,instance_masks=output_dict.get('detection_masks_reframed', None),use_normalized_coordinates=True,line_thickness=8)
		#plt.imshow(image_np)
		#plt.show()
		#display(Image.fromarray(image_np))
		#cv2.imwrite('img.jpg',image_np)
		#display(image_np)
		self.img=image_np
		self.image_resize(height=500, width=500)
		self.im_pil_2 = Image.fromarray(self.img)
		self.im_pil_2 = ImageTk.PhotoImage(self.im_pil_2)

		self.image_view_2.configure(image=self.im_pil_2)

	
	def single_image_pred(self):
		utils_ops.tf = tf.compat.v1

		tf.gfile = tf.io.gfile
		dir_labelmap =rf"{os.path.abspath(os.path.dirname(__file__))}\object_detection\images\labelmap.pbtxt"
		PATH_TO_LABELS = dir_labelmap
		self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
		self.show_inference(detection_model, self.img_path)


		


	def open_image(self):

		self.img_path = askopenfilename()
		self.image_view.configure(image='')
		self.image_view_2.configure(image='')
		self.img = cv2.imread(self.img_path)
		self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
		self.image_resize(height=500, width=500)
		self.im_pil = Image.fromarray(self.img)
		self.im_pil = ImageTk.PhotoImage(self.im_pil)
		self.image_view.configure(image=self.im_pil)


	def run_inference_for_single_image(self,model, image):

		image = np.asarray(image)
		# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
		input_tensor = tf.convert_to_tensor(image)
		# The model expects a batch of images, so add an axis with `tf.newaxis`.
		input_tensor = input_tensor[tf.newaxis,...]

		# Run inference
		model_fn = model.signatures['serving_default']
		output_dict = model_fn(input_tensor)


		num_detections = int(output_dict.pop('num_detections'))
		output_dict = {key:value[0, :num_detections].numpy() 
						for key,value in output_dict.items()}
		output_dict['num_detections'] = num_detections


		output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   

		if 'detection_masks' in output_dict:

			detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(output_dict['detection_masks'], output_dict['detection_boxes'],image.shape[0], image.shape[1])
			detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
									   tf.uint8)
			output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
	
		return output_dict


	def show_inference_cv2(self,model, image):

		image_np = image
		output_dict = self.run_inference_for_single_image(model, image_np)
		vis_util.visualize_boxes_and_labels_on_image_array(
			image_np,
			output_dict['detection_boxes'],
			output_dict['detection_classes'],
			output_dict['detection_scores'],
			self.category_index,
			instance_masks=output_dict.get('detection_masks_reframed', None),
			use_normalized_coordinates=True,
			line_thickness=8)

		return image_np


	def live_pred(self):
		# patch tf1 into `utils.ops`
		utils_ops.tf = tf.compat.v1

		# Patch the location of gfile
		tf.gfile = tf.io.gfile
		dir_labelmap =rf"{os.path.abspath(os.path.dirname(__file__))}\object_detection\images\labelmap.pbtxt"
		PATH_TO_LABELS = dir_labelmap
		self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
		cap = cv2.VideoCapture(0)

		while 1:

			_,img = cap.read()
	
			img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
			final_img = self.show_inference_cv2(detection_model,img)
		
			final_img = cv2.cvtColor(final_img,cv2.COLOR_RGB2BGR)
			cv2.imshow('img',final_img)

			if cv2.waitKey(1) == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()


	def image_resize(self, height=None, width=None,inter = cv2.INTER_AREA):
		# initialize the dimensions of the image to be resized and
		# grab the image size
		dim = None
		(h, w) = self.img.shape[:2]

		# if both the width and height are None, then return the
		# original image
		if width is None and height is None:
			return self.img

		# check to see if the width is None
		if width is None:
			# calculate the ratio of the height and construct the
			# dimensions
			r = height / float(h)
			dim = (int(w * r), height)

		# otherwise, the height is None
		else:
		# calculate the ratio of the width and construct the
		# dimensions
			r = width / float(w)
			dim = (width, int(h * r))

		# resize the image
		self.img = cv2.resize(self.img, dim, interpolation = inter)


	
if __name__ == '__main__':
	
	dir_model =rf"{os.path.abspath(os.path.dirname(__file__))}\object_detection\inference_graph\saved_model"
	detection_model = tf.saved_model.load(dir_model)
	Application().mainloop()
