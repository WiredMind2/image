import os
import sqlite3
import tqdm

import keras
import numpy as np

import tensorflow as tf
import cv2
from keras import layers
import einops

DB_PATH = r"D:\willi\Documents\Python\internet\web_server\media_browser.db"

TAGS_IGNORE = {"edit"}
VID_EXT = {"mp4", "mkv", "avi", "webm"}
IMG_EXT = {"jpg", "jpeg", "png", "gif"}
VID_N_FRAMES = 8
TRAINING_SIZE = 0.8
BATCH_SIZE = 8
HEIGHT = 224
WIDTH = 224


class FrameHandler:
	# Frames handler
	def get_frames(self, fp):
		_, ext = os.path.splitext(fp)
		if ext[1:] in VID_EXT:
			return self.get_video_frames(fp)
		elif ext[1:] in IMG_EXT:
			return []
			return self.get_image_frame(fp)
		else:
			return [] # Ignore weird files (.mov)

	def get_video_frames(self, fp):
		src = cv2.VideoCapture(fp)

		video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

		frame_step = int(video_length // VID_N_FRAMES)
		result = []

		for i in range(0, int(video_length), frame_step):
			src.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
			ret, frame = src.read()
			if not ret:
				result.append(np.zeros_like(result[0]))
				continue

			frame = self.format_frames(frame)
			result.append(frame)

		src.release()
		result = np.array(result)[..., [2, 1, 0]]

		return result

	def get_image_frame(self, fp):
		image = cv2.imread(fp)
		if image is None:
			return np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

		image = self.format_frames(image)
		return np.expand_dims(image, axis=0)

	def format_frames(self, frame, output_size=None):
		"""
		Pad and resize an image from a video.

		Args:
		frame: Image that needs to resized and padded.
		output_size: Pixel size of the output frame image.

		Return:
		Formatted frame with padding of specified output size.
		"""
		if output_size is None:
			output_size = (HEIGHT, WIDTH)

		frame = tf.image.convert_image_dtype(frame, tf.float32)
		frame = tf.image.resize_with_pad(frame, *output_size)
		return frame.numpy()


class DatasetHandler(FrameHandler):
	def __init__(self):
		self.get_db()

	def get_db(self):
		self.conn = sqlite3.connect(DB_PATH)
		self.cursor = self.conn.cursor()

	def get_medias(self):
		self.cursor.execute("SELECT id, filename FROM media")
		medias = self.cursor.fetchall()
		data = {}
		for id, fp in medias:
			if not os.path.exists(fp):
				continue

			self.cursor.execute(
				"SELECT name, tag_id FROM tags, media_tags WHERE id=tag_id AND media_id=?", (id,)
			)
			tags = []
			for tag, tag_id in self.cursor.fetchall():
				if tag in TAGS_IGNORE:
					break
				tags.append(tag_id)
			else:
				data[fp] = tags

		return data

	def get_dataset(self, start, size):
		medias = self.get_medias()

		start = start * len(medias)
		size = size * len(medias)

		dataset = []
		for i, (fp, tags) in enumerate(medias.items()):
			if i < start:
				continue
			if i >= start + size:
				break

			frames = self.get_frames(fp)
			for frame in frames:
				# dataset.append((frame, tags))
				yield (frame, tags)

		return dataset

	def get_training_dataset(self):
		return self.get_dataset(0, TRAINING_SIZE)

	def get_testing_dataset(self):
		return self.get_dataset(TRAINING_SIZE, 1)


class MediaClassifier(DatasetHandler):

	def build_model(self, frames):
		input_shape = (None, 10, HEIGHT, WIDTH, 3)
		input = layers.Input(shape=(input_shape[1:]))
		x = input

		x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding="same")(x)
		x = layers.BatchNormalization()(x)
		x = layers.ReLU()(x)
		x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

		# Block 1
		x = add_residual_block(x, 16, (3, 3, 3))
		x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

		# Block 2
		x = add_residual_block(x, 32, (3, 3, 3))
		x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

		# Block 3
		x = add_residual_block(x, 64, (3, 3, 3))
		x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

		# Block 4
		x = add_residual_block(x, 128, (3, 3, 3))

		x = layers.GlobalAveragePooling3D()(x)
		x = layers.Flatten()(x)

		self.model = keras.Model(input, x)
		self.model.build(frames)

	def train(self):
		train_dataset = self.get_training_dataset()
		test_dataset = self.get_testing_dataset()

		# TODO - Iterator dataset instead of loading 200gb+ of videos at once (lol)
		train_ds = tf.data.Dataset.from_generator(
			lambda: train_dataset,
			output_signature=(
				tf.TensorSpec(shape=(10, HEIGHT, WIDTH, 3), dtype=tf.float32),
				tf.TensorSpec(shape=(None,), dtype=tf.int32),
			),
		).batch(BATCH_SIZE)

		val_ds = tf.data.Dataset.from_generator(
			lambda: test_dataset,
			output_signature=(
				tf.TensorSpec(shape=(10, HEIGHT, WIDTH, 3), dtype=tf.float32),
				tf.TensorSpec(shape=(None,), dtype=tf.int32),
			),
		).batch(BATCH_SIZE)

		self.model.compile(
			loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			optimizer=keras.optimizers.Adam(learning_rate=0.0001),
			metrics=["accuracy"],
		)

		# Display progress in real-time
		callbacks = [
			keras.callbacks.ProgbarLogger()
		]

		history = self.model.fit(x=train_ds, epochs=50, validation_data=val_ds, callbacks=callbacks)

		return history


class Conv2Plus1D(keras.layers.Layer):
	def __init__(self, filters, kernel_size, padding):
		"""
		A sequence of convolutional layers that first apply the convolution operation over the
		spatial dimensions, and then the temporal dimension.
		"""
		super().__init__()
		self.seq = keras.Sequential(
			[
				# Spatial decomposition
				layers.Conv3D(
					filters=filters,
					kernel_size=(1, kernel_size[1], kernel_size[2]),
					padding=padding,
				),
				# Temporal decomposition
				layers.Conv3D(
					filters=filters, kernel_size=(kernel_size[0], 1, 1), padding=padding
				),
			]
		)

	def call(self, x):
		return self.seq(x)


class ResidualMain(keras.layers.Layer):
	"""
	Residual block of the model with convolution, layer normalization, and the
	activation function, ReLU.
	"""

	def __init__(self, filters, kernel_size):
		super().__init__()
		self.seq = keras.Sequential(
			[
				Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding="same"),
				layers.LayerNormalization(),
				layers.ReLU(),
				Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding="same"),
				layers.LayerNormalization(),
			]
		)

	def call(self, x):
		return self.seq(x)


class Project(keras.layers.Layer):
	"""
	Project certain dimensions of the tensor as the data is passed through different
	sized filters and downsampled.
	"""

	def __init__(self, units):
		super().__init__()
		self.seq = keras.Sequential([layers.Dense(units), layers.LayerNormalization()])

	def call(self, x):
		return self.seq(x)


def add_residual_block(input, filters, kernel_size):
	"""
	Add residual blocks to the model. If the last dimensions of the input data
	and filter size does not match, project it such that last dimension matches.
	"""
	out = ResidualMain(filters, kernel_size)(input)

	res = input
	# Using the Keras functional APIs, project the last dimension of the tensor to
	# match the new filter size
	if out.shape[-1] != input.shape[-1]:
		res = Project(out.shape[-1])(res)

	return layers.add([res, out])


class ResizeVideo(keras.layers.Layer):
	def __init__(self, height, width):
		super().__init__()
		self.height = height
		self.width = width
		self.resizing_layer = layers.Resizing(self.height, self.width)

	def call(self, video):
		"""
		Use the einops library to resize the tensor.

		Args:
		  video: Tensor representation of the video, in the form of a set of frames.

		Return:
		  A downsampled size of the video according to the new height and width it should be resized to.
		"""
		# b stands for batch size, t stands for time, h stands for height,
		# w stands for width, and c stands for the number of channels.
		old_shape = einops.parse_shape(video, "b t h w c")
		images = einops.rearrange(video, "b t h w c -> (b t) h w c")
		images = self.resizing_layer(images)
		videos = einops.rearrange(images, "(b t) h w c -> b t h w c", t=old_shape["t"])
		return videos


if __name__ == "__main__":
	mc = MediaClassifier()

	mc.build_model(mc.get_training_dataset())

	# keras.utils.plot_model(
	# 	mc.model, to_file="model.png", expand_nested=True, dpi=60, show_shapes=True
	# )
 
	mc.train()

	pass
