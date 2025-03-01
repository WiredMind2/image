from collections import defaultdict
from itertools import chain, islice
import os
import pickle
import random
import time
# from deepface import DeepFace
import cv2
import faiss
import matplotlib.pyplot as plt
import numpy as np

CACHE_PATH = "cache.pkl"

def load_cache():
	if os.path.exists(CACHE_PATH):
		with open(CACHE_PATH, "rb") as f:
			return pickle.load(f)
	return {}

def save_cache(cache):
	with open(CACHE_PATH, "wb") as f:
		pickle.dump(cache, f)


class FacialRecognition:
	def __init__(self, model_name, backend):
		self.model_name = model_name
		self.backend = backend
		self.index = None
		self.feature_img_paths = []
		self.feature_vectors = np.array([])
  
		self.load_libs()
  
		self.cache = load_cache()
  
	def load_libs(self):
		global DeepFace

		start = time.time()
		from deepface import DeepFace
		print(f"Loaded libs in {time.time() - start:.2f} s")


	def plot_faces(self, img_path, model_name, backend):

		# embeddings
		embedding_objs = self.get_face_embeddings(img_path, model_name, backend)
		# Load the image
		image = cv2.imread(img_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Plot the image
		plt.imshow(image)
		plt.axis("off")
		plt.title(f"Image: {img_path}")

		for embedding_obj in embedding_objs:
			embedding = embedding_obj["embedding"]
			bbox = embedding_obj["facial_area"]
			# Draw bounding box
			x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
			rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
			plt.gca().add_patch(rect)


	def benchmark(self, img_path):
		models = [
			"VGG-Face",
			"Facenet",
			"Facenet512",
			"OpenFace",
			"DeepFace",
			"DeepID",
			"ArcFace",
			"Dlib",
			"SFace",
			"GhostFaceNet",
		]

		backends = [
			"opencv",
			#   'ssd',
			"dlib",
			"mtcnn",  # fast and good so yeah
			"fastmtcnn",
			"retinaface",
		]

		model_count = 1
		backend_count = len(backends)

		for i in range(model_count):
			model = models[i]
			for j in range(backend_count):
				backend = backends[j]
				plt.subplot(model_count, backend_count, i * backend_count + j + 1)

				start = time.time()
				plot_faces(img_path, model, backend=backend)
				print(f"Time elapsed for {model} / {backend}: {time.time() - start}")
				plt.title(f"{model} / {backend}: {time.time() - start:.2f} s")
				plt.axis("off")

		plt.show()


	def get_infos(self, img_path, backend):
		objs = DeepFace.analyze(
			img_path,
			actions=["age", "gender", "race", "emotion"],
			detector_backend=backend,
		)
		return objs


	def get_face_embeddings(self, img_path, model_name, backend):
		if img_path in self.cache:
			return self.cache[img_path]

		out = DeepFace.represent(
			img_path=img_path,
			model_name=model_name,
			detector_backend=backend,
		)
		self.cache[img_path] = out
		save_cache(self.cache)
		return out

	def build_index(self, img_paths=None):
		# Load the model

		feature_img_paths, feature_vectors = self.get_feature_vectors(img_paths)

		if self.index is None:
			self.index = faiss.IndexFlatL2(feature_vectors.shape[1])

		self.feature_img_paths.extend(feature_img_paths)
		if len(self.feature_vectors) == 0:
			self.feature_vectors = feature_vectors
		else:
			self.feature_vectors = np.vstack([self.feature_vectors, feature_vectors])

		# Load the images
		self.index.add(feature_vectors)

	def get_feature_vector(self, img_path):
		feature_vectors = []
		image_out = []

		try:
			embedding_objs = self.get_face_embeddings(img_path, self.model_name, self.backend)
			for embedding_obj in embedding_objs:
				embedding = embedding_obj["embedding"]
				feature_vectors.append(embedding)
				image_out.append(img_path)

		except Exception as e:
			print(f"Error processing {path}: {e}")
			return [], []
		else:
			feature_vectors = np.array(feature_vectors)
			return image_out, feature_vectors


	def get_feature_vectors(self, image_paths=None):
		if not image_paths:
			return [], []

		feature_vectors = []  # List of feature vectors
		image_out = []
		for path in list(set(image_paths)):
			cur_image_out, cur_feature_vectors = self.get_feature_vector(path)
			image_out.extend(cur_image_out)
			feature_vectors.extend(cur_feature_vectors)

		feature_vectors = np.array(feature_vectors)
		return image_out, feature_vectors
	
	def search(self, img_path, top_k=5):
		# Get the feature vector
		image_out, feature_vector = self.get_feature_vector(img_path)

		feature_vector = np.array(feature_vector).reshape(1, -1)

		# Search
		distances, indices = self.index.search(feature_vector, top_k)
		return distances, [self.feature_img_paths[indices[0][i]] for i in range(top_k)]
 
if __name__ == "__main__":
	root = 'lfw_funneled'

	imgs = defaultdict(list)
	for file in os.listdir(root):
		path = os.path.join(root, file)
		if os.path.isdir(path):
			for file in os.listdir(path):
				imgs[path].append(os.path.join(path, file))

	imgs = dict(sorted(islice(imgs.items(), 100), key=lambda x: len(x[1]), reverse=True))
	
	img_paths = []
	for k, v in imgs.items():
		img_paths.extend(v)


	f = FacialRecognition("Facenet", "mtcnn")
	f.build_index(img_paths)

	plt.figure()
	iters = 5
	most_k = 5
	choices = random.choices(list(chain(*imgs.values())), k=iters)
	for i, img in enumerate(choices):
		distances, img_path = f.search(img, 5)
		print(distances, img_paths)

		# Plot the images
		plt.subplot(iters, most_k + 1, i * most_k + 1)
		plt.imshow(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
		plt.title("Query Image")
		plt.axis("off")

		for j, (distance, img_path) in enumerate(zip(distances[0], img_path)):
			plt.subplot(iters, most_k + 1, i * most_k + j + 2)
			plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
			plt.title(f"Distance: {distance:.2f}")
			plt.axis("off")


	plt.show()
	input()
pass
