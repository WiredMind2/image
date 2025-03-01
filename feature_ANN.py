import heapq
import os
import threading
import faiss
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = set([".png", ".jpg", ".jpeg", ".gif"])
VIDEO_EXTENSIONS = set([".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm"])

class ANN:
	def __init__(self, paths = None, vectors = None):
		self.index = None
		self.paths = []
		self.paths_cache = set()
		self.vectors = None

		self.get_ANN_index()

	def get_vector(self, path):
		raise NotImplementedError

	def add_paths(self, paths, vectors=None):
		vector_queue = []
		for i, path in enumerate(paths):
			if path not in self.paths_cache:
				if vectors is not None and i < len(vectors):
					vector = vectors[i]
				else:
					try:
						vector = self.get_vector(path)
					except Exception as e:
						print(f"Error processing {path}: {e}")
						vector = None

				if vector is not None:
					self.paths_cache.add(path)
					self.paths.append(path)
					vector_queue.append(vector)

		if self.vectors is None:
			self.vectors = np.array(vector_queue)
		else:
			self.vectors = np.vstack([self.vectors, vector_queue])
	
		self.index.add(vector_queue)

	def replace_images(self, image_paths):
		self.index = None
		self.paths = []
		self.paths_cache = set()
		self.vectors = None
		self.add_paths(image_paths)

	def remove_images(self, image_paths):
		indexes = []
		for path in image_paths:
			if path in self.paths_cache:
				index = self.paths.index(path)
				self.paths.pop(index)
				self.paths_cache.remove(path)
				indexes.append(index)

		self.vectors = np.delete(self.vectors, indexes, axis=0)

		self.index = None
		self.get_ANN_index()

	def get_ANN_index(self):
		# Build an ANN index
		
		if self.index is not None:
			return self.index

		self.index = faiss.IndexFlatL2(self.vectors.shape[1])

		# res = faiss.StandardGpuResources()  # use a single GPU
		# ngpus = faiss.get_num_gpus()

		# print("number of GPUs:", ngpus)

		# index = faiss.index_cpu_to_all_gpus(index_cpu)

		self.index.add(self.vectors)  # Add feature vectors to the ANN index

		return self.index

	def iter_closest(self, feature_vectors, image_paths):
		# Build an ANN index while finding the two nearest neighbors of each feature vector
		
		if self.index is not None:
			raise ValueError("The index is already built, so use it")
		
		self.index = faiss.IndexFlatL2(feature_vectors.shape[1])
		
		# res = faiss.StandardGpuResources()  # use a single GPU
		# ngpus = faiss.get_num_gpus()
		# print("number of GPUs:", ngpus)
		# index = faiss.index_cpu_to_all_gpus(index_cpu)

		for i, vector in enumerate(feature_vectors):
			distances, indices = self.index.search(vector.reshape(1, -1), 2)  # Find the nearest neighbor of the feature vector

			out = [(dist, image_paths[j]) for j, dist in zip(indices.flatten(), distances.flatten()) if j != i]
			yield image_paths[i], out

			self.index.add(vector.reshape(1, -1))

		# self.index.add(feature_vectors)  # Add feature vectors to the ANN index

	def get_cluster(self, num_clusters=None):
		# TODO - This doesn't use the ANN index
		self.index = self.get_ANN_index()
		
		# Perform clustering using Faiss's KMeans
		if not num_clusters:
			num_clusters = 5  # Define the number of clusters
		kmeans = faiss.Kmeans(d=self.vectors.shape[1], k=num_clusters, niter=20, verbose=True)
		kmeans.train(self.vectors)

		# Assign each feature vector to a cluster
		_, cluster_assignments = kmeans.index.search(self.vectors, 1)

		# Group image paths by cluster
		clusters = {i: [] for i in range(num_clusters)}
		for i, cluster_id in enumerate(cluster_assignments.flatten()):
			clusters[cluster_id].append(image_paths[i])
		
		return clusters

	def get_pairs(self):
		# Find the pairs of closest images from each other
		heap = []

		for i in range(len(self.vectors)):
			for j in range(i + 1, len(self.vectors)):
				distance = np.linalg.norm(self.vectors[i] - self.vectors[j])
				# heap.append((distance, image_paths[i], image_paths[j]))
				heapq.heappush(heap, (distance, image_paths[i], image_paths[j]))

		while heap:
			distance, img1, img2 = heapq.heappop(heap)
			yield distance, img1, img2

	def get_pairs_opti(self):
		self.index = self.get_ANN_index()
		
		while len(self.vectors) > 0:
			distances, indices = self.index.search(self.vectors, 2)  # Find top-2 similar images
			for i, (distance, j) in enumerate(zip(distances, indices)):
				img1 = image_paths[i]
				img2 = image_paths[j[1]]
				yield distance[1], img1, img2

	def find_closest(self, n=5):
		self.index = self.get_ANN_index()

		# Find the closest images for each image
		distances, indices = self.index.search(self.vectors, n + 1)  # Find top-n similar images
		for j, image_path in enumerate(image_paths):
			out = [(dist, image_paths[i]) for i, dist in zip(indices[j], distances[j]) if i != j]
			yield image_path, out

	def find_closest_to(self, image_path, n=5):
		self.index = self.get_ANN_index()
		try:
			feature_vector = self.extract_features(image_path)
		except Exception as e:
			print(f"Error processing {image_path}: {e}")
			return []
		# Find the closest images for the given image
		if self.index is None:
			return []

		distances, indices = self.index.search(feature_vector.reshape(1, -1), n + 1)  # Find top-n similar images

		out = []
		for i, dist in zip(indices.flatten(), distances.flatten()):
			# if i == image_index: continue
			
			if i < 0 or i >= len(self.image_paths):
				return []
	
			out.append((dist, self.image_paths[i]))
		return out

	def plot_images(self, img_paths, distances):
		import matplotlib.pyplot as plt

		fig, axes = plt.subplots(1, len(img_paths), figsize=(10, 5))
		for i, (img_path, dist) in enumerate(zip(img_paths, distances)):
			img = Image.open(img_path)

			axes[i].imshow(img)
			axes[i].set_title(f"Image 1\n{img_path}")
			axes[i].axis('off')
			
			axes[i].set_title(f"Distance: {dist:.2f}")

		plt.show(block=True)


class Image_Features(ANN):
	def __init__(self, image_paths=None, feature_vectors=None):
		super().__init__()
		
		self.model = None
		self.model_thread = None

		self.load_model(block=False)

		if image_paths is not None:
			self.add_paths(image_paths, feature_vectors)


	def load_model(self, block=True):
		if self.model is not None:
			return self.model

		if self.model_thread is None:
			os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
			# Load model
			def loader():
				while True:
					try:
						import torch
					except Exception as e:
						pass # Weird bug with breakpoints
					else:
						break

				from torchvision import models
				# Load a pre-trained model (e.g., ResNet)
				self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
				self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove the classification layer
				self.model.eval()

			thread = threading.Thread(target=loader)
			thread.start()
			self.model_thread = thread

		if block:
			self.model_thread.join()

			return self.model

		return None

	# Preprocess images
	def preprocess_image(self, image_path):
		from torchvision import transforms
		transform = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
		_, ext = os.path.splitext(image_path)
		if ext.lower() in IMAGE_EXTENSIONS:
			image = Image.open(image_path).convert("RGB")
		elif ext.lower() in VIDEO_EXTENSIONS:
			import cv2
			cap = cv2.VideoCapture(image_path)
			ret, frame = cap.read()
			if frame is None or ret is False:
				raise ValueError(f"Cannot read the first frame from {image_path}")
			image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		else:
			raise ValueError(f"Unsupported file format: {ext}")

		return transform(image).unsqueeze(0)

	def get_vector(self, image_path):
		# Extract features
		import torch
		model = self.load_model()
		with torch.no_grad():
			return model(self.preprocess_image(image_path)).squeeze().numpy()

	def get_vectors(self, image_paths):
		if not image_paths:
			return [], []
		feature_vectors = []  # List of feature vectors
		image_out = []
		for path in list(set(image_paths)):
			try:
				feature_vectors.append(self.get_vector(path))
				image_out.append(path)
			except Exception as e:
				print(f"Error processing {path}: {e}")

		feature_vectors = np.array(feature_vectors)
		return image_out, feature_vectors


if __name__ == "__main__":
	from tkinter import filedialog
	root = filedialog.askdirectory()
	
	image_paths = []
	for fp in os.listdir(root):
		base, ext = os.path.splitext(fp)
		if ext in [".png", ".jpg", ".jpeg"]:
			path = os.path.join(root, fp)
			image_paths.append(path)

	ann = ANN(image_paths)
	# a = get_cluster()
	# a = find_closest(feature_vectors, image_paths)


	for dist, img1, img2 in ann.get_pairs_opti():
		ann.plot_images(img1, img2, dist)
		# input("Press Enter to continue...")
	pass
