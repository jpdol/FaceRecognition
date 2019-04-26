input_embeddings = {}
import glob
import os
import cv2
for dirname in glob.glob("images/*"):
	embs = []
	person_name = os.path.splitext(os.path.basename(dirname))[0]
	for file in glob.glob(str(dirname) + '/*'):
		image_file = cv2.imread(file, 1)
		embs.append(file)
	input_embeddings[person_name] = embs
print(input_embeddings)