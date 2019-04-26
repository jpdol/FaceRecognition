import lib

input_embeddings = lib.create_input_image_embeddings()
lib.recognize_faces_in_cam(input_embeddings)