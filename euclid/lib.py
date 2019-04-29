from align import AlignDlib

def align_image(img):
	alignment = AlignDlib('models/landmarks.dat')
	return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)