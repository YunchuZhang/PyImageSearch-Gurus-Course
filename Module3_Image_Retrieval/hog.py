# import the necessary packages
from skimage import feature
import skimage

class HOG:
	def __init__(self, orientations=12, pixelsPerCell=(4, 4), cellsPerBlock=(2, 2), normalize=True):
		# store the number of orientations, pixels per cell, cells per block, and
		# whether normalization should be applied to the image
		self.orientations = orientations
		self.pixelsPerCell = pixelsPerCell
		self.cellsPerBlock = cellsPerBlock
		self.normalize = normalize

	def describe(self, image):
		# compute Histogram of Oriented Gradients features for scikit-image < 0.13
		if int(skimage.__version__.split(".")[1]) < 13:
			hist = feature.hog(image, orientations=self.orientations, pixels_per_cell=self.pixelsPerCell,
				cells_per_block=self.cellsPerBlock, transform_sqrt=self.normalize)

		# otherwise comput Histogram of Oriented Gradients features for scikit-image >= 0.13
		else:
			hist = feature.hog(image, orientations=self.orientations, pixels_per_cell=self.pixelsPerCell,
				cells_per_block = self.cellsPerBlock, transform_sqrt = self.normalize, block_norm="L1")


		hist[hist < 0] = 0

		# return the histogram
		return hist