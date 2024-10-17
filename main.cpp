#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
		return 1;
	}

	const char* input_image_path = argv[1];

	// Task 1: Load the image and convert into a m*n matrix. Compute the euclidean norm of At * A.
	int width;
	int height;
	int channels;
	unsigned char* original_image = stbi_load(input_image_path, &width, &height, &channels, 1);
	if (!original_image) {
		std::cerr << "Error: Could not load image " << input_image_path << std::endl;
		return 1;
	}

	std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m(height, width);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			const int index = (i * width + j) * channels;
			m(i, j) = static_cast<double>(original_image[index]);
		}
	}

	// we do not need the original image data
	stbi_image_free(original_image);

	const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mT(m.transpose());
	std::cout << "Norm of At * A : " << (mT * m).norm() << std::endl;

	return 0;
}