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
	
    // Compute At * A
    Eigen::MatrixXd ATA = m.transpose() * m;

    // Print the Euclidean norm of At * A
    std::cout << "Norm of At * A : " << ATA.norm() << std::endl;

    // Task 2: Solve the eigenvalue problem ATAx = λx using the proper solver provided by the Eigen
    //library. Report the two largest computed singular values of A.

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(ATA);

    if (eigensolver.info() != Eigen :: Success) {
        std::cerr << "Errore nel calcolo degli autovalori." << std::endl;
        return 1;
    }

    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
    Eigen::VectorXd singularValues = eigenvalues.array().sqrt();


    // Stampa i due valori singolari più grandi
    std::cout << "I due valori singolari più grandi sono: "
              << singularValues(singularValues.size() - 1) << " e "
              << singularValues(singularValues.size() - 2) << std::endl;

	
 return 0;

}