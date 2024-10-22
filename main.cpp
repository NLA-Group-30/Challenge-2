#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <unsupported/Eigen/SparseExtra>

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

	std::cout << std::endl;
	std::cout << " ### Task 1 ### " << std::endl;
	std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(height, width);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			const int index = (i * width + j) * channels;
			A(i, j) = static_cast<double>(original_image[index]);
		}
	}

	// we do not need the original image data
	stbi_image_free(original_image);

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ATA(A.transpose() * A);
	std::cout << "Norm of At * A : " << ATA.norm() << std::endl << std::endl;

	std::cout << " ### Task 2 ### " << std::endl;
	// Task 2: Solve the eigenvalue problem ATAx = λx using the proper solver provided by the Eigen library. Report the
	// two largest computed singular values of A.
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(ATA);
	if (eigensolver.info() != Eigen ::Success) {
		std::cerr << "Errore nel calcolo degli autovalori." << std::endl;
		return 1;
	}

	Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();

	// Eigenvalues of ATA are singular values (squared) of A.
	Eigen::VectorXd singularValues = eigenvalues.array().sqrt();

	// We know the singular values are sorted, so we just get the last two elements in the array
	std::cout << "Two largest singular values of A: " << singularValues(singularValues.size() - 1) << " e "
			  << singularValues(singularValues.size() - 2) << std::endl
			  << std::endl;

	std::cout << " ### Task 3 ### " << std::endl;
	Eigen::saveMarket(ATA, "ATA.mtx");

	/*
		You must have the folder 'lis-2.1.6' in your project directory in order for the following commands to work.
		You can obtain it with these commands:
		wget https://www.ssisc.org/lis/dl/lis-2.1.6.zip
		unzip lis-2.1.6.zip
	 */

	// eigenvalue: 1.045818e+09 (it is the square of the actual eigenvalue of A)
	system(
		"export LIS_ROOT=`realpath .`/lis-2.1.6 && "
		"mpirun -n 1 ${LIS_ROOT}/test/etest1 ATA.mtx eigvec.txt hist.txt -e pi -emaxiter 100 -etol 1.e-8");

	std::cout << std::endl;
	std::cout << " ### Task 4 ### " << std::endl;

	// find a shift which leads to an acceleration
	for (double shift = 32338.0; shift <= 32340.0; shift += 0.1) {
		std::cout << " Shift: " << shift << std::endl;
		std::string cmd = std::string(
							  "export LIS_ROOT=`realpath .`/lis-2.1.6 && "
							  "mpirun -n 1 ${LIS_ROOT}/test/etest1 ATA.mtx eigvec.txt hist.txt -e pi -shift ") +
						  std::to_string(shift) + std::string(" -emaxiter 100 -etol 1.e-8 | grep -i elapsed -B 1");
		system(cmd.c_str());
	}

	// Task 5: Using the SVD module of the Eigen library, perform a singular value decomposition of the matrix A. Report
	// the Euclidean norm of the diagonal matrix Σ of the singular values.
	std::cout << std::endl;
	std::cout << " ### Task 5 ### " << std::endl;

	return 0;
}