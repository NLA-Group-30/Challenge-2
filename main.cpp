#include <iostream>

#include <Eigen/Core>
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

void save_image(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& m,
				const std::string& file_name) {
	// convert original image in one made of bytes instead of doubles
	Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> tmp(m.rows(), m.cols());
	tmp = m.unaryExpr([](const double val) -> unsigned char { return static_cast<unsigned char>(val); });
	// Save the image
	if (stbi_write_png(file_name.c_str(), tmp.cols(), tmp.rows(), 1, tmp.data(), tmp.cols()) == 0) {
		std::cerr << " \u001b[31mERROR\u001b[0m: Could not save image to " << file_name << std::endl;
		return;
	}
	std::cout << "Image saved to " << file_name << std::endl;
}

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
	if (eigensolver.info() != Eigen::Success) {
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

	Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::VectorXd sigmaVector = svd.singularValues();
	Eigen::MatrixXd sigmaM = singularValues.asDiagonal();
	std::cout << "Norm of sigma matrix: " << sigmaM.norm() << std::endl;

	// Task 6: Compute the matrices C and D described in (1) assuming k = 40 and k = 80. Report the number of nonzero
	// entries in the matrices C and D.
	std::cout << std::endl;
	std::cout << " ### Task 6 ### " << std::endl;
	Eigen::MatrixXd U = svd.matrixU();
	Eigen::MatrixXd V = svd.matrixV();
	Eigen::MatrixXd C40(U.rows(), 40);
	Eigen::MatrixXd D40(sigmaM.rows(), 40);
	Eigen::MatrixXd C80(U.rows(), 80);
	Eigen::MatrixXd D80(sigmaM.rows(), 80);
	std::cout << "k = 40" << std::endl;
	for (int k = 0; k < 40; k++) {
		C40.col(k) = U.col(k);
		D40.col(k) = sigmaVector(k) * V.col(k);
	}
	std::cout << "  non-zero entries in C: " << C40.count() << std::endl;
	std::cout << "  non-zero entries in D: " << D40.count() << std::endl;

	std::cout << "k = 80" << std::endl;
	for (int k = 0; k < 80; k++) {
		C80.col(k) = U.col(k);
		D80.col(k) = sigmaVector(k) * V.col(k);
	}
	std::cout << "  non-zero entries in C: " << C80.count() << std::endl;
	std::cout << "  non-zero entries in D: " << D80.count() << std::endl;

	// Task 7: Compute the compressed images as the matrix product C * D^T (again for k = 40 and k = 80). Export and
	// upload the resulting images in .png.
	std::cout << std::endl;
	std::cout << " ### Task 7 ### " << std::endl;
	Eigen::MatrixXd CDT40 = C40 * (D40.transpose());
	save_image(CDT40, "CDT40.png");
	Eigen::MatrixXd CDT80 = C80 * (D80.transpose());
	save_image(CDT80, "CDT80.png");

	return 0;
}