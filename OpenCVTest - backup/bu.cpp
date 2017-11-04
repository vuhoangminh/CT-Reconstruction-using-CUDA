// OpenCVTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


/**
* @file Morphology_1.cpp
* @brief Erosion and Dilation sample code
* @author OpenCV team
*/

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdarg.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "fstream"
#include "iostream"
#include <complex>

using namespace std;
using namespace cv;




//-----------------------------------------------------------------------------------------------------
//	linspace function Matlab
//-----------------------------------------------------------------------------------------------------
template <typename T = double> vector<T> linspace(T a, T b, size_t N) {
	T h = (b - a) / static_cast<T>(N - 1);
	vector<T> xs(N);
	typename vector<T>::iterator x;
	T val;
	for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
		*x = val;
	return xs;
}

//-----------------------------------------------------------------------------------------------------
//	absolute value all elements in a vector
//-----------------------------------------------------------------------------------------------------
template <typename T = double> vector<T> absvector(vector <T> A) {
	vector<T> absA;
	for (std::vector<double>::const_iterator i = A.begin(); i != A.end(); ++i)
		absA.push_back(abs(*i));
	return absA;
}

//-----------------------------------------------------------------------------------------------------
//	repmat
//-----------------------------------------------------------------------------------------------------
vector<vector<double>> RepMat(vector<double> angle, int N)
{
	vector<vector<double>> angles;
	for (int i = 0; i < N; i++)
		angles.push_back(angle);
	return angles;
}

//-----------------------------------------------------------------------------------------------------
//	convert vector to mat opencv
//-----------------------------------------------------------------------------------------------------
Mat VectorToMat(vector<vector<double> > angles)
{
	Mat matAngles(angles.size(), angles.at(0).size(), CV_64FC1);
	for (int i = 0; i<matAngles.rows; ++i)
		for (int j = 0; j<matAngles.cols; ++j)
			matAngles.at<double>(i, j) = angles.at(i).at(j);
	return matAngles;
}


void convolveDFT(Mat& A, Mat& B, Mat& C)
{
	C.create(abs(A.rows - B.rows) + 1, abs(A.cols - B.cols) + 1, A.type());
	Size dftSize;
	// calculate the size of DFT transform
	dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
	dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);

	// allocate temporary buffers and initialize them with 0's
	Mat tempA(dftSize, A.type(), Scalar::all(0));
	Mat tempB(dftSize, B.type(), Scalar::all(0));

	// copy A and B to the top-left corners of tempA and tempB, respectively
	Mat roiA(tempA, Rect(0, 0, A.cols, A.rows));
	A.copyTo(roiA);
	Mat roiB(tempB, Rect(0, 0, B.cols, B.rows));
	B.copyTo(roiB);

	// now transform the padded A & B in-place;
	// use "nonzeroRows" hint for faster processing
	dft(tempA, tempA, 0, A.rows);
	dft(tempB, tempB, 0, B.rows);

	// multiply the spectrums;
	// the function handles packed spectrum representations well
	/*mulSpectrums(tempA, tempB, tempA);*/
	mulSpectrums(tempA, tempB, tempA, 0);

	// transform the product back from the frequency domain.
	// Even though all the result rows will be non-zero,
	// you need only the first C.rows of them, and thus you
	// pass nonzeroRows == C.rows
	dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);

	// now copy the result back to C.
	tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);
}


//-----------------------------------------------------------------------------------------------------
// Multiply 2 matrices element by element
//-----------------------------------------------------------------------------------------------------
Mat MultiplyMatrices(Mat& A, Mat& B)
{
	int Row = A.rows;
	int Col = A.cols;
	Mat C(Row, Col, CV_32FC2);
	for (int i = 0; i < Row; i++)
		for (int j = 0; j < Col; j++)
			C.at<double>(i, j) = A.at<double>(i, j)*B.at<double>(i, j);
	return C;
}


//-----------------------------------------------------------------------------------------------------
// Shift to center
//-----------------------------------------------------------------------------------------------------
void shift(Mat magI) {

	// crop if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                            // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                     // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}


Mat computeDFT(Mat image) {
	Mat padded;
	int m = getOptimalDFTSize(image.rows);
	int n = getOptimalDFTSize(image.cols);
	// create output image of optimal size
	copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
	// copy the source image, on the border add zero values
	Mat planes[] = { Mat_< float>(padded), Mat::zeros(padded.size(), CV_32F) };
	// create a complex matrix
	Mat complex;
	merge(planes, 2, complex);
	dft(complex, complex, DFT_COMPLEX_OUTPUT);  // fourier transform
	return complex;
}

void updateMag(Mat& complex) {
	Mat magI;
	Mat planes[] = {
		Mat::zeros(complex.size(), CV_32F),
		Mat::zeros(complex.size(), CV_32F)
	};
	split(complex, planes); // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], magI); // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
										   // switch to logarithmic scale: log(1 + magnitude)
	magI += Scalar::all(1);
	log(magI, magI);
	shift(magI); // rearrage quadrants
				 // Transform the magnitude matrix into a viewable image (float values 0-1)
	normalize(magI, magI, 1, 0, NORM_INF);
	imshow("spectrum", magI);
}

void updateResult(Mat& complex)
{
	Mat work;
	idft(complex, work);
	//  dft(complex, work, DFT_INVERSE + DFT_SCALE);
	Mat planes[] = { Mat::zeros(complex.size(), CV_32F), Mat::zeros(complex.size(), CV_32F) };
	split(work, planes);                // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))

	magnitude(planes[0], planes[1], work);    // === sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
	normalize(work, work, 0, 1, NORM_MINMAX);
	imshow("result", work);
}

void printMat(Mat& someMat)
{
	MatIterator_<double> _it = someMat.begin<double>();
	for (; _it != someMat.end<double>(); _it++) {
		std::cout << *_it << std::endl;
	}
}

//-----------------------------------------------------------------------------------------------------
// Apply Fourier Transform
//-----------------------------------------------------------------------------------------------------
void FourierTransform(Mat& src, Mat& dst)
{
	Mat I = src.clone();
	I.convertTo(I, CV_32F);
	Mat padded;								// expand input image to optimal size
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols);		// on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	// Make place for both the complex and the real values
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

										// Make the Discrete Fourier Transform
	dft(complexI, complexI, DFT_COMPLEX_OUTPUT);            // this way the result may fit in the source matrix

															// compute the magnitude and switch to logarithmic scale
															// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))

	Mat complexI_original = complexI.clone();

	// Transform the real and complex values to magnitude
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];

	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);

	// Crop and rearrange
	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
											// viewable image form (float between values 0 and 1).
	dst = magI.clone();

	imshow("spectrum magnitude", magI);
}


//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
void CreateFilter(Mat& src, Mat& dst)
{
	double N = src.cols;
	/*double Row = src.rows;*/


	vector<double> freqs = linspace(-1.0, 1.0, N);
	freqs = absvector(freqs);

	vector<vector<double>> freqs_rep = RepMat(freqs, 180);

	Mat freqs_mat = VectorToMat(freqs_rep);

	FourierTransform(freqs_mat, dst);

	//dst = freqs_mat;
}


//-----------------------------------------------------------------------------------------------------
// Ramp filter spatial domain
//-----------------------------------------------------------------------------------------------------
void CreateRampFilter(int kernalSize, Mat& filter)
{
	vector<double> freqs = linspace(-1.0, 1.0, kernalSize);
	freqs = absvector(freqs);
	vector<vector<double>> freqs_rep = RepMat(freqs, 1);
	Mat freqs_mat = VectorToMat(freqs_rep);
}


Mat createGausFilterMask(Size mask_size, int x, int y, int ksize, bool normalization, bool invert) {
	// Some corrections if out of bounds
	if (x < (ksize / 2)) {
		ksize = x * 2;
	}
	if (y < (ksize / 2)) {
		ksize = y * 2;
	}
	if (mask_size.width - x < ksize / 2) {
		ksize = (mask_size.width - x) * 2;
	}
	if (mask_size.height - y < ksize / 2) {
		ksize = (mask_size.height - y) * 2;
	}

	// call openCV gaussian kernel generator
	double sigma = -1;
	Mat kernelX = getGaussianKernel(ksize, sigma, CV_32F);
	Mat kernelY = getGaussianKernel(ksize, sigma, CV_32F);
	// create 2d gaus
	Mat kernel = kernelX * kernelY.t();
	// create empty mask
	Mat mask = Mat::zeros(mask_size, CV_32F);
	Mat maski = Mat::zeros(mask_size, CV_32F);

	// copy kernel to mask on x,y
	Mat pos(mask, Rect(x - ksize / 2, y - ksize / 2, ksize, ksize));
	kernel.copyTo(pos);

	// create mirrored mask
	Mat posi(maski, Rect((mask_size.width - x) - ksize / 2, (mask_size.height - y) - ksize / 2, ksize, ksize));
	kernel.copyTo(posi);
	// add mirrored to mask
	add(mask, maski, mask);

	// transform mask to range 0..1
	if (normalization) {
		normalize(mask, mask, 0, 1, NORM_MINMAX);
	}

	// invert mask
	if (invert) {
		mask = Mat::ones(mask.size(), CV_32F) - mask;
	}

	return mask;
}

//-----------------------------------------------------------------------------------------------------
// Apply Filter
//-----------------------------------------------------------------------------------------------------
void ApplyFilter(Mat& src, Mat& filter, Mat& dst)
{
	// Create Fourier transform of filter

	// Create Filter
	/*Mat filter;
	CreateFilter(src, filter);*/

	//printMat(src);
	//printf("\n");
	//printf("================ \n");
	//printMat(filter);

	// Apply Filter
	Mat filtered_projection;

	//mulSpectrums(src, filter, filtered_projection, DFT_ROWS); //only DFT_ROWS flag is accepted

	multiply(src, filter, filtered_projection);


	//convolveDFT(src, filter, filtered_projection);

	Mat inverseTransform;
	dft(filtered_projection, inverseTransform, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
	normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);

	dst = inverseTransform.clone();
}


//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
void ForwardRadonTransform(Mat& src, Mat &dst)
{
	double w = src.cols;
	double h = src.rows;
	// calculate the diagonal
	double d = sqrt(w*w + h*h);
	// find out how much to enlarge the image so that the original version can rotate inside without loss
	double dw = (d - w) / 2;
	double dh = (d - h) / 2;

	copyMakeBorder(src, src, dh, dh, dw, dw, cv::BORDER_CONSTANT, Scalar::all(0));
	// the size of the enlarged (square) image
	w = src.cols;
	h = src.rows;
	// This will put the result
	dst = Mat::zeros(h, w, CV_32FC1);


	// Center point of rotation
	Point center = Point(w / 2, h / 2);
	double angle = 0.0;
	double scale = 1.0;

	// Rotate the image
	double angleStep = 1;

	// Radon transformation (one line - one projection)
	Mat RadonImg(180.0 / angleStep, w, CV_32FC1);

	// Rotate the image, put the result in RadonImg
	for (int i = 0; i<RadonImg.rows; i++)
	{
		Mat rot_mat = getRotationMatrix2D(center, angle, scale);
		warpAffine(src, dst, rot_mat, dst.size());
		reduce(dst, RadonImg.row(i), 0, CV_REDUCE_SUM);
		angle += angleStep;
	}
	dst /= RadonImg.rows;
	dst = RadonImg.clone();
}
//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
void InverseRadonTransform(Mat& src, Mat &dst)
{
	src.convertTo(src, CV_32FC1);

	double d = src.cols;
	dst = Mat::zeros(d, d, CV_32FC1);
	Point center = Point(d / 2, d / 2);
	double angleStep = 1;
	double scale = 1.0;
	Mat rot_mat = getRotationMatrix2D(center, -angleStep, scale);
	for (int i = 0; i<src.rows; i++)
	{
		warpAffine(dst, dst, rot_mat, dst.size());
		for (int j = 0; j<dst.rows; j++)
		{
			dst.row(j) += src.row((src.rows - 1) - i);
		}
	}
	d = (d - d / (sqrt(2.0)))*0.55; // cut a little more
	double dw = d;
	double dh = d;
	dst = dst(Rect(dw, dh, dst.cols - 2 * dw, dst.rows - 2 * dh));
}

//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
	// Read and display image
	Mat res;
	Mat img = imread("C:\\Users\\RD\\Google Drive\\Working\\VS2015\\OpenCVTest\\OpenCVTest\\media\\ct.jpg", 0);
	resize(img, img, Size(), 0.1, 0.1);
	namedWindow("SourceImg");
	imshow("SourceImg", img);
	cout << img << endl;


	// Test Fourier Transform
	Mat dftInput1, dftImage1, inverseDFT, inverseDFTconverted;
	img.convertTo(dftInput1, CV_32F);
	dft(dftInput1, dftImage1, DFT_COMPLEX_OUTPUT);    // Applying DFT
	cout << "======" << endl;
	cout << "======" << endl;
	cout << dftImage1 << endl;


	// Reconstructing original imae from the DFT coefficients
	idft(dftImage1, inverseDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT
	inverseDFT.convertTo(inverseDFTconverted, CV_8U);
	imshow("Output", inverseDFTconverted);

	cout << "======" << endl;
	cout << "======" << endl;
	cout << inverseDFTconverted << endl;




	Mat res_print;
	ForwardRadonTransform(img, res);
	normalize(res, res_print, 0, 1, cv::NORM_MINMAX);
	imshow("RadonTransform", res_print);


	// Apply Fourier Transform and display magnitude
	Mat fourierimg;
	FourierTransform(img, fourierimg);
	imshow("Spectrum magnitude", fourierimg);






	InverseRadonTransform(res_print, img);

	normalize(img, img, 0, 1, cv::NORM_MINMAX);
	imshow("ReconstructedImage", img);

	waitKey(0);

	return 0;
}
