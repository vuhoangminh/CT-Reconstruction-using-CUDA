//#include <cuda.h>
//#include <device_functions.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//
//#include <cufft.h>
//
//#include <iostream>
//#include <time.h>
//
//
//#include <stdio.h>
//
//#include <vector>
//#include <stdio.h>
//#include <stdarg.h>
//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/core/cuda.hpp"
//#include "fstream"
//#include "iostream"
//#include <complex>
//#include <cmath>
//
////#include "opencv2/cudaarithm.hpp"
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/transform.h>
//#include <thrust/functional.h>
//#include <cstdlib>
//
//
//using namespace std;
//using namespace cv;
//
//
//
//#define PI 3.14159265
//#define SIZE 2
//#define TILE_DIM 16
//#define WIDTHPERBLOCK 32
//#define BLOCK_ROWS 8
//#define MATRIXSIZE 128
//#define MATRIXAREA MATRIXSIZE*MATRIXSIZE
//#define IMAGEWIDTH round(MATRIXSIZE/sqrt(2))
//
//__shared__ float As[WIDTHPERBLOCK][WIDTHPERBLOCK];
//__shared__ float Bs[WIDTHPERBLOCK][WIDTHPERBLOCK];
//
//extern __shared__ float a[];
//extern __shared__ float b[];
//
//
//
//
//
//void MatAddThrust(std::vector<float> & c, std::vector<float> a, std::vector<float> b, unsigned int size)
//{
//	thrust::device_vector<float> X(a.begin(), a.end());
//	thrust::device_vector<float> Y(b.begin(), b.end());
//	thrust::device_vector<float> Z(size);
//	thrust::transform(X.begin(), X.end(), Y.begin(), Z.begin(), thrust::plus<float>());
//	thrust::copy(Z.begin(), Z.end(), c.begin());
//}
//
//
//__global__ void MatAddWithoutSharedMemory(float A[], float B[], float C[], int m, int n) {
//	/* blockDim.x = threads_per_block                            */
//	/* First block gets first threads_per_block components.      */
//	/* Second block gets next threads_per_block components, etc. */
//	int idx = blockDim.x * blockIdx.x + threadIdx.x;
//
//	/* The test shouldn't be necessary */
//	if (blockIdx.x < m && threadIdx.x < n)
//		C[idx] = A[idx] + B[idx];
//}
//
//
//
//
//
//void Print_matrix(char title[], float A[], int m, int n) {
//	int i, j;
//
//	printf("%s\n", title);
//	for (i = 0; i < m; i++) {
//		for (j = 0; j < n; j++)
//			printf("%.1f ", A[i*n + j]);
//		printf("\n");
//	}
//}  /* Print_matrix */
//
//   //-----------------------------------------------------------------------------------------------------
//   // Shift to center
//   //-----------------------------------------------------------------------------------------------------
//void shiftDFT(Mat& fImage)
//{
//	Mat tmp, q0, q1, q2, q3;
//
//	// first crop the image, if it has an odd number of rows or columns
//
//	fImage = fImage(Rect(0, 0, fImage.cols & -2, fImage.rows & -2));
//
//	int cx = fImage.cols / 2;
//	int cy = fImage.rows / 2;
//
//	// rearrange the quadrants of Fourier image
//	// so that the origin is at the image center
//
//	q0 = fImage(Rect(0, 0, cx, cy));
//	q1 = fImage(Rect(cx, 0, cx, cy));
//	q2 = fImage(Rect(0, cy, cx, cy));
//	q3 = fImage(Rect(cx, cy, cx, cy));
//
//	q0.copyTo(tmp);
//	q3.copyTo(q0);
//	tmp.copyTo(q3);
//
//	q1.copyTo(tmp);
//	q2.copyTo(q1);
//	tmp.copyTo(q2);
//}
//
///******************************************************************************/
//// return a floating point spectrum magnitude image scaled for user viewing
//// complexImg- input dft (2 channel floating point, Real + Imaginary fourier image)
//// rearrange - perform rearrangement of DFT quadrants if true
//
//// return value - pointer to output spectrum magnitude image scaled for user viewing
//
//Mat CreateDisplaySpectrum(Mat& complexImg, bool rearrange)
//{
//	Mat planes[2];
//
//	// compute magnitude spectrum (N.B. for display)
//	// compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
//
//	split(complexImg, planes);
//	magnitude(planes[0], planes[1], planes[0]);
//
//	Mat mag = (planes[0]).clone();
//	mag += Scalar::all(1);
//	log(mag, mag);
//
//	if (rearrange)
//	{
//		// re-arrange the quaderants
//		shiftDFT(mag);
//	}
//
//	cv::normalize(mag, mag, 0, 1, CV_MINMAX);
//
//	return mag;
//
//}
//
//
////-----------------------------------------------------------------------------------------------------
//// Create Ramp Filter
////-----------------------------------------------------------------------------------------------------
//void CreateRampFilter(Mat& dft_Filter, double a)
//{
//	Mat tmp = Mat(dft_Filter.rows, dft_Filter.cols, CV_32F);
//	int N = dft_Filter.cols;
//	int M = dft_Filter.rows;
//	double step;
//	step = 2 * a / (N - 1);
//
//	for (int i = 0; i < dft_Filter.rows; i++)
//	{
//		for (int j = 0; j < dft_Filter.cols; j++)
//		{
//			tmp.at<float>(i, j) = abs(-a + j*step);
//		}
//	}
//
//	Mat abc = Mat::zeros(tmp.size(), CV_32F);
//
//
//	Mat toMerge[] = { tmp, abc };
//	merge(toMerge, 2, dft_Filter);
//}
//
//
////-----------------------------------------------------------------------------------------------------
//// Create Hann Filter and Ramp Filter
////-----------------------------------------------------------------------------------------------------
//void CreateHannFilter(Mat& dft_Filter, double a)
//{
//	Mat tmp = Mat(dft_Filter.rows, dft_Filter.cols, CV_32F);
//	int N = dft_Filter.cols; // length
//	int M = dft_Filter.rows; // 180% radon
//	double step, ramp, hann;
//	step = 2 * a / (N - 1);
//
//	for (int i = 0; i < dft_Filter.rows; i++)
//	{
//		for (int j = 0; j < dft_Filter.cols; j++)
//		{
//			ramp = abs(-a + j*step);
//			hann = sin(j*PI / (N - 1));
//			hann = pow(hann, 2.0);
//			tmp.at<float>(i, j) = ramp*hann;
//		}
//	}
//
//
//
//	Mat abc = Mat::zeros(tmp.size(), CV_32F);
//
//
//	Mat toMerge[] = { tmp, abc };
//	merge(toMerge, 2, dft_Filter);
//}
//
//
//
////-----------------------------------------------------------------------------------------------------
//// Apply Filter
////-----------------------------------------------------------------------------------------------------
//void ApplyFilter(Mat& img, string FilterMethod, Mat& filterOutput)
//{
//	Mat imgGray, imgOutput, filterOutput_dis;	// image object(s)
//
//	Mat padded;		// fourier image objects and arrays
//	Mat complexImg, filter;
//	Mat planes[2], mag;
//
//
//	int N, M; // fourier image sizes
//
//	int radius = 30;				// low pass filter parameter
//	int order = 2;				// low pass filter parameter
//
//	const string originalName = "Input Image (grayscale)"; // window name
//	const string spectrumMagName = "Magnitude Image (log transformed)"; // window name
//	const string lowPassName = "Ramp Filtered (grayscale)"; // window name
//	const string filterName = "Filter Image"; // window nam
//
//											  // setup the DFT image sizes
//
//	M = getOptimalDFTSize(img.rows);
//	N = getOptimalDFTSize(img.cols);
//
//	// convert input to grayscale if RGB
//	if (img.channels() == 3)
//		cvtColor(img, imgGray, CV_BGR2GRAY);
//	else
//		imgGray = img.clone();
//
//	// setup the DFT images
//	copyMakeBorder(imgGray, padded, 0, M - imgGray.rows, 0,
//		N - imgGray.cols, BORDER_CONSTANT, Scalar::all(0));
//	planes[0] = Mat_<float>(padded);
//	planes[1] = Mat::zeros(padded.size(), CV_32F);
//	merge(planes, 2, complexImg);
//
//	// do the DFT
//	/*int start_s = clock();*/
//
//	dft(complexImg, complexImg);
//
//	/*int stop_s = clock();
//	std::cout << "===================================================================" << endl;
//	std::cout << "Without CUDA FFT" << endl;
//	std::cout << "time: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) << " seconds" << endl;*/
//
//
//	// construct the filter (same size as complex image)
//	Mat ramp_filter, hann_filter;
//	filter = complexImg.clone();
//	ramp_filter = filter.clone();
//	hann_filter = filter.clone();
//	CreateRampFilter(ramp_filter, 1.0);
//	CreateHannFilter(hann_filter, 1.0);
//
//	//cout << "====================" << endl;
//	//cout << "Ramp" << endl;
//	//cout << ramp_filter << endl;
//
//	//cout << "====================" << endl;
//	//cout << "Hann" << endl;
//	//cout << hann_filter << endl;
//
//	/*if (FilterMethod == "Ramp")
//	filter = ramp_filter.clone();
//	else*/
//	/*filter = hann_filter.clone();*/
//
//	filter = ramp_filter.clone();
//
//	// apply filter
//	shiftDFT(complexImg);
//	mulSpectrums(complexImg, filter, complexImg, 1);
//	/*multiply(complexImg, filter, complexImg);*/
//	shiftDFT(complexImg);
//
//	// create magnitude spectrum for display
//	mag = CreateDisplaySpectrum(complexImg, true);
//
//	// do inverse DFT on filtered image
//	idft(complexImg, complexImg);
//
//	// split into planes and extract plane 0 as output image
//	split(complexImg, planes);
//	cv::normalize(planes[0], imgOutput, 0, 1, CV_MINMAX);
//
//	// do the same with the filter image
//	split(filter, planes);
//	cv::normalize(planes[0], filterOutput_dis, 0, 1, CV_MINMAX);
//
//	filterOutput = imgOutput.clone();
//
//
//	// ***
//
//	// display image in window
//
//	Mat rgb = img.clone();
//	cv::normalize(rgb, rgb, 0, 1, cv::NORM_MINMAX);
//	//imshow("rgb", rgb);
//	//imshow(originalName, imgGray);
//	cv::imshow(spectrumMagName, mag);
//	cv::imshow(lowPassName, imgOutput);
//	cv::imshow(filterName, filterOutput_dis);
//}
//
//
//
//
////-----------------------------------------------------------------------------------------------------
//// Forward Radon Transform
////-----------------------------------------------------------------------------------------------------
//void ForwardRadonTransform(Mat& src, Mat &dst)
//{
//	double w = src.cols;
//	double h = src.rows;
//	// calculate the diagonal
//	double d = sqrt(w*w + h*h);
//	// find out how much to enlarge the image so that the original version can rotate inside without loss
//	double dw = (d - w) / 2;
//	double dh = (d - h) / 2;
//
//	copyMakeBorder(src, src, dh, dh, dw, dw, cv::BORDER_CONSTANT, Scalar::all(0));
//	// the size of the enlarged (square) image
//	w = src.cols;
//	h = src.rows;
//	// This will put the result
//	dst = Mat::zeros(h, w, CV_32FC1);
//
//
//	// Center point of rotation
//	Point center = Point(w / 2, h / 2);
//	double angle = 0.0;
//	double scale = 1.0;
//
//	// Rotate the image
//	double angleStep = 1;
//
//	// Radon transformation (one line - one projection)
//	Mat RadonImg(180.0 / angleStep, w, CV_32FC1);
//
//	// Rotate the image, put the result in RadonImg
//	for (int i = 0; i < RadonImg.rows; i++)
//	{
//		Mat rot_mat = getRotationMatrix2D(center, angle, scale);
//		warpAffine(src, dst, rot_mat, dst.size());
//		reduce(dst, RadonImg.row(i), 0, CV_REDUCE_SUM);
//		angle += angleStep;
//	}
//	dst /= RadonImg.rows;
//	dst = RadonImg.clone();
//}
//
//
//void PrintArray(uchar* array)
//{
//	int numElements = (sizeof(array) / sizeof(*array));
//	for (int i = 0; i < numElements; i++)
//		cout << array[i] << " ";
//	cout << endl;
//}
//
//void PrintVector(vector<float> path)
//{
//	// ...
//	for (std::vector<float>::const_iterator i = path.begin(); i != path.end(); ++i)
//		std::cout << *i << ' ';
//}
//
//
//vector<float> ConvertMat2Vector(Mat & mat)
//{
//	std::vector<float> array;
//	if (mat.isContinuous()) {
//		array.assign((float*)mat.datastart, (float*)mat.dataend);
//	}
//	else {
//		for (int i = 0; i < mat.rows; ++i) {
//			array.insert(array.end(), (float*)mat.ptr<float>(i), (float*)mat.ptr<float>(i) + mat.cols);
//		}
//	}
//	return array;
//}
//
//
//vector<float> ConcatRow2Vector(Mat & mat, int numRow)
//{
//	std::vector<float> array;
//	for (int i = 0; i < mat.cols; ++i) {
//		array.insert(array.end(), (float*)mat.ptr<float>(numRow), (float*)mat.ptr<float>(numRow) + mat.cols);
//	}
//	return array;
//}
//
