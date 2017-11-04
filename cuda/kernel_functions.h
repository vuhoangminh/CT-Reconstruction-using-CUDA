// ====================================================================================================
// liba for cuda
#include <cuda.h>
#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

// libs common
#include <iostream>
#include <time.h>
#include <stdio.h>
#include <vector>
#include <stdio.h>
#include <stdarg.h>
#include "fstream"
#include "iostream"
#include <complex>
#include <cmath>
#include <cstdlib>

// libs for opencv
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/cuda.hpp"

// namespace
using namespace std;
using namespace cv;

// constants
#define PI 3.14159265
#define SIZE 2
#define TILE_DIM 16
#define WIDTHPERBLOCK 32
#define BLOCK_ROWS 8
#define MATRIXWIDTH 128
#define MATRIXAREA MATRIXWIDTH*MATRIXWIDTH
#define IMAGEWIDTH round(MATRIXWIDTH/sqrt(2))

// shred memory
__shared__ float As[WIDTHPERBLOCK][WIDTHPERBLOCK];
__shared__ float Bs[WIDTHPERBLOCK][WIDTHPERBLOCK];
extern __shared__ float a[];
extern __shared__ float b[];


// ====================================================================================================
// Thrust
vector<float> ConvertMat2Vector(Mat & im_in);
vector<float> ConcatRow2Vector(Mat & im_in, int num_row);
void MatAddThrust(std::vector<float> & v_out, std::vector<float> v_in1, std::vector<float> v_in2, unsigned int v_size);

// DFT and Forward Radon Transform
void ShiftDftToCenter(Mat & im_in);
Mat CreateDisplaySpectrum(Mat & complexImg, bool rearrange);
void CreateRampFilter(Mat & ramp_filter, double a);
void CreateHannFilter(Mat & hann_filter, double a);
void ApplyFilter(Mat & im_in, string filter_method, Mat & im_out);
void ForwardRadonTransform(Mat & im_in, Mat & im_out);
void PerformCuFFT();

// Add without shared memory
__global__ void MatAddWithoutSharedMemory(float A[], float B[], float C[], int m, int n);

// Add with shared memory
__global__ void MatAddSharedMemory(float A[], float B[], float C[], int m, int n);

// Methods
void InverseRadonTransformOpenCV(Mat & im_in, Mat & im_out);
void InverseRadonTransformArray(Mat & im_in, Mat & im_out);
void InverseRadonTransformCuda(Mat & im_in, Mat & im_out);
void InverseRadonTransformCudaSharedMemory(Mat & im_in, Mat & im_out);
void InverseRadonTransformCudaThrust(Mat & im_in, Mat & im_out);
void PerformRadonAndInverseRadon(string method);

// Addition, Cuda while multiplying 2 complex matrices
void ApplyFilterCuda(Mat & im_in, string filter_method, Mat & im_out);
__global__ void MatMulPixelwiseWithoutSharedMemory(float A_r[], float A_c[],
	float B_r[], float B_c[], float C_r[], float C_c[], int m, int n);
void MultiplySpectrums(Mat & planes_r, Mat & planes_c,
	Mat & planes_filter_r, Mat & planes_filter_c, Mat & img_complex);


// ====================================================================================================
int main()
{

	string method;
	
	
	method = "cudathrust";
	PerformRadonAndInverseRadon(method);
	method = "cudanoshare";
	PerformRadonAndInverseRadon(method);
	method = "cudashare";
	PerformRadonAndInverseRadon(method);
	method = "array";
	PerformRadonAndInverseRadon(method);
	method = "opencv";
	PerformRadonAndInverseRadon(method);


	waitKey(0);
	return 0;
}



// ====================================================================================================
//-----------------------------------------------------------------------------------------------------
/*
	Function: MatAddThrust
	Add 2 vectors using Thrust library
	Parameters:
		v_out			-	result vector
		v_in1			-	vector 1
		v_in2			-	vector 2
		v_size			-	vector size
*/
//-----------------------------------------------------------------------------------------------------
void MatAddThrust(std::vector<float> & v_out, std::vector<float> v_in1, std::vector<float> v_in2, unsigned int v_size)
{
	// Init vector in thrust
	thrust::device_vector<float> X(v_in1.begin(), v_in1.end());
	thrust::device_vector<float> Y(v_in2.begin(), v_in2.end());
	thrust::device_vector<float> Z(v_size);

	// Make adition
	thrust::transform(X.begin(), X.end(), Y.begin(), Z.begin(), thrust::plus<float>());

	// Copy data back
	thrust::copy(Z.begin(), Z.end(), v_out.begin());
}


//-----------------------------------------------------------------------------------------------------
/*
	Function: MatAddWithoutSharedMemory
	Add 2 matrices without using shared memory
	Parameters:
		A			-	Matrix input 1
		B			-	Matrix input 2
		C			-	Matrix output
		m			-	Matrix row
		n			-	Matrix column
*/
//-----------------------------------------------------------------------------------------------------
__global__ void MatAddWithoutSharedMemory(float A[], float B[], float C[], int m, int n) {
	// Get index
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	/* The test shouldn't be necessary */
	if (blockIdx.x < m && threadIdx.x < n)
		C[idx] = A[idx] + B[idx];
}


//-----------------------------------------------------------------------------------------------------
/*
	Function: MatAddSharedMemory
	Add 2 matrices using shared memory
	Parameters:
		A			-	Matrix input 1
		B			-	Matrix input 2
		C			-	Matrix output
		m			-	Matrix row
		n			-	Matrix column
*/
//-----------------------------------------------------------------------------------------------------
__global__ void MatAddSharedMemory(float A[], float B[], float C[], int m, int n)
{
	// Get index
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Copy data to shared memory
	a[idx] = A[idx];
	__syncthreads();
	b[idx] = B[idx];
	__syncthreads();

	/* The test shouldn't be necessary */
	if (blockIdx.x < m && threadIdx.x < n)
	{
		C[idx] = a[idx] + b[idx];
		__syncthreads();
	}
}


//-----------------------------------------------------------------------------------------------------
/*
	Function: ShiftDftToCenter
	Shift DFT to the center
	Parameters:
		im_in		-	input image
*/
//-----------------------------------------------------------------------------------------------------
void ShiftDftToCenter(Mat & im_in)
{
	// init vars
	Mat tmp, q0, q1, q2, q3;

	// first crop the image, if it has an odd number of rows or columns
	im_in = im_in(Rect(0, 0, im_in.cols & -2, im_in.rows & -2));

	// get size of image
	int cx = im_in.cols / 2;
	int cy = im_in.rows / 2;

	// rearrange the quadrants of Fourier image
	// so that the origin is at the image center
	q0 = im_in(Rect(0, 0, cx, cy));
	q1 = im_in(Rect(cx, 0, cx, cy));
	q2 = im_in(Rect(0, cy, cx, cy));
	q3 = im_in(Rect(cx, cy, cx, cy));

	// copy back
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	// shift
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}


//-----------------------------------------------------------------------------------------------------
/*
	Function: CreateDisplaySpectrum
	Create image to display the spectrum
	Parameters:
		complexImg		-	input dft (2 channel floating point, Real + Imaginary fourier image)
		rearrange		-	perform rearrangement of DFT quadrants if true
	Returns:
		pointer to output spectrum magnitude image scaled for user viewing
*/
//-----------------------------------------------------------------------------------------------------
Mat CreateDisplaySpectrum(Mat & complexImg, bool rearrange)
{
	// init vars
	Mat planes[2];

	// split plane
	split(complexImg, planes);
	magnitude(planes[0], planes[1], planes[0]);

	// for magnitude
	Mat mag = (planes[0]).clone();
	mag += Scalar::all(1);
	log(mag, mag);

	// shift to center
	if (rearrange)
	{
		ShiftDftToCenter(mag);
	}

	// normalize image
	cv::normalize(mag, mag, 0, 1, CV_MINMAX);

	// get result
	return mag;
}


//-----------------------------------------------------------------------------------------------------
/*
	Function: CreateRampFilter
	Create Ramp Filter
	Parameters:
		ramp_filter		-	ramp filter
		a				-	threshold
*/
//-----------------------------------------------------------------------------------------------------
void CreateRampFilter(Mat & ramp_filter, double a)
{
	// init image blocks
	Mat tmp = Mat(ramp_filter.rows, ramp_filter.cols, CV_32F);

	// get row and col of image
	int N = ramp_filter.cols;
	int M = ramp_filter.rows;

	// init vars
	double step;
	step = 2 * a / (N - 1);

	// make filter
	for (int i = 0; i < ramp_filter.rows; i++)
	{
		for (int j = 0; j < ramp_filter.cols; j++)
		{
			tmp.at<float>(i, j) = abs(-a + j*step);
		}
	}

	// patch real part with 0
	Mat abc = Mat::zeros(tmp.size(), CV_32F);

	// merge real and complex parts
	Mat toMerge[] = { tmp, abc };
	merge(toMerge, 2, ramp_filter);
}


//-----------------------------------------------------------------------------------------------------
/*
	Function: CreateHannFilter
	Create Hann Filter
	Parameters:
		hann_filter		-	hann filter
		a				-	threshold
*/
//-----------------------------------------------------------------------------------------------------
void CreateHannFilter(Mat & hann_filter, double a)
{
	// init image blocks
	Mat tmp = Mat(hann_filter.rows, hann_filter.cols, CV_32F);

	// get row and col of image
	int N = hann_filter.cols; // length
	int M = hann_filter.rows; // 180% radon

	// init vars
	double step, ramp, hann;
	step = 2 * a / (N - 1);

	// make filter
	for (int i = 0; i < hann_filter.rows; i++)
	{
		for (int j = 0; j < hann_filter.cols; j++)
		{
			ramp = abs(-a + j*step);
			hann = sin(j*PI / (N - 1));
			hann = pow(hann, 2.0);
			tmp.at<float>(i, j) = ramp*hann;
		}
	}

	// patch real part with 0
	Mat abc = Mat::zeros(tmp.size(), CV_32F);

	// merge real and complex parts
	Mat toMerge[] = { tmp, abc };
	merge(toMerge, 2, hann_filter);
}



//-----------------------------------------------------------------------------------------------------
/*
	Function: ApplyFilter
	Apply filter to sinogram
	Parameters:
		im_in			-	input image
		filter_method	-	filter method
		im_out			-	output image
*/
//-----------------------------------------------------------------------------------------------------
void ApplyFilter(Mat & im_in, string filter_method, Mat & im_out)
{
	// Init images objects
	Mat img_gray, img_output, img_output_display;
	Mat img_padded;	
	Mat img_complex, filter;
	Mat planes[2], img_magnitude;

	// Fourier image size
	int N, M;

	// String constant for image title
	const string originalName = "Input Image (grayscale)"; // window name
	const string spectrumMagName = "Magnitude Image (log transformed)"; // window name
	const string lowPassName = "Ramp Filtered (grayscale)"; // window name
	const string filterName = "Filter Image"; // window nam

	// setup the DFT image sizes
	M = getOptimalDFTSize(im_in.rows);
	N = getOptimalDFTSize(im_in.cols);

	// convert input to grayscale if RGB
	if (im_in.channels() == 3)
		cvtColor(im_in, img_gray, CV_BGR2GRAY);
	else
		img_gray = im_in.clone();

	// setup the DFT images
	copyMakeBorder(img_gray, img_padded, 0, M - img_gray.rows, 0,
		N - img_gray.cols, BORDER_CONSTANT, Scalar::all(0));
	planes[0] = Mat_<float>(img_padded);
	planes[1] = Mat::zeros(img_padded.size(), CV_32F);
	merge(planes, 2, img_complex);

	// do the DFT
	dft(img_complex, img_complex);

	// construct the filter (same size as complex image)
	Mat ramp_filter, hann_filter;
	filter = img_complex.clone();
	ramp_filter = filter.clone();
	hann_filter = filter.clone();
	CreateRampFilter(ramp_filter, 1.0);
	CreateHannFilter(hann_filter, 1.0);

	// select filter to apply
	filter = ramp_filter.clone();

	// apply filter
	ShiftDftToCenter(img_complex);

	// HERE! WE CAN PARALLELIZE THE MULTIPLICATION OF 2 COMPLEX MATRICES
	mulSpectrums(img_complex, filter, img_complex, 1);

	// shift dft to the center
	ShiftDftToCenter(img_complex);

	// create magnitude spectrum for display
	img_magnitude = CreateDisplaySpectrum(img_complex, true);

	// do inverse DFT on filtered image
	idft(img_complex, img_complex);

	// split into planes and extract plane 0 as output image
	split(img_complex, planes);
	cv::normalize(planes[0], img_output, 0, 1, CV_MINMAX);

	// do the same with the filter image
	split(filter, planes);
	cv::normalize(planes[0], img_output_display, 0, 1, CV_MINMAX);

	// save image output
	im_out = img_output.clone();

	// show result
	Mat rgb = im_in.clone();
	cv::normalize(rgb, rgb, 0, 1, cv::NORM_MINMAX);
	cv::imshow(spectrumMagName, img_magnitude);
	cv::imshow(lowPassName, img_output);
	cv::imshow(filterName, img_output_display);

	// Write result
	Mat img8bit;
	img_output_display.convertTo(img8bit, CV_8UC1, 255.0);
	imwrite( "filtered_sinogram.jpg", img8bit);
}


//-----------------------------------------------------------------------------------------------------
/*
	Function: ForwardRadonTransform
	Forward Radon Transform
	Parameters:
		im_in			-	input image for forward projection
		im_out			-	sinogram
*/
//-----------------------------------------------------------------------------------------------------
void ForwardRadonTransform(Mat & im_in, Mat & im_out)
{
	// Init
	double w = im_in.cols;
	double h = im_in.rows;

	// calculate the diagonal
	double d = sqrt(w*w + h*h);

	// find out how much to enlarge the image so that the original version can rotate inside without loss
	double dw = (d - w) / 2;
	double dh = (d - h) / 2;

	// make image wider
	copyMakeBorder(im_in, im_in, dh, dh, dw, dw, cv::BORDER_CONSTANT, Scalar::all(0));

	// the size of the enlarged (square) image
	w = im_in.cols;
	h = im_in.rows;

	// This will put the result
	im_out = Mat::zeros(h, w, CV_32FC1);

	// Center point of rotation
	Point center = Point(w / 2, h / 2);
	double angle = 0.0;
	double scale = 1.0;

	// Rotate the image
	double angleStep = 1;

	// Radon transformation (one line - one projection)
	Mat RadonImg(180.0 / angleStep, w, CV_32FC1);

	// Rotate the image, put the result in RadonImg
	for (int i = 0; i < RadonImg.rows; i++)
	{
		Mat rot_mat = getRotationMatrix2D(center, angle, scale);
		warpAffine(im_in, im_out, rot_mat, im_out.size());
		reduce(im_out, RadonImg.row(i), 0, CV_REDUCE_SUM);
		angle += angleStep;
	}

	// Average radon
	im_out /= RadonImg.rows;
	im_out = RadonImg.clone();
}


//-----------------------------------------------------------------------------------------------------
/*
	Function: ConvertMat2Vector
	Convert Mat image to vector for Thrust
	Parameters:
		im_in			-	input image for forward projection
	Returns:
		vector stores data of image
*/
//-----------------------------------------------------------------------------------------------------
vector<float> ConvertMat2Vector(Mat & im_in)
{
	// Init vector
	std::vector<float> array;

	// Convert to vector
	if (im_in.isContinuous()) {
		array.assign((float*)im_in.datastart, (float*)im_in.dataend);
	}
	else {
		for (int i = 0; i < im_in.rows; ++i) {
			array.insert(array.end(), (float*)im_in.ptr<float>(i), (float*)im_in.ptr<float>(i) + im_in.cols);
		}
	}
	return array;
}


//-----------------------------------------------------------------------------------------------------
/*
	Function: ConcatRow2Vector
	Concat Mat image using only 1 row to vector for Thrust
	Parameters:
		im_in			-	input image for forward projection
		num_row			-	row number to concat
	Returns:
		vector stores data of concat image
*/
//-----------------------------------------------------------------------------------------------------
vector<float> ConcatRow2Vector(Mat & im_in, int num_row)
{
	// Init vector
	std::vector<float> array;

	// Copy specific row and concat it into a predefine matrix
	for (int i = 0; i < im_in.cols; ++i) {
		array.insert(array.end(), (float*)im_in.ptr<float>(num_row), (float*)im_in.ptr<float>(num_row) + im_in.cols);
	}
	return array;
}


//-----------------------------------------------------------------------------------------------------
/*
	Function: InverseRadonTransformOpenCV
	Inverse radon transform using OpenCV
	Parameters:
		im_in			-	filtered sinogram
		im_out			-	reconstructed image
*/
//-----------------------------------------------------------------------------------------------------
void InverseRadonTransformOpenCV(Mat & im_in, Mat & im_out)
{
	// Init
	im_in.convertTo(im_in, CV_32FC1);
	double d = im_in.cols;
	im_out = Mat::zeros(d, d, CV_32FC1);

	// Center point
	Point center = Point(d / 2, d / 2);

	// Angle step rotation
	double angleStep = 1;

	// Scale 
	double scale = 1.0;

	// Rotation matrix
	Mat rot_mat = getRotationMatrix2D(center, -angleStep, scale);
	
	// Start back-projection
	for (int i = 0; i < im_in.rows; i++)
	{
		// Get rotated matrix
		warpAffine(im_out, im_out, rot_mat, im_out.size());

		// Add matrices using OpenCV
		for (int j = 0; j < im_out.rows; j++)
		{
			im_out.row(j) += im_in.row((im_in.rows - 1) - i);
		}
	}

	// Crop image
	d = (d - d / (sqrt(2.0)))*0.55; // cut a little more
	double dw = d;
	double dh = d;
	im_out = im_out(Rect(dw, dh, im_out.cols - 2 * dw, im_out.rows - 2 * dh));
}


//-----------------------------------------------------------------------------------------------------
/*
	Function: InverseRadonTransformArray
	Inverse radon transform using array addition
	Parameters:
		im_in			-	filtered sinogram
		im_out			-	reconstructed image
*/
//-----------------------------------------------------------------------------------------------------
void InverseRadonTransformArray(Mat & im_in, Mat & im_out)
{
	// Init
	im_in.convertTo(im_in, CV_32FC1);
	double d = im_in.cols;
	im_out = Mat::zeros(d, d, CV_32FC1);

	// Center point
	Point center = Point(d / 2, d / 2);

	// Angle step rotation
	double angleStep = 1;

	// Scale 
	double scale = 1.0;

	// Rotation matrix
	Mat rot_mat = getRotationMatrix2D(center, -angleStep, scale);
	
	// Start back-projection
	for (int i = 0; i < im_in.rows; i++)
	{
		// Get rotated matrix
		warpAffine(im_out, im_out, rot_mat, im_out.size());

		// Get row and col of matrix
		size_t row = im_out.rows, col = im_out.rows;
		size_t size_matrix = row*col;

		// Init arrays
		float *a = new float[size_matrix];
		float *b = new float[size_matrix];
		float *c = new float[size_matrix];

		// Copy data to array
		for (int j = 0; j < row; j++)
			for (int k = 0; k < col; k++)
			{
				a[j*col + k] = im_out.at<float>(j, k);
				b[j*col + k] = im_in.at<float>(im_in.rows - 1 - i, k);
			}

		// Add 2 arrays
		for (size_t j = 0; j < size_matrix; j++)
			c[j] = a[j] + b[j];

		// Copy result to image
		for (int j = 0; j < row; j++)
			for (int k = 0; k < col; k++)
			{
				im_out.at<float>(j, k) = c[j*col + k];
			}
	}

	// Crop image
	d = (d - d / (sqrt(2.0)))*0.55; // cut a little more
	double dw = d;
	double dh = d;
	im_out = im_out(Rect(dw, dh, im_out.cols - 2 * dw, im_out.rows - 2 * dh));
}


//-----------------------------------------------------------------------------------------------------
/*
	Function: InverseRadonTransformCuda
	Inverse radon transform using Cuda without using shared memory
	Parameters:
		im_in			-	filtered sinogram
		im_out			-	reconstructed image
*/
//-----------------------------------------------------------------------------------------------------
void InverseRadonTransformCuda(Mat & im_in, Mat & im_out)
{
	// Init
	im_in.convertTo(im_in, CV_32FC1);
	double d = im_in.cols;
	im_out = Mat::zeros(d, d, CV_32FC1);

	// Center point
	Point center = Point(d / 2, d / 2);

	// Angle step rotation
	double angleStep = 1;

	// Scale 
	double scale = 1.0;

	// Rotation matrix
	Mat rot_mat = getRotationMatrix2D(center, -angleStep, scale);
	
	// Start back-projection
	int start_s_cop = 0;
	for (int i = 0; i < im_in.rows; i++)
	{
		// Get rotated matrix
		warpAffine(im_out, im_out, rot_mat, im_out.size());

		// Init
		int m, n;
		float *h_A, *h_B, *h_C;
		float *d_A, *d_B, *d_C;
		size_t size;

		// Size of image
		m = im_out.rows;
		n = im_out.cols;
		size = m*n * sizeof(float);

		// Allocate host memory
		h_A = (float*)malloc(size);
		h_B = (float*)malloc(size);
		h_C = (float*)malloc(size);

		// Start clocking for copying data
		int start_s = clock();

		// Transfer data to host var
		for (int j = 0; j < m; j++)
			for (int k = 0; k < n; k++)
			{
				h_A[j*n + k] = im_out.at<float>(j, k);
				h_B[j*n + k] = im_in.at<float>(im_in.rows - 1 - i, k);
			}

		// Stop cloking for copying data
		int stop_s = clock();

		// Stack timing
		start_s_cop = start_s_cop + stop_s - start_s;

		// Allocate matrices in device memory
		cudaMalloc(&d_A, size);
		cudaMalloc(&d_B, size);
		cudaMalloc(&d_C, size);

		// Copy matrices from host memory to device memory
		cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

		// Kernel matrix add
		MatAddWithoutSharedMemory << <MATRIXWIDTH, MATRIXWIDTH >> > (d_A, d_B, d_C, m, n);
		
		// Wait for the kernel to complete
		cudaThreadSynchronize();

		// Copy result from device memory to host memory
		cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

		// Copy result from host memory to local var
		for (int j = 0; j < m; j++)
			for (int k = 0; k < n; k++)
			{
				im_out.at<float>(j, k) = h_C[j*n + k];
			}

		// Free host memory
		free(h_A);
		free(h_B);
		free(h_C);

		// Free device memory
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
	}

	// Crop image
	d = (d - d / (sqrt(2.0)))*0.55; // cut a little more
	double dw = d;
	double dh = d;
	im_out = im_out(Rect(dw, dh, im_out.cols - 2 * dw, im_out.rows - 2 * dh));
}


//-----------------------------------------------------------------------------------------------------
/*
	Function: InverseRadonTransformCudaSharedMemory
	Inverse radon transform using Cuda with using shared memory
	Parameters:
		im_in			-	filtered sinogram
		im_out			-	reconstructed image
*/
//-----------------------------------------------------------------------------------------------------
void InverseRadonTransformCudaSharedMemory(Mat & im_in, Mat & im_out)
{
	// Init
	im_in.convertTo(im_in, CV_32FC1);
	double d = im_in.cols;
	im_out = Mat::zeros(d, d, CV_32FC1);

	// Center point
	Point center = Point(d / 2, d / 2);

	// Angle step rotation
	double angleStep = 1;

	// Scale 
	double scale = 1.0;

	// Rotation matrix
	Mat rot_mat = getRotationMatrix2D(center, -angleStep, scale);
	
	// Start back-projection
	int start_s_cop = 0;
	for (int i = 0; i < im_in.rows; i++)
	{
		// Get rotated matrix
		warpAffine(im_out, im_out, rot_mat, im_out.size());

		// Init
		int m, n;
		float *h_A, *h_B, *h_C;
		float *d_A, *d_B, *d_C;
		size_t size;

		// Size of image
		m = im_out.rows;
		n = im_out.cols;
		size = m*n * sizeof(float);

		// Allocate host memory
		h_A = (float*)malloc(size);
		h_B = (float*)malloc(size);
		h_C = (float*)malloc(size);

		// Start clocking for copying data
		int start_s = clock();

		// Transfer data to host var
		for (int j = 0; j < m; j++)
			for (int k = 0; k < n; k++)
			{
				h_A[j*n + k] = im_out.at<float>(j, k);
				h_B[j*n + k] = im_in.at<float>(im_in.rows - 1 - i, k);
			}

		// Stop cloking for copying data
		int stop_s = clock();

		// Stack timing
		start_s_cop = start_s_cop + stop_s - start_s;

		// Allocate matrices in device memory
		cudaMalloc(&d_A, size);
		cudaMalloc(&d_B, size);
		cudaMalloc(&d_C, size);

		// Copy matrices from host memory to device memory
		cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

		// Init dimgrid and dimblock
		dim3 dimGrid(MATRIXWIDTH / 32, MATRIXWIDTH / 32);
		dim3 dimBlock(32, 32);

		// Kernel matrix add
		MatAddSharedMemory << <dimGrid, dimBlock >> > (d_A, d_B, d_C, m, n);

		// Wait for the kernel to complete
		cudaThreadSynchronize();

		// Copy result from device memory to host memory
		cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

		// Copy result from host memory to local var
		for (int j = 0; j < m; j++)
			for (int k = 0; k < n; k++)
			{
				im_out.at<float>(j, k) = h_C[j*n + k];
			}

		// Free host memory
		free(h_A);
		free(h_B);
		free(h_C);

		// Free device memory
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
	}

	// Crop image
	d = (d - d / (sqrt(2.0)))*0.55; // cut a little more
	double dw = d;
	double dh = d;
	im_out = im_out(Rect(dw, dh, im_out.cols - 2 * dw, im_out.rows - 2 * dh));
}


//-----------------------------------------------------------------------------------------------------
/*
	Function: InverseRadonTransformCudaThrust
	Inverse radon transform using Thrust library
	Parameters:
		im_in			-	filtered sinogram
		im_out			-	reconstructed image
*/
//-----------------------------------------------------------------------------------------------------
void InverseRadonTransformCudaThrust(Mat & im_in, Mat & im_out)
{
	// Init
	im_in.convertTo(im_in, CV_32FC1);
	double d = im_in.cols;
	im_out = Mat::zeros(d, d, CV_32FC1);

	// Center point
	Point center = Point(d / 2, d / 2);

	// Angle step rotation
	double angleStep = 1;

	// Scale 
	double scale = 1.0;

	// Rotation matrix
	Mat rot_mat = getRotationMatrix2D(center, -angleStep, scale);
	
	// Start back-projection
	for (int i = 0; i < im_in.rows; i++)
	{
		// Get rotated matrix
		warpAffine(im_out, im_out, rot_mat, im_out.size());

		// thrust method
		vector<float> v_thrust_in1 = ConvertMat2Vector(im_out);
		vector<float> v_thrust_in2 = ConcatRow2Vector(im_in, im_in.rows - 1 - i);
		vector<float> v_thrust_out(im_out.rows*im_out.cols);
		MatAddThrust(v_thrust_out, v_thrust_in1, v_thrust_in2, im_out.rows*im_out.cols);
	}

	// Crop image
	d = (d - d / (sqrt(2.0)))*0.55; // cut a little more
	double dw = d;
	double dh = d;
	im_out = im_out(Rect(dw, dh, im_out.cols - 2 * dw, im_out.rows - 2 * dh));
}


//-----------------------------------------------------------------------------------------------------
/*
	Function: PerformRadonAndInverseRadon
	Perform Radon and Inverse Radon transform using filtered-back-projection to reconstruct image
	Parameters:
		method		-	method used
*/
//-----------------------------------------------------------------------------------------------------
void PerformRadonAndInverseRadon(string method)
{
	// Read and display image
	Mat sinogram;
	Mat img = imread("C:\\Users\\RD\\Google Drive\\Working\\VS2015\\OpenCVTest\\OpenCVTest\\media\\ct2.jpg", 0);

	// Resize image to defined image width
	resize(img, img, Size(IMAGEWIDTH, IMAGEWIDTH));
	
	// Display input image
	namedWindow("SourceImg");
	cv::imshow("SourceImg", img);
	imwrite( "original.jpg", img);

	// Init vars
	Mat sinogram_display, sinogram_filtered, img_reconstructed, img_imwrite;

	// Forward radon transform
	ForwardRadonTransform(img, sinogram);
	cv::normalize(sinogram, sinogram_display, 0, 1, cv::NORM_MINMAX);
	cv::imshow("RadonTransform", sinogram_display);

	// Save sinogram
	sinogram_display.convertTo(img_imwrite, CV_8UC1, 255.0);
	imwrite( "sinogram.jpg", img_imwrite);

	// Apply Filter
	string FilterMethod = "Ramp";
	ApplyFilter(sinogram, FilterMethod, sinogram_filtered);

	// Start clocking
	int start_s = clock();
	
	// Inverse radon transform with different method
	if		(method.compare("opencv") == 0)			// opencv
		InverseRadonTransformOpenCV(sinogram_filtered, img_reconstructed);
	else if (method.compare("array") == 0)			// normal array
		InverseRadonTransformArray(sinogram_filtered, img_reconstructed);
	else if (method.compare("cudathrust") == 0)		// thrust
		InverseRadonTransformCudaThrust(sinogram_filtered, img_reconstructed);
	else if (method.compare("cudanoshare") == 0)	// cuda without shared memory
		InverseRadonTransformCuda(sinogram_filtered, img_reconstructed);
	else											// cuda with shared memory				
		InverseRadonTransformCudaSharedMemory(sinogram_filtered, img_reconstructed);

	// Stop clocking
	int stop_s = clock();

	// Method timing
	if (method.compare("opencv") == 0)				// opencv
	{
		std::cout << "===================================================================" << endl;
		std::cout << "Without CUDA using OpenCV" << endl;
		std::cout << "time: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) << " seconds" << endl;
	}
	else if (method.compare("array") == 0)			// normal array
	{
		std::cout << "===================================================================" << endl;
		std::cout << "Without CUDA using array" << endl;
		std::cout << "time: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) << " seconds" << endl;
	}
	else if (method.compare("cudathrust") == 0)		// thrust
	{
		std::cout << "===================================================================" << endl;
		std::cout << "With CUDA using Thrust" << endl;
		std::cout << "time: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) << " seconds" << endl;
	}
	else if (method.compare("cudanoshare") == 0)	// cuda without shared memory
	{
		std::cout << "===================================================================" << endl;
		std::cout << "With CUDA without using shared memory" << endl;
		std::cout << "time: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) << " seconds" << endl;
	}
	else 											// cuda with shared memory					
	{
		std::cout << "===================================================================" << endl;
		std::cout << "With CUDA using shared memory" << endl;
		std::cout << "time: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) << " seconds" << endl;
	}


	// Display and write reconstructed image
	cv::normalize(img_reconstructed, img_reconstructed, 0, 1, cv::NORM_MINMAX);
	cv::imshow("ReconstructedImage", img_reconstructed);
	img_reconstructed.convertTo(img_imwrite, CV_8UC1, 255.0);
	imwrite( "final.jpg", img_imwrite);
}


void PerformCuFFT()
{
	// Read image
	Mat des;
	Mat img = imread("C:\\Users\\RD\\Google Drive\\Working\\VS2015\\OpenCVTest\\OpenCVTest\\media\\ct.jpeg", 0);
	resize(img, img, Size(), 0.2, 0.2);
	namedWindow("SourceImg");
	imshow("SourceImg", img);


	// Radon transform
	Mat des_print, filterOutput, inverse_img;
	ForwardRadonTransform(img, des);
	normalize(des, des_print, 0, 1, cv::NORM_MINMAX);
	imshow("RadonTransform", des_print);


	int NX = des.rows;
	int NY = des.cols;
	int NN = 1000;

	std::cout << "NX=" << NX << " ; NY=" << NY << " ; NN=" << NN << std::endl;

	cufftHandle plan;
	cufftComplex *dev_data, *res;
	cudaMalloc((void**)&dev_data, sizeof(cufftComplex)*NX*NY);
	cudaMalloc((void**)&res, sizeof(cufftComplex)*NX*NY);

	/* Try to do the same thing than cv::randu() */
	cufftComplex* host_data;
	host_data = (cufftComplex *)malloc(sizeof(cufftComplex)*NX*NY);

	srand(time(NULL));

	float value;
	int idx;
	for (int i = 0; i < NX; i++)
	{
		for (int j = 0; j < NY; j++)
		{
			value = des.at<float>(i, j);
			idx = i*NX + j - 1;
			host_data[idx] = make_cuComplex(value, value);
		}
	}

	cudaMemcpy(dev_data, host_data, sizeof(cufftComplex)*NX*NY, cudaMemcpyHostToDevice);

	/* Warm up ? */

	double t = cv::getTickCount();

	int start_s = clock();

	/* Create a 2D FFT plan. */
	cufftPlan2d(&plan, NX, NY, CUFFT_C2C);

	cufftExecC2C(plan, dev_data, res, CUFFT_FORWARD);

	int stop_s = clock();
	cout << "===================================================================" << endl;
	cout << "With CUDA FFT" << endl;
	cout << "time: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) << " seconds" << endl;



	std::cout << "Cuda time=" << t << " ms" << std::endl;



	int size_of_one_set = sizeof(cufftComplex) * NX*NY;
	cufftComplex* result;
	result = (cufftComplex*)malloc(size_of_one_set);


	cudaMemcpy(result, res, sizeof(cufftComplex)*NX*NY, cudaMemcpyDeviceToHost);

	/* Destroy the cuFFT plan. */
	cufftDestroy(plan);
	cudaFree(dev_data);

}


//-----------------------------------------------------------------------------------------------------
/*
	Function: MatMulPixelwiseWithoutSharedMemory
	Multiply 2 complex matrices without using shared memory
	Parameters:
		A_r			-	Matrix input 1 real part
		A_c			-	Matrix input 1 complex part
		B_r			-	Matrix input 2 real part
		B_c			-	Matrix input 2 complex part
		C_r			-	Matrix output real part
		C_c			-	Matrix output complex part
		m			-	Matrix row
		n			-	Matrix column
*/
//-----------------------------------------------------------------------------------------------------
__global__ void MatMulPixelwiseWithoutSharedMemory(float A_r[], float A_c[], 
			float B_r[], float B_c[], float C_r[], float C_c[], int m, int n) {
	/* blockDim.x = threads_per_block                            */
	/* First block gets first threads_per_block components.      */
	/* Second block gets next threads_per_block components, etc. */
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	/* The test shouldn't be necessary */
	if (blockIdx.x < m && threadIdx.x < n)
	{
		C_r[idx] = A_r[idx] * B_r[idx] - A_c[idx] * B_c[idx];
		C_c[idx] = A_r[idx] * B_c[idx] - A_c[idx] * B_r[idx];
	}
		
}


//-----------------------------------------------------------------------------------------------------
/*
	Function: MultiplySpectrums
	Multiply 2 complex matrices
	Parameters:
		planes			-	planes of complex image
		planes_filter	-	planes of filter image
		img_complex		-	output image
*/
//-----------------------------------------------------------------------------------------------------
void MultiplySpectrums(Mat & planes_r, Mat & planes_c, 
		Mat & planes_filter_r, Mat & planes_filter_c, Mat & img_complex)
{
	vector<float> A_r = ConvertMat2Vector(planes_r);
	vector<float> A_c = ConvertMat2Vector(planes_c);
	vector<float> B_r = ConvertMat2Vector(planes_filter_r);
	vector<float> B_c = ConvertMat2Vector(planes_filter_c);


	// Init
		int m, n;
		float *h_A_r, *h_B_r, *h_C_r;
		float *h_A_c, *h_B_c, *h_C_c;
		float *d_A_r, *d_B_r, *d_C_r;
		float *d_A_c, *d_B_c, *d_C_c;
		size_t size;

		// Size of image
		m = planes_r.rows;
		n = planes_r.cols;
		size = m*n * sizeof(float);

		// Allocate host memory
		h_A_r = (float*)malloc(size);
		h_B_r = (float*)malloc(size);
		h_C_r = (float*)malloc(size);
		h_A_c = (float*)malloc(size);
		h_B_c = (float*)malloc(size);
		h_C_c = (float*)malloc(size);

		// Start clocking for copying data
		int start_s = clock();

		// Transfer data to host var
		for (int j = 0; j < m; j++)
			for (int k = 0; k < n; k++)
			{
				h_A_r[j*n + k] = planes_r.at<float>(j, k);
				h_B_r[j*n + k] = planes_filter_r.at<float>(j, k);
				h_A_c[j*n + k] = planes_c.at<float>(j, k);
				h_B_c[j*n + k] = planes_filter_c.at<float>(j, k);
			}

		// Stop cloking for copying data
		int stop_s = clock();

		// Allocate matrices in device memory
		cudaMalloc(&d_A_r, size);
		cudaMalloc(&d_B_r, size);
		cudaMalloc(&d_C_r, size);
		cudaMalloc(&d_A_c, size);
		cudaMalloc(&d_B_c, size);
		cudaMalloc(&d_C_c, size);

		// Copy matrices from host memory to device memory
		cudaMemcpy(d_A_r, h_A_r, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B_r, h_B_r, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_A_c, h_A_c, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B_c, h_B_c, size, cudaMemcpyHostToDevice);

		// Kernel matrix add
		MatMulPixelwiseWithoutSharedMemory << <MATRIXWIDTH, MATRIXWIDTH >> > (d_A_r, d_A_c, d_B_r, d_B_c, d_C_r, d_C_c, m, n);
		
		// Wait for the kernel to complete
		cudaThreadSynchronize();

		// Copy result from device memory to host memory
		cudaMemcpy(h_C_r, d_C_r, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_C_c, d_C_c, size, cudaMemcpyDeviceToHost);

		// Declare block vars
		Mat plane_temp[2];
		Mat m_A_r_temp = Mat::zeros(img_complex.size(), CV_32F);
		Mat m_A_c_temp = Mat::zeros(img_complex.size(), CV_32F);


		// Copy result from host memory to local var
		for (int j = 0; j < m; j++)
			for (int k = 0; k < n; k++)
			{
				m_A_r_temp.at<float>(j, k) = h_C_r[j*n + k];
				m_A_c_temp.at<float>(j, k) = h_C_c[j*n + k];
			}


		plane_temp[0] = m_A_r_temp;
		plane_temp[1] = m_A_c_temp;
		merge(plane_temp, 2, img_complex);

		// Free host memory
		free(h_A_r);
		free(h_B_r);
		free(h_C_r);
		free(h_A_c);
		free(h_B_c);
		free(h_C_c);

		// Free device memory
		cudaFree(d_A_r);
		cudaFree(d_B_r);
		cudaFree(d_C_r);
		cudaFree(d_A_c);
		cudaFree(d_B_c);
		cudaFree(d_C_c);
}


//-----------------------------------------------------------------------------------------------------
/*
	Function: ApplyFilterCuda
	Apply filter to sinogram using cuda when multiplying 2 complex matrices
	Parameters:
		im_in			-	input image
		filter_method	-	filter method
		im_out			-	output image
*/
//-----------------------------------------------------------------------------------------------------
void ApplyFilterCuda(Mat & im_in, string filter_method, Mat & im_out)
{
	// Init images objects
	Mat img_gray, img_output, img_output_display;
	Mat img_padded;	
	Mat img_complex, filter;
	Mat planes[2], planes_filter[2], img_magnitude;

	// Fourier image size
	int N, M;

	// String constant for image title
	const string originalName = "Input Image (grayscale)"; // window name
	const string spectrumMagName = "Magnitude Image (log transformed)"; // window name
	const string lowPassName = "Ramp Filtered (grayscale)"; // window name
	const string filterName = "Filter Image"; // window nam

	// setup the DFT image sizes
	M = getOptimalDFTSize(im_in.rows);
	N = getOptimalDFTSize(im_in.cols);

	// convert input to grayscale if RGB
	if (im_in.channels() == 3)
		cvtColor(im_in, img_gray, CV_BGR2GRAY);
	else
		img_gray = im_in.clone();

	// setup the DFT images
	copyMakeBorder(img_gray, img_padded, 0, M - img_gray.rows, 0,
		N - img_gray.cols, BORDER_CONSTANT, Scalar::all(0));
	planes[0] = Mat_<float>(img_padded);
	planes[1] = Mat::zeros(img_padded.size(), CV_32F);
	merge(planes, 2, img_complex);

	// do the DFT
	dft(img_complex, img_complex);

	// construct the filter (same size as complex image)
	Mat ramp_filter, hann_filter;
	filter = img_complex.clone();
	ramp_filter = filter.clone();
	hann_filter = filter.clone();
	CreateRampFilter(ramp_filter, 1.0);
	CreateHannFilter(hann_filter, 1.0);

	// select filter to apply
	filter = ramp_filter.clone();

	// apply filter
	ShiftDftToCenter(img_complex);

	// HERE! WE CAN PARALLELIZE THE MULTIPLICATION OF 2 COMPLEX MATRICES
	split(img_complex, planes);
	split(filter, planes_filter);

	mulSpectrums(img_complex, filter, img_complex, 1);

	// shift dft to the center
	ShiftDftToCenter(img_complex);

	// create magnitude spectrum for display
	img_magnitude = CreateDisplaySpectrum(img_complex, true);

	// do inverse DFT on filtered image
	idft(img_complex, img_complex);

	// split into planes and extract plane 0 as output image
	split(img_complex, planes);
	cv::normalize(planes[0], img_output, 0, 1, CV_MINMAX);

	// do the same with the filter image
	split(filter, planes);
	cv::normalize(planes[0], img_output_display, 0, 1, CV_MINMAX);

	// save image output
	im_out = img_output.clone();

	// show result
	Mat rgb = im_in.clone();
	cv::normalize(rgb, rgb, 0, 1, cv::NORM_MINMAX);
	cv::imshow(spectrumMagName, img_magnitude);
	cv::imshow(lowPassName, img_output);
	cv::imshow(filterName, img_output_display);

	// Write result
	Mat img8bit;
	img_output_display.convertTo(img8bit, CV_8UC1, 255.0);
	imwrite( "filtered_sinogram.jpg", img8bit);
}
