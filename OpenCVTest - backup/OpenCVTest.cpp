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
#include <cmath>

using namespace std;
using namespace cv;

#define PI 3.14159265


//-----------------------------------------------------------------------------------------------------
// Shift to center
//-----------------------------------------------------------------------------------------------------
void shiftDFT(Mat& fImage)
{
	Mat tmp, q0, q1, q2, q3;

	// first crop the image, if it has an odd number of rows or columns

	fImage = fImage(Rect(0, 0, fImage.cols & -2, fImage.rows & -2));

	int cx = fImage.cols / 2;
	int cy = fImage.rows / 2;

	// rearrange the quadrants of Fourier image
	// so that the origin is at the image center

	q0 = fImage(Rect(0, 0, cx, cy));
	q1 = fImage(Rect(cx, 0, cx, cy));
	q2 = fImage(Rect(0, cy, cx, cy));
	q3 = fImage(Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

/******************************************************************************/
// return a floating point spectrum magnitude image scaled for user viewing
// complexImg- input dft (2 channel floating point, Real + Imaginary fourier image)
// rearrange - perform rearrangement of DFT quadrants if true

// return value - pointer to output spectrum magnitude image scaled for user viewing

Mat CreateDisplaySpectrum(Mat& complexImg, bool rearrange)
{
	Mat planes[2];

	// compute magnitude spectrum (N.B. for display)
	// compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))

	split(complexImg, planes);
	magnitude(planes[0], planes[1], planes[0]);

	Mat mag = (planes[0]).clone();
	mag += Scalar::all(1);
	log(mag, mag);

	if (rearrange)
	{
		// re-arrange the quaderants
		shiftDFT(mag);
	}

	normalize(mag, mag, 0, 1, CV_MINMAX);

	return mag;

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
// Create Ramp Filter
//-----------------------------------------------------------------------------------------------------
void CreateRampFilter(Mat& dft_Filter, double a)
{
	Mat tmp = Mat(dft_Filter.rows, dft_Filter.cols, CV_32F);
	int N = dft_Filter.cols;
	int M = dft_Filter.rows;
	double step;
	step = 2*a/(N-1);

	for (int i = 0; i < dft_Filter.rows; i++)
	{
		for (int j = 0; j < dft_Filter.cols; j++)
		{
			tmp.at<float>(i, j) = abs(-a+j*step);
		}
	}

	Mat toMerge[] = { tmp, tmp };
	merge(toMerge, 2, dft_Filter);
}


//-----------------------------------------------------------------------------------------------------
// Create Hann Filter and Ramp Filter
//-----------------------------------------------------------------------------------------------------
void CreateHannFilter(Mat& dft_Filter, double a)
{
	Mat tmp = Mat(dft_Filter.rows, dft_Filter.cols, CV_32F);
	int N = dft_Filter.cols; // length
	int M = dft_Filter.rows; // 180% radon
	double step, ramp, hann;
	step = 2 * a / (N - 1);

	for (int i = 0; i < dft_Filter.rows; i++)
	{
		for (int j = 0; j < dft_Filter.cols; j++)
		{
			ramp = abs(-a + j*step);
			hann = sin(j*PI / (N-1));
			hann = pow(hann, 2.0);
			tmp.at<float>(i, j) = ramp*hann;
		}
	}

	Mat toMerge[] = { tmp, tmp };
	merge(toMerge, 2, dft_Filter);
}



//-----------------------------------------------------------------------------------------------------
// Apply Filter
//-----------------------------------------------------------------------------------------------------
void ApplyFilter (Mat& img, string FilterMethod, Mat& filterOutput)
{
	Mat imgGray, imgOutput, filterOutput_dis;	// image object(s)
	
	Mat padded;		// fourier image objects and arrays
	Mat complexImg, filter;
	Mat planes[2], mag;


	int N, M; // fourier image sizes

	int radius = 30;				// low pass filter parameter
	int order = 2;				// low pass filter parameter

	const string originalName = "Input Image (grayscale)"; // window name
	const string spectrumMagName = "Magnitude Image (log transformed)"; // window name
	const string lowPassName = "Ramp Filtered (grayscale)"; // window name
	const string filterName = "Filter Image"; // window nam

	// setup the DFT image sizes

	M = getOptimalDFTSize(img.rows);
	N = getOptimalDFTSize(img.cols);

	// convert input to grayscale if RGB
	if (img.channels() == 3)
		cvtColor(img, imgGray, CV_BGR2GRAY);
	else
		imgGray = img.clone();

	// setup the DFT images
	copyMakeBorder(imgGray, padded, 0, M - imgGray.rows, 0,
		N - imgGray.cols, BORDER_CONSTANT, Scalar::all(0));
	planes[0] = Mat_<float>(padded);
	planes[1] = Mat::zeros(padded.size(), CV_32F);
	merge(planes, 2, complexImg);

	// do the DFT
	dft(complexImg, complexImg);

	// construct the filter (same size as complex image)
	Mat ramp_filter, hann_filter;
	filter = complexImg.clone();
	ramp_filter = filter.clone();
	hann_filter = filter.clone();
	CreateRampFilter(ramp_filter, 1.0);
	CreateHannFilter(hann_filter, 1.0);

	//cout << "====================" << endl;
	//cout << "Ramp" << endl;
	//cout << ramp_filter << endl;

	//cout << "====================" << endl;
	//cout << "Hann" << endl;
	//cout << hann_filter << endl;


	filter = ramp_filter.clone();

	// apply filter
	shiftDFT(complexImg);
	mulSpectrums(complexImg, filter, complexImg, 0);
	shiftDFT(complexImg);

	// create magnitude spectrum for display
	mag = CreateDisplaySpectrum(complexImg, true);

	// do inverse DFT on filtered image
	idft(complexImg, complexImg);

	// split into planes and extract plane 0 as output image
	split(complexImg, planes);
	normalize(planes[0], imgOutput, 0, 1, CV_MINMAX);

	// do the same with the filter image
	split(filter, planes);
	normalize(planes[0], filterOutput_dis, 0, 1, CV_MINMAX);

	filterOutput = imgOutput.clone();


	// ***

	// display image in window

	Mat rgb = img.clone();
	normalize(rgb, rgb, 0, 1, cv::NORM_MINMAX);
	//imshow("rgb", rgb);
	//imshow(originalName, imgGray);
	imshow(spectrumMagName, mag);
	imshow(lowPassName, imgOutput);
	imshow(filterName, filterOutput_dis);
}


//-----------------------------------------------------------------------------------------------------
// Read Filtered and assign to Mat Opencv
//-----------------------------------------------------------------------------------------------------
Mat ReadMatFromTxt(string filename, int rows, int cols)
{
	double m;
	Mat out = Mat::zeros(rows, cols, CV_64FC1);//Matrix to store values

	ifstream fileStream(filename);
	int cnt = 0;//index starts from 0
	while (fileStream >> m)
	{
		int temprow = cnt / cols;
		int tempcol = cnt % cols;
		out.at<double>(temprow, tempcol) = m;
		cnt++;
	}
	return out;
}


//-----------------------------------------------------------------------------------------------------
// Write Mat to text file
//-----------------------------------------------------------------------------------------------------
void WriteMatToText(Mat&  someMatrixOfAnyType)
{
	// Declare what you need
	cv::FileStorage file("C:\\Users\\RD\\Google Drive\\Working\\VS2015\\OpenCVTest\\OpenCVTest\\media\\filteredCpp.txt", cv::FileStorage::WRITE);
	// Write to file!
	file << someMatrixOfAnyType;
}


//-----------------------------------------------------------------------------------------------------
// Forward Radon Transform
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
// Inverse Radon Transform
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
	Mat img = imread("C:\\Users\\RD\\Google Drive\\Working\\VS2015\\OpenCVTest\\OpenCVTest\\media\\ct.jpeg", 0);
	resize(img, img, Size(), 0.2, 0.2);
	namedWindow("SourceImg");
	imshow("SourceImg", img);

	
	// Radon and Inverse Radon transform
	Mat res_print, filterOutput, inverse_img;
	ForwardRadonTransform(img, res);
	normalize(res, res_print, 0, 1, cv::NORM_MINMAX);
	imshow("RadonTransform", res_print);


	// Apply Filter
	string FilterMethod = "Ramp";
	ApplyFilter(res, FilterMethod, filterOutput);

	//cout << filterOutput << endl;


	// Inverse Radon Trasnform
	InverseRadonTransform(filterOutput, inverse_img);
	normalize(inverse_img, inverse_img, 0, 1, cv::NORM_MINMAX);
	imshow("ReconstructedImage", inverse_img);


	waitKey(0);

	return 0;
}
