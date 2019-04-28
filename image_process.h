#pragma once

#define _USE_MATH_DEFINES

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <list>
#include <deque>
#include <assert.h>
#include <math.h>
#include <algorithm>
#include <numeric>
#include <functional>


#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class GaussianBlur
{

public:
	GaussianBlur(int blurRadius, double sigma) :
		mBlurRadius(blurRadius),
		mSigma(sigma)
	{
		CreateConvolutionMatrix();
	}

	cv::Mat BlurImage(const cv::Mat& in) {
		if (in.empty())
			return cv::Mat();

		cv::Mat image;
		in.copyTo(image);

		int matrixSize = mConvolutionMatrix.size();
		int halfMatrixSize = matrixSize / 2;

		float sumRed = 0.0f;
		float sumBlue = 0.0f;
		float sumGreen = 0.0f;
		float matrixValue = 0.0f;
		int x1 = 0, y1 = 0;

		int rows = in.rows;
		int cols = in.cols;
		int r_pos = 2;
		int g_pos = 1;
		int b_pos = 0;

		for (int x = 0; x < rows; ++x)
		{
			for (int y = 0; y < cols; ++y)
			{
				for (int kx = -halfMatrixSize; kx <= halfMatrixSize; ++kx)
				{
					x1 = ReflectIndex(x - kx, rows);

					cv::Vec3b color = in.at<cv::Vec3b>(x1, y);
			
					matrixValue = mConvolutionMatrix[kx + halfMatrixSize];

					sumRed += color[r_pos] * matrixValue;
					sumBlue += color[b_pos] * matrixValue;
					sumGreen += color[g_pos] * matrixValue;
				}
				image.at<cv::Vec3b>(x, y)[r_pos] = sumRed;
				image.at<cv::Vec3b>(x, y)[g_pos] = sumGreen;
				image.at<cv::Vec3b>(x, y)[b_pos] = sumBlue;

				sumRed = sumGreen = sumBlue = 0.0f;
			}
		}

		for (int x = 0; x < rows; ++x)
		{
			for (int y = 0; y < cols; ++y)
			{
				for (int ky = -halfMatrixSize; ky <= halfMatrixSize; ++ky)
				{
					y1 = ReflectIndex(y - ky, cols);

					cv::Vec3b color = in.at<cv::Vec3b>(x, y1);
					matrixValue = mConvolutionMatrix[ky + halfMatrixSize];

					sumRed += color[r_pos] * matrixValue;
					sumBlue += color[b_pos] * matrixValue;
					sumGreen += color[g_pos] * matrixValue;
				}

				image.at<cv::Vec3b>(x, y)[r_pos] = sumRed;
				image.at<cv::Vec3b>(x, y)[g_pos] = sumGreen;
				image.at<cv::Vec3b>(x, y)[b_pos] = sumBlue;

				sumRed = sumGreen = sumBlue = 0.0f;
			}
		}

		return image;
	}

	float GaussFunc(float x) {
		return (1 / sqrtf(2 * M_PI * mSigma * mSigma)) *
			exp(-(x*x) / (2 * mSigma*mSigma));
	}

	int getBlurRadius() const {
		return mBlurRadius;
	}

	void setBlurRadius(int value) {
		mBlurRadius = value;
		CreateConvolutionMatrix();
	}

	float getSigma() const {
		return mSigma;
	}
	void setSigma(float value) {
		mSigma = value;
		CreateConvolutionMatrix();
	}

	~GaussianBlur() {

	}

private:

	std::vector<float> mConvolutionMatrix;

	int ReflectIndex(int x, int length) {
		if (x < 0)
			return -x - 1;
		else if (x >= length)
			return 2 * length - x - 1;

		return x;
	}

	void CreateConvolutionMatrix() {
		int x = 0;
		size_t matrixSize, halfMatrixSize;

		matrixSize = (size_t)(2 * mBlurRadius + 1);
		halfMatrixSize = matrixSize / 2;

		mConvolutionMatrix.resize(matrixSize);

		std::vector<float>::iterator begin = mConvolutionMatrix.begin();
		std::vector<float>::iterator end = mConvolutionMatrix.end();

		x = -(int)halfMatrixSize;
		std::for_each(begin, end,
			[&](float& val) mutable
		{
			val = GaussFunc(x);
			x++;
		});

		// normalize the values in the convolution matrix
		float sum = std::accumulate(begin, end, 0.0f);

		std::for_each(begin, end, [&](float& val) { val /= sum; });
	}

	int mBlurRadius;
	float mSigma;
};


class ImageProcess
{
public:
	// ¶ÁÈ¡Í¼Ïñ
	static cv::Mat read_image(const std::string& name);
	// ÏÔÊ¾Í¼Ïñ
	static void show_image(const std::string& name, cv::Mat& pic);
	// °ÑÍ¼ÏñÐ´ÈëÎÄ¼þÖÐ
	static void write_image(const std::string& name, const cv::Mat& img);

	// »Ò¶È»Ò¶È±ä»»
	static cv::Mat GreyScale(const cv::Mat& origin);
	// ×ª³ÉÅ¯É«µ÷
	static cv::Mat Warm(int delta, const cv::Mat& origin);
	// ×ª³ÉÀäÉ«µ÷
	static cv::Mat Cool(int delta, const cv::Mat& origin);

	// ÁÁ¶Èµ÷Õû
	static cv::Mat Brightness(int delta, const cv::Mat& origin);
	// Ë®Æ½·­×ª
	static cv::Mat Horizontal(const cv::Mat& origin);
	// ´¹Ö±·­×ª
	static cv::Mat Vertical(const cv::Mat& origin);
	// ÏßÐÔ»Ò¶È±ä»» y = ax + b
	static cv::Mat LinearLevelTransformation( const cv::Mat& origin, double a, double b);
	static cv::Mat LogGreyLevelTransformation( const cv::Mat& origin, double a, double b);
	static cv::Mat PowerGreyLevelTransformation( const cv::Mat& origin, double c, double r, double b);
	static cv::Mat ExpTransform(const cv::Mat &origin, double b, double c, double a);
	static cv::Mat TwoThreshold(const cv::Mat &origin, double t1, double t2, int option);

	static cv::Mat StretchTransform(const cv::Mat &origin,
		int x1, int x2,
		double k1, double k2, double k3,
		double b1, double b2, double b3);

	// ¼òµ¥µÄÆ½»¬
	static cv::Mat SimpleSmooth(const cv::Mat &origin);

	// ÖÐÖµÂË²¨
	static cv::Mat MeidaFilter(const cv::Mat &origin);

	// LaplaceÈñ»¯
	static cv::Mat LaplaceSharpen(const cv::Mat &origin);

	// Sobel±ßÔµ¼ì²â
	static cv::Mat SobelEdge(const cv::Mat &origin);


	// ¸ßË¹ÂË²¨
	static cv::Mat GaussianSmoothing(const cv::Mat &origin, int radius, double sigma);

	static cv::Mat Binaryzation(const cv::Mat &origin);

	static cv::Mat Metal(cv::Mat& origin);

	static cv::Mat PrewittEdge(const cv::Mat &origin);

	static cv::Mat ContourExtraction(const cv::Mat &origin);
	static cv::Mat ConnectedDomain(const cv::Mat &origin);
	static cv::Mat Dilate(const cv::Mat &origin);
	static cv::Mat Expansion(const cv::Mat &origin);
	static cv::Mat Opening(const cv::Mat &origin);
	static cv::Mat Closing(const cv::Mat &origin);
	static cv::Mat Thinning(const cv::Mat &origin);
	static cv::Mat RGB2HSV(const cv::Mat &origin);
	static cv::Mat RGB2HSL(const cv::Mat &origin);
	static cv::Mat RGB2CMYK(const cv::Mat &origin);
	static cv::Mat Final(const cv::Mat &origin);
public:
	// 平移
	static cv::Mat Translation(const cv::Mat &origin, int x, int y);
	static cv::Mat Mirror(const cv::Mat &origin,bool horizontal = true,bool vertical = true);
	static cv::Mat Zoom(const cv::Mat &origin, double x_scale = 1.0, double y_scale = 1.0);
	static cv::Mat Rotate(const cv::Mat &origin, double angle = 0.0);
private:
	static uchar SafeValue(int value);
	static int Gray(const cv::Vec3b& oldColor);
	/*
	** ¾í»ýÔËËã
	** origin£ºÔ­Ê¼Í¼Ïñ
	** newImage£ºÄ¿±êÍ¼Ïñ
	** kernel£º¾í»ýºË£¨Õý·½ÐÎ£¬¶þÎ¬Êý×é£©
	** kernelSize£º¾í»ýºËµÄ±ä³¤£¨ÆæÊý£©
	** kernelSum£º¾í»ýºËÄÚËùÓÐÔªËØÖ®ºÍ
	*/
	static void KernalCalc(const cv::Mat &origin, 
		cv::Mat & newImage, 
		const int * kernel, 
		const int kernelSize, 
		const int kernelSum);
};

