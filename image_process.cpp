#include "image_process.h"
//Í¨µÀµÄË³ÐòÊÇBGR¶ø·ÇRGB

uchar ImageProcess::SafeValue(int value) {
	return cv::saturate_cast<uchar>(value);
}

int ImageProcess::Gray(const cv::Vec3b& oldColor) {
	int gray = (oldColor[0] * 299 + oldColor[1] * 587 + oldColor[2] * 114 + 500) / 1000;
	return gray;
}

void ImageProcess::KernalCalc(const cv::Mat &origin, cv::Mat & newImage, const int * kernel, const int kernelSize, const int kernelSum) {
	
	int rows = origin.rows;
	int cols = origin.cols;

	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	int r = 0;
	int g = 0;
	int b = 0;

	for (int i = kernelSize / 2; i < rows - kernelSize / 2; ++i)
	{
		for (int j = kernelSize / 2; j < cols - kernelSize / 2; ++j)
		{
			r = g = b = 0;
			for (int x = -kernelSize / 2; x <= kernelSize / 2; ++x)
			{
				for (int y = -kernelSize / 2; y <= kernelSize / 2; ++y)
				{
					const cv::Vec3b oldColor = origin.at<cv::Vec3b>(i + x, j + y);

					int val = 0; //kernel[kernelSize / 2 + x][kernelSize / 2 + y]
					{
						int temp_i = kernelSize / 2 + x;
						int temp_j = kernelSize / 2 + y;
						int pos = temp_i * temp_j + temp_j;
						val = kernel[pos];
					}
					r += oldColor[r_pos] * val;
					g += oldColor[g_pos] * val;
					b += oldColor[b_pos] * val;
				}
			}
			r = SafeValue((kernelSum ? r / kernelSum:r));
			g = SafeValue((kernelSum ? g / kernelSum:g));
			b = SafeValue((kernelSum ? b / kernelSum:b));

			newImage.at<cv::Vec3b>(i, j)[r_pos] = r;
			newImage.at<cv::Vec3b>(i, j)[g_pos] = g;
			newImage.at<cv::Vec3b>(i, j)[b_pos] = b;
		}
	}
}

cv::Mat ImageProcess::read_image(const std::string& name) {
	return cv::imread(name);
}

void ImageProcess::show_image(const std::string& name, cv::Mat& pic) {
	cv::namedWindow(name);
	cv::imshow(name, pic);
}

void ImageProcess::write_image(const std::string& name, const cv::Mat& img) {
	cv::imwrite(name, img);
}

cv::Mat ImageProcess::GreyScale(const cv::Mat& origin) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	if (origin.isContinuous()) {
		cols *= rows;
		rows = 1;
	}

	for (int i = 0; i < rows; ++i)
	{
		const cv::Vec3b* origin_ptr = origin.ptr<cv::Vec3b>(i);
		cv::Vec3b* new_ptr = newImage.ptr<cv::Vec3b>(i);

		for (int j = 0; j < cols; ++j)
		{
			const cv::Vec3b oldColor = origin_ptr[j];
			int average = Gray(oldColor);
			
			new_ptr[j][r_pos] = SafeValue(average);
			new_ptr[j][g_pos] = SafeValue(average);
			new_ptr[j][b_pos] = SafeValue(average);
		}
	}
	return newImage;
}

cv::Mat ImageProcess::Warm(int delta, const cv::Mat& origin) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	if (origin.isContinuous()) {
		cols *= rows;
		rows = 1;
	}

	for (int i = 0; i < rows; ++i)
	{
		const cv::Vec3b* origin_ptr = origin.ptr<cv::Vec3b>(i);
		cv::Vec3b* new_ptr = newImage.ptr<cv::Vec3b>(i);

		for (int j = 0; j < cols; ++j)
		{
			const cv::Vec3b oldColor = origin_ptr[j];
			new_ptr[j][r_pos] = SafeValue(oldColor[r_pos] + delta);
			new_ptr[j][g_pos] = SafeValue(oldColor[g_pos] + delta);
			new_ptr[j][b_pos] = SafeValue(oldColor[b_pos] + 0);
		}
	}
	return newImage;
}

cv::Mat ImageProcess::Cool(int delta, const cv::Mat& origin) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	if (origin.isContinuous()) {
		cols *= rows;
		rows = 1;
	}

	for (int i = 0; i < rows; ++i)
	{
		const cv::Vec3b* origin_ptr = origin.ptr<cv::Vec3b>(i);
		cv::Vec3b* new_ptr = newImage.ptr<cv::Vec3b>(i);

		for (int j = 0; j < cols; ++j)
		{
			const cv::Vec3b oldColor = origin_ptr[j];
			new_ptr[j][r_pos] = SafeValue(oldColor[r_pos] + 0);
			new_ptr[j][g_pos] = SafeValue(oldColor[g_pos] + 0);
			new_ptr[j][b_pos] = SafeValue(oldColor[b_pos] + delta);
		}
	}
	return newImage;
}

// ÁÁ¶Èµ÷Õû
cv::Mat ImageProcess::Brightness(int delta, const cv::Mat& origin) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	if (origin.isContinuous()) {
		cols *= rows;
		rows = 1;
	}

	for (int i = 0; i < rows; ++i)
	{
		const cv::Vec3b* origin_ptr = origin.ptr<cv::Vec3b>(i);
		cv::Vec3b* new_ptr = newImage.ptr<cv::Vec3b>(i);

		for (int j = 0; j < cols; ++j)
		{
			const cv::Vec3b oldColor = origin_ptr[j];
			new_ptr[j][r_pos] = SafeValue(oldColor[r_pos] + delta);
			new_ptr[j][g_pos] = SafeValue(oldColor[g_pos] + delta);
			new_ptr[j][b_pos] = SafeValue(oldColor[b_pos] + delta);
		}
	}
	return newImage;
}
cv::Mat ImageProcess::Horizontal(const cv::Mat& origin) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	/*if (origin.isContinuous()) {
		cols *= rows;
		rows = 1;
	}*/

	for (int i = 0; i < rows; ++i)
	{
		const cv::Vec3b* origin_ptr = origin.ptr<cv::Vec3b>(i);
		cv::Vec3b* new_ptr = newImage.ptr<cv::Vec3b>(rows - i - 1);

		for (int j = 0; j < cols; ++j)
		{
			const cv::Vec3b oldColor = origin_ptr[j];
			new_ptr[j] = oldColor;
		}
	}
	return newImage;
}
cv::Mat ImageProcess::Vertical(const cv::Mat& origin) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	/*if (origin.isContinuous()) {
		cols *= rows;
		rows = 1;
	}*/

	for (int i = 0; i < rows; ++i)
	{
		const cv::Vec3b* origin_ptr = origin.ptr<cv::Vec3b>(i);
		cv::Vec3b* new_ptr = newImage.ptr<cv::Vec3b>(i);

		for (int j = 0; j < cols; ++j)
		{
			const cv::Vec3b oldColor = origin_ptr[cols - j - 1];
			new_ptr[j] = oldColor;
		}
	}
	return newImage;
}
// ÏßÐÔ»Ò¶È±ä»» y = ax + b
cv::Mat ImageProcess::LinearLevelTransformation( const cv::Mat& origin, double a, double b) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	if (origin.isContinuous()) {
		cols *= rows;
		rows = 1;
	}

	for (int i = 0; i < rows; ++i)
	{
		const cv::Vec3b* origin_ptr = origin.ptr<cv::Vec3b>(i);
		cv::Vec3b* new_ptr = newImage.ptr<cv::Vec3b>(i);

		for (int j = 0; j < cols; ++j)
		{
			const cv::Vec3b oldColor = origin_ptr[j];
			int gray = Gray(oldColor);
			int y = a * gray + b;
			y = SafeValue(y);
			new_ptr[j][r_pos] = y;
			new_ptr[j][g_pos] = y;
			new_ptr[j][b_pos] = y;
		}
	}
	return newImage;
}

cv::Mat ImageProcess::LogGreyLevelTransformation( const cv::Mat& origin, double a, double b) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	if (origin.isContinuous()) {
		cols *= rows;
		rows = 1;
	}

	for (int i = 0; i < rows; ++i)
	{
		const cv::Vec3b* origin_ptr = origin.ptr<cv::Vec3b>(i);
		cv::Vec3b* new_ptr = newImage.ptr<cv::Vec3b>(i);

		for (int j = 0; j < cols; ++j)
		{
			const cv::Vec3b oldColor = origin_ptr[j];
			int gray = Gray(oldColor);
			int y = log(b + gray) / log(a);
			y = SafeValue(y);
			new_ptr[j][r_pos] = y;
			new_ptr[j][g_pos] = y;
			new_ptr[j][b_pos] = y;
		}
	}
	return newImage;
}
cv::Mat ImageProcess::PowerGreyLevelTransformation( const cv::Mat& origin, double c, double r, double b) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	if (origin.isContinuous()) {
		cols *= rows;
		rows = 1;
	}

	for (int i = 0; i < rows; ++i)
	{
		const cv::Vec3b* origin_ptr = origin.ptr<cv::Vec3b>(i);
		cv::Vec3b* new_ptr = newImage.ptr<cv::Vec3b>(i);

		for (int j = 0; j < cols; ++j)
		{
			const cv::Vec3b oldColor = origin_ptr[j];
			int gray = Gray(oldColor);
			int y = c * pow(gray,r) + b;
			y = SafeValue(y);
			new_ptr[j][r_pos] = y;
			new_ptr[j][g_pos] = y;
			new_ptr[j][b_pos] = y;
		}
	}
	return newImage;
}

cv::Mat ImageProcess::ExpTransform(const cv::Mat& origin, double b, double c, double a) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	if (origin.isContinuous()) {
		cols *= rows;
		rows = 1;
	}

	for (int i = 0; i < rows; ++i)
	{
		const cv::Vec3b* origin_ptr = origin.ptr<cv::Vec3b>(i);
		cv::Vec3b* new_ptr = newImage.ptr<cv::Vec3b>(i);

		for (int j = 0; j < cols; ++j)
		{
			const cv::Vec3b oldColor = origin_ptr[j];
			int gray = Gray(oldColor);
			int y = pow(b, c*(gray - a));
			y = SafeValue(y);
			new_ptr[j][r_pos] = y;
			new_ptr[j][g_pos] = y;
			new_ptr[j][b_pos] = y;
		}
	}
	return newImage;
}

cv::Mat ImageProcess::TwoThreshold(const cv::Mat &origin, double t1, double t2, int option) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	if (origin.isContinuous()) {
		cols *= rows;
		rows = 1;
	}

	for (int i = 0; i < rows; ++i)
	{
		const cv::Vec3b* origin_ptr = origin.ptr<cv::Vec3b>(i);
		cv::Vec3b* new_ptr = newImage.ptr<cv::Vec3b>(i);

		for (int j = 0; j < cols; ++j)
		{
			const cv::Vec3b oldColor = origin_ptr[j];
			int gray = Gray(oldColor);
			int y = 0;
			if (option == 0) {
				if (gray < t1 || gray > t2) {
					y = 0;
				}
				else {
					y = 255;
				}
			}
			else {
				if (gray >= t1 && gray <= t2) {
					y = 0;
				}
				else {
					y = 255;
				}
			}
			
			y = SafeValue(y);
			new_ptr[j][r_pos] = y;
			new_ptr[j][g_pos] = y;
			new_ptr[j][b_pos] = y;
		}
	}
	return newImage;
}

cv::Mat ImageProcess::StretchTransform(const cv::Mat &origin,
	int x1, int x2,
	double k1, double k2, double k3,
	double b1, double b2, double b3) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	if (origin.isContinuous()) {
		cols *= rows;
		rows = 1;
	}

	for (int i = 0; i < rows; ++i)
	{
		const cv::Vec3b* origin_ptr = origin.ptr<cv::Vec3b>(i);
		cv::Vec3b* new_ptr = newImage.ptr<cv::Vec3b>(i);

		for (int j = 0; j < cols; ++j)
		{
			const cv::Vec3b oldColor = origin_ptr[j];
			int gray = Gray(oldColor);
			int y = 0;
			if (gray < x1) {
				y = k1 * gray + b1;
			}
			else if (gray < x2) {
				y = k2 * gray + b2;
			}
			else {
				y = k3 * gray + b3;
			}
			y = SafeValue(y);
			new_ptr[j][r_pos] = y;
			new_ptr[j][g_pos] = y;
			new_ptr[j][b_pos] = y;
		}
	}
	return newImage;
}

cv::Mat ImageProcess::SimpleSmooth(const cv::Mat &origin) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	/*int kernel[5][5] = {
		{0,0,1,0,0},
		{0,1,3,1,0},
		{1,3,7,3,1},
		{0,1,3,1,0},
		{0,0,1,0,0}
	};*/
	int kernel[5][5] = {
		{0,0,1,0,0},
		{0,1,3,1,0},
		{1,3,1,3,1},
		{0,1,3,1,0},
		{0,0,1,0,0}
	};
	int kernelSize = 5;
	int sumKernel = 21;

	KernalCalc(origin, newImage, (const int*)kernel, kernelSize, sumKernel);
	return newImage;
}

cv::Mat ImageProcess::MeidaFilter(const cv::Mat &origin) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	int kernel[3][3] = {
		{1,1,1},
		{1,1,1},
		{1,1,1}
	};

	int kernelSize = 3;
	int kernelSum = 9;

	KernalCalc(origin, newImage, (int*)kernel, kernelSize, kernelSum);

	return newImage;
}

cv::Mat ImageProcess::LaplaceSharpen(const cv::Mat &origin) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	int kernel[3][3] = {
		{0,-1,0},
		{-1,4,-1},
		{0,-1,0}
	};

	int kernelSize = 3;
	int kernelSum = 0;
	KernalCalc(origin, newImage, (int*)kernel, kernelSize, kernelSum);
	return newImage;
}

cv::Mat ImageProcess::SobelEdge(const cv::Mat &origin) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	/* Sobel */
	double Gx[9] = {
		1.0,0.0,-1.0,
		2.0,0.0,-2.0,
		1.0,0.0,-1.0
	};
	double Gy[9] = {
		-1.0,-2.0,-1.0,
		0.0,0.0,0.0,
		1.0,2.0,1.0
	};

	cv::Mat grayImage = GreyScale(origin);

	float* sobel_norm = new float[rows * cols];
	float max = 0.0;

	for (int x = 0; x < rows -2; x++)
	{
		for (int y = 0; y < cols -2; y++)
		{
			double value_gx = 0.0;
			double value_gy = 0.0;

			for (int k = 0; k < 3; k++)
			{
				for (int p = 0; p < 3; p++)
				{
					cv::Vec3b grayColor = grayImage.at<cv::Vec3b>(x + 2 - k, y + 2 - p);
					value_gx += Gx[p * 3 + k] * grayColor[0];
					value_gy += Gy[p * 3 + k] * grayColor[0];
				}
				//sobel_norm[x + y * rows] = abs(value_gx) + abs(value_gy);
				//max = sobel_norm[x + y * rows] > max ? sobel_norm[x + y * rows] : max;

				sobel_norm[x * cols + y] = abs(value_gx) + abs(value_gy);
				max = sobel_norm[x * cols + y] > max ? sobel_norm[x * cols + y] : max;
			}
		}
	}

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			//int val = 255 - int(255.0*sobel_norm[i + j * rows] / max);
			int val = 255 - int(255.0*sobel_norm[i * cols + j] / max);
			newImage.at<cv::Vec3b>(i, j)[r_pos] = val;
			newImage.at<cv::Vec3b>(i, j)[g_pos] = val;
			newImage.at<cv::Vec3b>(i, j)[b_pos] = val;
		}
	}

	return newImage;
}

cv::Mat ImageProcess::GaussianSmoothing(const cv::Mat &origin, int radius, double sigma) {
	GaussianBlur blur(radius, sigma);
	cv::Mat newImage = blur.BlurImage(origin);
	return newImage;
}

cv::Mat ImageProcess::Binaryzation(const cv::Mat &origin) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	if (origin.isContinuous()) {
		cols *= rows;
		rows = 1;
	}

	for (int i = 0; i < rows; ++i)
	{
		const cv::Vec3b* origin_ptr = origin.ptr<cv::Vec3b>(i);
		cv::Vec3b* new_ptr = newImage.ptr<cv::Vec3b>(i);

		for (int j = 0; j < cols; ++j)
		{
			const cv::Vec3b oldColor = origin_ptr[j];
			int gray = Gray(oldColor);
			int newGray = 0;
			if (gray > 128) {
				newGray = 255;
			}
			else {
				newGray = 0;
			}
			new_ptr[j][r_pos] = newGray;
			new_ptr[j][g_pos] = newGray;
			new_ptr[j][b_pos] = newGray;
		}
	}

	return newImage;
}

cv::Mat ImageProcess::PrewittEdge(const cv::Mat& origin) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	/* Sobel */
	double Gx[9] = {
		-1.0,0.0,1.0,
		-1.0,0.0,1.0,
		-1.0,0.0,1.0
	};
	double Gy[9] = {
		1.0,1.0,1.0,
		0.0,0.0,0.0,
		-1.0,-1.0,-1.0
	};

	cv::Mat grayImage = GreyScale(origin);

	float* sobel_norm = new float[rows * cols];
	float max = 0.0;

	for (int x = 0; x < rows - 2; x++)
	{
		for (int y = 0; y < cols - 2; y++)
		{
			double value_gx = 0.0;
			double value_gy = 0.0;

			for (int k = 0; k < 3; k++)
			{
				for (int p = 0; p < 3; p++)
				{
					cv::Vec3b grayColor = grayImage.at<cv::Vec3b>(x + 2 - k, y + 2 - p);
					value_gx += Gx[p * 3 + k] * grayColor[0];
					value_gy += Gy[p * 3 + k] * grayColor[0];
				}
	
				sobel_norm[x * cols + y] = abs(value_gx) + abs(value_gy);
				max = sobel_norm[x * cols + y] > max ? sobel_norm[x * cols + y] : max;
			}
		}
	}

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int val = 255 - int(255.0*sobel_norm[i * cols + j] / max);
			newImage.at<cv::Vec3b>(i, j)[r_pos] = val;
			newImage.at<cv::Vec3b>(i, j)[g_pos] = val;
			newImage.at<cv::Vec3b>(i, j)[b_pos] = val;
		}
	}

	return newImage;
}

cv::Mat ImageProcess::ContourExtraction(const cv::Mat &origin) {
	cv::Mat newImage(origin.rows,origin.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	//origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// Èý¸öÑÕÉ«·ÖÁ¿µÄÎ»ÖÃ£¨BGR£¬¶ø²»ÊÇRGB£©
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	int pixel[8];   // µ±Ç°ÏñËØÖÜÎ§µÄ8¸öÏñËØµÄÏñËØÖµ
	cv::Mat binImg = Binaryzation(origin);

	for (int y = 1; y < rows -1; y++)
	{
		for (int x = 1; x < cols - 1; x++)
		{
			memset(pixel, 0, 8);

			int r_val = binImg.at<cv::Vec3b>(x, y)[r_pos];
			if (r_val == 0)
			{
				newImage.at<cv::Vec3b>(x, y)[r_pos] = 0;
				newImage.at<cv::Vec3b>(x, y)[g_pos] = 0;
				newImage.at<cv::Vec3b>(x, y)[b_pos] = 0;

				pixel[0] = binImg.at<cv::Vec3b>(x - 1, y - 1)[r_pos];
				pixel[1] = binImg.at<cv::Vec3b>(x - 1, y)[r_pos];
				pixel[2] = binImg.at<cv::Vec3b>(x - 1, y + 1)[r_pos];
				pixel[3] = binImg.at<cv::Vec3b>(x, y - 1)[r_pos];
				pixel[4] = binImg.at<cv::Vec3b>(x, y + 1)[r_pos];
				pixel[5] = binImg.at<cv::Vec3b>(x + 1, y - 1)[r_pos];
				pixel[6] = binImg.at<cv::Vec3b>(x + 1, y)[r_pos];
				pixel[7] = binImg.at<cv::Vec3b>(x + 1, y + 1)[r_pos];
				if (pixel[0] + pixel[1] + pixel[2] + pixel[3] + pixel[4] + pixel[5] + pixel[6] + pixel[7] == 0) {
					newImage.at<cv::Vec3b>(x, y) = cv::Vec3b(255, 255, 255);
					newImage.at<cv::Vec3b>(x, y)[r_pos] = 255;
					newImage.at<cv::Vec3b>(x, y)[g_pos] = 255;
					newImage.at<cv::Vec3b>(x, y)[b_pos] = 255;
				}	
			}
		}
	}

	return newImage;
}

cv::Mat ImageProcess::ConnectedDomain(const cv::Mat &origin){
  return cv::Mat();
}

cv::Mat ImageProcess::Dilate(const cv::Mat &origin){
    cv::Mat newImage(origin.rows,origin.cols, CV_8UC3, cv::Scalar(255, 255, 255));

    int channels = origin.channels();
    int rows = origin.rows;
    int cols = origin.cols;

    int r_pos = 2;
    int g_pos = 1;
    int b_pos = 0;

    int dilateItem[9] = {1,0,1,
                         0,0,0,
                         1,0,1};

    for (int i = 1; i < rows - 1; ++i)
    {
        for (int j = 1; j < cols - 1; ++j)
        {
            newImage.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);

            for(int m=0; m<3; m++)
            {
                for(int n=0; n<3; n++)
                {
                    if(dilateItem[m+n] == 1){
                        continue;
                    }

                    cv::Vec3b oldColor = origin.at<cv::Vec3b>(i+(n-1),j+(1-m));

                    if(oldColor[r_pos]> 128){
                        //newImage.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,255);
                      newImage.at<cv::Vec3b>(i,j)[r_pos] = 255;
                      newImage.at<cv::Vec3b>(i,j)[g_pos] = 255;
                      newImage.at<cv::Vec3b>(i,j)[b_pos] = 255;
                    }

                }
            }
        }
    }

    return newImage;
}

cv::Mat ImageProcess::Expansion(const cv::Mat &origin){
    cv::Mat newImage(origin.rows,origin.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    
    int channels = origin.channels();
    int rows = origin.rows;
    int cols = origin.cols;
    
    int r_pos = 2;
    int g_pos = 1;
    int b_pos = 0;
    
    int dilateItem[9] = {1,0,1,
        0,0,0,
        1,0,1};
    
    for (int i = 1; i < rows - 1; ++i)
    {
        for (int j = 1; j < cols - 1; ++j)
        {
            newImage.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,255);
            
            for(int m=0; m<3; m++)
            {
                for(int n=0; n<3; n++)
                {
                    if(dilateItem[m+n] == 1){
                        continue;
                    }
                    
                    cv::Vec3b oldColor = origin.at<cv::Vec3b>(i+(n-1),j+(1-m));
                    
                    if(oldColor[r_pos] < 128){
                        //newImage.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,255);
                        newImage.at<cv::Vec3b>(i,j)[r_pos] = 0;
                        newImage.at<cv::Vec3b>(i,j)[g_pos] = 0;
                        newImage.at<cv::Vec3b>(i,j)[b_pos] = 0;
                    }
                    
                }
            }
        }
    }
    
    
    return newImage;
}

cv::Mat ImageProcess::Opening(const cv::Mat &origin){
    cv::Mat afterDilate = Dilate(origin);
    cv::Mat afterExpansion = Expansion(afterDilate);
    
    return afterExpansion;
}
cv::Mat ImageProcess::Closing(const cv::Mat &origin){
    cv::Mat afterExpansion = Expansion(origin);
    cv::Mat afterDilate = Dilate(afterExpansion);
    
    return afterDilate;
}

cv::Mat ImageProcess::Thinning(const cv::Mat &origin){
    cv::Mat binImg = Binaryzation(origin);
    int rows = origin.rows;
    int cols = origin.cols;
    
    int r_pos = 2;
    int g_pos = 1;
    int b_pos = 0;
    
    int neighbor[8];
    cv::Mat mark(origin.rows,origin.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    
    bool loop = true;
    
    int markNum = 0;
    while(loop)
    {
        loop = false;
        markNum = 0;
        for(int y=1; y<cols-1; y++)
        {
            for(int x=1; x<rows-1; x++)
            {
                // 1
                // 如果当前像素是黑色（背景）
                if(binImg.at<cv::Vec3b>(x,y)[r_pos] == 0)  continue;
                
                // 根据当前像素以及领域构建一个窗口（核）
                neighbor[0] = binImg.at<cv::Vec3b>(x+1,y)[r_pos];
                neighbor[1] = binImg.at<cv::Vec3b>(x+1, y-1)[r_pos];
                neighbor[2] = binImg.at<cv::Vec3b>(x, y-1)[r_pos];
                neighbor[3] = binImg.at<cv::Vec3b>(x-1, y-1)[r_pos];
                neighbor[4] = binImg.at<cv::Vec3b>(x-1, y)[r_pos];
                neighbor[5] = binImg.at<cv::Vec3b>(x-1, y+1)[r_pos];
                neighbor[6] = binImg.at<cv::Vec3b>(x, y+1)[r_pos];
                neighbor[7] = binImg.at<cv::Vec3b>(x+1, y+1)[r_pos];
                
                // 2
                // 统计当前领域中白色（物体）像素的数量，如果小于2或者大于6，那么不处理
                int np = (neighbor[0]+neighbor[1]+neighbor[2]+neighbor[3]
                          +neighbor[4]+neighbor[5]+neighbor[6]+neighbor[7])/255;
                if (np<2|| np >6)   continue;
                
                // 3
                // 统计领域内，相邻像素由黑变白的变化次数
                int sp = 0;
                for(int i=1; i<8; i++)
                {
                    if(neighbor[i] - neighbor[i-1] == 255)
                        sp++;
                    
                }
                if(neighbor[0] - neighbor[7] == 255)
                    sp++;
                
                // 如果次数大于1，那么不处理
                if (sp!=1)  continue;
                
                // 4
                // 判断准则
                if(neighbor[2]&neighbor[0]&neighbor[4]!=0)
                    continue;
                //条件5：p2*p6*p4==0
                // 判断准则
                if(neighbor[2]&neighbor[6]&neighbor[4]!=0)
                    continue;
                
                // 如果条件都满足，那么删除该像素（不是立即删除，先打上标记，后面在删除）
                //标记删除
                mark.at<cv::Vec3b>(x,y) = cv::Vec3b(1,1,1);
                markNum ++;
                loop = true;
            }
        }
        
        // 将标记删除的点置为背景色
        
        for(int y=0; y< cols; y++)
        {
            for(int x=0; x< rows; x++)
            {
                if(mark.at<cv::Vec3b>(x,y)[r_pos] == 1)
                {
                    binImg.at<cv::Vec3b>(x,y) = cv::Vec3b(0,0,0);
                }
            }
        }
    }
    
    
    markNum = 0;
    
    return binImg;
    
}

//---------------------------------------
cv::Mat ImageProcess::Translation(const cv::Mat &origin, int x, int y) {
	cv::Mat newImage(origin.rows, origin.cols, CV_8UC3, cv::Scalar(0, 0, 0));
	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			cv::Vec3b oldColor = origin.at<cv::Vec3b>(i,j);
			int dst_i = i + x;
			int dst_j = j + y;
			if (dst_i >= rows || dst_j >= cols) {
				continue;
			}
			newImage.at<cv::Vec3b>(dst_i, dst_j) = oldColor;
		}
	}
	return newImage;
}

cv::Mat ImageProcess::Mirror(const cv::Mat &origin, bool horizontal, bool vertical) {
	cv::Mat newImage(origin.rows, origin.cols, CV_8UC3, cv::Scalar(0, 0, 0));
	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			cv::Vec3b oldColor = origin.at<cv::Vec3b>(i, j);
			
			int dst_i = i;
			int dst_j = j;

			if (vertical) {
				dst_i = rows - i - 1;
			}
			if (horizontal) {
				dst_j = cols - j - 1;
			}
			newImage.at<cv::Vec3b>(dst_i, dst_j) = oldColor;
		}
	}
	return newImage;
}

cv::Mat ImageProcess::Zoom(const cv::Mat &origin, double x_scale ,double y_scale ) {

	int newRows = origin.rows * x_scale + 0.5;
	int newCols = origin.cols * y_scale + 0.5;

	cv::Mat newImage(newRows, newCols, CV_8UC3, cv::Scalar(0, 0, 0));
	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	for (int i = 0; i < newRows; ++i) {
		for (int j = 0; j < newCols; ++j) {
			int origin_i = i / x_scale + 0.5;
			int origin_j = j / y_scale + 0.5;

			if (origin_i >= 0 && origin_i < rows && origin_j >= 0 && origin_j < cols) {
				cv::Vec3b oldColor = origin.at<cv::Vec3b>(origin_i, origin_j);
				newImage.at<cv::Vec3b>(i, j) = oldColor;
			}
			else {
				newImage.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
			}
		}
	}

	return newImage;
}

double angle_to_radian(double degree, double min = 0, double second = 0)
{
	double flag = (degree < 0) ? -1.0 : 1.0;          //判断正负  
	if (degree < 0)
	{
		degree = degree * (-1.0);
	}
	double angle = degree + min / 60 + second / 3600;
	double result = flag * (angle * M_PI) / 180;
	return result; 
}

// todo,还有问题
cv::Mat ImageProcess::Rotate(const cv::Mat &origin, double angle) {
	
	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	double srcx1, srcy1, srcx2, srcy2, srcx3, srcy3, srcx4, srcy4;
	double dstx1, dsty1, dstx2, dsty2, dstx3, dsty3, dstx4, dsty4;

	angle = angle_to_radian(angle);

	double cosa = (double)cos((double)angle);
	double sina = (double)sin((double)angle);

	srcx1 = -0.5 * cols;
	srcy1 = 0.5 * rows;
	srcx2 = 0.5 * cols;
	srcy2 = 0.5 * rows;
	srcx3 = -0.5 * cols;
	srcy3 = -0.5 * rows;
	srcx4 = 0.5 * cols;
	srcy4 = -0.5 * rows;

	dstx1 = cosa * srcx1 + sina * srcy1;
	dsty1 = -sina * srcx1 + cosa * srcy1;
	dstx2 = cosa * srcx2 + sina * srcy2;
	dsty2 = -sina * srcx2 + cosa * srcy2;
	dstx3 = cosa * srcx3 + sina * srcy3;
	dsty3 = -sina * srcx3 + cosa * srcy3;
	dstx4 = cosa * srcx4 + sina * srcy4;
	dsty4 = -sina * srcx4 + cosa * srcy4;

	int newRows = std::max<double>(fabs(dsty4 - dsty1), fabs(dsty3 - dsty2)) + 0.5;
	int newCols = std::max<double>(fabs(dstx4 - dstx1), fabs(dstx3 - dstx2)) + 0.5;

	double num1 = -0.5 * newCols* cosa - 0.5 * newRows * sina + 0.5 * cols;
	double num2 = -0.5 * newCols* sina - 0.5 * newRows * cosa + 0.5 * rows;

	cv::Mat newImage(newRows, newCols, CV_8UC3, cv::Scalar(0, 0, 0));

	for (int i = 0; i < newRows; ++i) {
		for (int j = 0; j < newCols; ++j) {
			int org_i = i * cosa + j * sina + num1;
			int org_j = -1.0 * i * sina + j * cosa + num2;

			if (org_i >= 0 && org_i < rows && org_j >= 0 && org_j < cols) {
				cv::Vec3b oldColor = origin.at<cv::Vec3b>(org_i, org_j);
				newImage.at<cv::Vec3b>(i, j) = oldColor;
			}
		}
	}
	return newImage;

}

void ImageProcess::Histogram(const cv::Mat &origin, double histogram[3][256]) {
	int gray[3][256] = { 0 };

	int rows = origin.rows;
	int cols = origin.cols;

	if (origin.isContinuous()) {
		cols *= rows;
		rows = 1;
	}

	for (int i = 0; i < rows; ++i)
	{
		const cv::Vec3b* origin_ptr = origin.ptr<cv::Vec3b>(i);

		for (int j = 0; j < cols; ++j)
		{
			gray[0][origin_ptr[j][0]]++;
			gray[1][origin_ptr[j][1]]++;
			gray[2][origin_ptr[j][2]]++;
		}
	}
	
	for (int i = 0; i < 256; ++i) {
		histogram[0][i] = gray[0][i] / (rows * cols * 1.0);
		histogram[1][i] = gray[1][i] / (rows * cols * 1.0);
		histogram[2][i] = gray[2][i] / (rows * cols * 1.0);
	}
}

cv::Mat ImageProcess::HistogramEqualization(const cv::Mat &origin) {
	cv::Mat newImage(origin.rows, origin.cols, CV_8UC3, cv::Scalar(0, 0, 0));

	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	double histogram[3][256];
	Histogram(origin, histogram);

	double newGray[3][256] = { 0 };
	double temp[3][256] = { 0 };
	for (int i = 0; i < 256; ++i) {
		if (i == 0) {
			temp[r_pos][i] = histogram[r_pos][i];
			temp[g_pos][i] = histogram[g_pos][i];
			temp[b_pos][i] = histogram[b_pos][i];
		}
		else {
			temp[r_pos][i] = temp[r_pos][i - 1] + histogram[r_pos][i];
			temp[g_pos][i] = temp[g_pos][i - 1] + histogram[g_pos][i];
			temp[b_pos][i] = temp[b_pos][i - 1] + histogram[b_pos][i];
		}
		newGray[r_pos][i] = (255.0 * temp[r_pos][i] + 0.5);
		newGray[g_pos][i] = (255.0 * temp[g_pos][i] + 0.5);
		newGray[b_pos][i] = (255.0 * temp[b_pos][i] + 0.5);
	}

	int rows = origin.rows;
	int cols = origin.cols;

	

	if (origin.isContinuous()) {
		cols *= rows;
		rows = 1;
	}

	for (int i = 0; i < rows; ++i)
	{
		const cv::Vec3b* src_ptr = origin.ptr<cv::Vec3b>(i);
		cv::Vec3b* dst_ptr = newImage.ptr<cv::Vec3b>(i);

		for (int j = 0; j < cols; ++j)
		{
			dst_ptr[j][r_pos] = newGray[r_pos][src_ptr[j][r_pos]];
			dst_ptr[j][g_pos] = newGray[g_pos][src_ptr[j][g_pos]];
			dst_ptr[j][b_pos] = newGray[b_pos][src_ptr[j][b_pos]];
		}
	}
	return newImage;
}

cv::Mat ImageProcess::Add(const cv::Mat &img1,double weight1,const cv::Mat &img2,double weight2){
    int newRows = std::max(img1.rows,img2.rows);
    int newCols = std::max(img1.cols,img2.cols);
    
    cv::Mat newImage(newRows, newCols, CV_8UC3, cv::Scalar(0, 0, 0));
    
    for(int i = 0; i < newRows; ++i){
        for(int j = 0; j < newCols; ++j){
            cv::Vec3b color;
            
            int i1 = i - (newRows - img1.rows) / 2;
            int j1 = j - (newCols - img1.cols) / 2;
            
            int i2 = i - (newRows - img2.rows) / 2;
            int j2 = j - (newCols - img2.cols) / 2;
            
            bool valid1 = false;
            bool valid2 = false;
            if(i1 >= 0 && j1 >= 0 && i1 < img1.rows && j1 < img1.cols){
                valid1 = true;
            }
            if(i2 >= 0 && j2 >= 0 && i2 < img2.rows && j2 < img2.cols){
                valid2 = true;
            }
            
            if(!valid1  && !valid2){
                continue;
            }
            
            if(valid1 && valid2){
                cv::Vec3b color1 = img1.at<cv::Vec3b>(i1,j1);
                cv::Vec3b color2 = img2.at<cv::Vec3b>(i2,j2);
                color = weight1 * color1 + weight2 * color2;
            }
            else {
                if(valid1){
                    cv::Vec3b color1 = img1.at<cv::Vec3b>(i1,j1);
                    color = color1;
                }
                else{
                    cv::Vec3b color2 = img2.at<cv::Vec3b>(i2,j2);
                    color = color2;
                }
            }
            
            newImage.at<cv::Vec3b>(i,j) = color;
        }
    }
    
    return newImage;
}

cv::Mat ImageProcess::Cut(const cv::Mat &origin,int width,int height,bool center ){
    assert(width <= origin.cols);
    assert(height <= origin.rows);
    
    int newRows = height;
    int newCols = width;
    
    cv::Mat newImage(newRows, newCols, CV_8UC3, cv::Scalar(0, 0, 0));
    
    int orgin_i = 0;
    int orgin_j = 0;
    if(center){
        orgin_i = (origin.rows - newRows) / 2;
        orgin_j = (origin.cols - newCols) / 2;
    }
    
    for(int i = 0; i < newRows; ++i){
        for(int j = 0; j < newCols; ++j){
            int orgin_i = i;
            int orgin_j = j;
            if(center){
                orgin_i = i + (origin.rows - newRows) / 2;
                orgin_j = j + (origin.cols - newCols) / 2;
            }
            uchar* new_ptr = newImage.ptr();
            newImage.at<cv::Vec3b>(i,j) = origin.at<cv::Vec3b>(orgin_i,orgin_j);
        }
    }
    return newImage;
}
