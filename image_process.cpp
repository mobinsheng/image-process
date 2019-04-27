#include "image_process.h"
//通道的顺序是BGR而非RGB

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

	// 三个颜色分量的位置（BGR，而不是RGB）
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

	// 三个颜色分量的位置（BGR，而不是RGB）
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

	// 三个颜色分量的位置（BGR，而不是RGB）
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

// 亮度调整
cv::Mat ImageProcess::Brightness(int delta, const cv::Mat& origin) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// 三个颜色分量的位置（BGR，而不是RGB）
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

	// 三个颜色分量的位置（BGR，而不是RGB）
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

	// 三个颜色分量的位置（BGR，而不是RGB）
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
// 线性灰度变换 y = ax + b
cv::Mat ImageProcess::LinearLevelTransformation( const cv::Mat& origin, double a, double b) {
	cv::Mat newImage;
	origin.copyTo(newImage);

	int channels = origin.channels();
	int rows = origin.rows;
	int cols = origin.cols;

	// 三个颜色分量的位置（BGR，而不是RGB）
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

	// 三个颜色分量的位置（BGR，而不是RGB）
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

	// 三个颜色分量的位置（BGR，而不是RGB）
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

	// 三个颜色分量的位置（BGR，而不是RGB）
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

	// 三个颜色分量的位置（BGR，而不是RGB）
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

	// 三个颜色分量的位置（BGR，而不是RGB）
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

	// 三个颜色分量的位置（BGR，而不是RGB）
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

	// 三个颜色分量的位置（BGR，而不是RGB）
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

	// 三个颜色分量的位置（BGR，而不是RGB）
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

	// 三个颜色分量的位置（BGR，而不是RGB）
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

	// 三个颜色分量的位置（BGR，而不是RGB）
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

	// 三个颜色分量的位置（BGR，而不是RGB）
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

	// 三个颜色分量的位置（BGR，而不是RGB）
	int r_pos = 2;
	int g_pos = 1;
	int b_pos = 0;

	int pixel[8];   // 当前像素周围的8个像素的像素值
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