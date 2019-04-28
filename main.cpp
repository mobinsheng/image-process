// OpenCV-vs-demo.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//


#include <iostream>

#include "image_process.h"

#define ROOT //"/Users/mobinsheng/Downloads/image-process/"

#define lena_name "img/Lena.jpg"
#define Airplane_name "img/Airplane.jpg"
#define Baboon_name "img/Baboon.jpg"
#define Fruits_name "img/Fruits.jpg"
#define logo_name "img/logo.png"
#define outfile_name "img/out.jpg"

#define water_mark "img/w.png"

#define src_win_name "src"
#define dst_win_name "dst"

int main()
{
    cv::Mat img = ImageProcess::read_image(ROOT lena_name);//lena_name logo_name
    cv::Mat dst = ImageProcess::HistogramEqualization(img);
    //cv::Mat dst2 = ImageProcess::Expansion(img);
	ImageProcess::show_image(src_win_name, img);
	ImageProcess::show_image(dst_win_name, dst);
    //ImageProcess::show_image("xxx", dst2);
	cv::waitKey(0);
	return 0;
}
