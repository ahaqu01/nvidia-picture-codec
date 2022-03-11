#include <iostream>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"
#include <cuda.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h>
#include <unistd.h>
#include <fstream>
#include "./bgr2yuv.cuh"

#include <fstream>  
#include <opencv2/opencv.hpp> 
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/cudacodec.hpp>
#include "opencv2/cudaimgproc.hpp"

//NVJPEG Header

#include "jpeg_encode_main.h"


using namespace std;
using namespace cv;

bool InitCUDA()
{
	int count;
	cudaGetDeviceCount(&count);
	if (count == 0)
	{
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	int i;
	for (i = 0; i < count; i++)
	{
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
		{

			/****
            printf("name:%s\n",prop.name);
            printf("totalGlobalMem:%d\n",prop.totalGlobalMem);
            printf("totalGlobalMem:%d\n",prop.totalGlobalMem/1024);
            printf("totalGlobalMem:%d\n",prop.totalGlobalMem/1024/1024);
            printf("totalGlobalMem:%d\n",prop.totalGlobalMem/1024/1024/1024);
            printf("multiProcessorCount:%d\n",prop.multiProcessorCount);
            printf("maxThreadsPerBlock:%d\n",prop.maxThreadsPerBlock);
            printf("major:%d,minor:%d\n",prop.major,prop.minor);
			***/
			
            if (prop.major >= 1)
			{
				break;
			}
		}
	}
	if (i == count)
	{
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}
	cudaSetDevice(i);
	return true;
}
void saveyuv(uchar* &yuvdata,int rows,int cols)
{
    //creat yuv file
    FILE *fp;
    fp=fopen("./output.yuv","w+");
    if(NULL==fp)
    {
        printf("The file doesn't exist!\n");
        return;
    }
    fwrite(yuvdata,1,rows*cols*3/2,fp);
	// for(int i = 0; i < rows * cols * 3 /2; ++i)
	// {
		// fwrite(yuvdata + i,1,1,fp);
	// }

    return;
}

/**
**Incoming parameters : yuv_data (yuv420p original data)
**Incoming and outgoing parameters : jpg_data (jpg data)
**Incoming parameters : w (picture width)
**Incoming parameters : h (picture height)
**/
int YUV_to_JPG(uchar * yuv_data, int w, int h, uchar * jpg_data)
{
	context_t ctx;
	clock_t start = clock();
	int buf_size =  jpeg_encode_proc(ctx, w, h, yuv_data, jpg_data);
	clock_t end = clock();
	std::cout << "yuv2jpg use = " << ((double)(end - start) / CLOCKS_PER_SEC) * 1000 << "ms" << std::endl;
	return buf_size;
}

void GPU_BGR_to_YUV(const char* FilePath)
{
	
    VideoCapture cap;
	cap.open(FilePath);
	if (!cap.isOpened())
	{
		std::cerr << "can not open camera or video file" << std::endl;
		return;
	}

    Mat  rgb_img;
    while(1)
	{
		cap >> rgb_img;
		if (rgb_img.empty())
		{
			break;
		}
		
		for(int i = 0; i < 1; ++i)
		{
			clock_t start0 = clock();
			uchar * yuv_img_buff = NULL;
			//bgr_to_yuv420p (use gpu cuda)
			rgb2yuv(rgb_img, yuv_img_buff);
			
			clock_t end0 = clock();
			std::cout << "bgr2yuv use = " << ((double)(end0 - start0) / CLOCKS_PER_SEC) * 1000 << "ms" << std::endl;
	
			saveyuv(yuv_img_buff,rgb_img.rows,rgb_img.cols);

			
			unsigned long buf_size = rgb_img.cols * rgb_img.rows * 3 / 2;
			uchar * jpg_data = new unsigned char[buf_size]; //jpg data 
			//yuv420p_to_jpg(use jeston nano NVJPEG)
			int size =  YUV_to_JPG(yuv_img_buff, rgb_img.cols, rgb_img.rows, jpg_data);
			
			
			//save jpg picture data
			FILE *fp;
			fp=fopen("./bbb.jpg","w+");
			if(NULL==fp)
			{
				printf("The file doesn't exist!\n");
				return;
			}
			fwrite(jpg_data, 1, size, fp);
			
			delete[] jpg_data;
			
			
			cout << "____________________________________________   " << i << endl;
		
		}
	
#ifdef SAVE_YUV
        //saveyuv
        saveyuv(yuv_img_buff,rgb_img.rows,rgb_img.cols);
#endif

#ifdef YUV2BGR
        //yuv_img_buff2BGR
        Mat rgbimg(rgb_img.rows,rgb_img.cols,CV_8UC3);
        yuv2bgr(yuv_img_buff,rgbimg);
#endif

        //free(yuv_img_buff);
	}
	exit(0);
    return;
}




int main(int argc, const char** argv)
{
    //初始化CUDA设备
	if (!InitCUDA())
	{
		return 0;
	}
	printf("HelloWorld, CUDA has been initialized.\n");


    GPU_BGR_to_YUV("./1.jpg");

    return 0;
}
