#include<time.h>
#include<iostream>
# include <stdio.h>
# include <stdlib.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
//#include<opencv/cxcore.h>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>


//CUDA RunTime API
#include <cuda_runtime.h>
using namespace std;
using namespace cv;
#define BLOCK_SIZE 32 //每个块的线程数 32*32
#define PIC_BLOCK 1 //每个线程处理图像块 5*5
//#define GRID_SIZE 16
//const int block_num=480;
//static const int N = 25;

//__device__ int flag;

#define uchar unsigned char

//超清公式

#define RGB2Y(R, G, B)  ( 16  + 0.183f * (R) + 0.614f * (G) + 0.062f * (B) )
#define RGB2U(R, G, B)  ( 128 - 0.101f * (R) - 0.339f * (G) + 0.439f * (B) )
#define RGB2V(R, G, B)  ( 128 + 0.439f * (R) - 0.399f * (G) - 0.040f * (B) )

#define YUV2R(Y, U, V) ( 1.164f *((Y) - 16) + 1.792f * ((V) - 128) )
#define YUV2G(Y, U, V) ( 1.164f *((Y) - 16) - 0.213f *((U) - 128) - 0.534f *((V) - 128) )
#define YUV2B(Y, U, V) ( 1.164f *((Y) - 16) + 2.114f *((U) - 128))

#define CLIPVALUE(x, minValue, maxValue) ((x) < (minValue) ? (minValue) : ((x) > (maxValue) ? (maxValue) : (x)))

__global__ static void __RgbToYuv420p(const unsigned char* dpRgbData, size_t rgbPitch, unsigned char* dpYuv420pData, size_t yuv420Pitch, int width, int height)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("index=%d\n",index);
	int w = index % yuv420Pitch; //线程对应的RGB图像列
	int h = index / yuv420Pitch; //线程对应的RGB图像行

	if (w >= width || h >= height)
		return;

    //printf("index=%d\t",index);
    //printf("w=%d,h=%d\n",w,h);
	unsigned char* dp_y_data = dpYuv420pData; //y通道存在前width*height数组中
	unsigned char* dp_u_data = dp_y_data + height * yuv420Pitch;  //yuv420Pitch RGB图像的列长
	unsigned char* dp_v_data = dp_u_data + height * yuv420Pitch / 4;
    //printf("h=%d,w=%d,rgbPitch=%d\t",h,w,rgbPitch);
	unsigned char b = dpRgbData[h * rgbPitch + w * 3 + 0]; //rgbPitch RGB图像的列长
	unsigned char g = dpRgbData[h * rgbPitch + w * 3 + 1];
	unsigned char r = dpRgbData[h * rgbPitch + w * 3 + 2];

	dp_y_data[h   * yuv420Pitch + w] = (unsigned char)(CLIPVALUE(RGB2Y(r, g, b), 0, 255));
	int num = h / 2 * width / 2 + w / 2;
	int offset = num / width * (yuv420Pitch - width);

	if (h % 2 == 0 && w % 2 == 0)
	{
		dp_u_data[num + offset] = (unsigned char)(CLIPVALUE(RGB2U(r, g, b), 0, 255));
		dp_v_data[num + offset] = (unsigned char)(CLIPVALUE(RGB2V(r, g, b), 0, 255));
	}
}



//void rgb2yuv(cv::Mat& rgb_img, uchar* &yuv_img_buff,const char* Flag)
void rgb2yuv(cv::Mat& rgb_img, uchar* &yuv_img_buff)
{
	//声明变量
	//clock_t start4 = clock();
    //printf("in GPU\n");
	//bgr图像
	uchar* cuda_src = NULL;
	//yuv图像 destination
	uchar* cuda_dst = NULL;
    int len_src=sizeof(uchar)*rgb_img.rows * rgb_img.cols * 3; //RGB图像大小
	
    cudaMalloc((void**)&cuda_src, len_src);
	//clock_t end4 = clock();
	//std::cout << "first cudaMalloc use = " << ((double)(end4 - start4) / CLOCKS_PER_SEC) * 1000 << "ms" << std::endl;
    
	//clock_t start5 = clock();
	//分块
    //bx*by块，每个块 BLOCK_SIZE*BLOCK_SIZE个线程（32的倍数最好），每个线程负责pic_block*pic_block小块
    int bx = ((rgb_img.cols + BLOCK_SIZE - 1) / BLOCK_SIZE + PIC_BLOCK - 1) / PIC_BLOCK;
    int by = ((rgb_img.rows + BLOCK_SIZE - 1) / BLOCK_SIZE + PIC_BLOCK - 1) / PIC_BLOCK;
    //printf("bx=%d,by=%d\n",bx,by);
    dim3 blocks(bx*by);
    dim3 threads(BLOCK_SIZE*BLOCK_SIZE);
	//clock_t end5 = clock();
	//std::cout << "block use = " << ((double)(end5 - start5) / CLOCKS_PER_SEC) * 1000 << "ms" << std::endl;

    /*
    YUV420图像的U/V分量在水平和垂直方向上downsample，在水平和垂直方向上的数据都只有Y分量的一半。
    因此总体来说，U/V分量的数据量分别只有Y分量的1/4，不能作为Mat类型的一个channel。
    所以通常YUV420图像的全部数据存储在Mat的一个channel，比如CV_8UC1，这样对于Mat来说，
    图像的大小就有变化。对于MxN（rows x cols，M行N列）的BGR图像（CV_8UC3)，
    其对应的YUV420图像大小是(3M/2)xN（CV_8UC1）。
    前MxN个数据是Y分量，后(M/2)xN个数据是U/V分量，UV数据各占一半。
    */
    //if(0==strcmp(Flag,"BGR_to_YUV_420P"))
    {
		//clock_t start3 = clock();
        //分配空间
        int len_dst=sizeof(uchar) * rgb_img.rows * rgb_img.cols * 3 / 2; //YUV图像大小
        cudaMalloc((void**)&cuda_dst,len_dst);
		//clock_t end3 = clock();
		//std::cout << "cudaMalloc use = " << ((double)(end3 - start3) / CLOCKS_PER_SEC) * 1000 << "ms" << std::endl;

        //初始化为0
        //cudaMemset(change, 0, sizeof(float)*bx*by);
		//clock_t start = clock();
        //cpu->gpu
        cudaMemcpy(cuda_src, rgb_img.data, len_src, cudaMemcpyHostToDevice);
		//clock_t end = clock();
		//std::cout << "cpu2gpu use = " << ((double)(end - start) / CLOCKS_PER_SEC) * 1000 << "ms" << std::endl;

        size_t rgbPitch=3*rgb_img.cols; //记得乘3！！！！！rgb为3通道

        //printf("rgbPitch=%zd\n",rgbPitch);
        size_t yuv420Pitch=rgb_img.cols;

		//clock_t start1 = clock();
        __RgbToYuv420p <<<blocks, threads >>> (cuda_src, rgbPitch,cuda_dst,yuv420Pitch,rgb_img.cols, rgb_img.rows);
        //clock_t end1 = clock();
		//std::cout << "gpu use = " << ((double)(end1 - start1) / CLOCKS_PER_SEC) * 1000 << "ms" << std::endl;
		
		//gpu->cpu
		//clock_t start2 = clock();
        yuv_img_buff= (uchar*)malloc(sizeof(uchar)*rgb_img.rows*rgb_img.cols*3/2);
        cudaMemcpy(yuv_img_buff, cuda_dst, len_dst, cudaMemcpyDeviceToHost);
		//clock_t end2 = clock();
		//std::cout << "gpu2cpu use = " << ((double)(end2 - start2) / CLOCKS_PER_SEC) * 1000 << "ms" << std::endl;
		//std::cout << endl;
    }




#if 0
    else if(0==strcmp(Flag,"BGR_to_YUV_NV12"))
    {
        //分配空间
        int len_dst=sizeof(uchar)*rgb_img.rows*rgb_img.cols*3/2; //YUV图像大小
        cudaMalloc((void**)&cuda_dst,len_dst);
        //初始化为0
        //cudaMemset(change, 0, sizeof(float)*bx*by);
        //cpu->gpu
        cudaMemcpy(cuda_src, rgb_img.data, len_src, cudaMemcpyHostToDevice);

        size_t rgbPitch=3*rgb_img.cols; //记得乘3！！！！！rgb为3通道
        printf("rgbPitch=%zd\n",rgbPitch);
        size_t yuvnv12Pitch=rgb_img.cols;

        __RgbToNv12 <<<blocks, threads >>> (cuda_src, rgbPitch,cuda_dst,yuvnv12Pitch,rgb_img.cols, rgb_img.rows);
        //gpu->cpu
        yuv_img_buff= (uchar*)malloc(sizeof(uchar)*rgb_img.rows*rgb_img.cols*3/2);
        cudaMemcpy(yuv_img_buff, cuda_dst, len_dst, cudaMemcpyDeviceToHost);

#if 0
        Mat yuv_img = Mat::zeros(rgb_img.rows*3/2, rgb_img.cols, CV_8UC1);
        cudaMemcpy(yuv_img.data, cuda_dst, len_dst, cudaMemcpyDeviceToHost);
        //printf("sizeof(uchar)*rgb_img->rows*rgb_img->cols*3/2=%d\n",sizeof(uchar)*rgb_img->rows*rgb_img->cols*3/2);
        printf("-----------------------");
        //yuv2BGR
        Mat rgbimg(rgb_img.rows,rgb_img.cols,CV_8UC3);
        cvtColor(yuv_img,rgbimg,COLOR_YUV2RGB_NV12);
        imwrite("yuv.jpg",rgbimg);
#endif

    }
    else if(0==strcmp(Flag,"BGR_to_YUV_422p"))
    {
        //分配空间
        int len_dst=sizeof(uchar)*rgb_img.rows*rgb_img.cols*2; //YUV图像大小
        cudaMalloc((void**)&cuda_dst,len_dst);
        //初始化为0
        //cudaMemset(change, 0, sizeof(float)*bx*by);
        //cpu->gpu
        cudaMemcpy(cuda_src, rgb_img.data, len_src, cudaMemcpyHostToDevice);

        size_t rgbPitch=3*rgb_img.cols; //记得乘3！！！！！rgb为3通道
        printf("rgbPitch=%zd\n",rgbPitch);
        size_t yuv422pPitch=rgb_img.cols;

        __RgbToYuv422p <<<blocks, threads >>> (cuda_src, rgbPitch,cuda_dst,yuv422pPitch,rgb_img.cols, rgb_img.rows);
        //gpu->cpu
        yuv_img_buff= (uchar*)malloc(sizeof(uchar)*rgb_img.rows*rgb_img.cols*2);
        cudaMemcpy(yuv_img_buff, cuda_dst, len_dst, cudaMemcpyDeviceToHost);
    }
#endif
	//free
	cudaFree(cuda_src);
	cudaFree(cuda_dst);

}

