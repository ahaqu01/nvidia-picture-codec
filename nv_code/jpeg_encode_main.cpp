
#include "jpeg_encode_main.h"


void abort(context_t * ctx)
{
    ctx->got_error = true;
    ctx->conv->abort();
}

/**
 * Callback function called after capture plane dqbuffer of NvVideoConverter class.
 * See NvV4l2ElementPlane::dqThread() in sample/common/class/NvV4l2ElementPlane.cpp
 * for details.
 *
 * @param v4l2_buf       : dequeued v4l2 buffer
 * @param buffer         : NvBuffer associated with the dequeued v4l2 buffer
 * @param shared_buffer  : Shared NvBuffer if the queued buffer is shared with
 *                         other elements. Can be NULL.
 * @param arg            : private data set by NvV4l2ElementPlane::startDQThread()
 *
 * @return               : true for success, false for failure (will stop DQThread)
 */
void set_defaults(context_t * ctx)
{
    memset(ctx, 0, sizeof(context_t));
    ctx->perf = false;
    //ctx->use_fd = true;
	ctx->use_fd = false;
    ctx->in_pixfmt = V4L2_PIX_FMT_YUV420M;
    ctx->stress_test = 1;
    ctx->quality = 50;
}

/**
 * Class NvJPEGEncoder encodes YUV420 image to JPEG.
 * NvJPEGEncoder::encodeFromBuffer() encodes from software buffer memory
 * which can be access by CPU directly.
 * NvJPEGEncoder::encodeFromFd() encodes from hardware buffer memory which is faster
 * than NvJPEGEncoder::encodeFromBuffer() since the latter involves conversion
 * from software buffer memory to hardware buffer memory.
 *
 * When using NvJPEGEncoder::encodeFromFd(), class NvVideoConverter is used to
 * convert MMAP buffer (CPU buffer holding YUV420 image) to hardware buffer memory
 * (DMA buffer fd). There may be YUV420 to NV12 conversion depends on commandline
 * argument.
 */
int jpeg_encode_proc(context_t& ctx, int w, int h, unsigned char * yuv_data, unsigned char * jpg_data)
{
    int ret = 0;
    int error = 0;
    int iterator_num = 1;

    set_defaults(&ctx);
    ctx.in_width = w;
    ctx.in_height = h;   
	
    //ctx.out_file = new ofstream("aaa.jpg");
    //TEST_ERROR(!ctx.out_file->is_open(), "Could not open output file", cleanup);

    ctx.jpegenc = NvJPEGEncoder::createJPEGEncoder("jpenenc");
    TEST_ERROR(!ctx.jpegenc, "Could not create Jpeg Encoder", cleanup);

    if (ctx.perf)
    {
        iterator_num = PERF_LOOP;
        ctx.jpegenc->enableProfiling();
    }

    ctx.jpegenc->setCropRect(ctx.crop_left, ctx.crop_top,
            ctx.crop_width, ctx.crop_height);

    if(ctx.scaled_encode)
    {
      ctx.jpegenc->setScaledEncodeParams(ctx.scale_width, ctx.scale_height);
    }


    /**
     * Case 1:
     * Read YUV420 image from file system to CPU buffer, encode by
     * encodeFromBuffer() then write to file system.
     */
	unsigned long out_buf_size = ctx.in_width * ctx.in_height * 3 / 2;
    if (!ctx.use_fd)
    {
       
        //unsigned char *out_buf = new unsigned char[out_buf_size];
		
        NvBuffer buffer(V4L2_PIX_FMT_YUV420M, ctx.in_width,
                ctx.in_height, 0);

        buffer.allocateMemory();		
		
        //ret = read_video_frame(ctx.in_file, buffer);
        ret = read_video_frame(yuv_data, buffer, ctx.in_width, ctx.in_height);
		
        TEST_ERROR(ret < 0, "Could not read a complete frame from file", cleanup);

		//std::cout << "original buf size = " << out_buf_size << endl;
		
            ret = ctx.jpegenc->encodeFromBuffer(buffer, JCS_YCbCr, &jpg_data,
                    out_buf_size, ctx.quality);
            //TEST_ERROR(ret < 0, "Error while encoding from buffer", cleanup);
		
		//strncpy((char *)jpg_data, (char *)out_buf, out_buf_size);
		//std::cout << "buf size = " << out_buf_size << endl;
        //ctx.out_file->write((char *) jpg_data, out_buf_size);
        
        goto cleanup;
    }

cleanup:
    if (ctx.perf)
    {
        ctx.jpegenc->printProfilingStats(cout);
    }

    if (ctx.conv && ctx.conv->isInError())
    {
        cerr << "VideoConverter is in error" << endl;
        error = 1;
    }

    if (ctx.got_error)
    {
        error = 1;
    }

    //delete ctx.in_file;
    //delete ctx.out_file;
    /**
     * Destructors do all the cleanup, unmapping and deallocating buffers
     * and calling v4l2_close on fd
     */
    delete ctx.conv;
    delete ctx.jpegenc;

    //free(ctx.in_file_path);
    //free(ctx.out_file_path);

    //return -error;
	return out_buf_size;
}

#if 0
int main(int argc, char *argv[])
{
    context_t ctx;
    int ret = 0;
    /* save iterator number */
    int iterator_num = 0;

	FILE* fp = fopen("/home/jdh/AQ200_test/nv_code/output.yuv", "rb+");	
	unsigned char* yuv_data = new unsigned char[1920 * 1080 * 3 / 2];
	fread(yuv_data, 1, 1920 * 1080 * 3 /2, fp);
	clock_t start = clock();

	jpeg_encode_proc(ctx, 1920, 1080, yuv_data);
    delete[] yuv_data;
	clock_t end   = clock();
	cout << "use" << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;	

		
    return 0;
}
#endif
