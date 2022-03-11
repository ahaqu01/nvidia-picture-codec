#ifndef JPEG_ENCODE_H
#define JPEG_ENCODE_H

#include "NvJpegEncoder.h"
#include "NvVideoConverter.h"
#include <fstream>
#include <queue>
#include <pthread.h>



#define TEST_ERROR(cond, str, label) if(cond) { \
                                        cerr << str << endl; \
                                        error = 1; \
                                        goto label; }

#define PERF_LOOP   300

using namespace std;

typedef struct
{
    NvVideoConverter *conv;
    NvJPEGEncoder *jpegenc;

    char *in_file_path;
    std::ifstream * in_file;
    uint32_t in_width;
    uint32_t in_height;
    uint32_t in_pixfmt;

    char *out_file_path;
    std::ofstream * out_file;

    bool got_error;
    bool use_fd;

    bool perf;

    uint32_t crop_left;
    uint32_t crop_top;
    uint32_t crop_width;
    uint32_t crop_height;
    int  stress_test;
    bool scaled_encode;
    uint32_t scale_width;
    uint32_t scale_height;
    int quality;
} context_t;

int parse_csv_args(context_t * ctx, int argc, char *argv[]);

#endif
