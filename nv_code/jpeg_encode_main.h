#ifndef JPEG_ENCODE_MAIN_H
#define JPEG_ENCODE_MAIN_H

#include "NvUtils.h"
#include <errno.h>
#include <fstream>
#include <iostream>
#include <malloc.h>
#include <string.h>
#include <unistd.h>
#include <ctime>

#include "jpeg_encode.h"

#define TEST_ERROR(cond, str, label) if(cond) { \
                                        cerr << str << endl; \
                                        error = 1; \
                                        goto label; }

#define PERF_LOOP   300

using namespace std;

void abort(context_t * ctx);
void set_defaults(context_t * ctx);
int jpeg_encode_proc(context_t& ctx, int w, int h, unsigned char * yuv_data, unsigned char * jpg_data);

#endif
