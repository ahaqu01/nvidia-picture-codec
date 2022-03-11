OPENCV_HEAD = -I/usr/local/opencv-4.2.0/include/opencv4
#OPENCV_LIBS = `pkg-config opencv4 --cflags --libs`
OPENCV_LIBS = -I/usr/local/opencv-4.2.0/include/opencv4 -L/usr/local/opencv-4.2.0/lib -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_stitching -lopencv_cudafeatures2d \
              -lopencv_superres -lopencv_cudacodec -lopencv_videostab -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_photo -lopencv_cudawarping -lopencv_cudaimgproc \
              -lopencv_cudafilters -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy \
              -lopencv_img_hash -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light \
              -lopencv_phase_unwrapping -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_xfeatures2d \
              -lopencv_shape -lopencv_video -lopencv_ml -lopencv_cudaarithm -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio \
              -lopencv_flann -lopencv_xobjdetect -lopencv_imgcodecs -lopencv_objdetect -lopencv_xphoto -lopencv_imgproc -lopencv_core -lopencv_cudev


CUDA_HEAD = -I/usr/local/cuda/include

CUDA_LIB = -L/usr/local/cuda/lib64/ -lcudart \
           -I/usr/include/eigen3 \
           -L/usr/lib/aarch64-linux-gnu -lcuda -ldl


include ./Rules.mk

APP := main

SRCS := \
        jpeg_encode_csvparser.cpp \
        jpeg_encode_main.cpp \
        $(wildcard $(CLASS_DIR)/*.cpp)

OBJS := $(SRCS:.cpp=.o)

all: $(APP)

$(CLASS_DIR)/%.o: $(CLASS_DIR)/%.cpp
	$(AT)$(MAKE) -C $(CLASS_DIR)

%.o: %.cpp
	@echo "Compiling: $<"
	$(CPP) $(CPPFLAGS) -c $< -fpermissive

#$(APP): $(OBJS)
#       @echo "Linking: $@"
#        $(CPP) -o $@ $(OBJS) $(CPPFLAGS) $(LDFLAGS)



$(APP):main.cpp bgr2yuv.o $(OBJS) 
	g++ $+ -o $@ \
        -I/usr/local/cuda/include ${OPENCV_LIBS} -I../header \
        -L/usr/local/cuda/lib64 -lcudart -ldl -pthread -std=c++11 ${LINK} -fpermissive $(CPPFLAGS) $(LDFLAGS)

bgr2yuv.o:bgr2yuv.cu
	nvcc -std=c++11 -c bgr2yuv.cu ${OPENCV_LIBS} -L/usr/local/cuda/lib64 -I../header

clean:
	rm *.o main


