TARGET = sw
CC = g++
CFLAGS = -g -Wall
LDFLAGS = -I "$(CUDA_INSTALL_PATH)/include" -L "$(CUDA_INSTALL_PATH)/lib64"

KERNEL = kernel.cl
OBJECTS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))
HEADERS = $(wildcard *.h)

ifeq ($(shell uname),Darwin)
        LIBS = -framework OpenCL
else
        LIBS = -lOpenCL -lm
endif


.PHONY: default all clean

default: $(TARGET)
all: default

%.o: %.cpp $(HEADERS)
	$(CC)  $(CFLAGS) $(LDFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS) $(KERNEL)
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJECTS) $(LIBS) -o $(TARGET)

EXEC=$(TARGET)
RUNSERVER_REQ_RUN_SETTINGS=True
RUNSERVER_DEPS=all
-include $(ACS_SHARED_PATH)/runserver.mk

clean:
	rm -f $(OBJECTS) $(TARGET)

