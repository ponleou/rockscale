#pragma once
#include "interpolation_utils.hpp"
#include <vector>
#include <functional>
#include <cmath>
#include <pthread.h>
using std::function;
using std::vector;

__global__ void lanczosRowInterpolate(const unsigned char *plane, const float *lanczos_kernel, unsigned char *output, const int width, const int height, const int scale, const int window)
{
    int thread_idx = threadIdx.x;
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    int row = global_idx / width; // which row we're in
    int col = global_idx % width; // which column in that row

    // check bounds
    if (row >= height || col >= width)
        return;

    const int kernel_size = (scale - 1) * window;

    extern __shared__ float local_kernel[];

    for (int i = thread_idx; i < kernel_size; i += blockDim.x)
    {
        local_kernel[i] = lanczos_kernel[i];
    }

    __syncthreads();

    // Starting position in output buffer
    int output_width = width * scale;
    int output_row = row * scale;
    int output_col = col * scale;
    int output_index = (output_row * output_width) + output_col;

    output[output_index] = plane[global_idx];

    unsigned char values[window];
    int a = window / 2;

    int row_min_index = row * width;
    int row_max_index = ((row + 1) * width) - 1;

    // for efficiency, we will start calculating for the first interpolation value with the loop that copies the values in
    float interpolated = 0;
    // loop for copying values + calculating first interpolation
    for (int i = 0; i < window; i++)
    {
        // for this forloop, the range of index would be (global_idx - a + 1) to (global_idx + a) inclusive
        int value_index = fminf(fmaxf(global_idx - a + 1 + i, row_min_index), row_max_index);
        values[i] = plane[value_index];

        // local kernel for the first interpolation value is from index 0 to window-1
        interpolated += (float)values[i] * local_kernel[i];
    }

    output[output_index + 1] = (unsigned char)fminf(fmaxf(interpolated, 0.0f), 255.0f);

    // first interpolation already done, so do for the rest
    for (int x = 2; x < scale; x++)
    {
        interpolated = 0;
        int kernel_index_offset = (x - 1) * window;

        for (int i = 0; i < window; i++)
            interpolated += (float)values[i] * local_kernel[i + kernel_index_offset];

        output[output_index + x] = (unsigned char)fminf(fmaxf(interpolated, 0.0f), 255.0f);
    }
}

__global__ void lanczosColumnInterpolate(unsigned char *row_interpolated, const float *lanczos_kernel, const int scaled_width, const int scaled_height, const int scale, const int window)
{
    int thread_idx = threadIdx.x;
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    int scaled_row = (global_idx / scaled_width) * scale; // which row we're in
    int col = global_idx % scaled_width;                  // which column in that row

    // check bounds
    if (scaled_row >= scaled_height - (scale - 1) || col >= scaled_width)
        return;

    const int kernel_size = (scale - 1) * window;

    extern __shared__ float local_kernel[];

    for (int i = thread_idx; i < kernel_size; i += blockDim.x)
    {
        local_kernel[i] = lanczos_kernel[i];
    }

    __syncthreads();

    // Starting position in output buffer
    int start_index = (scaled_row * scaled_width) + col;

    unsigned char values[window];
    int a = window / 2;

    int col_min_index = col;
    // total index - space between current column to last column - (scale-1) cropped columns
    int col_max_index = (scaled_width * scaled_height - 1) - (scaled_width - 1 - col) - (scaled_width * (scale - 1));

    // for efficiency, we will start calculating for the first interpolation value with the loop that copies the values in
    float interpolated = 0;
    // loop for copying values + calculating first interpolation
    for (int i = 0; i < window; i++)
    {
        // first index should be ((-a+1) * scale) columns from current column
        // then next interations should be (-a + 1 + i) * scale columns back
        int value_index = fminf(fmaxf(start_index + (((-a + 1 + i) * scale) * scaled_width), col_min_index), col_max_index);
        values[i] = row_interpolated[value_index];

        // local kernel for the first interpolation value is from index 0 to window-1
        interpolated += (float)values[i] * local_kernel[i];
    }

    row_interpolated[start_index + (1 * scaled_width)] = (unsigned char)fminf(fmaxf(interpolated, 0.0f), 255.0f);

    // first interpolation already done, so do for the rest
    for (int x = 2; x < scale; x++)
    {
        interpolated = 0;
        int kernel_index_offset = (x - 1) * window;

        for (int i = 0; i < window; i++)
            interpolated += (float)values[i] * local_kernel[i + kernel_index_offset];

        row_interpolated[start_index + (x * scaled_width)] = (unsigned char)fminf(fmaxf(interpolated, 0.0f), 255.0f);
    }
}

class LanczosKernel
{
private:
    int scale;
    int halfWindow;
    float *kernel;

    struct ThreadArgs
    {
        float *output;
        int startIndex;
        int jobCount;
        int halfWindow;
        int scale;
    };

    static void *kernelCalculator(void *args)
    {
        ThreadArgs *data = static_cast<ThreadArgs *>(args);

        for (int index = 0; index < data->jobCount; index++)
        {
            int outputIndex = data->startIndex + index;
            float a = data->halfWindow;

            int fractionPosition = (outputIndex) / (a * 2);               // start from 0, total is scale-1, so 0, 1, ..., (scale-1)-1
            int windowIndex = (outputIndex) - (fractionPosition * 2 * a); // start from 0, total is window*2, so 0, 1, ..., window-1

            float x = (float)(fractionPosition + 1) / (float)data->scale; // aka the fraction
            // i = floor(x) - window + 1 to floor(x) + window
            int iStart = floor(x) - a + 1;
            int i = iStart + windowIndex;

            // the parameter for the lanczos kernel
            float xl = x - (float)i;

            data->output[outputIndex] = a * (sin(M_PI * xl) * sin(M_PI * xl / a)) / (M_PI * xl * M_PI * xl);
        }

        return nullptr;
    }

    // for each fraction from scale, there is 2*halfWindow multiplier
    // for scale, there is (scale-1) new pixels, therefore, (scale-1) fractions
    // the foramt of the output will be {[window elements for fraction 1], ..., [window elements for fraction (scale)th]}
    void lanczosMultiplier(float *output, int threads)
    {
        int jobCount = (this->scale - 1) * 2 * this->halfWindow;
        int threadsJob = floor((float)jobCount / (float)threads);
        int mainsJob = jobCount - (threadsJob * (threads - 1));

        vector<pthread_t> threadsPool(threads - 1);
        vector<ThreadArgs> threadArgs(threads - 1);

        for (int i = 0; i < threads - 1; i++)
        {
            threadArgs[i] = {output, i * threadsJob, threadsJob, this->halfWindow, this->scale};
            pthread_create(&threadsPool[i], nullptr, kernelCalculator, &threadArgs[i]);
        }

        int mainStartIndex = (threads - 1) * threadsJob;
        ThreadArgs mainArgs = {output, mainStartIndex, mainsJob, this->halfWindow, this->scale};
        kernelCalculator((void *)&mainArgs);

        for (int i = 0; i < threads - 1; i++)
        {
            pthread_join(threadsPool[i], nullptr);
        }
    }

public:
    LanczosKernel(int scale, int halfWindow, int threads = 1)
    {
        this->scale = scale;
        this->halfWindow = halfWindow;
        this->kernel = new float[(scale - 1) * 2 * halfWindow];
        this->lanczosMultiplier(kernel, threads);
    }

    float *getKernel()
    {
        return this->kernel;
    }
};

class LanczosKernelGPUMemory : public GPUMemory<float, 1>
{
public:
    LanczosKernelGPUMemory(int scale, int halfWindow)
    {
        this->allocate({(scale - 1) * 2 * halfWindow});
    }

    enum MemoryPosition
    {
        KERNEL,
    };

    void validateMemorySizes(const int (&sizes)[1]) override
    {
        for (int i = 0; i < 1; i++)
            if (sizes[i] != this->sizes[i])
                throw invalid_argument("Memory sizes don't match at index " + to_string(i) + ", potential mismatched frames");
    }

    // the index is unused, always return KERNEL
    float *&operator[](int unused) override
    {
        return GPUMemory<float, 1>::operator[](KERNEL);
    }
};

class LanczosGPUMemory : public GPUMemory<uint8_t, 6>
{
public:
    LanczosGPUMemory(const AVFrame *frame, int scale)
    {
        int width = frame->width;
        int height = frame->height;
        int planeSize = width * height;

        // get chroma width and height
        int wChromaShift, hChromaShift;
        av_pix_fmt_get_chroma_sub_sample((enum AVPixelFormat)frame->format, &wChromaShift, &hChromaShift);
        int chromaWidth = AV_CEIL_RSHIFT(frame->width, wChromaShift);
        int chromaHeight = AV_CEIL_RSHIFT(frame->height, hChromaShift);
        int chromaPlaneSize = chromaHeight * chromaWidth;

        // only index that has a "next" row or column can interpolate, so the last col of any row, and last row of any col will be a straight map
        // a -1 because width and height is overlapping the very last index
        int scaledWidth = width * scale;
        int scaledHeight = height * scale;
        int scaledPlaneSize = scaledWidth * scaledHeight;
        // same idea here
        int scaledChromaWidth = chromaWidth * scale;
        int scaledChromaHeight = chromaHeight * scale;
        int scaledChromaPlaneSize = scaledChromaWidth * scaledChromaHeight;

        this->allocate({planeSize, scaledPlaneSize, chromaPlaneSize, scaledChromaPlaneSize, chromaPlaneSize, scaledChromaPlaneSize});
    }

    enum MemoryPosition
    {
        Y_PLANE,
        OUTPUT_Y_PLANE,
        U_PLANE,
        OUTPUT_U_PLANE,
        V_PLANE,
        OUTPUT_V_PLANE,
    };

    void validateMemorySizes(const int (&sizes)[6]) override
    {
        for (int i = 0; i < 6; i++)
            if (sizes[i] != this->sizes[i])
                throw invalid_argument("Memory sizes don't match at index " + to_string(i) + ", potential mismatched frames");
    }
};

void lanczosInterpolation(AVFrame **frame, int scale, int halfWindow, LanczosKernel &kernel, hipStream_t &stream, GPUMemory<uint8_t, 6> &memory, GPUMemory<float, 1> &kernelMemory)
{
    int window = 2 * halfWindow;

    int width = (*frame)->width;
    int height = (*frame)->height;
    int planeSize = width * height;

    // get chroma width and height
    int wChromaShift, hChromaShift;
    av_pix_fmt_get_chroma_sub_sample((enum AVPixelFormat)(*frame)->format, &wChromaShift, &hChromaShift);
    int chromaWidth = AV_CEIL_RSHIFT((*frame)->width, wChromaShift);
    int chromaHeight = AV_CEIL_RSHIFT((*frame)->height, hChromaShift);
    int chromaPlaneSize = chromaHeight * chromaWidth;

    // only index that has a "next" row or column can interpolate, so the last col of any row, and last row of any col will be a straight map
    // a -1 because width and height is overlapping the very last index
    int scaledWidth = width * scale;
    int scaledHeight = height * scale;
    int scaledPlaneSize = scaledWidth * scaledHeight;
    // same idea here
    int scaledChromaWidth = chromaWidth * scale;
    int scaledChromaHeight = chromaHeight * scale;
    int scaledChromaPlaneSize = scaledChromaWidth * scaledChromaHeight;

    memory.validateMemorySizes({planeSize, scaledPlaneSize, chromaPlaneSize, scaledChromaPlaneSize, chromaPlaneSize, scaledChromaPlaneSize});

    int kernelSize = (scale - 1) * window;

    kernelMemory.validateMemorySizes({kernelSize});

    float *kernelOutput = kernel.getKernel();

    HIP_CHECK(hipMemcpyAsync(kernelMemory[0], kernelOutput, kernelSize * sizeof(float), hipMemcpyHostToDevice, stream));

    uint8_t *YPlane = (*frame)->data[0];
    uint8_t *UPlane = (*frame)->data[1];
    uint8_t *VPlane = (*frame)->data[2];

    int YLinesize = (*frame)->linesize[0];
    int ULinesize = (*frame)->linesize[1];
    int VLinesize = (*frame)->linesize[2];

    HIP_CHECK(hipMemcpy2DAsync(memory[LanczosGPUMemory::Y_PLANE], width * sizeof(uint8_t), YPlane, YLinesize, width * sizeof(uint8_t), height, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpy2DAsync(memory[LanczosGPUMemory::U_PLANE], chromaWidth * sizeof(uint8_t), UPlane, ULinesize, chromaWidth * sizeof(uint8_t), chromaHeight, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpy2DAsync(memory[LanczosGPUMemory::V_PLANE], chromaWidth * sizeof(uint8_t), VPlane, VLinesize, chromaWidth * sizeof(uint8_t), chromaHeight, hipMemcpyHostToDevice, stream));

    int threadsPerBlock = 256;

    int rowThreadCountY = width * height;
    int rowNumBlocksY = ceil((float)rowThreadCountY / (float)threadsPerBlock);

    int rowThreadCountChroma = chromaWidth * chromaHeight;
    int rowNumBlocksChroma = ceil((float)rowThreadCountChroma / (float)threadsPerBlock);

    void *rowArgsY[] = {&memory[LanczosGPUMemory::Y_PLANE], &kernelMemory[0], &memory[LanczosGPUMemory::OUTPUT_Y_PLANE], &width, &height, &scale, &window};
    void *rowArgsU[] = {&memory[LanczosGPUMemory::U_PLANE], &kernelMemory[0], &memory[LanczosGPUMemory::OUTPUT_U_PLANE], &chromaWidth, &chromaHeight, &scale, &window};
    void *rowArgsV[] = {&memory[LanczosGPUMemory::V_PLANE], &kernelMemory[0], &memory[LanczosGPUMemory::OUTPUT_V_PLANE], &chromaWidth, &chromaHeight, &scale, &window};

    HIP_CHECK(hipLaunchKernel((const void *)lanczosRowInterpolate, dim3(rowNumBlocksY), dim3(threadsPerBlock), rowArgsY, kernelSize * sizeof(float), stream));
    HIP_CHECK(hipLaunchKernel((const void *)lanczosRowInterpolate, dim3(rowNumBlocksChroma), dim3(threadsPerBlock), rowArgsU, kernelSize * sizeof(float), stream));
    HIP_CHECK(hipLaunchKernel((const void *)lanczosRowInterpolate, dim3(rowNumBlocksChroma), dim3(threadsPerBlock), rowArgsV, kernelSize * sizeof(float), stream));

    int colThreadCountY = scaledWidth * height;
    int colNumBlocksY = ceil((float)colThreadCountY / (float)threadsPerBlock);

    int colThreadCountChroma = scaledChromaWidth * chromaHeight;
    int colNumBlocksChroma = ceil((float)colThreadCountChroma / (float)threadsPerBlock);

    void *colArgsY[] = {&memory[LanczosGPUMemory::OUTPUT_Y_PLANE], &kernelMemory[0], &scaledWidth, &scaledHeight, &scale, &window};
    void *colArgsU[] = {&memory[LanczosGPUMemory::OUTPUT_U_PLANE], &kernelMemory[0], &scaledChromaWidth, &scaledChromaHeight, &scale, &window};
    void *colArgsV[] = {&memory[LanczosGPUMemory::OUTPUT_V_PLANE], &kernelMemory[0], &scaledChromaWidth, &scaledChromaHeight, &scale, &window};

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipLaunchKernel((const void *)lanczosColumnInterpolate, dim3(colNumBlocksY), dim3(threadsPerBlock), colArgsY, kernelSize * sizeof(float), stream));
    HIP_CHECK(hipLaunchKernel((const void *)lanczosColumnInterpolate, dim3(colNumBlocksChroma), dim3(threadsPerBlock), colArgsU, kernelSize * sizeof(float), stream));
    HIP_CHECK(hipLaunchKernel((const void *)lanczosColumnInterpolate, dim3(colNumBlocksChroma), dim3(threadsPerBlock), colArgsV, kernelSize * sizeof(float), stream));

    AVFrame *newFrame = av_frame_alloc();
    av_frame_copy_props(newFrame, *frame);
    newFrame->width = scaledWidth;
    newFrame->height = scaledHeight;
    newFrame->format = (*frame)->format;
    av_frame_get_buffer(newFrame, 0);

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy2D(newFrame->data[0], newFrame->linesize[0], memory[LanczosGPUMemory::OUTPUT_Y_PLANE], scaledWidth * sizeof(uint8_t), scaledWidth * sizeof(uint8_t), scaledHeight, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy2D(newFrame->data[1], newFrame->linesize[1], memory[LanczosGPUMemory::OUTPUT_U_PLANE], scaledChromaWidth * sizeof(uint8_t), scaledChromaWidth * sizeof(uint8_t), scaledChromaHeight, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy2D(newFrame->data[2], newFrame->linesize[2], memory[LanczosGPUMemory::OUTPUT_V_PLANE], scaledChromaWidth * sizeof(uint8_t), scaledChromaWidth * sizeof(uint8_t), scaledChromaHeight, hipMemcpyDeviceToHost));

    av_frame_free(frame);
    *frame = newFrame;
}