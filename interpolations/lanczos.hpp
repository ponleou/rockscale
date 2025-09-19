#pragma once
#include "interpolation_utils.hpp"
#include <vector>
#include <functional>
#include <cmath>
#include <pthread.h>
using std::function;
using std::vector;

__global__ void lanczosRowInterpolation(const unsigned char *plane, const float *lanczos_kernel, unsigned char *output, const int width, const int height, const int scale, const int window)
{
    int thread_idx = threadIdx.x;
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    int row = global_idx / width; // which row we're in
    int col = global_idx % width; // which column in that row

    // check bounds
    if (row >= height || col >= width)
        return;

    assert((scale - 1) * window > blockDim.x);

    __shared__ float local_kernel[(scale - 1) * window];

    if (thread_idx < (scale - 1) * window)
        local_kernel[thread_idx] = lanczos_kernel[thread_idx];

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

__global__ void lanczosColumnInterpolation(unsigned char *row_interpolated, const float *lanczos_kernel, const int scaled_width, const int scaled_height, const int scale, const int window)
{
    int thread_idx = threadIdx.x;
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    int scaled_row = (global_idx / scaled_width) * scale; // which row we're in
    int col = global_idx % scaled_width;                  // which column in that row

    // check bounds
    if (scaled_row >= scaled_height - (scale - 1) || col >= scaled_width)
        return;

    assert((scale - 1) * window > blockDim.x);

    __shared__ float local_kernel[(scale - 1) * window];

    if (thread_idx < (scale - 1) * window)
        local_kernel[thread_idx] = lanczos_kernel[thread_idx];

    __syncthreads();

    // Starting position in output buffer
    int start_index = (scaled_row * scaled_width) + col;

    unsigned char values[window];
    int a = window / 2;

    int col_min_index = col;
    // total index - space between current column to last column - (scale-1) cropped columns
    int col_max_index = (scaled_width * scaled_height - 1) - (scaled_width - 1 - col) - (scaled_width * (scale - 1));

    float interpolated = 0;

    // for efficiency, we will start calculating for the first interpolation value with the loop that copies the values in
    float interpolated = 0;
    // loop for copying values + calculating first interpolation
    for (int i = 0; i < window; i++)
    {
        // first index should be ((-a+1) * scale) columns from current column
        // then next interations should be (-a + 1 + i) * scale columns back
        int value_index = fminf(fmaxf(global_idx + (((-a + 1 + i) * scale) * scaled_width), col_min_index), col_max_index);
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

// for each fraction from scale, there is 2*halfWindow multiplier
// for scale, there is (scale-1) new pixels, therefore, (scale-1) fractions
// the foramt of the output will be {[window elements for fraction 1], ..., [window elements for fraction (scale)th]}
template <int scale, int halfWindow>
void lanczosMultiplier(array<float, (scale - 1) * 2 * halfWindow> &output, int threads = 1)
{
    struct ThreadArgs
    {
        array<float, (scale - 1) * 2 * halfWindow> *output;
        int argScale;
        int argHalfWindow;
        int startIndex;
        int jobCount;
    };

    auto calculator = [](void *arg) -> void *
    {
        ThreadArgs *data = static_cast<ThreadArgs *>(arg);

        for (int index = 0; index < data->jobCount; index++)
        {
            int outputIndex = data->startIndex + index;
            float a = data->argHalfWindow;

            int fractionPosition = (outputIndex) / (a * 2);               // start from 0, total is scale-1, so 0, 1, ..., (scale-1)-1
            int windowIndex = (outputIndex) - (fractionPosition * 2 * a); // start from 0, total is window*2, so 0, 1, ..., window-1

            float x = (float)(fractionPosition + 1) / (float)scale; // aka the fraction
            // i = floor(x) - window + 1 to floor(x) + window
            int iStart = floor(x) - a + 1;
            int i = iStart + windowIndex;

            // the parameter for the lanczos kernel
            float xl = x - (float)i;

            (*data->output)[outputIndex] = a * (sin(M_PI * xl) * sin(M_PI * xl / a)) / (M_PI * xl * M_PI * xl);
        }

        return nullptr;
    };

    int jobCount = output.size();
    int threadsJob = floor((float)jobCount / (float)threads);
    int mainsJob = jobCount - (threadsJob * (threads - 1));

    vector<pthread_t> threadsPool(threads - 1);
    vector<ThreadArgs> threadArgs(threads - 1);

    for (int i = 0; i < threads - 1; i++)
    {
        threadArgs[i] = {&output, scale, halfWindow, i * threadsJob, threadsJob};
        pthread_create(&threadsPool[i], nullptr, calculator, &threadArgs[i]);
    }

    int mainStartIndex = (threads - 1) * threadsJob;
    ThreadArgs mainArgs = {&output, scale, halfWindow, mainStartIndex, mainsJob};
    calculator((void *)&mainArgs);

    for (int i = 0; i < threads - 1; i++)
    {
        pthread_join(threadsPool[i], nullptr);
    }
}

void lanczosInterpolation(AVFrame **frame, int scale, int window, hipStream_t &stream)
{
}