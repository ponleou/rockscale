#pragma once
#include "interpolation_utils.hpp"
#include <vector>
#include <functional>
#include <cmath>
#include <pthread.h>
using std::function;
using std::vector;

__global__ void lanczosRowInterpolation(const unsigned char *plane, unsigned char *output, const int width, const int height, const int scale, const int window)
{
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