#pragma once
extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}
#include <array>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
using std::array;
using std::cerr;
using std::copy;
using std::invalid_argument;
using std::to_string;

// just shutting up intellisense
#ifdef __INTELLISENSE__
#define threadIdx (dim3{})
#define blockIdx (dim3{})
#define blockDim (dim3{})
#define __syncthreads()
#endif

#define HIP_CHECK(expression)                 \
    {                                         \
        const hipError_t status = expression; \
        if (status != hipSuccess)             \
        {                                     \
            cerr << "HIP error "              \
                 << status << ": "            \
                 << hipGetErrorString(status) \
                 << " at " << __FILE__ << ":" \
                 << __LINE__ << std::endl;    \
        }                                     \
    }

template <typename T, int K>
class GPUMemory
{
private:
    bool allocated;
    array<T *, K> memory;

protected:
    array<int, K> sizes;

public:
    GPUMemory()
    {
        this->allocated = false;
    }

    GPUMemory(const int (&sizes)[K])
    {
        this->allocated = true;
        copy(sizes, sizes + K, this->sizes.begin());
        for (int i = 0; i < K; i++)
            HIP_CHECK(hipMalloc((void **)&this->memory[i], this->sizes[i] * sizeof(T)));
    }

    void allocate(const int (&sizes)[K])
    {
        if (this->allocated)
            throw invalid_argument("Memory is already allocated");

        copy(sizes, sizes + K, this->sizes.begin());
        for (int i = 0; i < K; i++)
            HIP_CHECK(hipMalloc((void **)&this->memory[i], this->sizes[i] * sizeof(T)));
    }

    ~GPUMemory()
    {
        for (int i = 0; i < K; i++)
        {
            HIP_CHECK(hipFree(this->memory[i]));
        }
    }

    virtual T *&operator[](int index)
    {
        return this->memory[index];
    }

    // allows function to check that memory sizes are as expected before passing to kernel
    virtual void validateMemorySizes(const int (&sizes)[K])
    {
        for (int i = 0; i < K; i++)
            if (sizes[i] != this->sizes[i])
                throw invalid_argument("Memory sizes don't match at index " + to_string(i));
    }
};