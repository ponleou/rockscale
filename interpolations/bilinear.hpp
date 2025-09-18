#pragma once
#include "interpolation_utils.hpp"

// TODO: output does not have the padding value
// TODO: the size is not exactly width and height times scale, because of the last row/col CHECK PROPERLY
__global__ void linearRowInterpolate(const unsigned char *plane, unsigned char *output, const int linesize, const int width, const int height, const int scale)
{
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    int row = global_idx / linesize; // which row we're in
    int col = global_idx % linesize; // which column in that row

    // check bounds
    if (row >= height || col >= width)
        return;

    // Calculate output linesize (scaled width)
    // int output_linesize = width * scale;

    // Starting position in output buffer
    int output_width = (width - 1) * scale + 1;
    int output_row = row * scale;
    int output_col = col * scale;
    int output_index = (output_row * output_width) + output_col;

    unsigned char start = plane[global_idx];

    // Set the original pixel in output
    output[output_index] = start;

    // if its the last pixel from width, it doesnt have a next pixel to interpolate from
    // but we still need to place it in the output, which we already did above
    if (col == width - 1)
        return;

    unsigned char end = plane[global_idx + 1];

    // gradient = change per step = (end - start) / scale
    float gradient = (float)(end - start) / (float)scale;

    for (int i = 1; i < scale; i++)
    {
        unsigned char interpolated = start + (unsigned char)(i * gradient);
        output[output_index + i] = interpolated;
    }
}

// TODO: make sure to give NO THREADS for the final row
// TODO: so we are passing one thread for all valued element in the rowInterpolated frame EXCEPT the final row
// TODO: also PASS THE SAME ROW INTERPOLATED FRAME, it will be modified on top
__global__ void linearColumnInterpolate(unsigned char *rowInterpolated, const int scaled_width, const int scaled_height, const int scale)
{
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    int row = (global_idx / scaled_width) * scale; // which row we're in
    int col = global_idx % scaled_width;           // which column in that row

    // check bounds
    if (row >= scaled_height || col >= scaled_width)
        return;

    // Starting position in output buffer
    int start_index = (row * scaled_width) + col;

    unsigned char start = rowInterpolated[start_index];
    unsigned char end = rowInterpolated[start_index + (scale * scaled_width)];

    // gradient = change per step = (end - start) / scale
    float gradient = (float)(end - start) / (float)scale;

    for (int i = 1; i < scale; i++)
    {
        unsigned char interpolated = start + (unsigned char)(i * gradient);
        rowInterpolated[start_index + (i * scaled_width)] = interpolated;
    }
}

void bilinearInterpolation(AVFrame **frame, int scale, hipStream_t &stream, GPUMemory<uint8_t, 6> &memory)
{
    int width = (*frame)->width;
    int height = (*frame)->height;
    int planeSize = width * height;

    // divide 2 and ceil
    int chromaWidth = (width + 1) / 2;
    int chromaHeight = (height + 1) / 2;
    int chromaPlaneSize = chromaHeight * chromaWidth;

    // only index that has a "next" row or column can interpolate, so the last col of any row, and last row of any col will be a straight map
    // a -1 because width and height is overlapping the very last index
    int newWidth = ((width - 1) * scale) + 1;
    int newHeight = ((height - 1) * scale) + 1;
    int newPlaneSize = newWidth * newHeight;
    // same idea here
    int newChromaWidth = ((chromaWidth - 1) * scale) + 1;
    int newChromaHeight = ((chromaHeight - 1) * scale) + 1;
    int newChromaPlaneSize = newChromaWidth * newChromaHeight;

    memory.validateMemorySizes({planeSize, chromaPlaneSize, chromaPlaneSize, newPlaneSize, newChromaPlaneSize, newChromaPlaneSize});

    uint8_t *YPlane = (*frame)->data[0];
    uint8_t *UPlane = (*frame)->data[1];
    uint8_t *VPlane = (*frame)->data[2];

    int YLinesize = (*frame)->linesize[0];
    int ULinesize = (*frame)->linesize[1];
    int VLinesize = (*frame)->linesize[2];

    HIP_CHECK(hipMemcpy2DAsync(memory[0], width * sizeof(uint8_t), YPlane, YLinesize, width * sizeof(uint8_t), height, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpy2DAsync(memory[1], chromaWidth * sizeof(uint8_t), UPlane, ULinesize, chromaWidth * sizeof(uint8_t), chromaHeight, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpy2DAsync(memory[2], chromaWidth * sizeof(uint8_t), VPlane, VLinesize, chromaWidth * sizeof(uint8_t), chromaHeight, hipMemcpyHostToDevice, stream));

    int threadsPerBlock = 256;

    int rowThreadCountY = width * height;
    int rowNumBlocksY = ceil((float)rowThreadCountY / (float)threadsPerBlock);
    void *rowArgsY[] = {&memory[0], &memory[3], &width, &width, &height, &scale};

    int rowThreadCountU = chromaWidth * chromaHeight;
    int rowNumBlocksU = ceil((float)rowThreadCountU / (float)threadsPerBlock);
    void *rowArgsU[] = {&memory[1], &memory[4], &chromaWidth, &chromaWidth, &chromaHeight, &scale};

    int rowThreadCountV = chromaWidth * chromaHeight;
    int rowNumBlocksV = ceil((float)rowThreadCountV / (float)threadsPerBlock);
    void *rowArgsV[] = {&memory[2], &memory[5], &chromaWidth, &chromaWidth, &chromaHeight, &scale};

    HIP_CHECK(hipLaunchKernel((const void *)linearRowInterpolate, dim3(rowNumBlocksY), dim3(threadsPerBlock), rowArgsY, 0, stream));
    HIP_CHECK(hipLaunchKernel((const void *)linearRowInterpolate, dim3(rowNumBlocksU), dim3(threadsPerBlock), rowArgsU, 0, stream));
    HIP_CHECK(hipLaunchKernel((const void *)linearRowInterpolate, dim3(rowNumBlocksV), dim3(threadsPerBlock), rowArgsV, 0, stream));

    int colThreadCountY = newWidth * (height - 1);
    int colNumBlocksY = ceil((float)colThreadCountY / (float)threadsPerBlock);
    void *colArgsY[] = {&memory[3], &newWidth, &newHeight, &scale};

    int colThreadCountU = newChromaWidth * (chromaHeight - 1);
    int colNumBlocksU = ceil((float)colThreadCountU / (float)threadsPerBlock);
    void *colArgsU[] = {&memory[4], &newChromaWidth, &newChromaHeight, &scale};

    int colThreadCountV = newChromaWidth * (chromaHeight - 1);
    int colNumBlocksV = ceil((float)colThreadCountV / (float)threadsPerBlock);
    void *colArgsV[] = {&memory[5], &newChromaWidth, &newChromaHeight, &scale};

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipLaunchKernel((const void *)linearColumnInterpolate, dim3(colNumBlocksY), dim3(threadsPerBlock), colArgsY, 0, stream));
    HIP_CHECK(hipLaunchKernel((const void *)linearColumnInterpolate, dim3(colNumBlocksU), dim3(threadsPerBlock), colArgsU, 0, stream));
    HIP_CHECK(hipLaunchKernel((const void *)linearColumnInterpolate, dim3(colNumBlocksV), dim3(threadsPerBlock), colArgsV, 0, stream));

    AVFrame *newFrame = av_frame_alloc();
    av_frame_copy_props(newFrame, *frame);
    newFrame->width = newWidth;
    newFrame->height = newHeight;
    newFrame->format = (*frame)->format;
    av_frame_get_buffer(newFrame, 0);

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy2D(newFrame->data[0], newFrame->linesize[0], memory[3], newWidth * sizeof(uint8_t), newWidth * sizeof(uint8_t), newHeight, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy2D(newFrame->data[1], newFrame->linesize[1], memory[4], newChromaWidth * sizeof(uint8_t), newChromaWidth * sizeof(uint8_t), newChromaHeight, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy2D(newFrame->data[2], newFrame->linesize[2], memory[5], newChromaWidth * sizeof(uint8_t), newChromaWidth * sizeof(uint8_t), newChromaHeight, hipMemcpyDeviceToHost));

    av_frame_free(frame);
    *frame = newFrame;
}
