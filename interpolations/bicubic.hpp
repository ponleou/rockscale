#pragma once
#include "interpolation_utils.hpp"

__global__ void rowTangent(const unsigned char *input, float *output, const int width, const int height)
{
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    int row = global_idx / width; // which row we're in
    int col = global_idx % width; // which column in that row

    // check bounds
    if (row >= height || col >= width)
        return;

    // using catmull-rom
    float tangent;
    if (col == 0)
        tangent = ((float)input[global_idx + 1] - (float)input[global_idx]) / 2.0;
    else if (col == width - 1)
        tangent = ((float)input[global_idx] - (float)input[global_idx - 1]) / 2.0;
    else
        tangent = ((float)input[global_idx + 1] - (float)input[global_idx - 1]) / 2.0;

    output[global_idx] = tangent;
}

// NOTE: the output is scaled up by the scale
// NOTE: plane and tangents is the same size
__global__ void cubicRowInterpolate(const unsigned char *plane, const float *tangents, unsigned char *output, const int width, const int height, const int scale)
{
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    int row = global_idx / width; // which row we're in
    int col = global_idx % width; // which column in that row

    // check bounds
    if (row >= height || col >= width)
        return;

    // Starting position in output buffer
    int output_width = ((width - 1) * scale) + 1;
    int output_row = row * scale;
    int output_col = col * scale;
    int output_index = (output_row * output_width) + output_col;

    unsigned char start = plane[global_idx];
    float start_tangent = tangents[global_idx];
    // Set the original pixel in output
    output[output_index] = start;

    // if its the last of width, it cant interpolate to the right, so no need
    if (col == width - 1)
        return;

    unsigned char end = plane[global_idx + 1];
    float end_tangent = tangents[global_idx + 1];

    float rank3Coeff = (2 * start) + start_tangent - (2 * end) + end_tangent;
    float rank2Coeff = (-3 * start + 3 * end - 2 * start_tangent - end_tangent);
    float rank1Coeff = start_tangent;
    float rank0Coeff = start;

    for (int i = 1; i < scale; i++)
    {
        float t = (float)i / (float)(scale - 1);
        // float interpolated = rank3Coeff * (t * t * t) + rank2Coeff * (t * t) + rank1Coeff * t + rank0Coeff;
        float interpolated = ((rank3Coeff * t + rank2Coeff) * t + rank1Coeff) * t + rank0Coeff; // more efficient
        float clamped = fminf(fmaxf(interpolated, 0.0f), 255.0f);
        output[output_index + i] = (unsigned char)clamped;
    }
}

// output will be scaled_width * height
// row_interpolated will be scaled_width * scaled_height
__global__ void columnTangent(const unsigned char *row_interpolated, float *output, const int scaled_width, const int height, int scale)
{
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    int row = global_idx / scaled_width; // which row we're in
    int col = global_idx % scaled_width; // which column in that row

    // check bounds
    if (row >= height || col >= scaled_width)
        return;

    // using catmull-rom
    float tangent;
    if (row == 0)
        tangent = ((float)row_interpolated[global_idx + (scaled_width * scale)] - (float)row_interpolated[global_idx]) / 2.0;
    else if (row == height - 1)
        tangent = ((float)row_interpolated[global_idx] - (float)row_interpolated[global_idx - (scaled_width * scale)]) / 2.0;
    else
        tangent = ((float)row_interpolated[global_idx + (scaled_width * scale)] - (float)row_interpolated[global_idx - (scaled_width * scale)]) / 2.0;

    output[global_idx] = tangent;
}

// NOTE: row_rec_tangents is the output from columnTangent, so its size is (scaled_width * height)
// NOTE: DO NOT SPAWN THREADS FOR THE LAST ROW, there IS check for last row but its less efficient
__global__ void cubicColumnInterpolate(unsigned char *row_interpolated, const float *row_rec_tangents, const int scaled_width, const int scaled_height, const int scale)
{
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    int scaled_row = (global_idx / scaled_width) * scale; // which row we're in, scaled
    int col = global_idx % scaled_width;                  // which column in that row

    // check bounds
    if (scaled_row >= scaled_height || col >= scaled_width)
        return;

    // skip last row
    if (scaled_row == scaled_height - 1)
        return;

    int row = global_idx / scaled_width;

    // starting position in output buffer by rows
    int output_index = (scaled_row * scaled_width) + col;
    // tangent's starting index by row
    // NOTE: tangent is only scaled by the width, height is not scaled
    int tangent_index = (row * scaled_width) + col;

    unsigned char start = row_interpolated[output_index];
    float start_tangent = row_rec_tangents[tangent_index];

    unsigned char end = row_interpolated[output_index + (scaled_width * scale)]; // we are skipping (scale-1) rows to the next available row, the ones in between are the ones to interpolate
    float end_tangent = row_rec_tangents[tangent_index + scaled_width];

    float rank3Coeff = (2 * start) + start_tangent - (2 * end) + end_tangent;
    float rank2Coeff = (-3 * start + 3 * end - 2 * start_tangent - end_tangent);
    float rank1Coeff = start_tangent;
    float rank0Coeff = start;

    for (int i = 1; i < scale; i++)
    {
        float t = (float)i / (float)(scale - 1);
        // float interpolated = rank3Coeff * (t * t * t) + rank2Coeff * (t * t) + rank1Coeff * t + rank0Coeff;
        float interpolated = ((rank3Coeff * t + rank2Coeff) * t + rank1Coeff) * t + rank0Coeff; // more efficient
        float clamped = fminf(fmaxf(interpolated, 0.0f), 255.0f);
        row_interpolated[output_index + (i * scaled_width)] = (unsigned char)clamped;
    }
}

class BicubicGPUMemory : public GPUMemory<uint8_t, 6>
{
public:
    BicubicGPUMemory(const AVFrame *frame, int scale)
    {
        int width = frame->width;
        int height = frame->height;
        int planeSize = width * height;

        // only index that has a "next" row or column can interpolate, so the last col of any row, and last row of any col will be a straight map
        // a -1 because width and height is overlapping the very last index
        int newWidth = ((width - 1) * scale) + 1;
        int newHeight = ((height - 1) * scale) + 1;
        int newPlaneSize = newWidth * newHeight;

        // get chroma width and height
        int wChromaShift, hChromaShift;
        av_pix_fmt_get_chroma_sub_sample((enum AVPixelFormat)frame->format, &wChromaShift, &hChromaShift);
        int chromaWidth = AV_CEIL_RSHIFT(frame->width, wChromaShift);
        int chromaHeight = AV_CEIL_RSHIFT(frame->height, hChromaShift);
        int chromaPlaneSize = chromaHeight * chromaWidth;

        // same idea here
        int newChromaWidth = ((chromaWidth - 1) * scale) + 1;
        int newChromaHeight = ((chromaHeight - 1) * scale) + 1;
        int newChromaPlaneSize = newChromaWidth * newChromaHeight;

        this->allocate({// Y Plane
                        planeSize,
                        newPlaneSize,
                        // U plane
                        chromaPlaneSize,
                        newChromaPlaneSize,
                        // V plane
                        chromaPlaneSize,
                        newChromaPlaneSize});
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

class BicubicTangentGPUMemory : public GPUMemory<float, 6>
{
public:
    BicubicTangentGPUMemory(const AVFrame *frame, int scale)
    {
        int width = frame->width;
        int height = frame->height;
        int planeSize = width * height;

        // only index that has a "next" row or column can interpolate, so the last col of any row, and last row of any col will be a straight map
        // a -1 because width and height is overlapping the very last index
        int newWidth = ((width - 1) * scale) + 1;
        int newHeight = ((height - 1) * scale) + 1;
        int newPlaneSize = newWidth * newHeight;

        int colTangentSize = newWidth * height;

        // get chroma width and height
        int wChromaShift, hChromaShift;
        av_pix_fmt_get_chroma_sub_sample((enum AVPixelFormat)frame->format, &wChromaShift, &hChromaShift);
        int chromaWidth = AV_CEIL_RSHIFT(frame->width, wChromaShift);
        int chromaHeight = AV_CEIL_RSHIFT(frame->height, hChromaShift);
        int chromaPlaneSize = chromaHeight * chromaWidth;

        // same idea here
        int newChromaWidth = ((chromaWidth - 1) * scale) + 1;
        int newChromaHeight = ((chromaHeight - 1) * scale) + 1;
        int newChromaPlaneSize = newChromaWidth * newChromaHeight;

        int chromaColTangentSize = newChromaWidth * chromaHeight;

        this->allocate({// Y Plane
                        planeSize,
                        colTangentSize,
                        // U plane
                        chromaPlaneSize,
                        chromaColTangentSize,
                        // V plane
                        chromaPlaneSize,
                        chromaColTangentSize});
    }

    enum MemoryPosition
    {
        Y_ROW_TANGENTS,
        Y_COL_TANGENTS,
        U_ROW_TANGENTS,
        U_COL_TANGENTS,
        V_ROW_TANGENTS,
        V_COL_TANGENTS,
    };

    void validateMemorySizes(const int (&sizes)[6]) override
    {
        for (int i = 0; i < 6; i++)
            if (sizes[i] != this->sizes[i])
                throw invalid_argument("Memory sizes don't match at index " + to_string(i) + ", potential mismatched frames");
    }
};

/**
 * for each plane, we need memory for
 * - the plane itself (width * height)
 * - the output plane ((width - 1) * scale + 1) * ((height - 1) * scale + 1)
 * - tangents by row (width * height)
 * - tangents by col as row_rectangles (((width - 1) * scale + 1) * height)
 *
 * we can do by row first, then by columns
 */

void bicubicInterpolation(AVFrame **frame, int scale, hipStream_t &stream, GPUMemory<uint8_t, 6> &memory, GPUMemory<float, 6> &tangentMemory)
{
    int width = (*frame)->width;
    int height = (*frame)->height;
    int planeSize = width * height;

    // only index that has a "next" row or column can interpolate, so the last col of any row, and last row of any col will be a straight map
    // a -1 because width and height is overlapping the very last index
    int scaledWidth = ((width - 1) * scale) + 1;
    int scaledHeight = ((height - 1) * scale) + 1;
    int scaledPlaneSize = scaledWidth * scaledHeight;

    int colTangentSize = scaledWidth * height;

    // get chroma width and height
    int wChromaShift, hChromaShift;
    av_pix_fmt_get_chroma_sub_sample((enum AVPixelFormat)(*frame)->format, &wChromaShift, &hChromaShift);
    int chromaWidth = AV_CEIL_RSHIFT((*frame)->width, wChromaShift);
    int chromaHeight = AV_CEIL_RSHIFT((*frame)->height, hChromaShift);
    int chromaPlaneSize = chromaHeight * chromaWidth;

    // same idea here
    int scaledChromaWidth = ((chromaWidth - 1) * scale) + 1;
    int scaledChromaHeight = ((chromaHeight - 1) * scale) + 1;
    int scaledChromaPlaneSize = scaledChromaWidth * scaledChromaHeight;

    int chromaColTangentSize = scaledChromaWidth * chromaHeight;

    // memory for the frame's plane, output, row tangents, col tangents
    memory.validateMemorySizes({// Y Plane
                                planeSize,
                                scaledPlaneSize,
                                // U plane
                                chromaPlaneSize,
                                scaledChromaPlaneSize,
                                // V plane
                                chromaPlaneSize,
                                scaledChromaPlaneSize});

    tangentMemory.validateMemorySizes({// Y Plane
                                       planeSize,
                                       colTangentSize,
                                       // U plane
                                       chromaPlaneSize,
                                       chromaColTangentSize,
                                       // V plane
                                       chromaPlaneSize,
                                       chromaColTangentSize});

    uint8_t *YPlane = (*frame)->data[0];
    uint8_t *UPlane = (*frame)->data[1];
    uint8_t *VPlane = (*frame)->data[2];

    int YLinesize = (*frame)->linesize[0];
    int ULinesize = (*frame)->linesize[1];
    int VLinesize = (*frame)->linesize[2];

    HIP_CHECK(hipMemcpy2DAsync(memory[BicubicGPUMemory::Y_PLANE], width * sizeof(uint8_t), YPlane, YLinesize, width * sizeof(uint8_t), height, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpy2DAsync(memory[BicubicGPUMemory::U_PLANE], chromaWidth * sizeof(uint8_t), UPlane, ULinesize, chromaWidth * sizeof(uint8_t), chromaHeight, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpy2DAsync(memory[BicubicGPUMemory::V_PLANE], chromaWidth * sizeof(uint8_t), VPlane, VLinesize, chromaWidth * sizeof(uint8_t), chromaHeight, hipMemcpyHostToDevice, stream));

    int threadsPerBlock = 256;

    int rowThreadCountY = width * height;
    int rowBlocksY = ceil((float)rowThreadCountY / (float)threadsPerBlock);

    int rowThreadCountChroma = chromaWidth * chromaHeight;
    int rowBlocksChroma = ceil((float)rowThreadCountChroma / (float)threadsPerBlock);

    void *rowTangentArgsY[] = {&memory[BicubicGPUMemory::Y_PLANE], &memory[BicubicTangentGPUMemory::Y_ROW_TANGENTS], &width, &height};
    void *rowTangentArgsU[] = {&memory[BicubicGPUMemory::U_PLANE], &memory[BicubicTangentGPUMemory::U_ROW_TANGENTS], &chromaWidth, &chromaHeight};
    void *rowTangentArgsV[] = {&memory[BicubicGPUMemory::V_PLANE], &memory[BicubicTangentGPUMemory::V_ROW_TANGENTS], &chromaWidth, &chromaHeight};

    HIP_CHECK(hipLaunchKernel((const void *)rowTangent, dim3(rowBlocksY), dim3(threadsPerBlock), rowTangentArgsY, 0, stream));
    HIP_CHECK(hipLaunchKernel((const void *)rowTangent, dim3(rowBlocksChroma), dim3(threadsPerBlock), rowTangentArgsU, 0, stream));
    HIP_CHECK(hipLaunchKernel((const void *)rowTangent, dim3(rowBlocksChroma), dim3(threadsPerBlock), rowTangentArgsV, 0, stream));

    void *rowInterpolateArgsY[] = {&memory[BicubicGPUMemory::Y_PLANE], &memory[BicubicTangentGPUMemory::Y_ROW_TANGENTS], &memory[BicubicGPUMemory::OUTPUT_Y_PLANE], &width, &height, &scale};
    void *rowInterpolateArgsU[] = {&memory[BicubicGPUMemory::U_PLANE], &memory[BicubicTangentGPUMemory::U_ROW_TANGENTS], &memory[BicubicGPUMemory::OUTPUT_U_PLANE], &chromaWidth, &chromaHeight, &scale};
    void *rowInterpolateArgsV[] = {&memory[BicubicGPUMemory::V_PLANE], &memory[BicubicTangentGPUMemory::V_ROW_TANGENTS], &memory[BicubicGPUMemory::OUTPUT_V_PLANE], &chromaWidth, &chromaHeight, &scale};

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipLaunchKernel((const void *)cubicRowInterpolate, dim3(rowBlocksY), dim3(threadsPerBlock), rowInterpolateArgsY, 0, stream));
    HIP_CHECK(hipLaunchKernel((const void *)cubicRowInterpolate, dim3(rowBlocksChroma), dim3(threadsPerBlock), rowInterpolateArgsU, 0, stream));
    HIP_CHECK(hipLaunchKernel((const void *)cubicRowInterpolate, dim3(rowBlocksChroma), dim3(threadsPerBlock), rowInterpolateArgsV, 0, stream));

    int colThreadCountY = scaledWidth * (height - 1);
    int colBlocksY = ceil((float)colThreadCountY / (float)threadsPerBlock);

    int colThreadCountChroma = scaledChromaWidth * (chromaHeight - 1);
    int colBlocksChroma = ceil((float)colThreadCountChroma / (float)threadsPerBlock);

    void *colTangentArgsY[] = {&memory[BicubicGPUMemory::OUTPUT_Y_PLANE], &memory[BicubicTangentGPUMemory::Y_COL_TANGENTS], &scaledWidth, &height, &scale};
    void *colTangentArgsU[] = {&memory[BicubicGPUMemory::OUTPUT_U_PLANE], &memory[BicubicTangentGPUMemory::U_COL_TANGENTS], &scaledChromaWidth, &chromaHeight, &scale};
    void *colTangentArgsV[] = {&memory[BicubicGPUMemory::OUTPUT_V_PLANE], &memory[BicubicTangentGPUMemory::V_COL_TANGENTS], &scaledChromaWidth, &chromaHeight, &scale};

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipLaunchKernel((const void *)columnTangent, dim3(colBlocksY), dim3(threadsPerBlock), colTangentArgsY, 0, stream));
    HIP_CHECK(hipLaunchKernel((const void *)columnTangent, dim3(colBlocksChroma), dim3(threadsPerBlock), colTangentArgsU, 0, stream));
    HIP_CHECK(hipLaunchKernel((const void *)columnTangent, dim3(colBlocksChroma), dim3(threadsPerBlock), colTangentArgsV, 0, stream));

    // NOTE: row_rec_tangents is the output from columnTangent, so its size is (scaled_width * height)
    // NOTE: DO NOT SPAWN THREADS FOR THE LAST ROW, there IS check for last row but its less efficient
    // __global__ void cubicColumnInterpolate(unsigned char *row_interpolated, const float *row_rec_tangents, const int scaled_width, const int scaled_height, const int scale)

    void *colInterpolateArgsY[] = {&memory[BicubicGPUMemory::OUTPUT_Y_PLANE], &memory[BicubicTangentGPUMemory::Y_COL_TANGENTS], &scaledWidth, &scaledHeight, &scale};
    void *colInterpolateArgsU[] = {&memory[BicubicGPUMemory::OUTPUT_U_PLANE], &memory[BicubicTangentGPUMemory::U_COL_TANGENTS], &scaledChromaWidth, &scaledChromaHeight, &scale};
    void *colInterpolateArgsV[] = {&memory[BicubicGPUMemory::OUTPUT_V_PLANE], &memory[BicubicTangentGPUMemory::V_COL_TANGENTS], &scaledChromaWidth, &scaledChromaHeight, &scale};

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipLaunchKernel((const void *)cubicColumnInterpolate, dim3(colBlocksY), dim3(threadsPerBlock), colInterpolateArgsY, 0, stream));
    HIP_CHECK(hipLaunchKernel((const void *)cubicColumnInterpolate, dim3(colBlocksChroma), dim3(threadsPerBlock), colInterpolateArgsU, 0, stream));
    HIP_CHECK(hipLaunchKernel((const void *)cubicColumnInterpolate, dim3(colBlocksChroma), dim3(threadsPerBlock), colInterpolateArgsV, 0, stream));

    AVFrame *newFrame = av_frame_alloc();
    av_frame_copy_props(newFrame, *frame);
    newFrame->width = scaledWidth;
    newFrame->height = scaledHeight;
    newFrame->format = (*frame)->format;
    av_frame_get_buffer(newFrame, 0);

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy2D(newFrame->data[0], newFrame->linesize[0], memory[BicubicGPUMemory::OUTPUT_Y_PLANE], scaledWidth * sizeof(uint8_t), scaledWidth * sizeof(uint8_t), scaledHeight, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy2D(newFrame->data[1], newFrame->linesize[1], memory[BicubicGPUMemory::OUTPUT_U_PLANE], scaledChromaWidth * sizeof(uint8_t), scaledChromaWidth * sizeof(uint8_t), scaledChromaHeight, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy2D(newFrame->data[2], newFrame->linesize[2], memory[BicubicGPUMemory::OUTPUT_V_PLANE], scaledChromaWidth * sizeof(uint8_t), scaledChromaWidth * sizeof(uint8_t), scaledChromaHeight, hipMemcpyDeviceToHost));

    av_frame_free(frame);
    *frame = newFrame;
}
