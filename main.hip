// just shutting up intellisense
#ifdef __INTELLISENSE__
#define threadIdx (dim3{})
#define blockIdx (dim3{})
#define blockDim (dim3{})
#endif

extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <mutex>
#include <string>
#include <vector>
#include <queue>
#include <condition_variable>
#include <iostream>
#include <pthread.h>
#include <thread>
#include <functional>
#include <hip/amd_detail/amd_hip_runtime.h>
using std::function;
using std::thread;
using namespace std::chrono;
using std::cerr;
using std::condition_variable;
using std::cout;
using std::endl;
using std::mutex;
using std::queue;
using std::string;
using std::unique_lock;
using std::vector;

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

#define MAX_THREADS thread::hardware_concurrency()

class VideoFFmpeg
{
private:
    AVFormatContext *file;
    AVCodecContext *codec;
    int videoIndex;

    mutex codecLocker;
    condition_variable codecCV;

    queue<AVFrame *> decodedBuffer;
    mutex bufferLocker;

    AVFrame *getFrameFromBuffer()
    {
        unique_lock<mutex> bufferLock(this->bufferLocker);
        if (!this->decodedBuffer.empty())
        {
            AVFrame *frame = this->decodedBuffer.front();
            this->decodedBuffer.pop();
            return frame;
        }
        return nullptr;
    }

    void raiseError(string message, int code = -1)
    {
        cerr << message << ", code: " << code << endl;
        exit(code);
    }

public:
    VideoFFmpeg(string fileName, int decodeThreads = 0)
    {
        int fileCode;

        this->file = avformat_alloc_context();
        fileCode = avformat_open_input(&this->file, fileName.c_str(), nullptr, 0);
        if (fileCode != 0)
            this->raiseError("Failed to open file", fileCode);
        fileCode = avformat_find_stream_info(this->file, 0);
        if (fileCode != 0)
            this->raiseError("Failed to open file", fileCode);

        int codecCode;
        for (int i = 0; i < this->file->nb_streams; i++)
        {
            if (this->file->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                const AVCodec *codecInfo = avcodec_find_decoder(this->file->streams[i]->codecpar->codec_id);
                this->codec = avcodec_alloc_context3(codecInfo);

                if (this->codec == NULL || (codecCode = avcodec_parameters_to_context(this->codec, this->file->streams[i]->codecpar)) < 0)
                    this->raiseError("Failed to start codec", codecCode);

                this->codec->thread_count = decodeThreads;
                this->codec->thread_type = FF_THREAD_FRAME;

                if ((codecCode = avcodec_open2(this->codec, codecInfo, nullptr)) < 0)
                    this->raiseError("Failed to start codec", codecCode);

                this->videoIndex = i;
                break;
            }
        }
    }

    ~VideoFFmpeg()
    {
        avformat_close_input(&this->file);
        avcodec_free_context(&this->codec);
    }

    void startDecode()
    {
        int sendCode;
        AVPacket *packet = av_packet_alloc();
        while (av_read_frame(this->file, packet) >= 0)
        {
            if (packet->stream_index == this->videoIndex)
            {
                unique_lock<mutex> codecLock(this->codecLocker);

                // codec buffer is full, need to get decoded frames out
                // TODO: find out how to eof codec
                while ((sendCode = avcodec_send_packet(this->codec, packet)) == AVERROR(EAGAIN))
                {
                    AVFrame *frame = av_frame_alloc();
                    int recvCode = avcodec_receive_frame(this->codec, frame);
                    // if we got one, then we put that to the buffer
                    if (recvCode == 0)
                    {
                        unique_lock<mutex> bufferLock(this->bufferLocker);
                        this->decodedBuffer.push(frame);
                        // cout << "Frame placed in buffer" << endl;
                    }

                    // the only scenario where both avcodec_receive_frame and avcodec_send_packet is both AVERROR(EAGAIN) is when buffer is full, but frames are still decoding
                    // so we give up lock, sleep for a bit, then try to send again
                    else if (recvCode == AVERROR(EAGAIN))
                    {
                        av_frame_free(&frame);
                        this->codecCV.wait_for(codecLock, milliseconds(10));
                        continue;
                    }

                    // impossible scenario tho?
                    else if (recvCode == AVERROR_EOF)
                        break;

                    // error
                    else if (recvCode == AVERROR(EINVAL) || recvCode < 0)
                        this->raiseError("Failed to get frame from codec", recvCode);
                }

                // FIXME: might fail when recvcode was EOF
                if (sendCode != 0)
                    this->raiseError("Failed to decode frame", sendCode);

                // TODO: notify one?
                this->codecCV.notify_all();
            }
            av_packet_unref(packet);
        }
        av_packet_free(&packet);

        if ((sendCode = avcodec_send_packet(this->codec, nullptr)) != 0)
            this->raiseError("Failed to EOF", sendCode);
    }

    // this function promises to always return a frame, unless theres none left, which it will return nullptr
    AVFrame *getFrame()
    {
        AVFrame *frame = getFrameFromBuffer();
        if (frame != nullptr)
        {
            // cout << "Read frame from buffer" << endl;
            return frame;
        }

        frame = av_frame_alloc();

        int recvCode;
        unique_lock<mutex> codecLock(this->codecLocker);
        while ((recvCode = avcodec_receive_frame(this->codec, frame)) != 0)
        {
            if (recvCode == AVERROR(EAGAIN))
            {
                this->codecCV.wait(codecLock);

                // try getting from queue again
                AVFrame *frameBuffer = getFrameFromBuffer();
                if (frameBuffer != nullptr)
                    return frameBuffer;
            }

            // end of file, return nullptr
            else if (recvCode == AVERROR_EOF)
                return nullptr;

            // errors
            else if (recvCode == AVERROR(EINVAL))
                exit(1);
            else if (recvCode < 0)
                exit(1);
        }
        // cout << "Read frame from codec" << endl;
        return frame;
    }

    AVFrame *getFrame(AVPixelFormat format)
    {
        AVFrame *frame = this->getFrame();
        if (frame == nullptr)
            return nullptr;

        if ((AVPixelFormat)frame->format == format)
            return frame;

        // prepare a new converted frame with original frame's metadata/info
        AVFrame *convertedFrame = av_frame_alloc();
        convertedFrame->format = format;
        convertedFrame->width = frame->width;
        convertedFrame->height = frame->height;
        av_frame_copy_props(convertedFrame, frame);
        av_frame_get_buffer(convertedFrame, 0);

        // converting original frame's data to pixel format
        struct SwsContext *sws_ctx = sws_getContext(
            frame->width, frame->height, (AVPixelFormat)frame->format,
            frame->width, frame->height, format,
            SWS_BICUBIC, nullptr, nullptr, nullptr);
        sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, convertedFrame->data, convertedFrame->linesize);

        sws_freeContext(sws_ctx);
        av_frame_free(&frame);

        return convertedFrame;
    }
};

class ThreadTask
{
private:
    vector<pthread_t> threads;
    vector<function<void()> *> tasks;
    bool started;

    static void *functionCaller(void *arg)
    {
        function<void()> *func = static_cast<function<void()> *>(arg);
        if (func)
            (*func)();
        return nullptr;
    }

public:
    const int size;

    ThreadTask(int threadCount) : size(threadCount)
    {
        this->threads = vector<pthread_t>(this->size);
        this->tasks = vector<function<void()> *>(this->size, nullptr);
        this->started = false;
    }

    ~ThreadTask()
    {
        for (int i = 0; i < this->tasks.size(); i++)
            delete this->tasks[i];
    }

    void addTask(function<void()> *task, int assignedThread)
    {
        this->tasks[assignedThread] = task;
    }

    void start()
    {
        if (started)
            throw std::logic_error("Instance is already executed.");

        this->started = true;
        for (int i = 0; i < this->tasks.size(); i++)
        {
            if (this->tasks[i] == nullptr)
                continue;
            pthread_create(&this->threads[i], nullptr, ThreadTask::functionCaller, this->tasks[i]);
        }
    }

    void join()
    {
        if (!started)
            return;

        for (int i = 0; i < this->tasks.size(); i++)
        {
            if (this->tasks[i] == nullptr)
                continue;

            pthread_join(this->threads[i], nullptr);
        }
    }
};

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

void bilinearInterpolation(const AVFrame *frame, int scale, hipStream_t &stream, uint8_t *&y, uint8_t *&u, uint8_t *&v)
{
    int width = frame->width;
    int height = frame->height;

    // divide 2 and ceil
    int chromaWidth = (width + 1) / 2;
    int chromaHeight = (height + 1) / 2;

    uint8_t *YPlane = frame->data[0];
    uint8_t *UPlane = frame->data[1];
    uint8_t *VPlane = frame->data[2];

    int YLinesize = frame->linesize[0];
    int ULinesize = frame->linesize[1];
    int VLinesize = frame->linesize[2];

    // only index that has a "next" row or column can interpolate, so the last col of any row, and last row of any col will be a straight map
    // a -1 because width and height is overlapping the very last index
    int newWidth = ((width - 1) * scale) + 1;
    int newHeight = ((height - 1) * scale) + 1;
    int newPlaneSize = newWidth * newHeight;
    // same idea here
    int newChromaWidth = ((chromaWidth - 1) * scale) + 1;
    int newChromaHeight = ((chromaHeight - 1) * scale) + 1;
    int newChromaPlaneSize = newChromaWidth * newChromaHeight;

    uint8_t *hipYPlane{};
    uint8_t *hipUPlane{};
    uint8_t *hipVPlane{};

    uint8_t *scaledYPlane{};
    uint8_t *scaledUPlane{};
    uint8_t *scaledVPlane{};

    HIP_CHECK(hipMalloc((void **)&hipYPlane, width * height * sizeof(uint8_t)));
    HIP_CHECK(hipMalloc((void **)&hipUPlane, chromaWidth * chromaHeight * sizeof(uint8_t)));
    HIP_CHECK(hipMalloc((void **)&hipVPlane, chromaWidth * chromaHeight * sizeof(uint8_t)));

    HIP_CHECK(hipMalloc((void **)&scaledYPlane, newPlaneSize * sizeof(uint8_t)));
    HIP_CHECK(hipMalloc((void **)&scaledUPlane, newChromaPlaneSize * sizeof(uint8_t)));
    HIP_CHECK(hipMalloc((void **)&scaledVPlane, newChromaPlaneSize * sizeof(uint8_t)));

    HIP_CHECK(hipMemcpyAsync(hipYPlane, YPlane, width * height * sizeof(uint8_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(hipUPlane, UPlane, chromaWidth * chromaHeight * sizeof(uint8_t), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(hipVPlane, VPlane, chromaWidth * chromaHeight * sizeof(uint8_t), hipMemcpyHostToDevice, stream));

    int threadsPerBlock = 256;

    int rowThreadCountY = YLinesize * height;
    int rowNumBlocksY = ceil((float)rowThreadCountY / (float)threadsPerBlock);
    void *rowArgsY[] = {&hipYPlane, &scaledYPlane, &YLinesize, &width, &height, &scale};
    HIP_CHECK(hipLaunchKernel((const void *)linearRowInterpolate, dim3(rowNumBlocksY), dim3(threadsPerBlock), rowArgsY, 0, stream));

    int rowThreadCountU = ULinesize * chromaHeight;
    int rowNumBlocksU = ceil((float)rowThreadCountU / (float)threadsPerBlock);
    void *rowArgsU[] = {&hipUPlane, &scaledUPlane, &ULinesize, &chromaWidth, &chromaHeight, &scale};
    HIP_CHECK(hipLaunchKernel((const void *)linearRowInterpolate, dim3(rowNumBlocksU), dim3(threadsPerBlock), rowArgsU, 0, stream));

    int rowThreadCountV = VLinesize * chromaHeight;
    int rowNumBlocksV = ceil((float)rowThreadCountV / (float)threadsPerBlock);
    void *rowArgsV[] = {&hipVPlane, &scaledVPlane, &VLinesize, &chromaWidth, &chromaHeight, &scale};
    HIP_CHECK(hipLaunchKernel((const void *)linearRowInterpolate, dim3(rowNumBlocksV), dim3(threadsPerBlock), rowArgsV, 0, stream));

    HIP_CHECK(hipStreamSynchronize(stream));

    int colThreadCountY = newWidth * height;
    int colNumBlocksY = ceil((float)colThreadCountY / (float)threadsPerBlock);
    void *colArgsY[] = {&scaledYPlane, &newWidth, &newHeight, &scale};
    HIP_CHECK(hipLaunchKernel((const void *)linearColumnInterpolate, dim3(colNumBlocksY), dim3(threadsPerBlock), colArgsY, 0, stream));

    int colThreadCountU = newChromaWidth * chromaHeight;
    int colNumBlocksU = ceil((float)colThreadCountU / (float)threadsPerBlock);
    void *colArgsU[] = {&scaledUPlane, &newChromaWidth, &newChromaHeight, &scale};
    HIP_CHECK(hipLaunchKernel((const void *)linearColumnInterpolate, dim3(colNumBlocksU), dim3(threadsPerBlock), colArgsU, 0, stream));

    int colThreadCountV = newChromaWidth * chromaHeight;
    int colNumBlocksV = ceil((float)colThreadCountV / (float)threadsPerBlock);
    void *colArgsV[] = {&scaledVPlane, &newChromaWidth, &newChromaHeight, &scale};
    HIP_CHECK(hipLaunchKernel((const void *)linearColumnInterpolate, dim3(colNumBlocksV), dim3(threadsPerBlock), colArgsV, 0, stream));

    HIP_CHECK(hipStreamSynchronize(stream));

    y = new uint8_t[newPlaneSize];
    HIP_CHECK(hipMemcpy(y, scaledYPlane, newPlaneSize * sizeof(uint8_t), hipMemcpyDeviceToHost));

    u = new uint8_t[newChromaPlaneSize];
    HIP_CHECK(hipMemcpy(u, scaledUPlane, newChromaPlaneSize * sizeof(uint8_t), hipMemcpyDeviceToHost));

    v = new uint8_t[newChromaPlaneSize];
    HIP_CHECK(hipMemcpy(v, scaledUPlane, newChromaPlaneSize * sizeof(uint8_t), hipMemcpyDeviceToHost));

    // DEBUG
    {
        // Final dimensions
        int finalWidth = ((width - 1) * scale) + 1;
        int finalHeight = ((height - 1) * scale) + 1;

        int chromaFinalWidth = ((chromaWidth - 1) * scale) + 1;
        int chromaFinalHeight = ((chromaHeight - 1) * scale) + 1;

        // Create PGM image
        FILE *pgmFileY = fopen("outputy.pgm", "wb");
        fprintf(pgmFileY, "P5\n%d %d\n255\n", finalWidth, finalHeight);
        fwrite(y, sizeof(uint8_t), finalWidth * finalHeight, pgmFileY);
        fclose(pgmFileY);

        FILE *pgmFileU = fopen("outputu.pgm", "wb");
        fprintf(pgmFileU, "P5\n%d %d\n255\n", chromaFinalWidth, chromaFinalHeight);
        fwrite(u, sizeof(uint8_t), chromaFinalWidth * chromaFinalHeight, pgmFileU);
        fclose(pgmFileU);

        FILE *pgmFileV = fopen("outputv.pgm", "wb");
        fprintf(pgmFileV, "P5\n%d %d\n255\n", chromaFinalWidth, chromaFinalHeight);
        fwrite(v, sizeof(uint8_t), chromaFinalWidth * chromaFinalHeight, pgmFileV);
        fclose(pgmFileV);

        cout << "Created " << finalWidth << "x" << finalHeight << " image" << endl;
    }

    HIP_CHECK(hipFree(hipYPlane));
    HIP_CHECK(hipFree(hipUPlane));
    HIP_CHECK(hipFree(hipVPlane));
    HIP_CHECK(hipFree(scaledYPlane));
    HIP_CHECK(hipFree(scaledUPlane));
    HIP_CHECK(hipFree(scaledVPlane));
}

void producer(VideoFFmpeg &video)
{
    video.startDecode();
}

void consumer(VideoFFmpeg &video, int scale)
{
    AVFrame *frame = video.getFrame(AV_PIX_FMT_YUV420P);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    uint8_t *Y_output;
    uint8_t *U_output;
    uint8_t *V_output;
    bilinearInterpolation(frame, scale, stream, Y_output, U_output, V_output);

    HIP_CHECK(hipStreamDestroy(stream));
    delete[] Y_output;
}

int main(int argc, char *argv[])
{
    HIP_CHECK(hipSetDevice(0)); // setting HIP device to default

    VideoFFmpeg video("input.mp4");
    video.startDecode();

    // ThreadTask tasks(MAX_THREADS - 1);

    consumer(video, 5);

    // int count = 0;
    // while (true)
    // {
    //     AVFrame *frame = video.getFrame(AV_PIX_FMT_YUV420P);
    //     if (frame != nullptr)
    //     {
    //         count++;
    //         AVPixelFormat pix_fmt = (AVPixelFormat)frame->format;
    //         const char *pix_fmt_name = av_get_pix_fmt_name(pix_fmt);
    //         cout << "Frame " << frame->pts << " type " << (pix_fmt_name ? pix_fmt_name : "unknown") << " " << frame->linesize[0] << endl;
    //         av_frame_free(&frame);
    //         auto test = frame->data[0];
    //         // auto test = frame->linesize[0];
    //     }
    //     else
    //     {
    //         exit(0);
    //     }
    // }
}