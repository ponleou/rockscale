extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
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

// Example HIP kernel - similar to OpenCL kernel
__global__ void processFrameKernel(uint8_t *input_data, uint8_t *output_data,
                                   int width, int height, int channels)
{
    // Get thread indices (like OpenCL get_global_id)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = (y * width + x) * channels;

        // Example: Simple brightness adjustment
        for (int c = 0; c < channels; c++)
        {
            int pixel_idx = idx + c;
            output_data[pixel_idx] = min(255, (int)(input_data[pixel_idx] * 1.2f));
        }
    }
}

class HIPConsumer
{
private:
    hipStream_t stream;
    uint8_t *d_input_buffer;
    uint8_t *d_output_buffer;
    size_t buffer_size;
    int frame_width, frame_height, frame_channels;

public:
    HIPConsumer(int width, int height, int channels = 3)
        : frame_width(width), frame_height(height), frame_channels(channels)
    {
        // Create non-blocking stream (like OpenCL command queue)
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

        // Allocate device memory (like clCreateBuffer)
        buffer_size = width * height * channels * sizeof(uint8_t);
        HIP_CHECK(hipMalloc(&d_input_buffer, buffer_size));
        HIP_CHECK(hipMalloc(&d_output_buffer, buffer_size));
    }

    ~HIPConsumer()
    {
        HIP_CHECK(hipFree(d_input_buffer));
        HIP_CHECK(hipFree(d_output_buffer));
        HIP_CHECK(hipStreamDestroy(stream));
    }

    void processFrame(AVFrame *frame)
    {
        if (!frame)
            return;

        // Copy frame data to device (like clEnqueueWriteBuffer)
        HIP_CHECK(hipMemcpyAsync(d_input_buffer, frame->data[0], buffer_size,
                                 hipMemcpyHostToDevice, stream));

        // Set up kernel launch parameters (like OpenCL work groups)
        dim3 blockSize(16, 16); // 16x16 threads per block
        dim3 gridSize((frame_width + blockSize.x - 1) / blockSize.x,
                      (frame_height + blockSize.y - 1) / blockSize.y);

        // Launch kernel (like clEnqueueNDRangeKernel)
        hipLaunchKernelGGL(processFrameKernel, gridSize, blockSize, 0, stream,
                           d_input_buffer, d_output_buffer,
                           frame_width, frame_height, frame_channels);

        // Optional: Copy result back to host (like clEnqueueReadBuffer)
        // HIP_CHECK(hipMemcpyAsync(host_output, d_output_buffer, buffer_size,
        //                         hipMemcpyDeviceToHost, stream));

        // Optional: Synchronize stream if needed
        // HIP_CHECK(hipStreamSynchronize(stream));
    }

    hipStream_t getStream() const { return stream; }
};

void consumer(VideoFFmpeg *video, int consumer_id)
{
    // Create HIP consumer for this thread
    HIPConsumer hip_consumer(1920, 1080, 3); // Adjust dimensions as needed

    cout << "Consumer " << consumer_id << " started with HIP stream" << endl;

    while (true)
    {
        AVFrame *frame = video->getFrame();
        if (frame == nullptr)
        {
            cout << "Consumer " << consumer_id << " finished - no more frames" << endl;
            break;
        }

        // Process frame with HIP kernel
        hip_consumer.processFrame(frame);

        cout << "Consumer " << consumer_id << " processed frame " << frame->pts << endl;
        av_frame_free(&frame);
    }
}

int main(int argc, char *argv[])
{
    // Initialize HIP device (like OpenCL context creation)
    HIP_CHECK(hipSetDevice(0));

    // Get device properties (optional, for debugging)
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    cout << "Using device: " << prop.name << endl;
    cout << "Compute capability: " << prop.major << "." << prop.minor << endl;

    VideoFFmpeg video("input.mp4");

    // Start video decoding in a separate thread
    std::thread decode_thread([&video]()
                              { video.startDecode(); });

    // Create consumer threads with HIP processing
    const int num_consumers = std::min((int)MAX_THREADS - 1, 4); // Leave one thread for decoding
    ThreadTask tasks(num_consumers);

    // Add consumer tasks
    for (int i = 0; i < num_consumers; i++)
    {
        auto *task = new function<void()>([&video, i]()
                                          { consumer(&video, i); });
        tasks.addTask(task, i);
    }

    // Start all consumer threads
    tasks.start();

    // Wait for all threads to complete
    tasks.join();
    decode_thread.join();

    cout << "All processing completed!" << endl;
    return 0;
}