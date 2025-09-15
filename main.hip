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

__global__ void linearRowInterpolate(const unsigned char *plane, unsigned char *output, const int linesize, const int width, const int height)
{
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
}

void producer(VideoFFmpeg &video)
{
    video.startDecode();
}

void consumer(VideoFFmpeg &video)
{
    AVFrame *frame = video.getFrame(AV_PIX_FMT_YUV420P);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    uint8_t *yPlane = frame->data[0]
}

int main(int argc, char *argv[])
{
    HIP_CHECK(hipSetDevice(0)); // setting HIP device to default

    VideoFFmpeg video("input.mp4");
    video.startDecode();

    // ThreadTask tasks(MAX_THREADS - 1);

    int count = 0;
    while (true)
    {
        AVFrame *frame = video.getFrame();
        if (frame != nullptr)
        {
            count++;
            AVPixelFormat pix_fmt = (AVPixelFormat)frame->format;
            const char *pix_fmt_name = av_get_pix_fmt_name(pix_fmt);
            cout << "Frame " << frame->pts << " type " << (pix_fmt_name ? pix_fmt_name : "unknown") << " " << frame->linesize[0] << endl;
            av_frame_free(&frame);
            auto test = frame->data[0];
            // auto test = frame->linesize[0];
        }
        else
        {
            exit(0);
        }
    }
}