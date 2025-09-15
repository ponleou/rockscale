// #include <hip/hip_runtime.h>
// #include <hip/hip_runtime_api.h>
extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

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

template <typename T>
class ThreadSafeQueue
{
private:
    const unsigned int SIZE;     // the actual size of the queue pointer (capacity +1 because it is circular)
    const unsigned int CAPACITY; // usable space of the queue

    T *queue;

    unsigned int head;
    unsigned int tail;

    mutex locker;
    condition_variable cv;

    int getSpace() const
    {
        if (this->tail < this->head)
        {
            return this->head - this->tail - 1; // no need to account for 0 index because index of tail itself is empty
        }
        else
        {
            return this->head + ((this->SIZE - 1) - this->tail); // count the space between 0th index and head, and the space between tail and the final index
        }
    }

public:
    ThreadSafeQueue(unsigned int capacity) : out(output), SIZE(capacity + 1), CAPACITY(capacity)
    {
        this->queue = new T[this->SIZE];
        this->head = 0;
        this->tail = 0;
    }

    ~ThreadSafeQueue()
    {
        delete[] queue;
    }

    void push(const T &data)
    {
        unique_lock<mutex> lock(this->locker); // automatically locks and unlocks in and out of scope

        while (this->getSpace() == 0)
        {
            // queue is full
            this->cv.wait(lock);
            // will recheck the condition once woken up (because it is in while)
        }

        this->queue[this->tail] = data;
        this->tail = (this->tail + 1) % this->SIZE; // wraps back to 0 if tail reaches to max of size
        this->cv.notify_one();                      // we just added value, so it can pop
    }

    T pop()
    {
        unique_lock<mutex> lock(this->locker);

        while (this->getSpace() == this->CAPACITY)
        {
            // queue is empty
            this->cv.wait(lock);
            // will recheck the condition once woken up (because it is in while)
        }

        T value = queue[this->head];

        this->head = (this->head + 1) % this->SIZE; // also wraps back to 0 if reaches end
        this->cv.notify_one();                      // we just moved head, so there should be a free space
        return value;
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

int main(int argc, char *argv[])
{
    VideoFFmpeg video("input.mp4");
    video.startDecode();

    ThreadTask tasks(MAX_THREADS - 1);

    int count = 0;
    while (true)
    {
        AVFrame *frame = video.getFrame();
        if (frame != nullptr)
        {
            count++;
            cout << "Frame " << frame->pts << endl;
            av_frame_free(&frame);
        }
        else
        {
            exit(0);
        }
    }
}