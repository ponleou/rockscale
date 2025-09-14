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
using namespace std::chrono;
using std::condition_variable;
using std::mutex;
using std::queue;
using std::string;
using std::unique_lock;
using std::vector;

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

public:
    VideoFFmpeg(string fileName)
    {
        int errorCode;

        this->file = avformat_alloc_context();
        errorCode = avformat_open_input(&this->file, fileName.c_str(), nullptr, 0);
        if (errorCode != 0)
        {
            exit(1);
        }
        errorCode = avformat_find_stream_info(this->file, 0);
        if (errorCode != 0)
        {
            exit(1);
        }

        for (int i = 0; i < this->file->nb_streams; i++)
        {
            if (this->file->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                const AVCodec *codecInfo = avcodec_find_decoder(this->file->streams[i]->codecpar->codec_id);
                this->codec = avcodec_alloc_context3(codecInfo);
                if (this->codec == NULL || avcodec_parameters_to_context(this->codec, this->file->streams[i]->codecpar) < 0)
                    exit(1);

                this->codec->thread_count = 16;
                this->codec->thread_type = FF_THREAD_FRAME;
                if (avcodec_open2(this->codec, codecInfo, nullptr) < 0)
                {
                    exit(1);
                }

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

    void sendPackets()
    {
        AVPacket *packet = av_packet_alloc();
        while (av_read_frame(this->file, packet) >= 0)
        {
            if (packet->stream_index == this->videoIndex)
            {
                unique_lock<mutex> codecLock(this->codecLocker);

                int sendCode;
                // codec buffer is full, need to get decoded frames out
                while ((sendCode = avcodec_send_packet(this->codec, packet)) == AVERROR(EAGAIN))
                {
                    AVFrame *frame = av_frame_alloc();
                    int recvCode = avcodec_receive_frame(this->codec, frame);
                    // if we got one, then we put that to the buffer
                    if (recvCode == 0)
                    {
                        unique_lock<mutex> bufferLock(this->bufferLocker);
                        this->decodedBuffer.push(frame);
                    }

                    // the only scenario where both avcodec_receive_frame and avcodec_send_packet is both AVERROR(EAGAIN) is when buffer is full, but frames are still decoding
                    // so we give up lock, sleep for a bit, then try to send again
                    else if (recvCode == AVERROR(EAGAIN))
                    {
                        av_frame_free(&frame);
                        this->codecCV.wait_for(codecLock, milliseconds(10));
                        continue;
                    }

                    else if (recvCode == AVERROR_EOF)
                        // TODO: handle
                        exit(0);
                    else if (recvCode == AVERROR(EINVAL))
                        exit(1);
                    else if (recvCode < 0)
                        exit(1);
                }

                if (sendCode != 0)
                    exit(1);

                // TODO: notify one?
                this->codecCV.notify_all();
            }
            av_packet_unref(packet);
        }
        av_packet_free(&packet);
    }

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

    AVFrame *getFrame()
    {
        AVFrame *frame = getFrameFromBuffer();
        if (frame != nullptr)
            return frame;

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
            else if (recvCode == AVERROR_EOF)
                // TODO: handle
                break;
            else if (recvCode == AVERROR(EINVAL))
                exit(1);
            else if (recvCode < 0)
                exit(1);
        }

        return frame;
    }
};

int main()
{
}