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
#include <utility>
using std::function;
using std::thread;
using namespace std::chrono;
using std::cerr;
using std::condition_variable;
using std::cout;
using std::endl;
using std::max;
using std::mutex;
using std::pair;
using std::priority_queue;
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
    struct ComparePTS
    {
        bool operator()(const AVFrame *a, const AVFrame *b) const
        {
            return a->pts > b->pts; // smaller pts first
        }
    };

    const int BATCH_SIZE;
    AVFormatContext *inFile;
    int videoIndex;

    AVFormatContext *outFile;
    AVStream *outStream;

    // FIXME: probably dont need this
    mutex codecLocker;
    condition_variable codecCV;

    AVCodecContext *decoder;
    condition_variable decoderCV; // NOTE: not for the buffer, its used by the decoder itself

    bool decoderFinished;
    queue<AVFrame *> decodeBuffer;
    mutex dbufferLocker; // NOTE: must mutex decoderFinished and decodeBuffer
    condition_variable dbufferCV;

    const AVCodec *encoderCodec;
    AVCodecContext *encoder;
    condition_variable encoderCV;

    bool encoderInitialised;
    bool encodeEnded;
    priority_queue<AVFrame *, vector<AVFrame *>, ComparePTS> encodeBuffer;
    mutex ebufferLocker; // NOTE: must mutex encoderInitialised and encodeBuffer
    condition_variable ebufferCV;

    void raiseError(string message, int code = -1)
    {
        cerr << message << ", code: " << code << endl;
        exit(code);
    }

    void prepareEncode(const char *fileName)
    {
        int fileCode;
        if ((fileCode = avformat_alloc_output_context2(&this->outFile, nullptr, nullptr, fileName)) < 0 || this->outFile == nullptr)
            this->raiseError("Error setting up output file", fileCode);

        this->outStream = avformat_new_stream(this->outFile, nullptr);
        if (outStream == nullptr)
            this->raiseError("Error creating stream");

        avcodec_parameters_from_context(this->outStream->codecpar, this->encoder);
        this->outStream->time_base = this->encoder->time_base;

        // check if we need to explicity call avio_open for the file
        if (!(this->outFile->oformat->flags & AVFMT_NOFILE))
        {
            if ((fileCode = avio_open(&this->outFile->pb, fileName, AVIO_FLAG_WRITE)) < 0)
                this->raiseError("Failed to open output file", fileCode);
        }

        if ((fileCode = avformat_write_header(this->outFile, nullptr)) < 0)
            this->raiseError("Failed to write header", fileCode);
    }

public:
    VideoFFmpeg(string fileName, int batchSize, int decodeThreads = 0) : BATCH_SIZE(batchSize)
    {
        int fileCode;

        this->inFile = avformat_alloc_context();
        fileCode = avformat_open_input(&this->inFile, fileName.c_str(), nullptr, 0);
        if (fileCode != 0)
            this->raiseError("Failed to open file", fileCode);
        fileCode = avformat_find_stream_info(this->inFile, 0);
        if (fileCode != 0)
            this->raiseError("Failed to open file", fileCode);

        int codecCode;
        for (int i = 0; i < this->inFile->nb_streams; i++)
        {
            if (this->inFile->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                const AVCodec *decoderInfo = avcodec_find_decoder(this->inFile->streams[i]->codecpar->codec_id);
                this->decoder = avcodec_alloc_context3(decoderInfo);
                if (this->decoder == nullptr || (codecCode = avcodec_parameters_to_context(this->decoder, this->inFile->streams[i]->codecpar)) < 0)
                    this->raiseError("Failed to find decoder", codecCode);

                this->decoder->thread_count = decodeThreads;
                this->decoder->thread_type = FF_THREAD_FRAME;

                if ((codecCode = avcodec_open2(this->decoder, decoderInfo, nullptr)) < 0)
                    this->raiseError("Failed to start decoder", codecCode);

                this->encoderCodec = avcodec_find_encoder(this->inFile->streams[i]->codecpar->codec_id);
                if (this->encoderCodec == nullptr)
                    this->encoderCodec = avcodec_find_encoder(AV_CODEC_ID_H264);

                this->encoder = avcodec_alloc_context3(this->encoderCodec);

                if (this->encoder == nullptr)
                    this->raiseError("Failed to find encoder", codecCode);

                // Use the original stream's timebase to preserve exact timing
                this->encoder->time_base = this->inFile->streams[i]->time_base;

                this->encoderInitialised = false; // need an initialise function
                this->encodeEnded = false;
                this->decoderFinished = false;

                this->videoIndex = i;
                break;
            }
        }
    }

    ~VideoFFmpeg()
    {
        avformat_close_input(&this->inFile);
        avcodec_free_context(&this->decoder);
        avcodec_free_context(&this->encoder);

        while (!this->encodeBuffer.empty())
        {
            AVFrame *frame = this->encodeBuffer.top();
            av_frame_free(&frame);
            this->encodeBuffer.pop();
        }
    }

    pair<int, int> getDimension()
    {
        return pair<int, int>(this->decoder->width, this->decoder->height);
    }

    void startDecode()
    {
        int sendCode;
        AVPacket *packet = av_packet_alloc();

        while (av_read_frame(this->inFile, packet) >= 0)
        {
            if (packet->stream_index == this->videoIndex)
            {
                unique_lock<mutex> codecLock(this->codecLocker);

                // codec buffer is full, need to get decoded frames out
                if ((sendCode = avcodec_send_packet(this->decoder, packet)) != 0)
                {
                    if (sendCode != AVERROR(EAGAIN))
                        this->raiseError("Failed to decode frame", sendCode);
                }

                while (true)
                {
                    AVFrame *frame = av_frame_alloc();
                    int recvCode = avcodec_receive_frame(this->decoder, frame);

                    // if we got one, then we put that to the buffer
                    if (recvCode == 0)
                    {
                        unique_lock<mutex> bufferLock(this->dbufferLocker);

                        // sleep if its full
                        while (this->decodeBuffer.size() > this->BATCH_SIZE)
                            this->decoderCV.wait(bufferLock);

                        this->decodeBuffer.push(frame);

                        // TODO: notify one?
                        // new frame incoming, wake all waiting for frame
                        this->dbufferCV.notify_all();
                    }

                    // no frames available currently, break
                    else if (recvCode == AVERROR(EAGAIN))
                    {
                        av_frame_free(&frame);
                        break;
                    }

                    // impossible scenario tho?
                    else if (recvCode == AVERROR_EOF)
                    {
                        av_frame_free(&frame);
                        break;
                    }

                    // error
                    else if (recvCode == AVERROR(EINVAL) || recvCode < 0)
                    {
                        av_frame_free(&frame);
                        this->raiseError("Failed to get frame from codec", recvCode);
                    }
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

        // flush remaining frames from decoder
        while (true)
        {
            AVFrame *frame = av_frame_alloc();
            int recvCode = avcodec_receive_frame(this->decoder, frame);
            if (recvCode == 0)
            {
                unique_lock<mutex> bufferLock(this->dbufferLocker);

                while (this->decodeBuffer.size() > this->BATCH_SIZE)
                    this->decoderCV.wait(bufferLock);

                this->decodeBuffer.push(frame);
                this->dbufferCV.notify_all();
            }
            else
            {
                av_frame_free(&frame);
                break;
            }
        }

        // EOF the decoder
        if ((sendCode = avcodec_send_packet(this->decoder, nullptr)) != 0)
            this->raiseError("Failed to EOF", sendCode);

        // lock when changing bool, and wake up for threads to check new bool
        unique_lock<mutex> lock(this->dbufferLocker);
        this->decoderFinished = true;
        this->dbufferCV.notify_all();
    }
    // this function promises to always return a frame, unless theres none left, which it will return nullptr
    AVFrame *getFrame()
    {
        AVFrame *frame = nullptr;

        unique_lock<mutex> bufferLock(this->dbufferLocker);
        while (this->decodeBuffer.empty())
        {
            if (this->decoderFinished)
                return nullptr;

            this->dbufferCV.wait(bufferLock);
        }
        frame = this->decodeBuffer.front();
        this->decodeBuffer.pop();

        this->decoderCV.notify_all();

        return frame;

        // TODO: remove commented code
        // frame = av_frame_alloc();

        // int recvCode;
        // unique_lock<mutex> codecLock(this->codecLocker);
        // while ((recvCode = avcodec_receive_frame(this->decoder, frame)) != 0)
        // {
        //     if (recvCode == AVERROR(EAGAIN))
        //     {
        //         cout << "this should never happen" << endl;
        //         this->codecCV.wait(codecLock);

        //         // try getting from queue again
        //         AVFrame *frameBuffer = getFrameFromBuffer();
        //         if (frameBuffer != nullptr)
        //             return frameBuffer;
        //     }

        //     // end of file, return nullptr
        //     else if (recvCode == AVERROR_EOF)
        //         return nullptr;

        //     // errors
        //     else if (recvCode == AVERROR(EINVAL))
        //         exit(1);
        //     else if (recvCode < 0)
        //         exit(1);
        // }
        // // cout << "Read frame from codec" << endl;
        // return frame;
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

        // Set proper color range to avoid deprecated pixel format warnings
        // int *inv_table, *table, srcRange, dstRange, brightness, contrast, saturation;
        // sws_getColorspaceDetails(sws_ctx, &inv_table, &srcRange, &table, &dstRange, &brightness, &contrast, &saturation);
        // sws_setColorspaceDetails(sws_ctx, inv_table, 1, table, 1, brightness, contrast, saturation);

        sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, convertedFrame->data, convertedFrame->linesize);

        sws_freeContext(sws_ctx);
        av_frame_free(&frame);

        return convertedFrame;
    }

    void initialiseEncoder(int width, int height, AVPixelFormat format)
    {
        {
            unique_lock<mutex> lock(this->ebufferLocker);
            if (encoderInitialised)
                this->raiseError("Encoder already initialised");
        }

        // force even for compatibility for specific encode codecs (h264)
        width &= ~1;
        height &= ~1;

        this->encoder->width = width;
        this->encoder->height = height;
        this->encoder->pix_fmt = format;

        // Add proper color space settings for H.264
        this->encoder->color_range = AVCOL_RANGE_MPEG;
        this->encoder->colorspace = AVCOL_SPC_BT709;
        this->encoder->color_primaries = AVCOL_PRI_BT709;
        this->encoder->color_trc = AVCOL_TRC_BT709;

        int decoderSize = this->decoder->width * this->decoder->height;
        int newSize = width * height;

        float scale = (float)newSize / (float)decoderSize;
        this->encoder->bit_rate = this->decoder->bit_rate * scale;

        unique_lock<mutex> lock(this->ebufferLocker);
        this->encoderInitialised = true;
        this->ebufferCV.notify_all();
    }

    void addEncodeFrames(AVFrame *frame)
    {
        {
            unique_lock<mutex> lock(this->ebufferLocker);
            if (!this->encoderInitialised)
                this->raiseError("Encoder not initialised");
        }

        // apply even for width and height
        if ((frame->width & ~1) != this->encoder->width || (frame->height & ~1) != this->encoder->height || frame->format != this->encoder->pix_fmt)
            this->raiseError("Frame doesn't match encoder");

        // change frame width and height to even for compatibility
        frame->width &= ~1;
        frame->height &= ~1;

        unique_lock<mutex> lock(this->ebufferLocker);

        while (this->encodeBuffer.size() > BATCH_SIZE)
        {
            this->ebufferCV.wait(lock);
        }
        this->encodeBuffer.push(frame);
        this->encoderCV.notify_all();

        // TODO: remove
        // int orderedIndex = -1;
        // for (int i = 0; i < this->encodeBuffer.size(); i++)
        //     if (this->encodeBuffer[i]->pts > frame->pts)
        //     {
        //         orderedIndex = i;
        //         break;
        //     }

        // if (orderedIndex != -1)
        //     this->encodeBuffer.insert(this->encodeBuffer.begin() + orderedIndex, frame);
        // else
        //     this->encodeBuffer.push_back(frame);
    }

    void notifyEndEncode()
    {
        unique_lock<mutex> lock(this->ebufferLocker);
        this->encodeEnded = true;
        this->ebufferCV.notify_all();
    }

    void startEncode(const char *fileName, int threads = 0)
    {
        {
            unique_lock<mutex> lock(this->ebufferLocker);
            if (!this->encoderInitialised)
                this->raiseError("Encoder not initialised");
        }

        this->encoder->thread_count = 0; // Disable encoder threading to avoid opaque pointer issues
        // this->encoder->thread_type = FF_THREAD_SLICE;

        int codecCode;
        if ((codecCode = avcodec_open2(this->encoder, this->encoderCodec, nullptr)) < 0)
            this->raiseError("Failed to start encoder", codecCode);

        this->prepareEncode(fileName);

        while (true)
        {
            AVFrame *frame = nullptr;
            {
                unique_lock<mutex> lock(this->ebufferLocker);

                while (this->encodeBuffer.empty())
                {
                    if (this->encodeEnded)
                        break;
                    this->encoderCV.wait(lock);
                }

                // check and break again
                if (this->encodeEnded)
                    break;

                frame = this->encodeBuffer.top();
                this->encodeBuffer.pop();
                this->ebufferCV.notify_all(); // Wake up processors waiting for buffer space
            }
            cout << "frame picked" << endl;

            int codecCode = avcodec_send_frame(this->encoder, frame);
            if (codecCode < 0)
                this->raiseError("Error sending frame to encoder", codecCode);

            AVPacket *pkt = av_packet_alloc();
            while ((codecCode = avcodec_receive_packet(this->encoder, pkt)) == 0)
            {
                pkt->stream_index = outStream->index;
                av_interleaved_write_frame(this->outFile, pkt);
                av_packet_unref(pkt);
            }

            if (codecCode != AVERROR(EAGAIN) && codecCode != AVERROR_EOF)
                this->raiseError("Error receiving packet from encoder", codecCode);

            av_frame_free(&frame);

            cout << "frame encoded" << endl;
        }
        cout << "encoder got end signal" << endl;

        // sending EOF to encoder
        avcodec_send_frame(this->encoder, nullptr);

        cout << "did we get here?" << endl;

        // just making sure all packets are recieved
        AVPacket *pkt = av_packet_alloc();
        int ret;
        while ((ret = avcodec_receive_packet(this->encoder, pkt)) == 0)
        {
            pkt->stream_index = outStream->index;
            av_interleaved_write_frame(this->outFile, pkt);
            av_packet_unref(pkt);
            cout << "flushing" << endl;
        }

        cout << "how about here?" << endl;

        // Check if we exited due to EOF (expected) or an error
        if (ret != AVERROR_EOF)
        {
            cout << "Flush error: " << ret << endl;
            av_packet_free(&pkt);
            this->raiseError("Error during encoder flush", ret);
        }

        av_packet_free(&pkt);

        cout << "encoder finished all frames" << endl;

        // finish file
        av_write_trailer(this->outFile);
        if (!(this->outFile->oformat->flags & AVFMT_NOFILE))
            avio_closep(&this->outFile->pb);
        avformat_free_context(this->outFile);
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
    void join(int assignedThread)
    {
        if (!started || this->tasks[assignedThread] == nullptr)
            return;

        pthread_join(this->threads[assignedThread], nullptr);
    }
    // join all
    void join()
    {
        if (!started)
            return;

        for (int i = 0; i < this->tasks.size(); i++)
            this->join(i);
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

void bilinearInterpolation(AVFrame **frame, int scale, hipStream_t &stream, uint8_t *hipYPlane, uint8_t *hipUPlane, uint8_t *hipVPlane, uint8_t *scaledYPlane, uint8_t *scaledUPlane, uint8_t *scaledVPlane)
{
    int width = (*frame)->width;
    int height = (*frame)->height;

    // divide 2 and ceil
    int chromaWidth = (width + 1) / 2;
    int chromaHeight = (height + 1) / 2;

    uint8_t *YPlane = (*frame)->data[0];
    uint8_t *UPlane = (*frame)->data[1];
    uint8_t *VPlane = (*frame)->data[2];

    int YLinesize = (*frame)->linesize[0];
    int ULinesize = (*frame)->linesize[1];
    int VLinesize = (*frame)->linesize[2];

    // only index that has a "next" row or column can interpolate, so the last col of any row, and last row of any col will be a straight map
    // a -1 because width and height is overlapping the very last index
    int newWidth = ((width - 1) * scale) + 1;
    int newHeight = ((height - 1) * scale) + 1;
    int newPlaneSize = newWidth * newHeight;
    // same idea here
    int newChromaWidth = ((chromaWidth - 1) * scale) + 1;
    int newChromaHeight = ((chromaHeight - 1) * scale) + 1;
    int newChromaPlaneSize = newChromaWidth * newChromaHeight;

    HIP_CHECK(hipMemcpy2DAsync(hipYPlane, width * sizeof(uint8_t), YPlane, YLinesize, width * sizeof(uint8_t), height, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpy2DAsync(hipUPlane, chromaWidth * sizeof(uint8_t), UPlane, ULinesize, chromaWidth * sizeof(uint8_t), chromaHeight, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpy2DAsync(hipVPlane, chromaWidth * sizeof(uint8_t), VPlane, VLinesize, chromaWidth * sizeof(uint8_t), chromaHeight, hipMemcpyHostToDevice, stream));

    int threadsPerBlock = 256;

    int rowThreadCountY = width * height;
    int rowNumBlocksY = ceil((float)rowThreadCountY / (float)threadsPerBlock);
    void *rowArgsY[] = {&hipYPlane, &scaledYPlane, &width, &width, &height, &scale};

    int rowThreadCountU = chromaWidth * chromaHeight;
    int rowNumBlocksU = ceil((float)rowThreadCountU / (float)threadsPerBlock);
    void *rowArgsU[] = {&hipUPlane, &scaledUPlane, &chromaWidth, &chromaWidth, &chromaHeight, &scale};

    int rowThreadCountV = chromaWidth * chromaHeight;
    int rowNumBlocksV = ceil((float)rowThreadCountV / (float)threadsPerBlock);
    void *rowArgsV[] = {&hipVPlane, &scaledVPlane, &chromaWidth, &chromaWidth, &chromaHeight, &scale};

    HIP_CHECK(hipLaunchKernel((const void *)linearRowInterpolate, dim3(rowNumBlocksY), dim3(threadsPerBlock), rowArgsY, 0, stream));
    HIP_CHECK(hipLaunchKernel((const void *)linearRowInterpolate, dim3(rowNumBlocksU), dim3(threadsPerBlock), rowArgsU, 0, stream));
    HIP_CHECK(hipLaunchKernel((const void *)linearRowInterpolate, dim3(rowNumBlocksV), dim3(threadsPerBlock), rowArgsV, 0, stream));

    int colThreadCountY = newWidth * (height - 1);
    int colNumBlocksY = ceil((float)colThreadCountY / (float)threadsPerBlock);
    void *colArgsY[] = {&scaledYPlane, &newWidth, &newHeight, &scale};

    int colThreadCountU = newChromaWidth * (chromaHeight - 1);
    int colNumBlocksU = ceil((float)colThreadCountU / (float)threadsPerBlock);
    void *colArgsU[] = {&scaledUPlane, &newChromaWidth, &newChromaHeight, &scale};

    int colThreadCountV = newChromaWidth * (chromaHeight - 1);
    int colNumBlocksV = ceil((float)colThreadCountV / (float)threadsPerBlock);
    void *colArgsV[] = {&scaledVPlane, &newChromaWidth, &newChromaHeight, &scale};

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
    HIP_CHECK(hipMemcpy2D(newFrame->data[0], newFrame->linesize[0], scaledYPlane, newWidth * sizeof(uint8_t), newWidth * sizeof(uint8_t), newHeight, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy2D(newFrame->data[1], newFrame->linesize[1], scaledUPlane, newChromaWidth * sizeof(uint8_t), newChromaWidth * sizeof(uint8_t), newChromaHeight, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy2D(newFrame->data[2], newFrame->linesize[2], scaledVPlane, newChromaWidth * sizeof(uint8_t), newChromaWidth * sizeof(uint8_t), newChromaHeight, hipMemcpyDeviceToHost));

    av_frame_free(frame);
    *frame = newFrame;

    cout << "finished frame" << endl;
    // DEBUG
    // {
    //     // Final dimensions
    //     int finalWidth = ((width - 1) * scale) + 1;
    //     int finalHeight = ((height - 1) * scale) + 1;

    //     int chromaFinalWidth = ((chromaWidth - 1) * scale) + 1;
    //     int chromaFinalHeight = ((chromaHeight - 1) * scale) + 1;

    //     // Create PGM image
    //     FILE *pgmFileY = fopen("outputy.pgm", "wb");
    //     fprintf(pgmFileY, "P5\n%d %d\n255\n", finalWidth, finalHeight);
    //     for (int y = 0; y < (*frame)->height; y++)
    //     {
    //         fwrite((*frame)->data[0] + y * (*frame)->linesize[0], 1, finalWidth, pgmFileY);
    //     }
    //     fclose(pgmFileY);

    //     FILE *pgmFileU = fopen("outputu.pgm", "wb");
    //     FILE *pgmFileV = fopen("outputv.pgm", "wb");
    //     fprintf(pgmFileU, "P5\n%d %d\n255\n", chromaFinalWidth, chromaFinalHeight);
    //     fprintf(pgmFileV, "P5\n%d %d\n255\n", chromaFinalWidth, chromaFinalHeight);

    //     for (int y = 0; y < chromaFinalHeight; y++)
    //     {
    //         fwrite((*frame)->data[1] + y * (*frame)->linesize[1], 1, chromaFinalWidth, pgmFileU);
    //         fwrite((*frame)->data[2] + y * (*frame)->linesize[2], 1, chromaFinalWidth, pgmFileV);
    //     }
    //     fclose(pgmFileU);

    //     fclose(pgmFileV);

    //     cout << "Created " << (*frame)->linesize[0] << " - " << (*frame)->width << " image" << endl;
    // }
}

void producer(VideoFFmpeg &video, int scale)
{
    video.startDecode();
}

void processor(VideoFFmpeg &video, int scale)
{
    AVFrame *frame = nullptr;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    pair<int, int> dimension = video.getDimension();
    int width = dimension.first;
    int height = dimension.second;

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

    while (true)
    {
        cout << "getting frame" << endl;
        frame = video.getFrame(AV_PIX_FMT_YUV420P);
        cout << "got frame" << endl;
        if (frame == nullptr)
        {
            cout << "breaking" << endl;
            break;
        }
        cout << "processing frame" << endl;
        bilinearInterpolation(&frame, scale, stream, hipYPlane, hipUPlane, hipVPlane, scaledYPlane, scaledUPlane, scaledVPlane);
        cout << "finished processing frame" << endl;

        video.addEncodeFrames(frame);
    }

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(hipYPlane));
    HIP_CHECK(hipFree(hipUPlane));
    HIP_CHECK(hipFree(hipVPlane));
    HIP_CHECK(hipFree(scaledYPlane));
    HIP_CHECK(hipFree(scaledUPlane));
    HIP_CHECK(hipFree(scaledVPlane));
}

void consumer(VideoFFmpeg &video, const char *fileName)
{
    video.startEncode(fileName);
}

int main(int argc, char *argv[])
{
    // 4 threads total: main, producer, processor, consumer
    int threadCount = max(static_cast<int>(MAX_THREADS), 4);
    int scale = 2;

    HIP_CHECK(hipSetDevice(0)); // setting HIP device to default

    VideoFFmpeg video("input.mp4", MAX_THREADS * 2);
    pair<int, int> dimension = video.getDimension();
    int scaledWidth = (dimension.first - 1) * scale + 1;
    int scaledHeight = (dimension.second - 1) * scale + 1;
    video.initialiseEncoder(scaledWidth, scaledHeight, AV_PIX_FMT_YUV420P);
    // video.startDecode();
    // TODO: call producer

    // ThreadTask tasks(MAX_THREADS - 1);

    ThreadTask threads(threadCount - 1);

    // cout << "Starting decode..." << endl;
    // video.startDecode();
    // cout << "Decode complete, starting upscale threads..." << endl;

    // producer becomes processor once its finished with its task
    auto producerTask = [&video, scale]()
    {
        producer(video, scale);
        cout << "producer finished" << endl;

        // processor(video, scale);
    };

    auto processorTask = [&video, scale]()
    {
        processor(video, scale);
        cout << "processor finished" << endl;
    };

    auto consumerTask = [&video]()
    {
        consumer(video, "output.mp4");
        cout << "consumer finished" << endl;
    };

    // one thread for producer and consumer
    threads.addTask(new function<void()>(producerTask), 0);
    threads.addTask(new function<void()>(consumerTask), 1);

    // the rest are processors
    for (int i = 2; i < threads.size; i++)
        threads.addTask(new function<void()>(processorTask), i);

    threads.start();

    cout << "Total threads created: " << threads.size << endl;
    cout << "Joining processor threads from 2 to " << (threads.size - 1) << endl;

    // once processors are all done, notify end to encoder
    for (int i = 2; i < threads.size; i++)
    {
        cout << "Joining thread " << i << endl;
        threads.join(i);
        cout << "Thread " << i << " joined" << endl;
    }

    // this will allow the consumer/encoder thread to end (or else it will keep running forever)
    cout << "calling encoder to end" << endl;
    video.notifyEndEncode();

    // Wait for the producer thread to finish
    cout << "Joining producer thread (0)" << endl;
    threads.join(0);
    cout << "Producer thread joined" << endl;

    // Wait for the consumer thread to finish
    cout << "Joining consumer thread (1)" << endl;
    threads.join(1);
    cout << "Consumer thread joined" << endl;

    cout << "All threads completed successfully" << endl;

    return 0;

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