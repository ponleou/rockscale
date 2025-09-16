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
    AVFormatContext *inFile;
    AVCodecContext *decoder;
    int videoIndex;

    mutex codecLocker;
    condition_variable codecCV;

    queue<AVFrame *> decodedBuffer;
    mutex decodedBufferLocker;

    const AVCodec *encoderCodec;
    AVCodecContext *encoder;
    bool encoderInitialised;

    vector<AVFrame *> encodeBuffer;
    mutex encodeBufferLocker;

    AVFormatContext *outFile;
    AVStream *outStream;

    AVFrame *getFrameFromBuffer()
    {
        unique_lock<mutex> bufferLock(this->decodedBufferLocker);
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
    VideoFFmpeg(string fileName, int decodeThreads = 0)
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

        for (int i = 0; i < this->encodeBuffer.size(); i++)
            av_frame_free(&encodeBuffer[i]);

        this->encodeBuffer.clear();
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
                // TODO: find out how to eof codec
                while ((sendCode = avcodec_send_packet(this->decoder, packet)) == AVERROR(EAGAIN))
                {
                    AVFrame *frame = av_frame_alloc();
                    int recvCode = avcodec_receive_frame(this->decoder, frame);
                    // if we got one, then we put that to the buffer
                    if (recvCode == 0)
                    {
                        unique_lock<mutex> bufferLock(this->decodedBufferLocker);
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

        if ((sendCode = avcodec_send_packet(this->decoder, nullptr)) != 0)
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
        while ((recvCode = avcodec_receive_frame(this->decoder, frame)) != 0)
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

    void initialiseEncoder(int width, int height, AVPixelFormat format)
    {
        this->encoder->width = width;
        this->encoder->height = height;
        this->encoder->pix_fmt = format;

        int decoderSize = this->decoder->width * this->decoder->height;
        int newSize = width * height;

        float scale = (float)newSize / (float)decoderSize;
        this->encoder->bit_rate = this->decoder->bit_rate * scale;

        unique_lock<mutex> lock(this->encodeBufferLocker);
        this->encoderInitialised = true;
    }

    void addEncodeFrames(AVFrame *frame)
    {
        {
            unique_lock<mutex> lock(this->encodeBufferLocker);
            if (!this->encoderInitialised)
                this->raiseError("Encoder not initialised");
        }

        if (frame->width != this->encoder->width || frame->height != this->encoder->height || frame->format != this->encoder->pix_fmt)
            this->raiseError("Frame doesn't match encoder");

        int orderedIndex = -1;
        for (int i = 0; i < this->encodeBuffer.size(); i++)
            if (this->encodeBuffer[i]->pts > frame->pts)
            {
                orderedIndex = i;
                break;
            }

        if (orderedIndex != -1)
            this->encodeBuffer.insert(this->encodeBuffer.begin() + orderedIndex, frame);
        else
            this->encodeBuffer.push_back(frame);
    }

    void startEncode(const char *fileName, int threads)
    {
        {
            unique_lock<mutex> lock(this->encodeBufferLocker);
            if (!this->encoderInitialised)
                this->raiseError("Encoder not initialised");
        }

        this->encoder->thread_count = threads;
        this->encoder->thread_type = FF_THREAD_FRAME;

        int codecCode;
        if ((codecCode = avcodec_open2(this->encoder, this->encoderCodec, nullptr)) < 0)
            this->raiseError("Failed to start encoder", codecCode);

        this->prepareEncode(fileName);

        unique_lock<mutex> lock(this->encodeBufferLocker);

        for (AVFrame *frame : this->encodeBuffer)
        {
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
        }
        this->encodeBuffer.clear();

        // sending EOF to encoder
        avcodec_send_frame(this->encoder, nullptr);

        // just making sure all packets are recieved
        AVPacket *pkt = av_packet_alloc();
        while (avcodec_receive_packet(this->encoder, pkt) == 0)
        {
            pkt->stream_index = outStream->index;
            av_interleaved_write_frame(this->outFile, pkt);
            av_packet_unref(pkt);
        }
        av_packet_free(&pkt);

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

void bilinearInterpolation(AVFrame **frame, int scale, hipStream_t &stream)
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

    int rowThreadCountU = ULinesize * chromaHeight;
    int rowNumBlocksU = ceil((float)rowThreadCountU / (float)threadsPerBlock);
    void *rowArgsU[] = {&hipUPlane, &scaledUPlane, &ULinesize, &chromaWidth, &chromaHeight, &scale};

    int rowThreadCountV = VLinesize * chromaHeight;
    int rowNumBlocksV = ceil((float)rowThreadCountV / (float)threadsPerBlock);
    void *rowArgsV[] = {&hipVPlane, &scaledVPlane, &VLinesize, &chromaWidth, &chromaHeight, &scale};

    HIP_CHECK(hipLaunchKernel((const void *)linearRowInterpolate, dim3(rowNumBlocksY), dim3(threadsPerBlock), rowArgsY, 0, stream));
    HIP_CHECK(hipLaunchKernel((const void *)linearRowInterpolate, dim3(rowNumBlocksU), dim3(threadsPerBlock), rowArgsU, 0, stream));
    HIP_CHECK(hipLaunchKernel((const void *)linearRowInterpolate, dim3(rowNumBlocksV), dim3(threadsPerBlock), rowArgsV, 0, stream));

    int colThreadCountY = newWidth * height;
    int colNumBlocksY = ceil((float)colThreadCountY / (float)threadsPerBlock);
    void *colArgsY[] = {&scaledYPlane, &newWidth, &newHeight, &scale};

    int colThreadCountU = newChromaWidth * chromaHeight;
    int colNumBlocksU = ceil((float)colThreadCountU / (float)threadsPerBlock);
    void *colArgsU[] = {&scaledUPlane, &newChromaWidth, &newChromaHeight, &scale};

    int colThreadCountV = newChromaWidth * chromaHeight;
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
    HIP_CHECK(hipMemcpy2D(newFrame->data[1], newFrame->linesize[1], scaledUPlane, newChromaWidth * sizeof(uint8_t), newChromaWidth * sizeof(uint8_t), newChromaHeight, hipMemcpyDeviceToHost))
    HIP_CHECK(hipMemcpy2D(newFrame->data[2], newFrame->linesize[2], scaledVPlane, newChromaWidth * sizeof(uint8_t), newChromaWidth * sizeof(uint8_t), newChromaHeight, hipMemcpyDeviceToHost))

    av_frame_free(frame);
    *frame = newFrame;

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

    HIP_CHECK(hipFree(hipYPlane));
    HIP_CHECK(hipFree(hipUPlane));
    HIP_CHECK(hipFree(hipVPlane));
    HIP_CHECK(hipFree(scaledYPlane));
    HIP_CHECK(hipFree(scaledUPlane));
    HIP_CHECK(hipFree(scaledVPlane));
}

void producer(VideoFFmpeg &video, int scale)
{
    video.startDecode();
}

void consumer(VideoFFmpeg &video, int scale)
{
    AVFrame *frame = nullptr;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    while (true)
    {
        frame = video.getFrame(AV_PIX_FMT_YUV420P);
        if (frame == nullptr)
            break;

        bilinearInterpolation(&frame, scale, stream);

        video.addEncodeFrames(frame);
    }

    HIP_CHECK(hipStreamDestroy(stream));
}

int main(int argc, char *argv[])
{
    int scale = 5;

    HIP_CHECK(hipSetDevice(0)); // setting HIP device to default

    VideoFFmpeg video("input.mp4");
    pair<int, int> dimension = video.getDimension();
    int scaledWidth = (dimension.first - 1) * scale + 1;
    int scaledHeight = (dimension.second - 1) * scale + 1;
    video.initialiseEncoder(scaledWidth, scaledHeight, AV_PIX_FMT_YUV420P);
    // video.startDecode();
    // TODO: call producer

    // ThreadTask tasks(MAX_THREADS - 1);

    ThreadTask threads(MAX_THREADS - 1);

    auto producerTask = [&video, scale]()
    {
        producer(video, scale);
        consumer(video, scale);
    };
    auto consumerTask = [&video, scale]()
    {
        consumer(video, scale);
    };

    // first thread is producer
    threads.addTask(new function<void()>(producerTask), 0);

    for (int i = 1; i < threads.size; i++)
        threads.addTask(new function<void()>(consumerTask), i);

    threads.start();
    threads.join();

    video.startEncode("output.mp4", MAX_THREADS - 1);

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