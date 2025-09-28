#pragma once
extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}
#include <vector>
#include <queue>
#include <condition_variable>
#include <iostream>

#include "logger.hpp"

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
using std::to_string;
using std::unique_lock;
using std::vector;

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
    AVFormatContext *mergeInFile; // Separate input file for mergeStreams
    int videoIndex;

    AVFormatContext *outFile;
    vector<AVStream *> outStreamsOrdered; // this contains outFile's stream in order of inFile's streams

    AVCodecContext *decoder;
    condition_variable decoderCV; // NOTE: not for the buffer, its used by the decoder itself

    bool decoderFinished;
    queue<AVFrame *> decodeBuffer;
    mutex dbufferLocker; // NOTE: must mutex decoderFinished and decodeBuffer
    condition_variable dbufferCV;
    int decodeFramesCount;

    const AVCodec *encoderCodec;
    AVCodecContext *encoder;
    condition_variable encoderCV;
    bool hwEncoder;
    AVHWFramesContext *hwFramesCtx;

    bool encoderInitialised;
    bool encodeEnded;
    priority_queue<AVFrame *, vector<AVFrame *>, ComparePTS> encodeBuffer;
    mutex ebufferLocker; // NOTE: must mutex encoderInitialised and encodeBuffer
    condition_variable ebufferCV;
    int encodeFramesCount;

    int misplaceFramesCount;

    void raiseError(string message, int code = -1)
    {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(code, errorBuf, AV_ERROR_MAX_STRING_SIZE);

        cout << endl;
        cout << "ERROR: " << message << ", code: " << code << " (" << errorBuf << ")" << endl;
        exit(code);
    }

    // interleaved packets until we put the passed videopacket, then we return and wait for the next ones
    void mergeStreams(AVPacket *videoPacket)
    {
        AVPacket *pkt = av_packet_alloc();

        while (av_read_frame(this->mergeInFile, pkt) >= 0)
        {

            AVStream *inStream = this->mergeInFile->streams[pkt->stream_index];
            AVStream *outStream = this->outStreamsOrdered[pkt->stream_index];

            if (pkt->stream_index == this->videoIndex)
            {
                if (videoPacket->pts != pkt->pts || videoPacket->dts != pkt->dts)
                    this->misplaceFramesCount++;

                // Copy the original video packet's timing to our encoded packet
                videoPacket->pts = av_rescale_q_rnd(pkt->pts, inStream->time_base, outStream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
                videoPacket->dts = av_rescale_q_rnd(pkt->dts, inStream->time_base, outStream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
                videoPacket->duration = av_rescale_q(pkt->duration, inStream->time_base, outStream->time_base);

                av_interleaved_write_frame(this->outFile, videoPacket);
                av_packet_unref(videoPacket);
                av_packet_unref(pkt);
                av_packet_free(&pkt);

                return;
            }

            pkt->pts = av_rescale_q_rnd(pkt->pts, inStream->time_base, outStream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
            pkt->dts = av_rescale_q_rnd(pkt->dts, inStream->time_base, outStream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
            pkt->duration = av_rescale_q(pkt->duration, inStream->time_base, outStream->time_base);
            pkt->pos = -1;

            av_interleaved_write_frame(this->outFile, pkt);
            av_packet_unref(pkt);
        }
        av_packet_free(&pkt);
    }

    // opens encoder and sets up file
    void prepareEncode(const char *fileName)
    {
        int fileCode;
        if ((fileCode = avformat_alloc_output_context2(&this->outFile, nullptr, nullptr, fileName)) < 0 || this->outFile == nullptr)
            this->raiseError("Error setting up output file", fileCode);

        // setup streams
        for (unsigned int i = 0; i < this->inFile->nb_streams; i++)
        {
            // prepate video stream for output file
            if (i == this->videoIndex)
            {
                AVStream *outputStream = avformat_new_stream(this->outFile, nullptr);
                if (outputStream == nullptr)
                    this->raiseError("Error creating stream");

                // just pushing for index, wont actually be used
                this->outStreamsOrdered.push_back(outputStream);
            }
            // other streams
            else
            {
                AVStream *inputStream = this->inFile->streams[i];
                AVStream *outputStream = avformat_new_stream(this->outFile, nullptr);

                if (outputStream == nullptr)
                    this->raiseError("Error creating stream");

                avcodec_parameters_copy(outputStream->codecpar, inputStream->codecpar);
                outputStream->time_base = inputStream->time_base;
                outputStream->r_frame_rate = inputStream->r_frame_rate;
                outputStream->avg_frame_rate = inputStream->avg_frame_rate;

                this->outStreamsOrdered.push_back(outputStream);
            }
        }

        // set global header flag for containers that require it (like mkv)
        if (this->outFile->oformat->flags & AVFMT_GLOBALHEADER)
            this->encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

        // NOTE: opening encoder cant be done in initialiseEncoder, hevc fails
        int codecCode;
        if ((codecCode = avcodec_open2(this->encoder, this->encoderCodec, nullptr)) < 0)
            this->raiseError("Failed to start encoder", codecCode);

        // setup video stream
        avcodec_parameters_from_context(this->outStreamsOrdered[this->videoIndex]->codecpar, this->encoder);
        this->outStreamsOrdered[this->videoIndex]->time_base = this->encoder->time_base;
        this->outStreamsOrdered[this->videoIndex]->r_frame_rate = this->encoder->framerate;
        this->outStreamsOrdered[this->videoIndex]->avg_frame_rate = this->encoder->framerate;

        // check if we need to explicity call avio_open for the file
        if (!(this->outFile->oformat->flags & AVFMT_NOFILE))
        {
            if ((fileCode = avio_open(&this->outFile->pb, fileName, AVIO_FLAG_WRITE)) < 0)
                this->raiseError("Failed to open output file", fileCode);
        }

        // write file header
        if ((fileCode = avformat_write_header(this->outFile, nullptr)) < 0)
            this->raiseError("Failed to write header", fileCode);
    }

    // this will change the threshold so that it remains at a level that the buffer barely gets to full
    // this means that it will give the best accuracy possible for the performance
    // also means that higher performance would impact accuracy, hence, minThreshold to guard accuracy
    void encoderAdaptiveThreshold(int &threshold, int minThreshold = 0)
    {
        if (this->encodeBuffer.size() == BATCH_SIZE)
        {
            threshold = max(threshold - 1, minThreshold);
        }
    }

    // unsafe because no lock
    void pushDecodeBufferUnsafe(AVFrame *frame)
    {
        this->decodeBuffer.push(frame);

        // new frame incoming, wake all waiting for frame
        this->dbufferCV.notify_all();

        this->decodeFramesCount += 1;
    }

    void pushDecodeBuffer(AVFrame *frame)
    {
        unique_lock<mutex> bufferLock(this->dbufferLocker);

        // sleep if its full
        while (this->decodeBuffer.size() > this->BATCH_SIZE)
            this->decoderCV.wait(bufferLock);

        this->pushDecodeBufferUnsafe(frame);
    }

    void pushDecodeBuffer(AVFrame *frame, Logger *logger, int line)
    {
        unique_lock<mutex> bufferLock(this->dbufferLocker);

        // sleep if its full
        while (this->decodeBuffer.size() > this->BATCH_SIZE)
            this->decoderCV.wait(bufferLock);

        this->pushDecodeBufferUnsafe(frame);

        if (logger != nullptr)
        {
            logger->log(line, "Decoder: " + to_string(this->decodeFramesCount) + " frames");
        }
    }

    // unsafe because it needs a locker
    AVFrame *popEncodeBufferUnsafe()
    {
        AVFrame *frame = this->encodeBuffer.top();
        this->encodeBuffer.pop();
        this->ebufferCV.notify_all();
        this->encodeFramesCount += 1;

        if (this->hwEncoder && frame->pts == 0)
        {
            frame->pts = 1;
            cout << "Adjusted PTS from 0 to 1 for hardware encoder" << endl;
        }

        if (this->hwEncoder)
        {
            // Convert YUV420P to NV12 for VAAPI compatibility
            if (frame->format == AV_PIX_FMT_YUV420P)
            {
                cout << "Converting YUV420P to NV12 for VAAPI" << endl;

                AVFrame *nv12Frame = av_frame_alloc();
                nv12Frame->format = AV_PIX_FMT_NV12;
                nv12Frame->width = frame->width;
                nv12Frame->height = frame->height;
                av_frame_get_buffer(nv12Frame, 0);

                SwsContext *swsCtx = sws_getContext(frame->width, frame->height, AV_PIX_FMT_YUV420P,
                                                    frame->width, frame->height, AV_PIX_FMT_NV12,
                                                    SWS_BILINEAR, nullptr, nullptr, nullptr);
                sws_scale(swsCtx, frame->data, frame->linesize, 0, frame->height,
                          nv12Frame->data, nv12Frame->linesize);
                sws_freeContext(swsCtx);

                av_frame_copy_props(nv12Frame, frame);
                av_frame_free(&frame);
                frame = nv12Frame;

                cout << "Converted to NV12: format=" << frame->format << " (should be " << AV_PIX_FMT_NV12 << ")" << endl;
            }

            // Create hardware frame
            AVFrame *hwFrame = av_frame_alloc();

            int code;
            if ((code = av_hwframe_get_buffer(this->encoder->hw_frames_ctx, hwFrame, 0)) < 0)
            {
                av_frame_free(&hwFrame);
                this->raiseError("Failed to get hardware frame buffer", code);
            }

            cout << "HW frame format after get_buffer: " << hwFrame->format << endl;
            cout << "Encoder expected format: " << this->encoder->pix_fmt << endl;
            cout << "Encoder sw_pix_fmt: " << this->encoder->sw_pix_fmt << endl;

            if ((code = av_hwframe_transfer_data(hwFrame, frame, 0)) < 0)
            {
                char errorBuf[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(code, errorBuf, AV_ERROR_MAX_STRING_SIZE);
                cout << "Transfer data failed: " << code << " (" << errorBuf << ")" << endl;
                av_frame_free(&hwFrame);
                this->raiseError("Failed to transfer data to hardware frame", code);
            }

            av_frame_copy_props(hwFrame, frame);
            av_frame_free(&frame);
            frame = hwFrame;
        }

        return frame;
    }

    AVFrame *popEncodeBufferUnsafe(Logger *logger, int line)
    {
        AVFrame *frame = this->popEncodeBufferUnsafe();

        if (logger != nullptr)
        {
            logger->log(line, "Encoder: " + to_string(this->encodeFramesCount) + " frames");
        }

        return frame;
    }

public:
    VideoFFmpeg(string fileName, int batchSize, int decodeThreads = 0) : BATCH_SIZE(batchSize)
    {
        this->decodeFramesCount = 0;
        this->encodeFramesCount = 0;
        this->misplaceFramesCount = 0;
        this->hwEncoder = false;

        int fileCode;

        this->inFile = avformat_alloc_context();
        fileCode = avformat_open_input(&this->inFile, fileName.c_str(), nullptr, 0);
        if (fileCode != 0)
            this->raiseError("Failed to open file", fileCode);
        fileCode = avformat_find_stream_info(this->inFile, 0);
        if (fileCode != 0)
            this->raiseError("Failed to open file", fileCode);

        // Open separate input file for mergeStreams
        this->mergeInFile = avformat_alloc_context();
        fileCode = avformat_open_input(&this->mergeInFile, fileName.c_str(), nullptr, 0);
        if (fileCode != 0)
            this->raiseError("Failed to open merge input file", fileCode);
        fileCode = avformat_find_stream_info(this->mergeInFile, 0);
        if (fileCode != 0)
            this->raiseError("Failed to open merge input file", fileCode);

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
        avformat_close_input(&this->mergeInFile);
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
        return pair<int, int>(this->inFile->streams[this->videoIndex]->codecpar->width, this->inFile->streams[this->videoIndex]->codecpar->height);
    }

    AVPixelFormat getPixelFormat()
    {
        return (enum AVPixelFormat)this->inFile->streams[this->videoIndex]->codecpar->format;
    }

    void startDecode(Logger *logger = nullptr, int line = 0)
    {
        int sendCode;
        AVPacket *packet = av_packet_alloc();

        while (av_read_frame(this->inFile, packet) >= 0)
        {
            if (packet->stream_index == this->videoIndex)
            {
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
                        pushDecodeBuffer(frame, logger, line);

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
                pushDecodeBuffer(frame, logger, line);
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

    // FIXME: this is not finished yet, nearly none of the hardware encoders use YUV pixel format
    // this is a WIP
    void findHwEncoder(int width, int height, AVPixelFormat format, AVCodecID codecId)
    {
        void *i = nullptr;
        while ((this->encoderCodec = av_codec_iterate(&i)))
        {
            // checking if the codec supports encoding, is the same codecId we are looking for, and is hardware
            const AVCodecHWConfig *config = nullptr;
            if (av_codec_is_encoder(this->encoderCodec) && this->encoderCodec->id == codecId && (config = avcodec_get_hw_config(this->encoderCodec, 0)) != nullptr)
                // checking if we can open this encoder codec
                if ((this->encoder = avcodec_alloc_context3(this->encoderCodec)) != nullptr)
                {

                    AVBufferRef *hwDevice = nullptr;
                    if (av_hwdevice_ctx_create(&hwDevice, config->device_type, nullptr, nullptr, 0) != 0)
                    {
                        avcodec_free_context(&this->encoder);
                        this->encoder = nullptr;
                        continue;
                    }

                    // setting necessary parameters to check if we can open encoder
                    this->encoder->width = width;
                    this->encoder->height = height;
                    // NOTE: hw encoder have a width and height limit apparently

                    this->encoder->time_base = this->inFile->streams[this->videoIndex]->time_base;
                    this->encoder->pix_fmt = config->pix_fmt;
                    this->encoder->hw_device_ctx = hwDevice;
                    this->encoder->sw_pix_fmt = AV_PIX_FMT_NV12;

                    AVBufferRef *hwBufferRef = av_hwframe_ctx_alloc(hwDevice);
                    AVHWFramesContext *hwFramesCtx = (AVHWFramesContext *)hwBufferRef->data;
                    hwFramesCtx->format = config->pix_fmt;
                    hwFramesCtx->width = this->encoder->width;
                    hwFramesCtx->height = this->encoder->height;
                    hwFramesCtx->sw_format = AV_PIX_FMT_NV12;

                    if (av_hwframe_ctx_init(hwBufferRef) < 0)
                    {
                        av_buffer_unref(&hwBufferRef);
                        av_buffer_unref(&hwDevice);
                        avcodec_free_context(&this->encoder);
                        this->encoder = nullptr;
                        continue;
                    }

                    this->encoder->hw_frames_ctx = hwBufferRef;

                    // if we can open, then we will use this codec
                    if (avcodec_open2(this->encoder, this->encoderCodec, nullptr) == 0)
                    {
                        // reset
                        avcodec_free_context(&this->encoder);
                        this->encoder = nullptr;
                        this->encoder = avcodec_alloc_context3(this->encoderCodec);
                        this->hwEncoder = true;
                        return;
                    }

                    // if this codec doesnt work, then we go next
                    avcodec_free_context(&this->encoder);
                    this->encoder = nullptr;
                }
        }
    }

    void initialiseEncoder(int width, int height, AVPixelFormat format, bool useHardware = false, int threads = 0)
    {
        {
            unique_lock<mutex> lock(this->ebufferLocker);
            if (encoderInitialised)
                this->raiseError("Encoder already initialised");
        }

        // force even for compatibility for specific encode codecs (h264)
        width &= ~1;
        height &= ~1;

        // try to find original codec, if not, theres a list of fallbacks
        AVCodecID codecId = this->inFile->streams[this->videoIndex]->codecpar->codec_id;

        if (useHardware)
            // this will make hwEncoder true if theres one available
            this->findHwEncoder(width, height, format, codecId);

        // if we are not using hwEncoder, we fallback to software
        if (!this->hwEncoder)
        {
            this->encoderCodec = avcodec_find_encoder(codecId);
            if (this->encoderCodec == nullptr)
            {
                // trying different ones, in order of most compatible
                AVCodecID fallbackCodecs[] = {AV_CODEC_ID_H264, AV_CODEC_ID_H265, AV_CODEC_ID_VP9, AV_CODEC_ID_AV1};
                for (AVCodecID codecId : fallbackCodecs)
                {
                    this->encoderCodec = avcodec_find_encoder(codecId);
                    if (this->encoderCodec != nullptr)
                        break;
                }
            }

            if (this->encoderCodec == nullptr)
                this->raiseError("No suitable encoder codec found");

            this->encoder = avcodec_alloc_context3(this->encoderCodec);

            if (this->encoder == nullptr)
                this->raiseError("Failed to find encoder");
        }

        cout << this->encoderCodec->name << endl;

        this->encoder->width = width;
        this->encoder->height = height;
        this->encoder->pix_fmt = format;
        this->encoder->max_b_frames = 0;
        this->encoder->flags = 0;

        AVCodecParameters *inputParams = this->inFile->streams[this->videoIndex]->codecpar;
        this->encoder->color_range = inputParams->color_range;
        this->encoder->color_primaries = inputParams->color_primaries;
        this->encoder->color_trc = inputParams->color_trc;
        this->encoder->colorspace = inputParams->color_space;
        this->encoder->chroma_sample_location = inputParams->chroma_location;

        this->encoder->framerate = this->inFile->streams[this->videoIndex]->r_frame_rate;

        // fallback if framerate is invalid or sth
        if (this->encoder->framerate.num == 0 || this->encoder->framerate.den == 0)
        {
            this->encoder->framerate = av_guess_frame_rate(this->inFile, this->inFile->streams[this->videoIndex], nullptr);
        }

        this->encoder->time_base = this->inFile->streams[this->videoIndex]->time_base;

        // sometimes timebase is not divisble by framerate, and becomes wrong
        // AVRational timebase = this->inFile->streams[this->videoIndex]->time_base;
        // AVRational framerate = this->encoder->framerate;
        // float ratioRounded = round((float)(timebase.den * framerate.den) / (float)(timebase.num * framerate.num));
        // int newTimebaseDen = (float)(ratioRounded * framerate.num * timebase.num) / (float)framerate.den;
        // this->encoder->time_base.den = newTimebaseDen;

        float fps = av_q2d(this->encoder->framerate);

        // TODO: NOTE: i honestly have no idea what GOP size is, this is purely AI generated
        // Set appropriate GOP size for different codecs
        if (this->encoderCodec->id == AV_CODEC_ID_H264 || this->encoderCodec->id == AV_CODEC_ID_H265)
        {
            if (fps > 0)
                this->encoder->gop_size = (int)(fps * 2); // 2 seconds GOP
            else
                this->encoder->gop_size = 50; // Default fallback
        }
        else if (this->encoderCodec->id == AV_CODEC_ID_VP9 || this->encoderCodec->id == AV_CODEC_ID_AV1)
        {
            if (fps > 0)
                this->encoder->gop_size = (int)(fps * 1); // 1 second GOP for VP9/AV1
            else
                this->encoder->gop_size = 30;
        }

        int64_t fileSize_bits = avio_size(this->inFile->pb) * 8;            // original is in bytes
        double fileDuration_s = (double)this->inFile->duration / 1000000.0; // original duration is in microsections
        int fileBitrate = (int)(fileSize_bits / fileDuration_s);

        pair<int, int> fileDimension = this->getDimension();
        int originalSize = fileDimension.first * fileDimension.second;
        int newSize = width * height;

        float scale = (float)newSize / (float)originalSize;
        this->encoder->bit_rate = fileBitrate * scale;

        this->encoder->thread_count = threads;
        this->encoder->thread_type = FF_THREAD_FRAME;

        if (this->hwEncoder)
        {
            int code;
            // NOTE: none of these stuff should fail, we already tested in findHwEncoder function
            const AVCodecHWConfig *config = nullptr;
            config = avcodec_get_hw_config(this->encoderCodec, 0);

            AVBufferRef *hwDevice = nullptr;
            if ((code = av_hwdevice_ctx_create(&hwDevice, config->device_type, nullptr, nullptr, 0)) != 0)
                this->raiseError("Unexpected error in initialising hardware encoder", code);

            // change some of the encoder parameters
            this->encoder->pix_fmt = config->pix_fmt;
            this->encoder->hw_device_ctx = hwDevice;
            this->encoder->sw_pix_fmt = AV_PIX_FMT_NV12;

            cout << "format " << av_pix_fmt_desc_get(format)->name << endl;

            AVBufferRef *hwBufferRef = av_hwframe_ctx_alloc(hwDevice);
            AVHWFramesContext *hwFramesCtx = (AVHWFramesContext *)hwBufferRef->data;
            hwFramesCtx->format = config->pix_fmt;
            hwFramesCtx->width = this->encoder->width;
            hwFramesCtx->height = this->encoder->height;
            hwFramesCtx->sw_format = AV_PIX_FMT_NV12;
            if ((code = av_hwframe_ctx_init(hwBufferRef)) != 0)
                this->raiseError("Unexpected error in initialising hardware encoder", code);

            this->encoder->hw_frames_ctx = hwBufferRef;

            this->hwFramesCtx = hwFramesCtx;
        }

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

        cout << "SW frame pushed to buffer with pts=" << frame->pts << endl;

        // // prepare frame for hw encoder
        // if (this->hwEncoder)
        // {
        //     AVFrame *hwFrame = av_frame_alloc();
        //     hwFrame->format = this->encoder->pix_fmt;
        //     hwFrame->width = this->encoder->width;
        //     hwFrame->height = this->encoder->height;
        //     hwFrame->hw_frames_ctx = this->encoder->hw_frames_ctx;

        //     int code;
        //     if ((code = av_hwframe_get_buffer(encoder->hw_frames_ctx, hwFrame, 0)) < 0)
        //     {
        //         av_frame_free(&hwFrame);
        //         this->raiseError("Failed to convert software frame to hardware frame", code);
        //     }

        //     if ((code = av_hwframe_transfer_data(hwFrame, frame, 0)) < 0)
        //     {
        //         av_frame_free(&hwFrame);
        //         this->raiseError("Failed to convert software frame to hardware frame", code);
        //     }

        //     av_frame_free(&frame);
        //     frame = hwFrame;
        // }

        // apply even for width and height
        // if ((frame->width & ~1) != this->encoder->width || (frame->height & ~1) != this->encoder->height || frame->format != this->encoder->pix_fmt)
        //     this->raiseError("Frame doesn't match encoder");

        unique_lock<mutex> lock(this->ebufferLocker);

        // TODO: encoder is the bottle neck, so limiting the size also slows down processors and decoders
        // but unlimiting the size is a disaster, dont
        while (this->encodeBuffer.size() > BATCH_SIZE)
        {
            this->ebufferCV.wait(lock);
        }
        this->encodeBuffer.push(frame);
        this->encoderCV.notify_all();
    }

    void notifyEndEncode()
    {
        unique_lock<mutex> lock(this->ebufferLocker);
        this->encodeEnded = true;
        this->ebufferCV.notify_all();
        this->encoderCV.notify_all();
    }

    void startEncode(const char *fileName, float thresholdLevel, Logger *logger = nullptr, int line = 0)
    {
        {
            unique_lock<mutex> lock(this->ebufferLocker);
            if (!this->encoderInitialised)
                this->raiseError("Encoder not initialised");
        }

        // this threshold measures how many elements in the buffer at minimum to pick out a frame and encode
        // the less frames there are, the more likely that the inordered frame as not arrived in the buffer yet
        int threshold = this->BATCH_SIZE;
        const int minThreshold = this->BATCH_SIZE * thresholdLevel;

        this->prepareEncode(fileName);

        while (true)
        {
            AVFrame *frame = nullptr;
            {
                unique_lock<mutex> lock(this->ebufferLocker);

                // Wait until we have frames
                while (this->encodeBuffer.empty())
                {
                    if (this->encodeEnded)
                        break;
                    this->encoderCV.wait(lock);
                }

                // if its empty and ended, break, theres a step at the end that works with any remaining frames in the buffer
                // TODO: might need to remove empty check
                if (this->encodeEnded && this->encodeBuffer.empty())
                    break;

                // only encode if the buffer has a threshold amount of frames
                // NOTE: we are relying on the fact that if enough frames accumulate, the order will be near guarenteed
                // ALSO NOTE: the buffer size of the encode and decode buffer is the same, so if we wait until buffer is full, then it is 100% guarenteed to be correct positions

                encoderAdaptiveThreshold(threshold, minThreshold);
                while (this->encodeBuffer.size() < threshold && !this->encodeEnded)
                {
                    this->encoderCV.wait(lock);
                }

                frame = this->popEncodeBufferUnsafe(logger, line);
            }

            // send a frame, if it is successful go next
            int codecCode;
            if ((codecCode = avcodec_send_frame(this->encoder, frame)) == 0)
            {
                av_frame_free(&frame);
                continue;
            }

            // from this point, a frame wasnt successfully sent to encoder

            // time to drain
            while (codecCode == AVERROR(EAGAIN))
            {
                int recvCode;
                AVPacket *pkt = av_packet_alloc();
                while ((recvCode = avcodec_receive_packet(this->encoder, pkt)) == 0)
                {
                    // getting a packet and processing it
                    pkt->stream_index = outStreamsOrdered[this->videoIndex]->index;

                    this->mergeStreams(pkt);

                    // try to send the packet again, if its successful, no need to get more packets
                    if ((codecCode = avcodec_send_frame(this->encoder, frame)) == 0)
                    {
                        av_frame_free(&frame);
                        break;
                    }
                }

                // definitely some sort of error if the encoder is full (cannot take frames) AND the encoder has no packets to give
                if (recvCode == AVERROR(EAGAIN))
                    this->raiseError("Error receiving packet from encoder", recvCode);

                if (recvCode == AVERROR_EOF)
                    this->raiseError("Something unexpected happened", recvCode);
            }

            // this shouldnt happen? because notification would always likely to come first
            if (codecCode == AVERROR_EOF)
                break;

            else if (codecCode < 0)
                this->raiseError("Error sending frame to encoder", codecCode);
        }

        // Process any remaining frames when ending
        {
            unique_lock<mutex> lock(this->ebufferLocker);
            this->encodeBuffer.push(nullptr); // add to EOF encoder
            while (!this->encodeBuffer.empty())
            {
                AVFrame *frame = this->popEncodeBufferUnsafe(logger, line);
                lock.unlock();

                int codecCode;
                if ((codecCode = avcodec_send_frame(this->encoder, frame)) == 0)
                {
                    av_frame_free(&frame);
                    continue;
                }

                // from this point, a frame wasnt successfully sent to encoder

                // time to drain
                while (codecCode == AVERROR(EAGAIN))
                {
                    int recvCode;
                    AVPacket *pkt = av_packet_alloc();
                    while ((recvCode = avcodec_receive_packet(this->encoder, pkt)) == 0)
                    {
                        // getting a packet and processing it
                        this->mergeStreams(pkt);

                        // try to send the packet again, if its successful, no need to get more packets
                        if ((codecCode = avcodec_send_frame(this->encoder, frame)) == 0)
                        {
                            av_frame_free(&frame);
                            break;
                        }
                    }

                    // definitely some sort of error if the encoder is full (cannot take frames) AND the encoder has no packets to give
                    if (recvCode == AVERROR(EAGAIN))
                        this->raiseError("Error receiving packet from encoder", recvCode);

                    if (recvCode == AVERROR_EOF)
                        this->raiseError("Something unexpected happened", recvCode);
                }

                if (codecCode == AVERROR_EOF)
                    break;

                else if (codecCode < 0)
                    this->raiseError("Error sending frame to encoder", codecCode);

                lock.lock(); // lock for checking while loop condition
            }
        }

        int ret;

        // just making sure all packets are recieved
        AVPacket *pkt = av_packet_alloc();
        while ((ret = avcodec_receive_packet(this->encoder, pkt)) == 0)
        {
            pkt->stream_index = this->outStreamsOrdered[this->videoIndex]->index;

            this->mergeStreams(pkt);
        }

        av_packet_free(&pkt);

        // finish file
        av_write_trailer(this->outFile);
        if (!(this->outFile->oformat->flags & AVFMT_NOFILE))
            avio_closep(&this->outFile->pb);
        avformat_free_context(this->outFile);
    }

    int getFrameCount()
    {
        return max(this->encodeFramesCount, this->decodeFramesCount);
    }

    int getMisplaceFrameCount()
    {
        return this->misplaceFramesCount;
    }
};
