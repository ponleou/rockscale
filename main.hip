extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include "interpolations/bilinear.hpp"
#include "interpolations/bicubic.hpp"
#include "interpolations/lanczos.hpp"
#include "utils/video.hpp"
#include "utils/logger.hpp"

#include <mutex>
#include <string>
#include <cstdlib>
#include <vector>
#include <queue>
#include <condition_variable>
#include <iostream>
#include <pthread.h>
#include <thread>
#include <functional>
#include <utility>
#include <algorithm>
#include <iomanip>
#include <map>
using std::function;
using std::thread;
using namespace std::chrono;
using std::cerr;
using std::condition_variable;
using std::cout;
using std::endl;
using std::fixed;
using std::flush;
using std::left;
using std::logic_error;
using std::map;
using std::max;
using std::min;
using std::mutex;
using std::pair;
using std::priority_queue;
using std::queue;
using std::setprecision;
using std::setw;
using std::stof;
using std::stoi;
using std::string;
using std::unique_lock;
using std::vector;

#define MAX_THREADS thread::hardware_concurrency()

class ThreadTask
{
private:
    struct Task
    {
        function<void(int)> *function;
        int num;
    };

    vector<pthread_t> threads;
    vector<Task *> tasks;
    bool started;

    static void *functionCaller(void *arg)
    {
        Task *task = static_cast<Task *>(arg);
        if (task->function != nullptr)
            (*task->function)(task->num);
        return nullptr;
    }

public:
    const int size;

    ThreadTask(int threadCount) : size(threadCount)
    {
        this->threads = vector<pthread_t>(this->size);
        this->tasks = vector<Task *>(this->size, nullptr);
        this->started = false;
    }

    ~ThreadTask()
    {
        for (int i = 0; i < this->tasks.size(); i++)
        {
            if (this->tasks[i] != nullptr)
            {
                delete this->tasks[i]->function;
                delete this->tasks[i];
            }
        }
    }

    void addTask(function<void(int)> *task, int assignedThread)
    {
        this->tasks[assignedThread] = new Task{task, assignedThread};
    }

    void start()
    {
        if (started)
            throw logic_error("Instance is already executed.");

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

enum Interpolation
{
    BILINEAR,
    BICUBIC,
    LANCZOS,
};

struct InterpolationOptions
{
    int scale;
    AVPixelFormat format;
    int halfWindow;
    LanczosKernel *kernel;
};

struct ProcessorOptions : InterpolationOptions
{
    Interpolation interpolation;
};

void postInterpolation(VideoPipeline &video, AVFrame *&frame, AVPixelFormat format, Logger *logger = nullptr, int line = 0, int *frameCount = nullptr)
{
    *frameCount += 1;
    if (logger != nullptr)
        logger->log(line, "Processor " + to_string(line - 1) + ": " + to_string(*frameCount) + " frames");

    video.addEncodeFrames(frame);
    frame = video.getFrame(format);
}

void interpolateVideo(VideoPipeline &video, Interpolation type, InterpolationOptions &options, Logger *logger = nullptr, int line = 0)
{
    int frameCount = 0;
    AVFrame *frame = video.getFrame(options.format);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    if (type == BILINEAR)
    {
        BilinearGPUMemory bilinearMemory(frame, options.scale);

        do
        {
            bilinearInterpolation(&frame, options.scale, stream, bilinearMemory);
            postInterpolation(video, frame, options.format, logger, line, &frameCount);
        } while (frame != nullptr);
    }

    else if (type == BICUBIC)
    {
        BicubicGPUMemory bicubicMemory(frame, options.scale);
        BicubicTangentGPUMemory bicubicTangent(frame, options.scale);

        do
        {
            bicubicInterpolation(&frame, options.scale, stream, bicubicMemory, bicubicTangent);
            postInterpolation(video, frame, options.format, logger, line, &frameCount);
        } while (frame != nullptr);
    }

    else if (type == LANCZOS)
    {
        LanczosGPUMemory lanczosMemory(frame, options.scale);
        LanczosKernelGPUMemory lanczosKernel(options.scale, options.halfWindow);
        lanczosKernel.memcpyKernel(*options.kernel, stream);

        do
        {
            lanczosInterpolation(&frame, options.scale, options.halfWindow, *options.kernel, stream, lanczosMemory, lanczosKernel);
            postInterpolation(video, frame, options.format, logger, line, &frameCount);
        } while (frame != nullptr);
    }

    HIP_CHECK(hipStreamDestroy(stream));
}

void processor(VideoPipeline &video, ProcessorOptions &options, Logger &logger, int threadNum)
{
    interpolateVideo(video, options.interpolation, options, &logger, threadNum + 2);
}

void producer(VideoPipeline &video, int scale, Logger &logger, int threadNum)
{
    video.startDecode(&logger, threadNum + 2);
}

void consumer(VideoPipeline &video, const char *fileName, float thresholdLevel, Logger &logger, int threadNum)
{
    video.startEncode(fileName, thresholdLevel, &logger, threadNum + 2);
}

// just matching whatever format it is with the correct chroma size, but in YUV
AVPixelFormat convertToYUV(AVPixelFormat format)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(format);
    if (!desc)
        return AV_PIX_FMT_NONE;

    if (desc->log2_chroma_w == 0 && desc->log2_chroma_h == 0)
        return AV_PIX_FMT_YUV444P;
    else if (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 0)
        return AV_PIX_FMT_YUV422P;
    else if (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 1)
        return AV_PIX_FMT_YUV420P;
    else if (desc->log2_chroma_w == 0 && desc->log2_chroma_h == 1)
        return AV_PIX_FMT_YUV440P;
    else if (desc->log2_chroma_w == 2 && desc->log2_chroma_h == 0)
        return AV_PIX_FMT_YUV411P;

    return AV_PIX_FMT_NONE;
}

struct FlagOptions
{
    const char *input;
    const char *output;
    int scale;
    Interpolation type;
    int halfWindow;
    int device;
    bool hardwareEncode;
    int unsafeProcessThreads;
    int decodeThreads;
    int encodeThreads;
};

struct Flags
{
    enum class OPTION_POS
    {
        FULL,
        SHORT,
        TYPE,
        DESC,
    };

    enum OPTION_ORDER
    {
        HELP,
        INPUT,
        OUTPUT,
        SCALING,
        TYPE,
        WINDOW,
        DEVICE,
        HARDWARE_ENCODE,
        UNSAFE_PROCESS_THREADS,
        DECODE_THREADS,
        ENCODE_THREADS,
    };

    inline static vector<map<OPTION_POS, string>> options = {
        {{OPTION_POS::SHORT, "-h"},
         {OPTION_POS::FULL, "--help"},
         {OPTION_POS::TYPE, ""},
         {OPTION_POS::DESC, "Show this help message and exit"}},

        {{OPTION_POS::SHORT, "-i"},
         {OPTION_POS::FULL, "--input"},
         {OPTION_POS::TYPE, "<string>"},
         {OPTION_POS::DESC, "Input video file"}},

        {{OPTION_POS::SHORT, "-o"},
         {OPTION_POS::FULL, "--output"},
         {OPTION_POS::TYPE, "<string>"},
         {OPTION_POS::DESC, "Output video file"}},

        {{OPTION_POS::SHORT, "-s"},
         {OPTION_POS::FULL, "--scaling"},
         {OPTION_POS::TYPE, "<int>"},
         {OPTION_POS::DESC, "Scaling factor (integer, > 1)"}},

        {{OPTION_POS::SHORT, "-t"},
         {OPTION_POS::FULL, "--type"},
         {OPTION_POS::TYPE, "<string>"},
         {OPTION_POS::DESC, "Upscaling interpolation type (\"bilinear\", \"bicubic\", or \"lanczos\")"}},

        {{OPTION_POS::SHORT, "-w"},
         {OPTION_POS::FULL, "--window"},
         {OPTION_POS::TYPE, "<int>"},
         {OPTION_POS::DESC, "Window size, only for \"lanczos\" interpolation (integer, > 1, default: 10)"}},

        {{OPTION_POS::SHORT, "-d"},
         {OPTION_POS::FULL, "--device"},
         {OPTION_POS::TYPE, "<int>"},
         {OPTION_POS::DESC, "Device index to use (integer, default: 0)"}},

        {{OPTION_POS::SHORT, "-hwe"},
         {OPTION_POS::FULL, "--hardware-encode"},
         {OPTION_POS::TYPE, ""},
         {OPTION_POS::DESC, "Use hardware acceleration for encoding (default: disabled)"}},

        {{OPTION_POS::SHORT, "-upt"},
         {OPTION_POS::FULL, "--unsafe-process-threads"},
         {OPTION_POS::TYPE, "<int>"},
         {OPTION_POS::DESC, "Number of threads for processor, may be faster but may cause frame misplacements (integer, > 0, default: 1)"}},

        {{OPTION_POS::SHORT, "-dt"},
         {OPTION_POS::FULL, "--decode-threads"},
         {OPTION_POS::TYPE, "<int>"},
         {OPTION_POS::DESC, "Number of threads for decoder (integer, > 0, default: 1)"}},

        {{OPTION_POS::SHORT, "-et"},
         {OPTION_POS::FULL, "--uencode-threads"},
         {OPTION_POS::TYPE, "<int>"},
         {OPTION_POS::DESC, "Number of threads for encoder (integer, > 0, default: automatic)"}}};
};

void unknownFlag(const char *flag)
{
    cout << "Error: unknown option " << "\"" << flag << "\"" << endl;
    exit(1);
}

void unknownValue(map<Flags::OPTION_POS, string> option, const char *value, string expected)
{
    cout << "Error: unknown value for " << option[Flags::OPTION_POS::SHORT] << ", " << option[Flags::OPTION_POS::FULL] << " flag: " << "\"" << value << "\"" << ", expected " << expected << endl;
    exit(1);
}

void printHelp(const char *programName)
{
    cout << "Usage: " << programName << " [options]\n\n";
    cout << "Options:\n";

    size_t maxShortFull = 0;
    for (auto &flag : Flags::options)
    {
        size_t len = flag[Flags::OPTION_POS::SHORT].length() + 2 + flag[Flags::OPTION_POS::FULL].length() + 1 + flag[Flags::OPTION_POS::TYPE].length();
        if (len > maxShortFull)
            maxShortFull = len;
    }

    const int extraGap = 3;

    for (auto &flag : Flags::options)
    {
        string shortFull = flag[Flags::OPTION_POS::SHORT] + ", " + flag[Flags::OPTION_POS::FULL] + " " + flag[Flags::OPTION_POS::TYPE];
        cout << "  "
             << left << setw(maxShortFull + extraGap) << shortFull
             << flag[Flags::OPTION_POS::DESC] << endl;
    }

    exit(0);
}

bool checkFlagCondition(const char *arg, map<Flags::OPTION_POS, string> option)
{
    return strcmp(arg, option[Flags::OPTION_POS::SHORT].c_str()) == 0 || strcmp(arg, option[Flags::OPTION_POS::FULL].c_str()) == 0;
}

void checkFlags(int argc, char *argv[], FlagOptions &flags)
{
    auto options = Flags::options;

    // default values
    flags.device = 0;
    flags.unsafeProcessThreads = 1;
    flags.halfWindow = 5;
    flags.hardwareEncode = false;
    flags.decodeThreads = 1;

    // unused default, default to automatic
    flags.encodeThreads = -1;

    // required flags
    bool input = false;
    bool output = false;
    bool scaling = false;
    bool type = false;

    for (int i = 1; i < argc; i++)
    {
        if (checkFlagCondition(argv[i], options[Flags::HELP]))
            printHelp(argv[0]);
        else if (checkFlagCondition(argv[i], options[Flags::INPUT]))
        {
            i++;
            flags.input = argv[i];
            input = true;
        }
        else if (checkFlagCondition(argv[i], options[Flags::OUTPUT]))
        {
            i++;
            flags.output = argv[i];
            output = true;
        }
        else if (checkFlagCondition(argv[i], options[Flags::SCALING]))
        {
            i++;
            try
            {
                flags.scale = stoi(argv[i]);

                if (!(flags.scale > 1))
                    unknownValue(options[Flags::SCALING], argv[i], "integer, must be more than 1");
            }
            catch (invalid_argument)
            {
                unknownValue(options[Flags::SCALING], argv[i], "integer, > 1");
            }
            scaling = true;
        }
        else if (checkFlagCondition(argv[i], options[Flags::TYPE]))
        {
            i++;
            if (strcmp(argv[i], "bilinear") == 0)
                flags.type = BILINEAR;
            else if (strcmp(argv[i], "bicubic") == 0)
                flags.type = BICUBIC;
            else if (strcmp(argv[i], "lanczos") == 0)
                flags.type = LANCZOS;
            else
                unknownValue(options[Flags::TYPE], argv[i], "\"bilinear\", \"bicubic\", or \"lanczos\"");

            type = true;
        }
        else if (checkFlagCondition(argv[i], options[Flags::WINDOW]))
        {
            i++;
            try
            {
                int window = stof(argv[i]);

                flags.halfWindow = floor(stof(argv[i]) / 2.0);

                if (!(flags.halfWindow > 0))
                    unknownValue(options[Flags::WINDOW], argv[i], "integer, must be more than 1");
            }
            catch (invalid_argument)
            {
                unknownValue(options[Flags::WINDOW], argv[i], "integer");
            }
        }
        else if (checkFlagCondition(argv[i], options[Flags::DEVICE]))
        {
            i++;
            try
            {
                flags.device = stoi(argv[i]);
            }
            catch (invalid_argument)
            {
                unknownValue(options[Flags::DEVICE], argv[i], "integer");
            }
        }
        else if (checkFlagCondition(argv[i], options[Flags::HARDWARE_ENCODE]))
        {
            flags.hardwareEncode = true;
        }
        else if (checkFlagCondition(argv[i], options[Flags::UNSAFE_PROCESS_THREADS]))
        {
            i++;
            try
            {
                flags.unsafeProcessThreads = stoi(argv[i]);

                if (!(flags.unsafeProcessThreads > 0))
                    unknownValue(options[Flags::UNSAFE_PROCESS_THREADS], argv[i], "integer, must be more than 0");
            }
            catch (invalid_argument)
            {
                unknownValue(options[Flags::UNSAFE_PROCESS_THREADS], argv[i], "integer");
            }
        }
        else if (checkFlagCondition(argv[i], options[Flags::DECODE_THREADS]))
        {
            i++;
            try
            {
                flags.decodeThreads = stoi(argv[i]);

                if (!(flags.decodeThreads > 0))
                    unknownValue(options[Flags::DECODE_THREADS], argv[i], "integer, must be more than 0");
            }
            catch (invalid_argument)
            {
                unknownValue(options[Flags::DECODE_THREADS], argv[i], "integer");
            }
        }
        else if (checkFlagCondition(argv[i], options[Flags::ENCODE_THREADS]))
        {
            i++;
            try
            {
                flags.encodeThreads = stoi(argv[i]);

                if (!(flags.encodeThreads > 0))
                    unknownValue(options[Flags::ENCODE_THREADS], argv[i], "integer, must be more than 0");
            }
            catch (invalid_argument)
            {
                unknownValue(options[Flags::ENCODE_THREADS], argv[i], "integer");
            }
        }
        else
            unknownFlag(argv[i]);
    }

    if (!(input && output && scaling && type))
    {
        cout << "Error: missing options" << endl;
        exit(1);
    }
}

int main(int argc, char *argv[])
{
    // av_log_set_level(AV_LOG_DEBUG);
    av_log_set_level(AV_LOG_FATAL);
#ifdef _WIN32
    freopen("NUL", "w", stderr);
#else
    freopen("/dev/null", "w", stderr);
#endif

    // we need at least 3 threads total: producer, processor, consumer
    int usableThreads = max((int)(MAX_THREADS), 3);

    FlagOptions settings;
    checkFlags(argc, argv, settings);

    float thresholdLevel = 0.75;

    // settings
    int scale = settings.scale;
    const char *input = settings.input;
    const char *output = settings.output;
    Interpolation type = settings.type;
    int halfWindow = settings.halfWindow;
    bool hardwareEncode = settings.hardwareEncode;

    int maxProcessor = usableThreads - 2;
    int processorThreads = min(settings.unsafeProcessThreads, maxProcessor);

    float remainingThreads = usableThreads - processorThreads;
    // decoding is not the bottleneck
    int decoderThreads = max(1, settings.decodeThreads);
    // encoding is the main bottleneck, we will give it all remaining threads
    int encoderThreads = remainingThreads - decoderThreads;
    if (settings.encodeThreads > 0)
        encoderThreads = settings.encodeThreads;

    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    if (!(settings.device < deviceCount))
    {
        cout << "Device " << settings.device << " not found, using default device" << endl;
        settings.device = 0;
    }

    HIP_CHECK(hipSetDevice(settings.device));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, settings.device))

    // 4 is a good multiplier, idk tho, if u want accuracy, just make it slower
    int batchSize = processorThreads * 4;
    VideoPipeline video(input, batchSize, decoderThreads);

    pair<int, int> dimension = video.getDimension();
    int scaledWidth = dimension.first * scale;
    int scaledHeight = dimension.second * scale;
    AVPixelFormat format = convertToYUV(video.getPixelFormat());

    video.initialiseEncoder(scaledWidth, scaledHeight, format, hardwareEncode, encoderThreads);

    int threadCount = min(usableThreads, processorThreads + 2); // +2 because one for consumer one for producer
    ThreadTask threads(threadCount - 1);                        // -1 because producer is running in main thread

    cout << "========== DEVICES ==========" << endl;
    cout << "Decoder: " << video.getDecoderCodec() << endl;
    cout << "Processor device " << settings.device << ": " << prop.name << endl;
    cout << "Encoder: " << video.getEncoderCodec() << endl;
    cout << endl;

    cout << "========== SETTINGS ==========" << endl;
    cout << "Max concurrency threads: " << MAX_THREADS << " threads" << endl;
    cout << "Using " << (decoderThreads + processorThreads + encoderThreads)
         << " threads: " << decoderThreads << " decoder threads, " << processorThreads << " processor threads, " << encoderThreads << " encoder threads" << endl;

    Logger logger;
    cout << endl;
    cout << "========== PROGRESS ==========" << endl;

    auto start = high_resolution_clock::now();

    int threadNum = 0; // main thread
    auto producerTask = [&video, scale, &logger](int threadNum)
    {
        producer(video, scale, logger, threadNum);
    };

    ProcessorOptions options = {scale, format, 0, nullptr, type};
    if (type == LANCZOS)
    {
        LanczosKernel *kernel = new LanczosKernel(options.scale, halfWindow);
        options.kernel = kernel;
        options.halfWindow = halfWindow;
    }
    auto processorTask = [&video, &options, &logger](int threadNum)
    {
        processor(video, options, logger, threadNum);
    };

    auto consumerTask = [&video, output, thresholdLevel, &logger](int threadNum)
    {
        consumer(video, output, thresholdLevel, logger, threadNum);
    };

    // all but the last one is the processor
    for (int i = 0; i < threads.size - 1; i++)
        threads.addTask(new function<void(int)>(processorTask), i);

    // the last thread is the consumer
    threads.addTask(new function<void(int)>(consumerTask), threads.size - 1);

    threads.start();

    // main thread is the producer, -1 is for main thread
    producerTask(-1);

    // once processors are all done, notify end to encoder
    for (int i = 0; i < threads.size - 1; i++)
        threads.join(i);

    // this will allow the consumer/encoder thread to end (or else it will keep running forever)
    video.notifyEndEncode();

    // wait for the consumer thread to finish
    threads.join(threads.size - 1);

    auto end = high_resolution_clock::now();

    duration<double> duration = end - start;
    double seconds = duration.count();
    int totalFrames = video.getFrameCount();
    float fps = (float)totalFrames / duration.count();

    cout << endl;
    cout << "========== RESULTS ==========" << endl;
    cout << "Duration: " << fixed << setprecision(2) << seconds << " seconds, " << fixed << setprecision(2) << fps << " fps" << endl;
    cout << totalFrames << " total frames, " << video.getMisplaceFrameCount() << " misplaced frames" << endl;
    return 0;
}