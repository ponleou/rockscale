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
#include "utils/video.hpp" // FIXME: experimental
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
using std::function;
using std::thread;
using namespace std::chrono;
using std::cerr;
using std::condition_variable;
using std::cout;
using std::endl;
using std::fixed;
using std::flush;
using std::logic_error;
using std::max;
using std::min;
using std::mutex;
using std::pair;
using std::priority_queue;
using std::queue;
using std::setprecision;
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
    int unsafeThreads;
    bool hardwareEncode;
};

void unknownFlag(const char *flag)
{
    cout << "Error: unknown option " << "\"" << flag << "\"" << endl;
    exit(1);
}

void unknownValue(string flag, const char *value, string expected)
{
    cout << "Error: unknown value for " << flag << " flag: " << "\"" << value << "\"" << ", expected " << expected << endl;
    exit(1);
}

void printHelp(const char *programName)
{
    cout << "Usage: " << programName << " [options]\n\n";
    cout << "Options:\n";
    cout << "  -h, --help                  Show this help message and exit\n";
    cout << "  -i, --input <file>          Input video file\n";
    cout << "  -o, --output <file>         Output video file\n";
    cout << "  -s, --scaling <int>         Scaling factor (integer)\n";
    cout << "  -t, --type <mode>           Upscaling interpolation type (\"bilinear\", \"bicubic\", or \"lanczos\")\n";
    cout << "  -w, --window <int>          Window size, only in use for \"lanczos\" interpolation type (integer, default: 10)\n";
    cout << "  -d, --device <int>          Device index to use (integer, default: 0)\n";
    cout << "  -hwe, --hardware-encode     Use hardware acceleration for encoding (default: disabled)\n";
    cout << "  -ut, --unsafe-threads <int> Number of threads, may be faster but causes frame misplacement, keep at default for safe value (integer, default: 1)\n";

    exit(0);
}

void checkFlags(int argc, char *argv[], FlagOptions &flags)
{
    // default values
    flags.device = 0;
    flags.unsafeThreads = 1;
    flags.halfWindow = 5;
    flags.hardwareEncode = false;

    // required flags
    bool input = false;
    bool output = false;
    bool scaling = false;
    bool type = false;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
            printHelp(argv[0]);
        else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0)
        {
            i++;
            flags.input = argv[i];
            input = true;
        }
        else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0)
        {
            i++;
            flags.output = argv[i];
            output = true;
        }
        else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--scaling") == 0)
        {
            i++;
            try
            {
                flags.scale = stoi(argv[i]);
            }
            catch (invalid_argument)
            {
                unknownValue("--scaling", argv[i], "integer");
            }
            scaling = true;
        }
        else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--type") == 0)
        {
            i++;
            if (strcmp(argv[i], "bilinear") == 0)
                flags.type = BILINEAR;
            else if (strcmp(argv[i], "bicubic") == 0)
                flags.type = BICUBIC;
            else if (strcmp(argv[i], "lanczos") == 0)
                flags.type = LANCZOS;
            else
                unknownValue("--type", argv[i], "\"bilinear\", \"bicubic\", or \"lanczos\"");

            type = true;
        }
        else if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--window") == 0)
        {
            i++;
            try
            {
                flags.halfWindow = floor(stof(argv[i]) / 2.0);
            }
            catch (invalid_argument)
            {
                unknownValue("--window", argv[i], "integer");
            }
        }
        else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0)
        {
            i++;
            try
            {
                flags.device = stoi(argv[i]);
            }
            catch (invalid_argument)
            {
                unknownValue("--device", argv[i], "integer");
            }
        }
        else if (strcmp(argv[i], "-hwe") == 0 || strcmp(argv[i], "--hardware-encode") == 0)
        {
            flags.hardwareEncode = true;
        }
        else if (strcmp(argv[i], "-ut") == 0 || strcmp(argv[i], "--unsafe-threads") == 0)
        {
            i++;
            try
            {
                flags.unsafeThreads = stoi(argv[i]);
            }
            catch (invalid_argument)
            {
                unknownValue("--unsafe-threads", argv[i], "integer");
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
    int processorThreads = min(settings.unsafeThreads, maxProcessor);

    float remainingThreads = usableThreads - processorThreads;
    // decoding is not the bottleneck
    int decoderThreads = 1;
    // encoding is the main bottleneck, we will give it all remaining threads
    int encoderThreads = remainingThreads - decoderThreads;

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
    cout << "Decoder: " << video.getDecoder() << endl;
    cout << "Processor device " << settings.device << ": " << prop.name << endl;
    cout << "Encoder: " << video.getEncoder() << endl;
    cout << endl;

    cout << "========== SETTINGS ==========" << endl;
    cout << "Max concurrency threads: " << MAX_THREADS << " threads" << endl;
    cout << "Using " << usableThreads << " threads, " << decoderThreads << " decoder threads, " << processorThreads << " processor threads, " << encoderThreads << " encoder threads" << endl;

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