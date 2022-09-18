#include "XAImodel.h"
#include "XAImodel_Yolov5.h"
#include<assert.h>
#include<opencv2/opencv.hpp>

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING) : reportableSeverity(severity)
    {

    }

    void log(Severity severity, char const* msg) noexcept
        // void log(Severity severity, const char* msg) noexcept
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "UNKNOWN: ";
            break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};



static Logger g_logger_;

XAImodel::XAImodel()
{

}

XAImodel::~XAImodel()
{

}

 XAImodel* XAImodel::Get(XAIModelType t)
{
    static XAImodel_Yolov5 yolov5;

    switch (t)
    {
    case XYOLOV5:
        return &yolov5;
        break;
    default:
        return nullptr;
        break;
    }
}



bool XAImodel::XAImodelInit()
{
    if (onnxModelPath.empty() && mEnginePath.empty())
    {
        std::cout << "[ERROE]:Please Input onnx model file path or TensorRT model file path." << std::endl;
        return false;
    }
    if (!mEnginePath.empty())
    {
        //std::cout << "[INFO]:Begin to Load TensorRT model File:" << mEnginePath << std::endl;
        loadRTModel();
    }
    else
    {
        onnxToTRTModel();
        loadRTModel();
    }
    
}




bool XAImodel::onnxToTRTModel() // output buffer for the TensorRT model
{
    std::cout << "[INFO]:开始转换onnxToTRTModel..." << std::endl;
    /*调用方式：
        int main() {
        nvinfer1::IHostMemory* trt_model_stream;
        onnxToTRTModel("../../config/pfe.onnx", trt_model_stream,"trt_model.trt");
    }*/
    int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;

    // create the builder
    std::cout << "[INFO]:创建构建器Builder..." << std::endl;
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(g_logger_);
    // 创建INetworkDefinition 对象
    std::cout << "[INFO]:创建INetworkDefinition对象..." << std::endl;
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    //nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_PRECISION;
    //const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_PRECISION);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    // 创建解析器
    std::cout << "[INFO]:创建onnx解析器..." << std::endl;
    auto parser = nvonnxparser::createParser(*network, g_logger_);

    // 解析onnx文件，并填充网络
    std::cout << "[INFO]:解析onnx文件，并填充网络..." << std::endl;
    if (!parser->parseFromFile(onnxModelPath.c_str(), verbosity))
    {
        std::string msg("[failed to parse onnx file");
        g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
        return false;
    }

    // Build the engine
    std::cout << "[INFO]:解析onnx文件成功，开始构建TRT模型..." << std::endl;
    builder->setMaxBatchSize(1);
    builder->setMaxThreads(builder->getMaxThreads());
    // 创建iBuilderConfig对象
    std::cout << "[INFO]:创建iBuilderConfig对象..." << std::endl;
    nvinfer1::IBuilderConfig* iBuilderConfig = builder->createBuilderConfig();
    // 设置engine可使用的最大GPU临时值
    std::cout << "[INFO]:设置engine可使用的最大GPU临时值..." << std::endl;
    iBuilderConfig->setMaxWorkspaceSize(1 << 30);//1024
    //iBuilderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);//FP16模式推理精度
    std::cout << "[INFO]:正在构建TRT模型..." << std::endl;
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *iBuilderConfig);
    std::cout << "[INFO]:构建TRT模型完毕..." << std::endl;
    // 将engine序列化，保存到文件中
    std::cout << "[INFO]:将engine序列化，保存到文件中..." << std::endl;
    nvinfer1::IHostMemory * trt_model_stream = NULL;
    trt_model_stream = engine->serialize();
    // save engine
    mEnginePath = onnxModelPath + ".TensorRT_model";
    std::ofstream p(mEnginePath, std::ios::binary);
    if (!p) 
    {
        std::cerr << "could not open plan output file" << std::endl;
        return false;
    }
    p.write(reinterpret_cast<const char*>(trt_model_stream->data()), trt_model_stream->size());
    std::cout << "[INFO]:engine序列化，保存到文件完毕..." << std::endl;
    parser->destroy();
    engine->destroy();
    network->destroy();
    builder->destroy();
    iBuilderConfig->destroy();
    std::cout << "[INFO]:onnx模型转TRT模型成功..." << std::endl;
    std::cout << "[INFO]:TRT模型名称:" << mEnginePath << std::endl;
    return true;
}


bool XAImodel::loadRTModel()
{
    //需要注意的是其中几个跟待部署模型强相关的参数：
    //以yolov5为例
    // InputSize：模型输入的大小，由于我们这个场景一次输入进一张图像即可，
    // yolov5s的输入就是3 * 640 * 640，因此我们的输入尺寸也应该是1 * 3 * 640 * 640，
    // 即我们要创建一个这么大的一维数组；
    // 
    //OutputSize：模型输出的大小，YOLOv5的输出头一共有三个，
    //每个输出头的大小分别为20 * 20、40 * 40、80 * 80，
    //一共是有8400个栅格(grid)，每个栅格有3个anchor，
    //每个anchor会输出85个信息（80个类别 + xywh + confidence）。
    //YOLOv5的作者为了方便我们做后处理，因此在模型导出的时候，
    //将三个输出头Concat在了一起，形成了1 * 25200 * 85大小的输出。
    //其中25200就是20 * 20 + 40 * 40 + 80 * 80。


    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    std::cout << "[INFO]:开始加载TensorRT 模型..." << std::endl;
    std::cout << "[INFO]:选择GPU..." << std::endl;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return false;
    }
    char* trtModelStream{ nullptr };
    size_t size{ 0 };
    std::cout << "[INFO]:查找TRT模型文件..." << std::endl;
    std::ifstream file(mEnginePath, std::ios::binary);
    std::cout << "[I] Detection model creating...\n";
    std::cout << "[INFO]:EnginePath is:" << mEnginePath << std::endl;
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    else
    {
        std::cout << "TRTModel File Not Good!" << std::endl;
        return false;
    }
    
    nvinfer1::IRuntime *mRuntime = nvinfer1::createInferRuntime(g_logger_);
    if (!mRuntime)
    {
        std::cout << "CreateInferRuntime Failed!";
        return false; 
    }

    std::cout << "[I] Detection engine creating...\n";
    engine = mRuntime->deserializeCudaEngine(trtModelStream, size);
    if (!engine)
    {
        std::cout << "DeserializeCudaEngine failed!" << std::endl;
        return false;    
    }
    std::cout << "[INFO]:模型反序列化成功!"<<std::endl;
    Context = engine->createExecutionContext();
    if (!Context)
    {
        std::cout << "CreateExecutionContext failed!" << std::endl;
        return false;
    }
    std::cout << "[INFO]:模型推理上下文创建成功!" << std::endl;

    delete[] trtModelStream;

    nvinfer1::Dims input_dim = engine->getBindingDimensions(0);
    nvinfer1::Dims output_dim = engine->getBindingDimensions(1);
    
    int input_size = 1;
    std::cout << "[INFO]:模型输入维度:";
    for (int j = 0; j < input_dim.nbDims; ++j) 
    {
        //NCHW
        input_size  *= input_dim.d[j];
        if (j == input_dim.nbDims - 1)
        {
            std::cout << input_dim.d[j];
        }
        else
        {
            std::cout << input_dim.d[j] << " X ";
        }
         
    }
    std::cout << " = " << input_size << std::endl;
    std::cout << "[INFO]:模型输出维度:";
    int output_size = 1;
    for (int j = 0; j < output_dim.nbDims; ++j)
    {
        output_size *= output_dim.d[j];
        if (j == output_dim.nbDims - 1)
        {
            std::cout << output_dim.d[j];
        }
        else
        {
            std::cout << output_dim.d[j] << " X ";
        }
    }
    std::cout<<" = "<< output_size << std::endl;
    Blob = new float[input_size];
    Prob = new float[output_size];
    InferenceInputSize = input_size;
    InferenceOutputSize = output_size;
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    if (engine->getNbBindings() == 2)
    {
        std::cout << "[I] engine->getNbBindings() == 2\n";
    }
    else
    {
        std::cout << "[I] engine->getNbBindings() != 2\n";
        return false;
    }
    std::cout << "[I] Cuda buffer creating...\n";

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    InputIndex = engine->getBindingIndex(InputName.c_str());
    std::cout << "[INFO]:输入索引:" << InputIndex << std::endl;
    assert(engine->getBindingDataType(InputIndex) == nvinfer1::DataType::kFLOAT);

    OutputIndex = engine->getBindingIndex(OutputName.c_str());
    std::cout << "[INFO]:输出索引:" << OutputIndex << std::endl;
    assert(engine->getBindingDataType(OutputIndex) == nvinfer1::DataType::kFLOAT);

    // Create GPU buffers on device
    cudaStatus = cudaMalloc(&Buffers[InputIndex], InferenceInputSize * sizeof(float));

    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "InputIndex cudaMalloc failed!");
        return false;
    }
    cudaStatus = cudaMalloc(&Buffers[OutputIndex], InferenceOutputSize * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "OutputIndex cudaMalloc failed!");
        return false;
    }

    // Create stream
    std::cout << "[I] Cuda stream creating...\n";
    cudaStatus = cudaStreamCreate(&Stream);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaStreamCreate failed!");
        return false;
    }
    std::cout << "[I] Detection engine created!\n";
    return true;
}

void XAImodel::blobFromImage(cv::Mat& img)
{
    //preprocess(img, img);
    int channels = img.channels();
    int cols = img.cols;
    int rows = img.rows;

    for (int c = 0; c < channels; c++)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                Blob[c * rows * cols + row * cols + col] = img.at<cv::Vec3f>(row, col)[c];
            }
        }
    }
    //std::cout << "[INFO]:图片送入Blob中" << std::endl;
}



bool XAImodel::doinference()
{
    /*TensorRT的推理非常精简，一共就是三步走：

     1、将Blob里面的数据拷贝至显卡；
     2、用Context->enqueueV2方法执行推理；
     3、将推理的结果拷贝至内存（也就是Prob里）*/


    cudaError_t cudaStatus;
    cudaStatus = cudaMemcpyAsync(Buffers[InputIndex], Blob, InferenceInputSize * sizeof(float), cudaMemcpyHostToDevice, Stream);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpyAsync Buffers[InputIndex] to Blob failed!");
        return false;
    }
     Context->enqueueV2(Buffers, Stream, nullptr);

     cudaStatus=cudaMemcpyAsync(Prob, Buffers[OutputIndex], InferenceOutputSize * sizeof(float), cudaMemcpyDeviceToHost, Stream);
     
     if (cudaStatus != cudaSuccess) 
     {
         fprintf(stderr, "cudaMemcpyAsync Prob to Buffers[OutputIndex] failed!");
         return false;
     }
     cudaStreamSynchronize(Stream);
     return true;
}


