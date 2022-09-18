#pragma once
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferRuntimeCommon.h"
#include <string>
#include <iostream>
#include <fstream>


namespace cv
{
	class Mat;
}

enum XAIModelType
{
	XYOLOV5, 
	XYOLOV7,
	//yolo
};

class XAImodel
{
public:
	static  XAImodel* Get(XAIModelType t= XYOLOV5);//产生工厂

	bool XAImodelInit();

	std::string mEnginePath = "";
	std::string onnxModelPath = "";
	std::string InputName = "image";
	std::string OutputName = "output";

	float* Blob = NULL;
	float* Prob = NULL;
	
	//前处理，纯虚函数，由子类实现,不同的模型对应的前处理方法不同
	void virtual preprocess(cv::Mat& src, cv::Mat& dst) = 0;
	//后处理，纯虚函数，由子类实现，不同的模型对应的后处理方法不同
	void virtual endprocess(cv::Mat& inputarray, cv::Mat& outputarray) = 0;
	bool virtual Filter(cv::Mat& src, cv::Mat& des) = 0;
	//将image放到Blob中
	void blobFromImage(cv::Mat& img);
	//正向推理
	bool doinference();
	~XAImodel();
protected:
	XAImodel();
private:
	int InferenceInputSize = 0;

	int InferenceOutputSize = 0;

	//nvinfer1::IHostMemory* trt_model_stream = new nvinfer1::IHostMemory;
	
	bool onnxToTRTModel(); // output buffer for the TensorRT model
	bool loadRTModel();

	nvinfer1::ICudaEngine* engine = NULL;
	nvinfer1::IExecutionContext* Context = NULL;
	int32_t InputIndex = 0;
	int32_t OutputIndex = 0;
	//输入输出指针
	void * Buffers[2];

	cudaStream_t Stream;


};

