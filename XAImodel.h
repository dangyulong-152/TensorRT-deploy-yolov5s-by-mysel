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
	static  XAImodel* Get(XAIModelType t= XYOLOV5);//��������

	bool XAImodelInit();

	std::string mEnginePath = "";
	std::string onnxModelPath = "";
	std::string InputName = "image";
	std::string OutputName = "output";

	float* Blob = NULL;
	float* Prob = NULL;
	
	//ǰ�������麯����������ʵ��,��ͬ��ģ�Ͷ�Ӧ��ǰ��������ͬ
	void virtual preprocess(cv::Mat& src, cv::Mat& dst) = 0;
	//�������麯����������ʵ�֣���ͬ��ģ�Ͷ�Ӧ�ĺ�������ͬ
	void virtual endprocess(cv::Mat& inputarray, cv::Mat& outputarray) = 0;
	bool virtual Filter(cv::Mat& src, cv::Mat& des) = 0;
	//��image�ŵ�Blob��
	void blobFromImage(cv::Mat& img);
	//��������
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
	//�������ָ��
	void * Buffers[2];

	cudaStream_t Stream;


};

