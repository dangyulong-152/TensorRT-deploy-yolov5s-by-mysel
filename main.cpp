#include<iostream>
#include<string>
#include "XAImodel.h"
#include<opencv2/opencv.hpp>
int main(int argv, char* argc[])
{
	XAImodel* model = XAImodel::Get(XYOLOV5);
	model->onnxModelPath = "yolov5s_simplified.onnx";
	//model->mEnginePath = "yolov5s_simplified.onnx.TensorRT_model";
	model->InputName = "image";
	model->OutputName = "output";
	model->XAImodelInit();
	cv::Mat image = cv::imread("./inference/car.jpg");
	cv::Mat output;
	//cv::imshow("image", image);
	//cv::waitKey(0);
	double timebase = cv::getTickFrequency();
	double begintime = cv::getTickCount();
	model->Filter(image, output);
	double endtime = cv::getTickCount();
	std::cout << "[INFO]:模型推理用时:" << (endtime - begintime) / timebase * 1000 << "毫秒" << std::endl;
	std::cout << "[INFO]:输出帧率FPS:" << 1000/((endtime - begintime) / timebase * 1000) << std::endl;
	cv::imshow("output", output);
	cv::waitKey(0);
	std::cout << "[INFO]:摄像头测试:" << std::endl;
	cv::VideoCapture cap;
	if (cap.open(0))
	{
		cv::Mat frame;
		cv::Mat frame_fliped;
		cv::Mat cap_output;
		for (;;)
		{
			cap.read(frame);
			if (!frame.empty())
			{
				cv::flip(frame, frame_fliped, 1);
				model->Filter(frame_fliped, cap_output);
				cv::imshow("cap_output", cap_output);
				cv::waitKey(1);
			}
		}
	}
	else
	{
		std::cout << "[ERROR]:摄像头打开失败！"<<std::endl;
	}
	return 0;
}