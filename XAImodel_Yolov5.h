#pragma once
#include "XAImodel.h"
#include<opencv2/opencv.hpp>
#include<iostream>
//标签
static const char* cocolabels[] = {
	"person", "bicycle", "car", "motorcycle", "airplane",
	"bus", "train", "truck", "boat", "traffic light", "fire hydrant",
	"stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
	"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
	"umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
	"snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
	"cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
	"orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
	"laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
	"oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
	"scissors", "teddy bear", "hair drier", "toothbrush"
};

static std::vector<cv::Scalar> cocolabelscolor;






/*yolov5的部署需要注意其中几个跟待部署模型强相关的参数：
    mInputSize：模型输入的大小，由于我们这个场景一次输入进一张图像即可，
yolov5s的输入就是3 * 640 * 640，
因此我们的输入尺寸也应该是1 * 3 * 640 * 640，
即我们要创建一个这么大的一维数组；
    mOutputSize：模型输出的大小，YOLOv5的输出头一共有三个，
每个输出头的大小分别为20 * 20、40 * 40、80 * 80，
一共是有8400个栅格(grid)，每个栅格有3个anchor，
每个anchor会输出85个信息（80个类别 + xywh + confidence）。
YOLOv5的作者为了方便我们做后处理，因此在模型导出的时候，
将三个输出头Concat在了一起，形成了1 * 25200 * 85大小的输出。
其中25200就是20 * 20 + 40 * 40 + 80 * 80。*/

//在resize时标记有效区域
//等比例resize,padding=0
struct object_rect 
{
	int x ;
	int y;
	int width ;
	int height ;
};

//yolov5输出的目标，标签，置信度
struct Object
{
    cv::Rect rect;
    int label;
    float conf;
};


class XAImodel_Yolov5 :public XAImodel
{
public:
	XAImodel_Yolov5();
	~XAImodel_Yolov5();

	//输入图像大小
	cv::Size mCvInputSize = cv::Size(640, 640);
	//等比例缩放
	int resize_uniform(cv::Mat& src, cv::Mat& dst, cv::Size dst_size, object_rect& effect_area);
	//等比例缩放后恢复
	int crop_effect_area(cv::Mat& uniform_scaled, cv::Mat& dst, cv::Size ori_size, object_rect effect_area);
	//前处理，纯虚函数，由子类实现
	void  preprocess(cv::Mat& src, cv::Mat& dst);
	//后处理，纯虚函数，由子类实现
	void endprocess(cv::Mat &inputarray, cv::Mat &outputarray);
	bool Filter(cv::Mat& src, cv::Mat& des);
private:
	std::pair<int, float>  argmax(std::vector<float>& vSingleProbs);
	//计算交集部分的面积
	float intersection_area(cv::Rect box1, cv::Rect box2);
	//过滤低置信度目标,结果存贮到std::vector<Object>& objects中，过滤置信度低于confThresh的候选框
	void generate_proposals(std::vector<Object>& objects, float confThresh);
	//非极大值抑制，结果放到picked中，新框与同一类中置信度最大的框的iou大于nms_threshold，则表明两个框表示一个目标，剔除该框
	void nms_sorted_bboxes(const std::vector<Object>& vObjects, std::vector<int>& picked, float nms_threshold);
	//对候选框做排序，快速排序
	void qsort_descent_inplace(std::vector<Object>& objects);
	void qsort_descent_inplace(std::vector<Object>& objects, int left, int right);
	//对结果解码
	std::vector<Object> decodeOutputs();

	cv::Size mCvOriginSize;
	object_rect effect_area;
};

  