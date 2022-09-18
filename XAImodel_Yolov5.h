#pragma once
#include "XAImodel.h"
#include<opencv2/opencv.hpp>
#include<iostream>
//��ǩ
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






/*yolov5�Ĳ�����Ҫע�����м�����������ģ��ǿ��صĲ�����
    mInputSize��ģ������Ĵ�С�����������������һ�������һ��ͼ�񼴿ɣ�
yolov5s���������3 * 640 * 640��
������ǵ�����ߴ�ҲӦ����1 * 3 * 640 * 640��
������Ҫ����һ����ô���һά���飻
    mOutputSize��ģ������Ĵ�С��YOLOv5�����ͷһ����������
ÿ�����ͷ�Ĵ�С�ֱ�Ϊ20 * 20��40 * 40��80 * 80��
һ������8400��դ��(grid)��ÿ��դ����3��anchor��
ÿ��anchor�����85����Ϣ��80����� + xywh + confidence����
YOLOv5������Ϊ�˷������������������ģ�͵�����ʱ��
���������ͷConcat����һ���γ���1 * 25200 * 85��С�������
����25200����20 * 20 + 40 * 40 + 80 * 80��*/

//��resizeʱ�����Ч����
//�ȱ���resize,padding=0
struct object_rect 
{
	int x ;
	int y;
	int width ;
	int height ;
};

//yolov5�����Ŀ�꣬��ǩ�����Ŷ�
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

	//����ͼ���С
	cv::Size mCvInputSize = cv::Size(640, 640);
	//�ȱ�������
	int resize_uniform(cv::Mat& src, cv::Mat& dst, cv::Size dst_size, object_rect& effect_area);
	//�ȱ������ź�ָ�
	int crop_effect_area(cv::Mat& uniform_scaled, cv::Mat& dst, cv::Size ori_size, object_rect effect_area);
	//ǰ�������麯����������ʵ��
	void  preprocess(cv::Mat& src, cv::Mat& dst);
	//�������麯����������ʵ��
	void endprocess(cv::Mat &inputarray, cv::Mat &outputarray);
	bool Filter(cv::Mat& src, cv::Mat& des);
private:
	std::pair<int, float>  argmax(std::vector<float>& vSingleProbs);
	//���㽻�����ֵ����
	float intersection_area(cv::Rect box1, cv::Rect box2);
	//���˵����Ŷ�Ŀ��,���������std::vector<Object>& objects�У��������Ŷȵ���confThresh�ĺ�ѡ��
	void generate_proposals(std::vector<Object>& objects, float confThresh);
	//�Ǽ���ֵ���ƣ�����ŵ�picked�У��¿���ͬһ�������Ŷ����Ŀ��iou����nms_threshold��������������ʾһ��Ŀ�꣬�޳��ÿ�
	void nms_sorted_bboxes(const std::vector<Object>& vObjects, std::vector<int>& picked, float nms_threshold);
	//�Ժ�ѡ�������򣬿�������
	void qsort_descent_inplace(std::vector<Object>& objects);
	void qsort_descent_inplace(std::vector<Object>& objects, int left, int right);
	//�Խ������
	std::vector<Object> decodeOutputs();

	cv::Size mCvOriginSize;
	object_rect effect_area;
};

  