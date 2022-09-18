#include<opencv2/opencv.hpp>
#include "XAImodel_Yolov5.h"

XAImodel_Yolov5::XAImodel_Yolov5()
{
    cv::RNG rng(255);
    for (int i = 0; i < 80; i++)
    {
        cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cocolabelscolor.push_back(color);
    }

}
XAImodel_Yolov5::~XAImodel_Yolov5()
{

}

//等比例缩放函数
int XAImodel_Yolov5::resize_uniform(cv::Mat& src, cv::Mat& dst, cv::Size dst_size, object_rect& effect_area)
{
    int w = src.cols;
    int h = src.rows;
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    //std::cout << "src: (" << h << ", " << w << ")" << std::endl;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    double ratio_src = w * 1.0 / h;
    double ratio_dst = dst_w * 1.0 / dst_h;

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst) 
    {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0 / w) * h);
    }
    else if (ratio_src < ratio_dst) {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    }
    else
    {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    //std::cout << "tmp: (" << tmp_h << ", " << tmp_w << ")" << std::endl;
    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));

    if (tmp_w != dst_w) 
    { //高对齐，宽没对齐
        int index_w = floor((dst_w - tmp_w) / 2.0);
        //std::cout << "index_w: " << index_w << std::endl;
        for (int i = 0; i < dst_h; i++) 
        {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else if (tmp_h != dst_h) 
    { //宽对齐， 高没有对齐
        int index_h = floor((dst_h - tmp_h) / 2.0);
        //std::cout << "index_h: " << index_h << std::endl;
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else 
    {
        printf("error\n");
    }
    return 0;
}
//等比例缩放后的恢复函数
int XAImodel_Yolov5::crop_effect_area(cv::Mat& uniform_scaled, cv::Mat& dst, cv::Size ori_size, object_rect effect_area)
{
    cv::Rect ori_rect(effect_area.x, effect_area.y, effect_area.width, effect_area.height);
    cv::Mat crop = uniform_scaled(ori_rect);
    cv::resize(crop, dst, ori_size);
    return 0;
}




//根据yolov5对图像的前处理工作将图像做前处理
void  XAImodel_Yolov5::preprocess(cv::Mat& src, cv::Mat& dst)
{
    mCvOriginSize = src.size();
    cv::Mat src_temp = src.clone();
    cv::Mat dst_temp;
    cv::cvtColor(src_temp, dst_temp, cv::COLOR_BGR2RGB);
    //等比例缩放并padding，有利于提高mAP
    resize_uniform(dst_temp, dst, mCvInputSize, effect_area);
    //cv::resize(dst, dst, mCvInputSize);
    //cv::imshow("resized image", dst);
    //cv::waitKey(0);
    dst.convertTo(dst, CV_32FC3);
    dst = dst / 255.0f;
    
    //std::cout << "Input Channels:" << dst.channels() << std::endl;
    //std::cout << "Input Width:" << dst.cols << std::endl;
    //std::cout << "Input Height:" << dst.rows << std::endl;
    
}







std::pair<int, float> XAImodel_Yolov5::argmax(std::vector<float>& vSingleProbs)
{
    std::pair<int, float> result;
    auto iter = std::max_element(vSingleProbs.begin(), vSingleProbs.end());
    result.first = static_cast<int>(iter - vSingleProbs.begin());
    result.second = *iter;

    return result;
}

void XAImodel_Yolov5::generate_proposals(std::vector<Object> & objects, float confThresh)
{
    //过滤低置信度目标
    /*
    1遍历所有结果，先取排在下标为4的置信度（顺序是x y w h conf），判断是否高于置信度的阈值；
    2如果高于阈值，按照顺序取xywh（注意是xy是中心点坐标，但是cv::Rect的xy是左上角点坐标）；
    3将xywh整理进cv::Rect数据结构内；
    4用argmax方法从后80个数据内获得类别的label；
    结果放到objects中*/
    int nc = 80;
    for (int i = 0; i < 25200; i++)
    {
        float conf = Prob[i * (nc + 5) + 4];
        if (conf > confThresh)
        {
            Object obj;
            float cx = Prob[i * (nc + 5)];
            float cy = Prob[i * (nc + 5) + 1];
            float w = Prob[i * (nc + 5) + 2];
            float h = Prob[i * (nc + 5) + 3];
            obj.rect.x = static_cast<int>(cx - w * 0.5f);
            obj.rect.y = static_cast<int>(cy - h * 0.5f);
            obj.rect.width = static_cast<int>(w);
            obj.rect.height = static_cast<int>(h);

            std::vector<float> vSingleProbs(nc);
            for (int j = 0; j < vSingleProbs.size(); j++)
            {
                vSingleProbs[j] = Prob[i * 85 + 5 + j];
            }

            auto max = argmax(vSingleProbs);
            obj.label = max.first;
            obj.conf = conf;

            objects.push_back(obj);
        }
    }
}


void XAImodel_Yolov5::qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    //对输出结果做快速排序
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].conf;

    while (i <= j)
    {
        while (objects[i].conf > p)
            i++;

        while (objects[j].conf < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

void XAImodel_Yolov5::qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}


float min(float a, float b)
{
    return a < b ? a : b;
}

float max(float a, float b)
{
    return a > b ? a : b;
}


float XAImodel_Yolov5::intersection_area(cv::Rect box1, cv::Rect box2)
{
    //计算交集部分的面积
    if (box1.x > box2.x + box2.width) { return 0.0; }
    if (box1.y > box2.y + box2.height) { return 0.0; }
    if (box1.x + box1.width < box2.x) { return 0.0; }
    if (box1.y + box1.height < box2.y) { return 0.0; }
    float colInt = min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x);
    float rowInt = min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y);
    return  colInt * rowInt;
}

void XAImodel_Yolov5::nms_sorted_bboxes(const std::vector<Object>& vObjects, std::vector<int>& picked, float nms_threshold)
{
    /*非极大值抑制的基本原理：首先，将所有的矩形框按照不同的类别标签分组，组内按分数

   高低进行排序，取得分最高的矩形框先放入结果序列，接着，遍历剩余矩形框，计算与当前得

   分最高的矩形框的交并比，若大于预设的阈值则剔除，然后对剩余的检测框重复上述操作，直

   到处理完图像内所有的候选框，即可得到最后的框序列信息。*/
    picked.clear();
    //目标个数
    const int n = vObjects.size();
    //std::vector<float> intersection_area(n);
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        //获得每个矩形的面积，存入areas[]中
        areas[i] = vObjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = vObjects[i];

        bool keep = true;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = vObjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a.rect, b.rect);//计算交集的面积
            float union_area = areas[i] + areas[picked[j]] - inter_area;//计算并集的面积
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = false;
        }

        if (keep)
            picked.push_back(i);
    }
}



//模型输出结果解码器
std::vector<Object> XAImodel_Yolov5::decodeOutputs()
{
    /*首先调用了generate_proposals进行低置信度目标的过滤，
    接着调用qsort_descent_inplace方法做快速排序，
    再调用nms_sorted_bboxes方法做nms。
    然后我们获得640与原图的宽高的比例，
    将框映射回原图，
    最后纠正下过大的框和负数坐标*/
    std::vector<Object> objects;

    generate_proposals(objects, 0.4f);
    qsort_descent_inplace(objects);

    std::vector<int> picked;
    nms_sorted_bboxes(objects, picked, 0.5f);

    int count = picked.size();

    int img_w = mCvOriginSize.width;
    int img_h = mCvOriginSize.height;
    float scaleH = static_cast<float>(effect_area.height) / static_cast<float>(img_h);
    float scaleW = static_cast<float>(effect_area.width) / static_cast<float>(img_w);

    std::vector<Object> results;
    results.resize(count);
    for (int i = 0; i < count; i++)
    {
        Object obj = objects[picked[i]];

        // adjust offset to original unpadded
        float x0 = static_cast<float>(obj.rect.x - effect_area.x) / scaleW;
        float y0 = static_cast<float>(obj.rect.y - effect_area.y) / scaleH;
        float x1 = static_cast<float>(obj.rect.x - effect_area.x + obj.rect.width) / scaleW;
        float y1 = static_cast<float>(obj.rect.y - effect_area.y + obj.rect.height) / scaleH;

        // clip
        /*x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.0f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.0f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.0f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.0f);*/

        obj.rect.x = static_cast<int>(x0);
        obj.rect.y = static_cast<int>(y0);
        obj.rect.width = static_cast<int>(x1 - x0);
        obj.rect.height = static_cast<int>(y1 - y0);
        results[i] = obj;
    }
    return results;  
}

//后处理，纯虚函数，由子类实现
void  XAImodel_Yolov5::endprocess(cv::Mat& inputarray, cv::Mat& outputarray)
{
    /*后处理
    1、类别概率，有25200 * 80个；
    2、位置信息，是相对于Anchor和Grid的偏移量，有25200 * 4个；
    3、置信度，有25200 * 1个；
    但是YOLOv5团队为了方便我们做后处理，
    已经将第2点的位置信息的解码过程一并导出到了ONNX中，
    自然也随着ONNX一并转到了TensorRT里面。
    也就是说，Engine推理后输出的位置信息，
    就是真实的位置信息（相对于640 * 640而言），
    不需要我们再费劲写位置信息的解码过程。
    我们再明确下一共要干哪几件事情：

        1、整理输出结果，置信度低于置信度阈值的不保留；
        2、做NMS；
        我们先做第一件事情：*/
        //std::vector<Object>  objects;//存储候选框的队列

        //cv::cvtColor(inputarray, outputarray, cv::COLOR_RGB2BGR);
    outputarray = inputarray;
    std::vector<Object>  objects_result = decodeOutputs();

    for (int i = 0; i < objects_result.size(); i++)
    {
        //cv::fillPoly(outputarray, cv::Rect(objects_result[i].rect.x, objects_result[i].rect.y - 10, 50, 30), cv::Scalar(255, 0, 0));

        int baseline;
        cv::Size textsize = cv::getTextSize(cocolabels[objects_result[i].label], cv::FONT_HERSHEY_TRIPLEX, 0.8, 1, &baseline);
        cv::Rect rec = cv::Rect(objects_result[i].rect.x, objects_result[i].rect.y - 30, textsize.width, 30);
        std::vector<cv::Point>  contour;
        contour.push_back(rec.tl());
        contour.push_back(cv::Point(rec.tl().x + rec.width, rec.tl().y));
        contour.push_back(cv::Point(rec.tl().x + rec.width, rec.tl().y + rec.height));
        contour.push_back(cv::Point(rec.tl().x, rec.tl().y + rec.height));
        cv::fillConvexPoly(outputarray, contour, cv::Scalar(255, 0, 0));//fillPoly函数的第二个参数是二维数组！！

        cv::Scalar color = cocolabelscolor[objects_result[i].label];
        //std::cout << objects_result[i].label << ":" << objects_result[i].conf << std::endl;
        cv::rectangle(outputarray, objects_result[i].rect, color, 2, 8, 0);
        cv::Point textlocation(objects_result[i].rect.x, objects_result[i].rect.y-10);
        cv::putText(outputarray, cocolabels[objects_result[i].label], textlocation, cv::FONT_HERSHEY_TRIPLEX, 0.8, cv::Scalar(255,255,255), 1, 8);
    }
}

bool XAImodel_Yolov5::Filter(cv::Mat& src, cv::Mat& des)
{
    cv::Mat processed_src;
    cv::Mat output_ori_size;
    preprocess(src, processed_src);//图像前处理
    blobFromImage(processed_src);//将图像放入blob中
    doinference();//正向推理，结果放到prob中
    endprocess(src, des);
    //crop_effect_area(output, output_ori_size, cv::Size(src.cols, src.rows), effect_area);
    //cv::imshow("output", output);
    //cv::imshow("output_ori_size", output_ori_size);
    //cv::waitKey(0);
    return true;
}




