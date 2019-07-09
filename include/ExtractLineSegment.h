//
// Created by max on 19-6-24.
//

// 这个类就是用来提取直线的 LSD方法

#ifndef ORB_SLAM2_EXTRACTLINESEGMENT_H
#define ORB_SLAM2_EXTRACTLINESEGMENT_H

#include <iostream>
#include <chrono>

#include "auxiliar.h"  //包含这个辅助头文件，因为cmake文件中包含了这个路径这里直接就可以调用
#include <eigen3/Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;
//using namespace line_descriptor;  //Thirdparty中

namespace ORB_SLAM2
{
    class LineSegment
    {
    public:
        LineSegment();

        ~LineSegment(){};

        // 注意这里有层数信息了！
        void ExtractLineSegment(const Mat& img, vector<KeyLine> &vkeyLines, Mat &ldesc, vector<Vector3d> &vkeylineFunctions, int scale=1.2, int numOctaves=1);

        // 线特征的匹配，这个函数不需要其他参数吗？
        void LineSegmentMathch(Mat &ldesc1, Mat &ldesc2);

        // 线特征描述子距离中位值
        void LineDescriptorMAD();  //这个函数没用到？

        // 求线特征观测线段和重投影线段的重合率 todo 思考这里写这个函数有什么用途呢
        double LineSegmentOverlap(double spl_obs, double epl_obs, double spl_proj, double edl_proj );
        // 这个函数传入的是四个数值？

    protected:
        vector<vector<DMatch> > mvlineMatches;
        double mnnMad, mnn12Mad;    // 样本差异性度量？
        // 这里我遵循ORBSLAM中函数命名规则，成员函数名首字母都是大写，变量命名使用驼峰命名，变量首字母为小写，且前面说明类型 比如mn mp vp vl mvp v l等等

    };

}

#endif //ORB_SLAM2_EXTRACTLINESEGMENT_H
