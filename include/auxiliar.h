//
// Created by max on 19-6-24.
//
// 这个函数名为辅助函数，在src中是没有相应的实现源文件的 可见头文件应该包含定义和实现. 是为了函数ExtracLineSegment中的一些调用

#pragma once

#include <cv.h>
#include <opencv2/features2d/features2d.hpp>
//#include <line_descriptor_custom.hpp>   //如果使用第三方库中的直线描述子 这两行是包含该库中的头文件
//#include <line_descriptor/descriptor_custorm.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
using namespace cv;
using namespace cv::line_descriptor;
//using namespace line_descriptor;

#include <iostream>
#include <vector>
using namespace std;

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
using namespace Eigen;

typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 6, 6> Matrix6d;

// 第一种方式：按照每个特征对应的最小匹配距离进行排序
struct compare_descriptor_by_NN_dist
{
    inline bool operator()(const vector<DMatch>& a, const vector<DMatch>& b) {
        return ( a[0].distance < b[0].distance );  //从小到大排列
    }
};

// 第二种方式： 按照每个特征对应的最小和第二小之间的差值进行排序，差值大的排在前面
struct compare_descriptor_by_NN12_dist
{
    inline bool operator()(const vector<DMatch>& a, const vector<DMatch>& b) {
        return ((a[1].distance - a[0].distance) < (b[1].distance - b[0].distance) );
    }
};

// 按描述子之间的距离从小到大排序？ 这里存疑，可能是按描述子左匹配的序号进行排序
struct sort_descriptor_by_queryIdx
{
    inline bool operator()(const vector<DMatch>& a, const vector<DMatch>& b){
        return ( a[0].queryIdx < b[0].queryIdx );
    }
};

//按照直线的响应排序直线特征
struct sort_lines_by_response
{
    inline bool operator() (const KeyLine& a, const KeyLine& b) {
        return ( a.response > b.response );
    }
};

inline Mat SkewSymmetricMatrix(const cv::Mat &v)  //传入的v为向量
{
    return (
            cv::Mat_<float>(3, 3)
            << 0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2), 0, -v.at<float>(0),
            -v.at<float>(1), v.at<float>(0), 0
            );
}

/**
 * @brief 求一个vector数组的中位数绝对偏差MAD
 * 中位数绝对偏差MAD——median absolute deviation, 是单变量数据集中样本差异性的稳健度量。
 * MAD是一个健壮的统计量，对于数据集中异常值的处理比标准差更具有弹性，可以大大减少异常值对于数据集的影响
 * 对于单变量数据集 X={X1,X2,X3,...,Xn}, MAD的计算公式为：MAD(X)=median(|Xi-median(X)|)
 * @param residues
 * @return
 */
inline double vector_mad(vector<double> residues)
{
    if(residues.size() != 0)
    {
        int n = residues.size();
        sort(residues.begin(), residues.end());
        double median = residues[ n/2 ];
        for(int i=0; i < n; i++)
            residues[i] = fabs(residues[i] - median);
        sort(residues.begin(), residues.end());
        return 1.4826 * residues[n/2];
    }
    else
        return 0.0;
}






















