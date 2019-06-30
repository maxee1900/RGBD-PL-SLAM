/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include "ExtractLineSegment.h"
#include "MapLine.h"
#include "auxiliar.h"

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;

class Frame
{

//***********************************************
public:
    Frame();

    // Copy constructor. 拷贝构造函数
    Frame(const Frame &frame);

    // Constructor for stereo cameras.
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);   //thDepth是远近点的分界线

    /// Constructor for RGB-D cameras.
    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);
    //todo 按照这个逻辑，构造函数中不传入直线提取器吗？

    // Constructor for Monocular cameras.
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Extract ORB on the image. 0 for left image and 1 for right image.
    // 提取的关键点存放在mvKeys和mDescriptors中
    // ORB是直接调orbExtractor提取的
    void ExtractORB(int flag, const cv::Mat &im);

    //--line---
    void ExtractLSD(const cv::Mat &im);

    //--line--
    // 计算线特征端点的3D坐标，自己添加的
    void ComputeLine3D(Frame &frame1, Frame &frame2);  //为什么传入两个帧？

    //--line--
    // 自己添加的，计算线特征描述子MAD
    void lineDescriptorMAD( vector<vector<DMatch>> matches, double &nn_mad, double &nn12_mad) const;


    // Compute Bag of Words representation.
    // 存放在mBowVec中
    void ComputeBoW();  //计算词典表示　　

    // Set the camera pose.
    // 用Tcw更新mTcw
    void SetPose(cv::Mat Tcw);  //用转换矩阵来更新旋转和平移等，其中会调用 UpdatePoseMatrices()函数

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline cv::Mat GetCameraCenter()
	{
        return mOw.clone();
    }

    // Returns inverse of rotation.  Rcw
    inline cv::Mat GetRotationInverse()
	{
        return mRwc.clone();   //旋转的逆矩阵 就是 Rwc^-1 = Rcw
    }

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    // 判断路标点是否在帧的视野中
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    //---line---
    bool isInFrustum(MapLine* pML, float viewingCosLimit);


    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;

    ///---line--- 这个函数要好好注意了！
    vector<size_t> GetLinesInArea(const float &x1, const float &y1, const float &x2, const float &y2,
                                  const float &r, const int minLevel=-1, const int maxLevel=-1) const;
    //参数变化为传入的坐标变为了四个


    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    void ComputeStereoMatches();  ///RGBD相机会用到吗

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);  ///重要

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    cv::Mat UnprojectStereo(const int &i);  //i可能是keypoint的索引吧


//********************************************************
public:   //以下的数据成员，其他类的函数可能会访问，所以都是公有的

    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    //---line---
    // line feature extractor
    LineSegment* mpLineSegment;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view. get!
    // Far points are inserted as in the monocular case from 2 views.
    //注意上面这句话：近点通过单张图像得到深度，远点通过两张图像得到深度。在三角化生成地图点的时候可能就是这样的策略
    float mThDepth;

    // Number of KeyPoints.
    int N; //KeyPoints数量
    int NL; //KeyLines数量

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    // mvKeys:原始左图像提取出的特征点（未校正）
    // mvKeysRight:原始右图像提取出的特征点（未校正）
    // mvKeysUn:校正mvKeys后的特征点，对于双目摄像头，一般得到的图像都是校正好的，再校正一次有点多余
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
    std::vector<cv::KeyPoint> mvKeysUn;  //去畸变之后的特征点

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    // 对于双目，mvuRight存储了左目像素点在右目中的对应点的横坐标
    // mvDepth对应的深度
    // 单目摄像头，这两个容器中存的都是-1
    std::vector<float> mvuRight;  //对每一个特征点 把右目坐标和深度给对应起来
    std::vector<float> mvDepth;

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.
    // 左目摄像头和右目摄像头特征点对应的描述子
    cv::Mat mDescriptors, mDescriptorsRight;
    //应该是一个二维矩阵，每一行表示一个关键点的描述子向量

    /// MapPoints associated to keypoints, NULL pointer if no association.
    // 每个特征点对应的MapPoint
    std::vector<MapPoint*> mvpMapPoints;

    // Flag to identify outlier associations. 是否为外点的标识
    // 观测不到Map中的3D点
    std::vector<bool> mvbOutlier;


    //---line--- 对于线同样具有上面的很多属性
    std::vector<KeyLine> mvKeylinesUn;
    //todo 提取线特征时图像去畸变的操作在哪里？
    vector<bool> mvbLineOutlier;
    Mat mLdesc; //每一行表示一个特征线的描述子，等价于mDescriptors
    vector<Vector3d> mvKeyLineFunctions; //每个特征线的直线系数
    vector<MapLine*> mvpMapLines;  //与地图线的关联
    //---止---


    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    // 坐标乘以mfGridElementWidthInv和mfGridElementHeightInv就可以确定在哪个格子
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    // 每个格子分配的特征点数，将图像分成格子，保证提取的特征点比较均匀
    // FRAME_GRID_ROWS 48
    // FRAME_GRID_COLS 64
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];  //二维数组, 二维数组中存储的变量为vector<size_t>  也就是说总共是一个三维变量

    // Camera pose.
    cv::Mat mTcw; ///< 相机姿态 世界坐标系到相机坐标坐标系的变换矩阵

    // Current and Next Frame id.
    static long unsigned int nNextId; ///< Next Frame id. 为啥定义为静态成员呢
    long unsigned int mnId; ///< Current Frame id.

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;


    // Scale pyramid info.
    int mnScaleLevels;//图像提金字塔的层数
    float mfScaleFactor;//图像提金字塔的尺度因子
    float mfLogScaleFactor;//
    vector<float> mvScaleFactors;  //相对于原图像每一层的缩放因子
    vector<float> mvInvScaleFactors; //反缩放因子
    vector<float> mvLevelSigma2; //存卡方分布的相关信息
    vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once). 去畸变之后的图像便捷
    // 用于确定画格子时的边界
    static float mnMinX;  //这里应该是指整个图像上的边界
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;
    //静态数据成员 属于类 所有对象可以共享这个变量

//***********************************************************
private:

    // Undistort keypoints given OpenCV distortion parameters.
    /// Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    // 这些都是在构造函数中会调用的工具函数
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    // Rotation, translation and camera center
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mOw;  //mtwc,Translation from camera to world， 相当于相机光心在世界下的坐标
};

}// namespace ORB_SLAM

#endif // FRAME_H
