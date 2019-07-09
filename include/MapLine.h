//
// Created by max on 19-6-24.
//

#ifndef ORB_SLAM2_MAPLINE_H
#define ORB_SLAM2_MAPLINE_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"

//#include "line_descriptor/descriptor_custom.hpp"
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/core/core.hpp>
#include <eigen3/Eigen/Core>
#include <map>

#include <mutex>

using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;


namespace ORB_SLAM2
{
class KeyFrame;
class Frame;
class Map;

typedef Matrix<double, 6, 1> Vector6d;

class MapLine
{
//*********************************************************************
public:

    /// 类比PL-SLAM
    MapLine(int idx_, Vector6d line3D_, Mat desc_, int kf_obs_, Vector3d obs_, Vector3d dir_, Vector4d pts_, double sigma2_=1.f);

    ~MapLine(){};

    void addMapLineObervation(Mat desc_, int kf_obs_, Vector3d obs_, Vector3d dir_, Vector4d pts_, double sigma2_=1.f);  //这个函数似乎没用到 先写着


//    int idx;                //线特征索引
//    bool inlier;            //是否属于内点
//    bool local;             //是否属于局部地图特征
//    Vector6d line3D;        // 3D endpoints of the line segment, 线段的两个端点的3D坐标
//    Vector3d med_obs_dir;   //观察该特征线段的方向

    vector<Vector4d> pts_list;  //线特征两个端点的坐标，一共是4个坐标点。2D平面上，该变量似乎没用到
    vector<Vector3d> obs_list;  //每个观察点的坐标，2D线方程的参数，用sqrt(lx2+ly2)归一化，该变量似乎没用到

    vector<int> kf_obs_list;    //观测到该线特征的KF的ID列表
    vector<double> sigma_list;  //每个观测值的sigma尺度

    /// 类比ORB-SLAM
    // 关键帧创建MapLine, 原orb中第一个参数是Mat类型
    MapLine(const Vector6d &Pos, KeyFrame* pRefKF, Map* pMap);
    // 普通帧创建MapLine
    MapLine(const Vector6d &Pos, Map* pMap, Frame* pFrame, const int &idxF);


    void SetWorldPos(const Vector6d &Pos);
    Vector6d GetWorldPos();  // 这两处ORB中Pos都是用Mat类型

    Vector3d GetNormal(); // 平均观测方向吧
    KeyFrame* GetReferenceKeyFrame();

    map<KeyFrame*,size_t> GetObservations();
    int Observations();

    void AddObservation(KeyFrame* pKF,size_t idx);
    void EraseObservation(KeyFrame* pKF);

    int GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapLine* pML);
    MapLine* GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
            return mnFound;
    }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    void UpdateAverageDir();  //这里只更新线段的平均观测方向，ORB中还要更新深度

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, const float &logScaleFactor);   //这里的尺度是LSD检测直线时候用到的尺度？

//*************************************************************************
public:

    long unsigned int mnId; ///< Global ID for MapPoint
    static long unsigned int nNextId;
    const long int mnFirstKFid; ///< 创建该MapPoint的关键帧ID
    const long int mnFirstFrame; ///< 创建该MapPoint的帧ID（即每一关键帧有一个帧ID）  ?
    int nObs;

    /// Variables used by tracking
    float mTrackProjX1;
    float mTrackProjY1;
    float mTrackProjX1R;  // 仿照ORB 这里是右目的横坐标

    float mTrackProjX2;
    float mTrackProjY2;
    float mTrackProjX2R;

    int   mnTrackScaleLevel;
    float mTrackViewCos;

    // TrackLocalMap - SearchByProjection中决定是否对该线进行投影的变量
    // mbTrackInView==false的点有几种：
    // a 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
    // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影
    // c 不在当前相机视野中的点（即未通过isInFrustum判断）
    bool mbTrackInView;

    // TrackLocalMap - UpdateLocalLines中防止将MapLines重复添加至mvpLocalMapLines的标记
    long unsigned int mnTrackReferenceForFrame;

    // TrackLocalMap - SearchLocalPoints中决定是否进行isInFrustum判断的变量
    // mnLastFrameSeen==mCurrentFrame.mnId的点有几种：
    // a 已经和当前帧经过匹配（TrackReferenceKeyFrame，TrackWithMotionModel）但在优化过程中认为是外点
    // b 已经和当前帧经过匹配且为内点，这类点也不需要再进行投影
    long unsigned int mnLastFrameSeen;


    /// Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    /// Variables used by loop closing
    long unsigned int mnLoopLineForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;

    static std::mutex mGlobalMutex;

//**********************************************************************
public:  //这里我将protected修改为了public

    // position in absolute coordinates
    Vector6d mWorldPos;  //这个变量到底指的是？？ todo，中点在世界坐标系下的坐标但是不应该为3d吗
    /// 根据optimizer.cc第2265行 pML->SetWorldPos(LinePos)这一句推断，mWorldPos应该指两端点构成的向量，要检查凡是使用到这个变量的地方，使用是否正确！！！
    Vector3d mStart3D;
    Vector3d mEnd3D;

    // 观测到该MapLine的KF和该MapLine在KF中的索引
    map<KeyFrame*, size_t> mObservations;

    Vector3d mNormalVector;  // 该线段的平均观测方向

    Mat mLDescriptor;  // 最优描述子

    // Reference KeyFrame
    KeyFrame* mpRefKF;

    vector<Mat> mvDescList;  // 线特征的描述子集
    vector<Vector3d> mvDirList;  // 每个观测线段的单位方向向量？ 这个成员似乎没有用到过

    // Tracking counts
    int mnVisible;
    int mnFound;

    // Bad flag (we do not currently erase MapPoint from memory) 这里仿照ORB
    bool mbBad;
    MapLine* mpReplaced;

    // Scale invariance distances
    float mfMinDistance;
    float mfMaxDistance;

    Map* mpMap;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;

};


}


#endif //ORB_SLAM2_MAPLINE_H
