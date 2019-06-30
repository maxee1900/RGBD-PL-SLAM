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


#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"  //MapDrawer类中没有用到Tracking类
#include "System.h"  // 在构造函数中会用到该类

//add
#include "auxiliar.h"
#include "ExtractLineSegment.h"
#include "MapLine.h"
#include "LSDmatcher.h"

#include <mutex>

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;
//tracking线程是不需要与地图点和地图线进行交互的

class Tracking
{  
//**********************************************************************************
public:
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
            KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor);  //参数里面凡是指针变量都带了p真是好习惯

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp);
    /// 终点关注
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp);
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);

    //设置其他三个线程，实际上就是把该类中的线程指针定义了
    void SetLocalMapper(LocalMapping* pLocalMapper);  //@该类中有其他线程的指针的话可能是为了通信吧
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);

    // Load new settings
    // The focal length should be similar or scale prediction will fail when projecting points 意思fx fy相近
    // TODO: Modify MapPoint::PredictScale to take into account focal lenght
    void ChangeCalibration(const string &strSettingPath);  //这个函数没用到

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);  //修改mbOnlyTracking = flag;


//************************************************************************************
public:  //Tracking类的公有数据成员

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor:MONOCULAR, STEREO, RGBD
    int mSensor;  //this is int

    // Current Frame
    Frame mCurrentFrame;  //包含了头文件所以这里可以定义对象
    cv::Mat mImGray;  //当前帧的灰度图

    // Initialization Variables (Monocular)。 RGBD相机的话这一块数据成员可以跳过
    // 初始化时前两帧相关变量
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;    // 跟踪初始化时前两帧之间的匹配
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;
    // --line--
    vector<pair<int, int>> mvLineMatches;
    vector<cv::Point3f> mvLineS3D;    //初始化时线段起始点的3D位置
    vector<cv::Point3f> mvLineE3D;    //初始化时线段终止点的3D位置
    vector<bool> mvbLineTriangulated; //匹配的线特征是否能够三角化


    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses;
    //每一帧相对于参考帧的位姿(即参考帧到当前帧)。这几处在System::SaveTrajectoryTUM()函数中会用到
    list<KeyFrame*> mlpReferences; //关键帧链表（顺序容器）.和上一个list一一对应的
    list<double> mlFrameTimes;     //时间戳
    list<bool> mlbLost;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset();


//************************************************************************************
protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();   //这里不是Tracking，不是构造函数

    /// Map initialization for stereo and RGB-D,RGBD也会调用这一函数
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization();
    void CreateInitialMapMonocular();
    // --line--
    void CreateInitialMapMonoWithLine();
    //todo 思考怎么可以使得初始化的时候只用点不用线 线只是之后才用到

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();

    bool Relocalization();

    //更新局部地图 局部点 局部关键帧
    void UpdateLocalMap();
    void UpdateLocalPoints();
    // --line--
    void UpdateLocalLines();
    void UpdateLocalKeyFrames();


    bool TrackLocalMap();
    void SearchLocalPoints();
    // --line--
    bool TrackLocalMapWithLines();   //代替上个函数
    void SearchLocalLines();    //Tracking类中有个特点，很多函数都是不带参数的


    bool NeedNewKeyFrame();   //todo 加入了线之后不知道关键帧的判断是否有变化？
    void CreateNewKeyFrame();

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;  //定位模式，该标识为true

    //Other Thread Pointers
    // 线程的指针，当前线程是主线程，通过这两个数据成员可以完成线程之间的交互是吗
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;


    //ORB
    // orb特征提取器，不管单目还是双目，mpORBextractorLeft都要用到
    // 如果是双目，则要用到mpORBextractorRight.  RGBD是不是也是这样的？
    // 如果是单目，在初始化的时候使用mpIniORBextractor而不是mpORBextractorLeft，
    // mpIniORBextractor属性中提取的特征点个数是mpORBextractorLeft的两倍
    ORBextractor *mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //BoW。相当于有两个独立的信息库，要查询或者更改
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    // 单目初始器
    Initializer* mpInitializer;  //todo 不知道lan版本中的单目初始化器有没有做更改

    //Local Map, 局部关键帧和局部地图点构成了局部地图
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    // --line--
    std::vector<MapLine*> mvpLocalMapLines;
    
    // System
    System* mpSystem; //可见System和Tracking类是互相用到的,两个头文件中都是既有头文件又有前向声明
    
    //Drawers.
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    //Map
    Map* mpMap;

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;  // IR projector baseline times fx (aprox.)

    // --line--
    // 两个用于纠正畸变的映射矩阵
    Mat mUndistX, mUndistY;  //todo 这两个矩阵怎么用

    //New KeyFrame rules (according to fps)
    int mMinFrames;  //frames的id间隔最小值，来评价是否为关键帧
    int mMaxFrames;



    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    /// and inserted from just one frame. Far points requiere a match in two keyframes. 近点直接深度图中，远点由三角化得到？ todo 注意这句话，在localMapping中检查是否符合这样的思想
    float mThDepth;  //近点还是远点的分界

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;  //深度图的尺度因子

    //Current matches in frame 当前帧上的匹配数
    int mnMatchesInliers;
    // --line--
    int mnLineMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;  //上一关键帧，可见关键帧的表示都是用到指针，而帧直接表示为了Frame的对象
    Frame mLastFrame;          //上一帧。注意这里不是指针变量了！！
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    //Motion Model 运动模型
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;  //这个变量主要是作为一种标识，在Tracking的实现文件中会多次判断以明确颜色通道的输入顺序

    list<MapPoint*> mlpTemporalPoints;  //地图点链式存储，方便插入删除
    //lan版本中没有类似这个数据成员的线的操作，TODO 是不是应该加上
    //list<MapLine*> mlpTemporalLines;

    //Actually,以上这些数据成员都是只在Tracking类的实现文件中用到，所以可以写作private,文中写成了protected但是也没有哪个类继承Tracking啊

    ///可见，Tracking里的数据成员是非常全的，基本上和其他的库都有关联，和优化库没有关系，优化库和pnp库在Tracking.cpp中包含用来写头文件中函数的实现
};

} //namespace ORB_SLAM

#endif // TRACKING_H
