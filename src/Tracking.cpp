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


#include "Tracking.h"
//Tracking是VO的主干，该源文件完成对Tracking.h中的函数的实现

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"

#include"Optimizer.h"
#include"PnPsolver.h"  //这个头文件在Tracking.h中没有包含，所以类的实现文件中如有用到新的功能可以再包含各种头文件也就是各种库

#include<iostream>
#include<cmath>
#include<mutex>


using namespace std;

// 程序中变量名的第一个字母如果为"m"则表示为类中的成员变量，member
// 第一个、第二个字母:
// "p"表示指针数据类型
// "n"表示int类型
// "b"表示bool类型
// "s"表示set类型
// "v"表示vector数据类型
// 'l'表示list数据类型
// "KF"表示KeyPoint数据类型
// "t"表示thread

namespace ORB_SLAM2
{

    // Tracking类的构造函数
Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    //     |fx  0   cx|
    // K = |0   fy  cy|
    //     |0   0   1 |
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    // 图像矫正系数
    // [k1 k2 p1 p2 k3]
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        //矫正矩阵最多有五个参数，这里不用管为了和cv中保持对齐后面直接调用cv的矫正函数
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    // --line-- 获取img的width和height
    int img_width = fSettings["Camera.width"];
    int img_height = fSettings["Camera.height"];

    //这部分代码我做了修改，相比lan版本，我会沿用ORB中的做法，提取特征后再进行去畸变操作
    //但是有个问题，直线未去畸变之前有可能是弯曲的，影响直线提取吗？ TODO 问一下李言
//#if 0
//    // --line-- 在这里先得到映射值
//    initUndistortRectifyMap(mK, mDistCoef, Mat_<double>::eye(3,3), mK, Size(img_width, img_height), CV_32F, mUndistX, mUndistY);
//
//    cout << "mUndistX size = " << mUndistX.size << endl;
//    cout << "mUndistY size = " << mUndistY.size << endl;
//
//#endif


    // 双目摄像头baseline * fx 50 。也就是z = fb / d 公式中的分子项，方便之后的计算
    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    // 1:RGB 0:BGR
    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    // 每一帧提取的特征点数 1000
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    // 图像建立金字塔时的变化尺度 1.2
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    // 尺度金字塔的层数 8
    int nLevels = fSettings["ORBextractor.nLevels"];
    // 提取fast特征点的默认阈值 20. 这里应该是每一个格子内FAST提取的阈值，这里只是提取的特征点不是匹配点
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 7
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    // tracking过程都会用到mpORBextractorLeft作为特征点提取器
    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);   //对ORB提取器做了封装，这里用指针申请一个动态提取器类

    // 如果是双目，tracking过程中还会用用到mpORBextractorRight作为右目特征点提取器
    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    // 在单目初始化的时候，会用mpIniORBextractor来作为特征点提取器。区别在于提取2倍特征
    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        // 判断一个3D点远/近的阈值: mbf * 40 / fx = baseline * 40（也就是40倍的baseline）
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }   //这里没问题，与程序运行的输出数据一致

    if(sensor==System::RGBD)
    {
        // 深度相机disparity转化为depth时的因子
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
            //除以5000， 用的时候直接乘以mDepthMapFactor即可
    }

}

//设置其他三个线程,在System的实现中会用到，相当于Tracking线程与其他线程进行交互
void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


//// 这个函数相当于一个纽带！很重要，是理解系统运作的关键函数
// 输入左目RGB或RGBA图像和深度图
// 1、将图像转为mImGray和imDepth并初始化mCurrentFrame
// 2、进行tracking过程
// 输出世界坐标系到该帧相机坐标系的变换矩阵
cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;   //mImGray是公有数据成员，这里赋值类的数据成员
    cv::Mat imDepth = imD; //Tracking类中是没有imDepth成员的，这里定义局部变量后面会用来构造frame

    // 步骤1：将RGB或RGBA图像转为灰度图像
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        //第一个参数为输入图像 第二个为输出图像 将颜色图转换为深度图
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    // 步骤2：将深度相机的disparity转为Depth
    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);
        //调用cv函数，把深度图转化为无尺度因子，格式为CV_32F（相当于像素深度值会除以5000）

    // 步骤3：构造Frame
    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    // 步骤4：跟踪
    Track();  //完成对Tracking类的很多数据成员的修改，这个函数也是该类的成员函数（protected）,可以随意访问该类的共有或者私有数据成员

    return mCurrentFrame.mTcw.clone();  //返回值为当前帧的世界位姿
}


//******************************************************************************

/**
 * @brief Main tracking function. It is independent of the input sensor.
 *
 * Tracking 线程. 追踪的主函数！！
 *///函数无参数也无返回值, Track()函数在GrabImageRGBD函数（封装在Tracking类中）中被调用，因此Track是对当前帧而言的
void Tracking::Track()

{
    // track包含两部分：估计运动、跟踪局部地图
    
    // mState为tracking的状态机
    // SYSTME_NOT_READY, NO_IMAGE_YET, NOT_INITIALIZED, OK, LOST
    // 如果图像复位过、或者第一次运行，则为NO_IMAGE_YET状态
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    // mLastProcessedState存储了Tracking最新的状态，用于FrameDrawer中的绘制
    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
    ///把lock看成是局部变量，局部变量没有消亡之前这个锁一直都在的

    // 步骤1：初始化
    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();  /// RGBD相机的初始化


//            ofstream file("InitialPoseEstimationTime.txt", ios::app);
//            chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

//            MonocularInitialization();

//            chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//            chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
//            cout << "Initialize time: " << time_used.count() << endl;
//            file << time_used.count() << endl;
//            file.close();

        mpFrameDrawer->Update(this);  //this指针指向当前类
        //函数原型： FrameDrawer::Update(Tracking *pTracker)

        if(mState!=OK)
            return;
    }
    else// 步骤2：跟踪
    {
        // System is initialized. Track Frame.

        // bOK为临时变量，用于表示每个函数是否执行成功
        bool bOK=false;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        // 在viewer中有个开关menuLocalizationMode，有它控制是否ActivateLocalizationMode，并最终管控mbOnlyTracking
        // mbOnlyTracking等于false表示正常VO模式（有地图更新），mbOnlyTracking等于true表示用户手动选择定位模式

        if(!mbOnlyTracking)   //mbOnlyTracking 在构造函数中默认为了false，默认slam模式
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            /// 其实这个代码块很重要！
            ///我们可以看到追踪一共有三种：追踪上一帧、追踪参考关键帧、重定位

            // 正常初始化成功
            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                // 检查并更新上一帧被替换的MapPoints
                // 更新Fuse函数和SearchAndFuse函数替换的MapPoints
                CheckReplacedInLastFrame();  //更新上一帧的地图点，因为上一帧的地图点有可能被localMap线程进行了替换

                // 步骤2.1：跟踪上一帧或者参考帧或者重定位

                // 运动模型是空的或刚完成重定位
                // mCurrentFrame.mnId<mnLastRelocFrameId+2这个判断不应该有
                // 应该只要mVelocity不为空，就优先选择TrackWithMotionModel
                // mnLastRelocFrameId上一次重定位的那一帧
                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    ///@@ 将上一帧的位姿作为当前帧的初始位姿?? todo 有个问题是什么时候运动模型会为空
                    // 通过BoW的方式在参考帧中找当前帧特征点的匹配点
                    // 优化每个特征点都对应3D点重投影误差即可得到位姿
                    bOK = TrackReferenceKeyFrame();  ///详细看这个函数实现,核心函数
                }
                else  //运动模型不为空
                {

//                    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

                    // 根据恒速模型设定当前帧的初始位姿
                    // 通过投影的方式在参考帧中找当前帧特征点的匹配点 todo 这里所说的参考帧是？
                    // 优化每个特征点所对应3D点的投影误差即可得到位姿
                    bOK = TrackWithMotionModel();   ///详细看这个函数的实现,核心函数
                    if(!bOK)
                        // TrackReferenceKeyFrame是跟踪参考帧，不能根据固定运动速度模型预测当前帧的位姿态，通过bow加速匹配（SearchByBow）
                        // 最后通过优化得到优化后的位姿
                        bOK = TrackReferenceKeyFrame();

//                    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//                    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
//                    cout << "Track time: " << time_used.count() << endl;

                }
            }
            else //LOST
            {
                // BOW搜索，PnP求解位姿,开启重定位
                bOK = Relocalization();
            }
        }


        mCurrentFrame.mpReferenceKF = mpReferenceKF;

//-----------------上面的步骤只是得到了当前帧的初始估计位姿--------------------------------------


//        ofstream file("TrackLocalMapTime.txt", ios::app);

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        /// 步骤2.2：在帧间匹配得到初始的姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
        // local map:当前帧、当前帧的MapPoints、当前关键帧与其它关键帧共视关系
        // 在步骤2.1中主要是两两跟踪（恒速模型跟踪上一帧、跟踪参考帧），这里搜索局部关键帧后搜集所有局部MapPoints，
        // 然后将局部MapPoints和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化
        if(!mbOnlyTracking)  //slam模式
        {
            if(bOK) //追踪运动模型或者追踪参考关键帧成功了，再追踪局部地图！
//                bOK = TrackLocalMap();   //原ORB中

//                chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
                bOK = TrackLocalMapWithLines();
//                chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//                chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
//                cout << "trackLocalMap time: " << time_used.count() << endl;
//                file << time_used.count() << endl;
//                file.close();
        }


        if(bOK)  //跟踪局部地图也成功了
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);  //this指针是属于类对象的

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())  //上一帧的位姿不为空
            {
                // 步骤2.3：更新恒速运动模型TrackWithMotionModel中的mVelocity
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
                /// Tcl=Tcw*Twl， mVelocity上一帧到当前帧的转换矩阵,按照恒速模型这个将作为下一帧的初始位姿。 todo 这里注意检查一下Tcl是怎么用的，两帧之间的位姿怎么能作为下一帧的位姿呢
            }
            else
                mVelocity = cv::Mat();  //上一帧位姿为空的时候运动模型即为空

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);


            // Clean VO matches
            // 步骤2.4：清除UpdateLastFrame中为当前帧临时添加的MapPoints
            // UpdateLastFrame在TrackingWithMotionModel函数中用到过！
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    // 排除UpdateLastFrame函数中为了跟踪增加的MapPoints
                    if(pMP->Observations()<1)  //说明这个地图点不是局部地图中的地图点，是的话观测数肯定大于1
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }


            /// --line--
            // 步骤2.4：清除UpdateLastFrame中为当前帧临时添加的MapLines
            for(int i=0; i<mCurrentFrame.NL; i++)
            {
                MapLine* pML = mCurrentFrame.mvpMapLines[i];
                if(pML)
                    if(pML->Observations()<1)
                    {
                        mCurrentFrame.mvbLineOutlier[i] = false;
                        mCurrentFrame.mvpMapLines[i]= static_cast<MapLine*>(NULL);
                    }
            }

            // Delete temporal MapPoints （全部删除了，删除的是MapPoint类型）
            // 步骤2.5：清除临时的MapPoints，这些MapPoints在TrackWithMotionModel的UpdateLastFrame函数里生成（仅双目和rgbd）
            // 步骤2.4中只是在当前帧中将这些MapPoints剔除，这里从MapPoints数据库中删除
            // 这里生成的仅仅是为了提高双目或rgbd摄像头的帧间跟踪效果，用完以后就扔了，没有添加到地图中
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            // 这里不仅仅是清除mlpTemporalPoints，通过delete pMP还删除了指针指向的MapPoint
            mlpTemporalPoints.clear();

            /// --line-- 考虑这里是不是也要mlpTemporalLines lan版本中没有
            // todo_ 根据上面的这部分代码，增加mlpTemporalLines的操作
            //todo_ 有个重要的问题：追踪过程中最终有没有生成地图点加入到地图中
            for(list<MapLine*>::iterator lit = mlpTemporalLines.begin(), lend =  mlpTemporalLines.end(); lit!=lend; lit++)
            {
                MapLine* pML = *lit;
                delete pML;
            }
            // 这里不仅仅是清除mlpTemporalPoints，通过delete pMP还删除了指针指向的MapPoint
            mlpTemporalLines.clear();


            // Check if we need to insert a new keyframe
            /// 步骤2.6：检测并插入关键帧，对于双目会产生新的MapPoints，同样得增加MapLine部分
            if(NeedNewKeyFrame())
//                cout << endl << "create new kf " << endl;
                CreateNewKeyFrame();


            // We allow points with high innovation (considererd outliers by the Huber Function) 也就是在核函数中显示为误匹配的点
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            // 删除那些在bundle adjustment中检测为outlier的3D map点
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    //如果该地图点不为空而且是outlier
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }

            /// --line--
            for(int i=0; i<mCurrentFrame.NL; i++)
            {
                if(mCurrentFrame.mvpMapLines[i] && mCurrentFrame.mvbLineOutlier[i])
                    mCurrentFrame.mvpMapLines[i]= static_cast<MapLine*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        // 跟踪失败，并且relocation也没有搞定，只能重新Reset
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();  //System的重新开始
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;  //mpReferenceKF在哪里定义? 在CreatNewKeyFrame()函数中会定义

        // 保存上一帧的数据
        mLastFrame = Frame(mCurrentFrame);
    }


    //----------------------------------------------------

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    // 步骤3：记录位姿信息，用于轨迹复现
    if(!mCurrentFrame.mTcw.empty())
    {
        // 计算相对姿态T_currentFrame_referenceKeyFrame
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();  //mpReferenceKF相当于Trw,求逆后Trw, Tcr=Tcw*Twr
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);  //参考关键帧的位姿 Trw
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else  //跟踪失败，也就是mCurrentFrame.mTcw为空
    {
        // This can happen if tracking is lost
        // 如果跟踪失败，则相对位姿使用上一次值
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);  //对当前帧保存是否丢失的bool
    }

}

/**
 * @brief 双目和rgbd的地图初始化
 *
 * 由于具有深度信息，直接生成MapPoints
 * 一定注意：初始化的时候，凡是具有深度信息统统将特征转化为了地图特征，而没有考虑远近
 */
void Tracking::StereoInitialization()  //RGBD初始化的时候也要相应加入线的一些操作！
{
    if(mCurrentFrame.N>500)  //当前帧提取的关键点大于500，不到500不会初始化？
    {
        // Set Frame pose to the origin
        // 步骤1：设定初始位姿
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));  //位姿是单位阵1

        // Create KeyFrame
        // 步骤2：将当前帧构造为初始关键帧
        // mCurrentFrame的数据类型为Frame
        // KeyFrame包含Frame、地图3D点、以及BoW
        // KeyFrame里有一个mpMap，Tracking里有一个mpMap，而KeyFrame里的mpMap都指向Tracking里的这个mpMap  ?
        // KeyFrame里有一个mpKeyFrameDB，Tracking里有一个mpKeyFrameDB，而KeyFrame里的mpMap都指向Tracking里的这个mpKeyFrameDB
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);
        // 这么说第一个关键帧的位姿也是1，所以相当于第一个关键帧就是世界坐标系

        // Insert KeyFrame in the map
        // KeyFrame中包含了地图、反过来地图中也包含了KeyFrame，相互包含
        // 步骤3：在地图中添加该初始关键帧
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        // 步骤4：为每个特征点构造MapPoint
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                // 步骤4.1：通过反投影得到该特征点的3D坐标
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                //todo 注意这个函数的实现并对比关键帧反投影函数一样吗……

                // 步骤4.2：将3D点构造为MapPoint
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);

                // 步骤4.3：为该MapPoint添加属性：
                // a.观测到该MapPoint的关键帧
                // b.该MapPoint的描述子
                // c.该MapPoint的平均观测方向和深度范围

                // a.表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到，第i个特征点
                pNewMP->AddObservation(pKFini,i);
                /// b.从众多观测到该MapPoint的特征点中挑选区分读最高的描述子,这就解释了3D地图点的描述子是怎么来的
                pNewMP->ComputeDistinctiveDescriptors();
                // c.更新该MapPoint平均观测方向以及观测距离的范围
                pNewMP->UpdateNormalAndDepth();

                // 步骤4.4：在地图中添加该MapPoint
                mpMap->AddMapPoint(pNewMP);
                // 步骤4.5：表示该KeyFrame的哪个特征点可以观测到哪个3D点，建立关键帧上特征点与地图点的对应关系
                pKFini->AddMapPoint(pNewMP,i);

                // 步骤4.6：将该MapPoint添加到当前帧的mvpMapPoints中
                // 为当前Frame的特征点与MapPoint之间建立索引
                mCurrentFrame.mvpMapPoints[i]=pNewMP;  //这里可以知道，mvpMapPoints的size是和mCurrentFrame.N一致的，也就是当前帧上提取的关键点的数量
            }
        }

        /// --line--
        for(int i=0; i<mCurrentFrame.NL; ++i)
        {
            float zs = mCurrentFrame.mvDepthLineStart[i];
//            cout << "zs: " << zs << endl;
            float ze = mCurrentFrame.mvDepthLineEnd[i];
//            cout << "ze: " << ze << endl;
            if(zs>0 && ze>0)
            {
                cv::Mat x3Ds = mCurrentFrame.UnprojectStereoLineStart(i);
                cv::Mat x3De = mCurrentFrame.UnprojectStereoLineEnd(i);

                Vector6d worldPos;
                worldPos << x3Ds.at<float>(0), x3Ds.at<float>(1), x3Ds.at<float>(2), x3De.at<float>(0), x3De.at<float>(1), x3De.at<float>(2);  //这里Mat的访问只写一个参数可以吗 todo
                MapLine* pNewML = new MapLine(worldPos, pKFini, mpMap);

                //为该MapLine增加属性
                pNewML->AddObservation(pKFini, i);
                pNewML->ComputeDistinctiveDescriptors();
                pNewML->UpdateAverageDir();

                mpMap->AddMapLine(pNewML);

                pKFini->AddMapLine(pNewML, i);

                mCurrentFrame.mvpMapLines[i]=pNewML;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points, " << mpMap->MapLinesInMap() << "lines. " << endl;  //初始化时第一帧上的关键点数量


        // 步骤4：在局部地图中添加该初始关键帧
        mpLocalMapper->InsertKeyFrame(pKFini);

        //这三步操作是更新上一帧的状态包括：上一帧、上一关键帧id、上一关键帧
        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);   //属于Tracking类的数据成员，整个系统Tracking类只调用了一次，因此这里会随着运行一直进行操作
        mvpLocalMapPoints=mpMap->GetAllMapPoints();  //函数的返回是vector<MapPoint* >
        mvpLocalMapLines=mpMap->GetAllMapLines();  /// line

        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;  //对于第一帧，当前帧的参考关键帧就是第一个关键帧，而第一个关键帧又是当前帧构建来的

        // 把当前（最新的）局部MapPoints作为ReferenceMapPoints
        // ReferenceMapPoints是DrawMapPoints函数画图的时候用的
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->SetReferenceMapLines(mvpLocalMapLines);  /// line

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);  //@ 关键帧的起源这个函数有什么用呢

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}


//************************主要的函数就是以上*****************************************
//***********************下面是一些辅助函数的实现*************************************


/**
 * @brief 检查上一帧中的MapPoints是否被替换
 * 
 * Local Mapping线程可能会将某些MapPoints进行替换，由于tracking中需要用到mLastFrame，这里检查并更新上一帧中被替换的MapPoints
 * @see LocalMapping::SearchInNeighbors()
 */
void Tracking::CheckReplacedInLastFrame()  //这个函数可以理解为更新上一帧的最新的地图点匹配
{
    for(int i =0; i<mLastFrame.N; i++)  //遍历上一帧中的所有关键点
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];  //取出上一帧地图点列表中的地图点，如果不为空说明是一个地图点

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();  //调用该地图点的成员函数GetReplaced,这个函数返回地图点的mReplaced成员。地图点要被替换为的地图点
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }

    // --line--
    for(int i=0; i<mLastFrame.NL; i++)
    {
        MapLine* pML = mLastFrame.mvpMapLines[i];

        if(pML)
        {
            MapLine* pRepL = pML->GetReplaced() ;
            if(pRepL)
                mLastFrame.mvpMapLines[i] = pRepL;
        }
    }

}

/**
 * @brief 对参考关键帧的MapPoints进行跟踪. 这个函数是重点！
 * 
 * 1. 计算当前帧的词包，将当前帧的特征点分到特定层的nodes上
 * 2. 对属于同一node的描述子进行匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return 如果匹配数大于10，返回true
 */
 //todo_ lan版本中和ORB一样，但是考虑到线，是不是在这个函数中加入类似线的操作，又他妈要改！ 草
 // :加入线的部分
 //也就是说对参考关键帧的MapLines也进行追踪，最后也剔除误匹配
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    // 步骤1：将当前帧的描述子转化为BoW向量。 相当于是整个图像的描述子向量
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;
    LSDmatcher line_matcher;  //构造函数不需要参数？
    vector<MapLine*> vpMapLineMatches;

    // 步骤2：通过特征点的BoW加快当前帧与参考帧之间的特征点匹配
    // 特征点的匹配关系由MapPoints进行维护
    // todo_ 是不是先匹配然后把参考关键帧上的匹配点对应的地图点赋给vpMapPointMatches
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
    ///追踪关键帧的最重要的还是在这一步，搞清楚当前帧和参考关键帧怎么做2D与3D的匹配！
//    int line_nmatches = line_matcher.SearchByProjection(mpReferenceKF,mCurrentFrame,vpMapLineMatches);  //todo 这里要再添加一个找关键帧和当前帧之间线匹配的函数

    if(nmatches<15)
        return false;

    // 步骤3:将上一帧的位姿态作为当前帧位姿的初始值
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    //vector存每个特征点对应的MapPoint, vpMapPointMatches的维度是和mCurrentFrame.N一致的。
    mCurrentFrame.mvpMapLines = vpMapLineMatches;  //这样，后面的优化就会点线都用

    mCurrentFrame.SetPose(mLastFrame.mTcw); // 用上一次的Tcw设置初值，在PoseOptimization可以收敛快一些

    // 步骤4:通过优化3D-2D的重投影误差来获得位姿，仅优化位姿？
    ///优化函数在track中封装的很完美，只需在这里调用一次，而且参数只有一个
    Optimizer::PoseOptimization(&mCurrentFrame);  //会更新mCurrentFrame的位姿, 不管函数的返回值

    // Discard outliers
    // 步骤5：剔除优化后的outlier匹配点（MapPoints）。去除的是当前帧上的对应的地图点,而不是地图中的地图点
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i]) //第i特征点有地图点的匹配，先判断特征点有没有匹配到地图点
        {
            if(mCurrentFrame.mvbOutlier[i])  //是误匹配,在之前的优化函数中做了更新
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                //@ 这两步是说去除误匹配的时候也要更新地图点以下两个属性
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId; //虽然是误匹配但还是看到的
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;  //当前帧特征点与地图点的正确匹配数
        }
    }

    /// line
    int line_nmatchesMap = 0;
    for(int i=0; i<mCurrentFrame.NL; i++)
    {
        if(mCurrentFrame.mvpMapLines[i])
        {
            if(mCurrentFrame.mvbLineOutlier[i])
            {
                MapLine* pML = mCurrentFrame.mvpMapLines[i];

                mCurrentFrame.mvpMapLines[i]=static_cast<MapLine*>(NULL);
                mCurrentFrame.mvbLineOutlier[i]=false;
                //@ 这两步是说去除误匹配的时候也要更新地图点以下两个属性
                pML->mbTrackInView = false;
                pML->mnLastFrameSeen = mCurrentFrame.mnId; //虽然是误匹配但还是看到的
                line_nmatchesMap--;
            }
            else if (mCurrentFrame.mvpMapLines[i]->Observations()>0)
                line_nmatchesMap++;
        }
    }

    return nmatchesMap + line_nmatchesMap >= 10; //总的匹配数大于10 说明追踪关键帧成功
    // todo 这里的参数要不改为20 测试之
}


// 生成临时的地图点，这里我把线的内容也加进去
/**
 * @brief 双目或rgbd摄像头根据深度值为上一帧产生新的MapPoints.
 *        这个函数在TrackingWithMotionModel中会用到
 * 在双目和rgbd情况下，选取一些深度小一些的点（可靠一些） \n
 * 可以通过深度值产生一些新的MapPoints,这些点加入TemprolMapPoints
 *
 * 函数最终的效果是为上一帧增加了一些新的地图点匹配 mLastFrame.mvpMapPoints[i]=pNewMP
 * 同时做了记录 mlpTemporalPoints.push_back(pNewMP)
 */
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    // 步骤1：更新最近一帧的位姿
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();  //链表的结尾，参考帧到上一帧的位姿

    mLastFrame.SetPose(Tlr*pRef->GetPose()); // Tlr*Trw = Tlw 1:last r:reference w:world。 也就是上一帧的绝对位姿Tlw，更新上一帧的绝对位姿

    // 如果上一帧为关键帧，或者单目的情况，则退出
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR)
        return;

    // 步骤2：对于双目或rgbd摄像头，为上一帧临时生成新的VO MapPoints
    // 注意这些MapPoints不加入到Map中，在tracking的最后会删除
    // 跟踪过程中需要将将上一帧的MapPoints投影到当前帧可以缩小匹配范围，加快当前帧与上一帧进行特征点匹配

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    // 步骤2.1：得到上一帧有深度值的特征点
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);

    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i)); //存深度值和特征点索引
        }
    }

    if(vDepthIdx.empty())
        return;

    // 步骤2.2：按照深度从小到大排序
    sort(vDepthIdx.begin(),vDepthIdx.end());  //sort按照关键字来排序也就是pair<float, int>中的float

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    // 步骤2.3：将距离比较近的点包装成MapPoints
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        //如果第i个特征点没有对应的地图点，则插入新的点，这里重要，对于已有地图点匹配的不操作

        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            // 这些生成MapPoints后并没有通过：
            // a.AddMapPoint、
            // b.AddObservation、
            // c.ComputeDistinctiveDescriptors、
            // d.UpdateNormalAndDepth添加属性，
            // 这些MapPoint仅仅为了提高双目和RGBD的跟踪成功率
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP; // 根据上一帧的反投影添加新的临时的MapPoint

            // 标记为临时添加的MapPoint，之后在CreateNewKeyFrame之前会全部删除
            mlpTemporalPoints.push_back(pNewMP);  //临时地图点链表
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)  //加载距离最近的100个点
            break;
    }

    // todo_ 这里也一样对于线特征，是不是也要仿照上面的操作，生成一些临时的地图线
    // 仿照上面对点的操作，增加线的内容

    /// Create "visual odometry" MapLines
    // We sort lines according to their measured depth by the stereo/RGB-D sensor
    vector<pair<pair<float,float>, int>> vDepthIdxLine;  //第一个成员两端点深度，第二个成员线的索引
    vDepthIdxLine.reserve(mLastFrame.NL);

    for(int i=0; i<mLastFrame.NL; i++)
    {
        float zs =mLastFrame.mvDepthLineStart[i];
        float ze =mLastFrame.mvDepthLineEnd[i];
        if(zs>0 && ze>0)
            vDepthIdxLine.push_back(make_pair(make_pair(zs,ze), i));

    }

    if(vDepthIdxLine.empty())
        return;   //这里的返回条件可能需要改一下比较好

    //起点和终点先找出较大的深度值，对这个深度值进行从小到大的排序
    sort(vDepthIdxLine.begin(), vDepthIdxLine.end(), compare_by_maxDepth());

    //将距离较近的线包装为MapLine
    int nLines = 0;
    for(size_t j=0; j<vDepthIdxLine.size(); j++)
    {
        int i = vDepthIdxLine[j].second;

        bool bCreateNew = false;

        MapLine* pML = mLastFrame.mvpMapLines[i];  //如果第i个特征点没有对应的地图点，则插入新的点
        if(!pML)
            bCreateNew = true;
        else if(pML->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereoLine(i);
            Vector6d worldPos;
            worldPos << x3D.at<float>(0,0), x3D.at<float>(1,0), x3D.at<float>(2,0), x3D.at<float>(3,0), x3D.at<float>(4,0), x3D.at<float>(5,0);

            MapLine* pNewML = new MapLine(worldPos, mpMap, &mLastFrame, i);

            mLastFrame.mvpMapLines[i] = pNewML;
            mlpTemporalLines.push_back(pNewML);
            nLines++;
        }
        else
        {
            nLines++;
        }

        if(max(vDepthIdxLine[j].first.first, vDepthIdxLine[j].first.second) > mThDepth && nLines>30)  //如果线的端点的最大深度超过了阈值，或者加载了超过30线就终止了
            break;

    }
}




/**
 * @brief 根据匀速度模型对上一帧的MapPoints进行跟踪。   这个函数和 TrackReferenceKeyFrame()同地位！
 * 
 * 1. 非单目情况，需要对上一帧产生一些新的MapPoints（临时）
 * 2. 将上一帧的MapPoints投影到当前帧的图像平面上，在投影的位置进行区域匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return 如果匹配数大于10，返回true
 * @see V-B Initial Pose Estimation From Previous Frame
 */
 //todo 这个函数lan版本中加入了线的操作，可以参照对TrackWithRefKeyFrame进行修改
bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // --line--
    LSDmatcher lmatcher;  //这是lan写的，可见构造的时候也没有传入参数

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points
    // 步骤1：对于双目或rgbd摄像头，根据深度值为上一关键帧生成新的MapPoints
    // （跟踪过程中需要将当前帧与上一帧进行特征点匹配，将上一帧的MapPoints投影到当前帧可以缩小匹配范围）
    // 在跟踪过程中，去除outlier的MapPoint，如果不及时增加MapPoint会逐渐减少
    // 这个函数的功能就是补充增加RGBD和双目相机上一帧的MapPoints数
    UpdateLastFrame();

    // 根据Const Velocity Model(认为这两帧之间的相对运动和之前两帧间相对运动相同)估计当前帧的位姿
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);  //假设上一帧到当前帧的位位姿等于mVelocity(上上一帧到上一帧的位姿)

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));  //当前帧的地图点列表为空
    // todo_ 这里是不是也需要更新一下mvpMapLines lan版本中没做
    /// --line--
    fill(mCurrentFrame.mvpMapLines.begin(),mCurrentFrame.mvpMapLines.end(),static_cast<MapLine*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;  //RGBD下的th
    else
        th=7;

    // 步骤2：根据匀速度模型进行对上一帧的MapPoints进行跟踪
    // 根据上一帧特征点对应的3D点投影的位置缩小特征点匹配范围
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);  ///@ 看这个函数的具体实现

    // --line-- todo 检查
    int lnmatches = lmatcher.SearchByProjection(mCurrentFrame, mLastFrame);

//    cout << "tracking points = " << nmatches << endl;
//    cout << "tracking lmatches = " << lmatches << endl;


    // If few matches, uses a wider window search
    // 如果跟踪的点少，则扩大搜索半径再来一次
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR); // 2*th
    }

    // --line--
    if(mCurrentFrame.mvKeylinesUn.size()==0)
        cerr << "error: Tracking::TrackWithMotionModel(),mCurrentFrame.mvKeylinesUn.size() = 0" << endl;
    //或者使用
//    assert(mCurrentFrame.mvKeylinesUn.size() != 0);
    double lmatch_ratio = lnmatches*1.0/mCurrentFrame.mvKeylinesUn.size();
//    cout << "line match ratio in current frame: " << lmatch_ratio << endl;

    // --line-- 点匹配数少于20或者直线匹配数少于当前帧检测出的直线数量的一半？？
    if(nmatches<20 || lmatch_ratio<0.3)  //todo 这个参数可以调节，可以考虑将两个匹配数量加起来
        return false;


    // Optimize frame pose with all matches
    // 步骤3：优化位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    // 步骤4：优化位姿后剔除outlier的mvpMapPoints
    int nmatchesMap = 0;
    //todo_ 思考如果不剔除的话影响其他操作吗，可能会吧比如当前帧被判断为关键帧后会生成地图点？
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;  //计算优化后和地图点的正确匹配
        }
    }

    /// --line-- todo_ 同样滴 lan版本中这里没有去除误匹配线的操作 考虑是否增加 要添加！
    // 误匹配不一定是整条直线 而是直线上的某个采样点
    int lnmatchesMap = 0;
     for(int i =0; i<mCurrentFrame.NL; i++)
     {
         if(mCurrentFrame.mvpMapLines[i])
         {
             if(mCurrentFrame.mvbLineOutlier[i])
             {
                 MapLine* pML = mCurrentFrame.mvpMapLines[i];

                 mCurrentFrame.mvpMapLines[i]=static_cast<MapLine*>(NULL);
                 mCurrentFrame.mvbLineOutlier[i]=false;
                 pML->mbTrackInView = false;
                 pML->mnLastFrameSeen = mCurrentFrame.mnId;
                 lnmatches--;
             }
             else if(mCurrentFrame.mvpMapLines[i]->Observations()>0)
                 lnmatchesMap++;  //计算优化后和地图点的正确匹配
         }
     }

//   return nmatchesMap>=10;   //位姿优化之后，当前帧的特征点与地图点的正确匹配数量
    return nmatchesMap+lnmatches >= 15;
}


/**
 * @brief 对Local Map的MapPoints进行跟踪, 加入线的内容
 *
 * 1. 更新局部地图，包括局部关键帧和关键点
 * 2. 对局部MapPoints进行投影匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return true if success
 * @see V-D track Local Map
 */
/// --line-- 对局部地图中点和线进行追踪
bool Tracking::TrackLocalMapWithLines()
{
    // step1：更新局部关键帧mvpLocalKeyFrames和局部地图点mvpLocalMapPoints，局部地图线mvpLocalMapLines
    UpdateLocalMap(); //这个函数要重写加入线

//    // step2：在局部地图中查找与当前帧匹配的MapPoints
//    SearchLocalPoints();
//
//    // step3: 在局部地图中查找与当前帧匹配的MapLines
//    SearchLocalLines();

    //多线程操作，可以可以，666
    thread threadPoints(&Tracking::SearchLocalPoints, this);
    thread threadLines(&Tracking::SearchLocalLines, this);
    threadPoints.join();
    threadLines.join();

    //现在已经完成了当前帧上点和线与局部地图中的点线匹配

    // step4：更新局部所有MapPoints和MapLines后对位姿再次优化
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;
    mnLineMatchesInliers = 0;

    // Update MapPoints Statistics
    // step5：更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i]) //位姿优化之后这个匹配是内点
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)  //slam模式
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        //这个判断说明匹配的地图点是地图中的地图点而不是临时生成的地图点
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
          /*  else if(mSensor==System::STEREO)  //todo_ 这个双目是真的双目还是RGBD都算
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                //如果这个匹配点是外点而且是双目相机，就把当前帧上的点匹配置空
                // todo_ 这里是不是有问题：如果是RGBD相机的话，如果这个点是外点当前帧的点匹配不置空吗
                // 难道说，这里即使匹配的是外点，不置空也没有关系？*/

        }
    }

    // 更新MapLines Statistics
    // step6：更新当前帧的MapLines被观测程度，并统计跟踪局部地图的效果
    for(int i=0; i<mCurrentFrame.NL; i++)
    {
        if(mCurrentFrame.mvpMapLines[i])
        {
            if(!mCurrentFrame.mvbLineOutlier[i])
            {
                mCurrentFrame.mvpMapLines[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapLines[i]->Observations()>0)
                        mnLineMatchesInliers++;
                }
                else
                    mnLineMatchesInliers++;
            }
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    // step7：决定是否跟踪成功。 距离重定位帧比较近的时候要求匹配足够大才认为是追踪成功
//    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)

    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers+mnLineMatchesInliers<60)  //该参数可调
        return false;

    // todo_ 这里不需要写关于线的操作吗，不然上面算线的匹配数量干嘛  lan版本中没有写
    // 可以考虑将上面的判断条件改为 mnMatchesInliers+mnLineMatchesInliers < 50

    if(mnMatchesInliers+mnLineMatchesInliers<40)  //todo 同样的这里的判断条件也可以修改
        return false;
    else
        return true;
}


//以下是ORB中的实现，这个函数如果有问题可以参考wubo版本中的该函数
//todo_ 这个函数中没有考虑到线的因素，想一想要不要增加
// :这里暂时不加入线的内容，也就是说判断是否需要关键帧目前只与点有关系
/// 注意：这个函数lan版本中和wubo版本中不一样，这里我完全用wubo版
bool Tracking::NeedNewKeyFrame()
{
    // 步骤1：如果用户在界面上选择重定位，那么将不插入关键帧
    // 由于插入关键帧过程中会生成MapPoint，因此用户选择重定位后地图上的点云和关键帧都不会再增加
    if(mbOnlyTracking)  //定位模式下不需要插入关键帧
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    // 如果局部地图被闭环检测使用，则不插入关键帧
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())  ///局部地图停止或者有停止请求的时候不插入关键帧
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    // 步骤2：判断是否距离上一次插入关键帧的时间太短，过短不会插入关键帧
    // mCurrentFrame.mnId是当前帧的ID
    // mnLastRelocFrameId是最近一次重定位帧的ID
    // mMaxFrames等于图像输入的帧率
    // 如果关键帧比较少，则考虑插入关键帧
    // 或距离上一次重定位超过1s，则考虑插入关键帧
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)  //当前地图中关键帧很多（>mMaxFrames）而且 当前帧位于上一次重定位到上一次重定位后的30帧这个范围内，不会插入关键帧
        return false;

    // Tracked MapPoints in the reference keyframe
    // 步骤3：得到参考关键帧跟踪到的MapPoints数量

    int nMinObs = 3;  //大于等于这个阈值（最小观测数），说明关键帧上的这个地图点就是个高质量的地图点
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);  //当前帧中高质量地图点的数量
    // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧。

    // Local Mapping accept keyframes?
    // 步骤4：查询局部地图管理器是否繁忙
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Stereo & RGB-D: Ratio of close "matches to map"/"total matches"
    // "total matches = matches to map + visual odometry matches"
    /// Visual odometry matches will become MapPoints if we insert a keyframe.
    // This ratio measures how many MapPoints we could create if we insert a keyframe.
    // 步骤5：对于双目或RGBD摄像头，统计总的可以添加的MapPoints数量和跟踪到地图中的MapPoints数量
    int nMap = 0;
    int nTotal= 0;
    if(mSensor!=System::MONOCULAR)// 双目或rgbd
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                nTotal++;// 总的可以投影得到可靠3D位置的点
                if(mCurrentFrame.mvpMapPoints[i])  //第i个特征点有对应的地图点
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)  //该地图点有被其他关键帧观测
                        nMap++;// 即能观测到的地图中的MapPoints数量
            }
        }
    }
    else
    {
        // There are no visual odometry matches in the monocular case
        nMap=1;
        nTotal=1;
    }

    assert(max(1, nTotal) != 0);
    const float ratioMap = (float)nMap/(float)(std::max(1, nTotal));  //

    // 步骤6：决策是否需要插入关键帧
    // Thresholds
    // 设定inlier阈值，和之前帧特征点匹配的inlier比例
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;// 关键帧只有一帧，那么插入关键帧的阈值设置很低
    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // MapPoints中和地图关联的比例阈值
    float thMapRatio = 0.35f;
    if(mnMatchesInliers>300)
        thMapRatio = 0.20f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    // 很长时间没有插入关键帧
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    // localMapper处于空闲状态
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    // Condition 1c: tracking is weak
    // 跟踪要跪的节奏，0.25和0.3是一个比较低的阈值
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || ratioMap<0.3f) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    // 内点匹配数量与之前参考帧（最近的一个关键帧）较少或者观测到的地图点所占比例较少 && 内点匹配数量大于15
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio || ratioMap<thMapRatio) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)  //c2是必要条件
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();  //局部地图不空闲的时候中断BA？
            if(mSensor!=System::MONOCULAR)
            {
                // 队列里不能阻塞太多关键帧
                // tracking插入关键帧不是直接插入，而且先插入到mlNewKeyFrames中，
                // 然后localmapper再逐个pop出来插入到mspKeyFrames
                if(mpLocalMapper->KeyframesInQueue()<3)  ///std::list<KeyFrame*> mlNewKeyFrames; 队列中的关键帧就是这个链表的size
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

/**
 * @brief 创建新的关键帧
 *
 * 对于非单目的情况，同时创建新的MapPoints！！ 加入线的内容
 */
 /// 可见双目情况下，在Tracking 线程中的创建关键帧的函数中会生成地图点 同样地 地图线
 /// 其中线的部分我增加的代码和UpdateLastFrame函数的内容基本一致，如果要修改这两个地方应该都修改
void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true)) //局部地图有设置停，则不会创建关键帧 ？
        return;

    // 步骤1：将当前帧构造成关键帧
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB); //单个参数为当前帧、地图、关键帧库

    // 步骤2：将当前关键帧设置为当前帧的参考关键帧
    // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧 ？？（两个问号表示对吴博的注释有疑问）

    mpReferenceKF = pKF;  //这是Tracking类中的数据成员，也就是当前追踪线程的参考关键帧
    mCurrentFrame.mpReferenceKF = pKF;  //当前帧的参考关键帧是由自身所创建而来的关键帧

    // 这段代码和UpdateLastFrame中的那一部分代码功能相同
    // 步骤3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
    if(mSensor!=System::MONOCULAR)
    {
        // 根据Tcw计算mRcw、mtcw和mRwc、mOw
        mCurrentFrame.UpdatePoseMatrices();
        ///这里的目的是因为当前帧的点反投影得到3D点的时候需要知道当前帧的绝对位姿Tcw, 在下面的UnprojectStereo函数中会用到

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        // 步骤3.1：得到当前帧深度小于阈值的特征点
        // 创建新的MapPoint, depth < mThDepth
        vector<pair<float,int> > vDepthIdx;  //第一个为深度 第二个为特征点的序号
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            // 步骤3.2：按照深度从小到大排序
            sort(vDepthIdx.begin(),vDepthIdx.end());

            // 步骤3.3：将距离比较近的点包装成MapPoints
            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];  //第i特征点所对应的地图点
                if(!pMP)
                    bCreateNew = true;  //第i特征点所对应的地图点为空的话说明可以创建为3D点
                else if(pMP->Observations()<1)  //第i特征点对应的地图点不为空但是观测次数小于1（不就是0嘛）。这样写可能是为了更鲁棒吧。 这个地方说明pMP是个临时地图点,这个点也要生成真正的地图点
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    // 这些添加属性的操作是每次创建MapPoint后都要做的。  对这个新创建的地图点更新三个属性！
                    pNewMP->AddObservation(pKF,i);            //i表示关键帧上特征点的序号
                    pKF->AddMapPoint(pNewMP,i);               //该关键帧上增加地图点
                    pNewMP->ComputeDistinctiveDescriptors();  //该地图点计算最佳描述子
                    pNewMP->UpdateNormalAndDepth();           //该地图点更新方位信息和深度信息
                    mpMap->AddMapPoint(pNewMP);               //最后地图中加入这个点

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;    //是以当前帧创建的关键帧所以当前帧的地图点列表更新对应关系
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                // 这里决定了双目和rgbd摄像头时地图点云的稠密程度
                // 但是仅仅为了让地图稠密直接改这些不太好，
                // 因为这些MapPoints会参与之后整个slam过程
                if(vDepthIdx[j].first>mThDepth && nPoints>100)  //如果创建了超过100地图点而且距离大于阈值的话，不会再创建地图点。所以这里就算是深度大于阈值了也有可能会创建，因为至少要创建100个点
                    break;
            }   ///这里就说明RGBD相机中，根据深度生成地图点是在AddKeyFrame中进行了，其他地方再说
        }


        //todo_ 同样的根据深度信息生成真正的地图线这里是不是也要写上！仿照上面的操作！
        // 因为lan版本是针对单目所有这里没有写
        /// ---------------------------line-------------------------------------
        // We sort lines according to their measured depth by the stereo/RGB-D sensor
        vector<pair<pair<float,float>, int>> vDepthIdxLine;  //第一个成员两端点深度，第二个成员线的索引
        vDepthIdxLine.reserve(mCurrentFrame.NL);

        for(int i=0; i<mCurrentFrame.NL; i++)
        {
            float zs =mCurrentFrame.mvDepthLineStart[i];
            float ze =mCurrentFrame.mvDepthLineEnd[i];
            if(zs>0 && ze>0)   //当两个端点的深度都有时才会加入
                vDepthIdxLine.push_back(make_pair(make_pair(zs,ze), i));

        }

        if(!vDepthIdxLine.empty())
        {
            //起点和终点先找出较大的深度值，对这个深度值进行从小到大的排序
            sort(vDepthIdxLine.begin(), vDepthIdxLine.end(), compare_by_maxDepth());

            int nLines = 0;
            for(size_t j=0; j<vDepthIdxLine.size(); j++)
            {
                int i = vDepthIdxLine[j].second;

                bool bCreateNew = false;

                MapLine* pML = mCurrentFrame.mvpMapLines[i];  //如果第i个特征点没有对应的地图点，则插入新的点
                if(!pML)
                    bCreateNew = true;
                else if(pML->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereoLine(i); //这里要确保两个端点都有深度值
                    Vector6d worldPos;
                    //todo_ 段错误！ 原来是上面应为mCurrentFrame not mLastFrame
                    worldPos << x3D.at<float>(0,0), x3D.at<float>(1,0), x3D.at<float>(2,0), x3D.at<float>(3,0), x3D.at<float>(4,0), x3D.at<float>(5,0);

                    MapLine* pNewML = new MapLine(worldPos, pKF, mpMap);
                    //这一步构造地图线和UpdateLastFrame中的构造函数不一样

                    // 更新属性
                    pNewML->AddObservation(pKF, i);           //地图线增加关键帧的观测
                    pKF->AddMapLine(pNewML, i);               //关键帧增加地图线的观测
                    pNewML->ComputeDistinctiveDescriptors();  //地图线更新最佳描述子
                    pNewML->UpdateAverageDir();               //地图线更新平均观测方位

                    mpMap->AddMapLine(pNewML);

                    mCurrentFrame.mvpMapLines[i] = pNewML;

                    nLines++;
                }
                else
                {
                    nLines++;
                }

                if(max(vDepthIdxLine[j].first.first, vDepthIdxLine[j].first.second) > mThDepth && nLines>40)
                    break;
                    //如果线的端点的最大深度超过了阈值，或者加载了超过30线就终止了。40可调！
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);  //@ 是不是意思对局部地图的设置到这里就停了（插入关键帧的操作结束了）

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}


/**
 * @brief 对Local MapPoints进行跟踪（即找到当前帧2D特征点和局部地图中地图点的匹配）
 * 
 * 在局部地图中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影匹配
 */
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    // 步骤1：遍历当前帧的mvpMapPoints，标记这些MapPoints不参与之后的搜索
    // 因为当前的mvpMapPoints一定在当前帧的视野中
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)  //说明所对应的地图点不为空，已经有了匹配。这个匹配可能是TrackWithRefrenceKF或者TrackWithMotionModel而来的
        {
            if(pMP->isBad()) //这个匹配的地图点是坏的，把这个点置为空
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                // 更新能观测到该点的帧数加1
                pMP->IncreaseVisible();   //该地图点是好的而且跟当前帧有匹配，那么这个点的观测数加1
                // 标记该点被当前帧观测到
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                // 标记该点将来不被投影，因为已经匹配过
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    // 步骤2：将所有局部MapPoints投影到当前帧，判断是否在视野范围内，然后进行投影匹配
    //感觉这一步的操作，真正有变化的就是 pMP->IncreaseVisible();
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        // 已经被当前帧观测到MapPoint不再判断是否能被当前帧观测到
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        
        // Project (this fills MapPoint variables for matching)
        // 步骤2.1：判断LocalMapPoints中的点是否在在视野内
        if(mCurrentFrame.isInFrustum(pMP,0.5))  //0.5表示cos(60) 。地图点的平均视角向量和当前帧下观测该点的向量的夹角
        {
        	// 观测到该点的帧数加1，该MapPoint在某些帧的视野范围内
            pMP->IncreaseVisible();
            // 只有在视野范围内的MapPoints才参与之后的投影匹配
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);  //第一个参数含义: @param nnratio  ratio of the best and the second score
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;

        // If the camera has been relocalised recently, perform a coarser search
        // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)  //是重定位后的前两帧
            th=5;

        // 步骤2.2：对视野范围内的MapPoints通过投影进行特征点匹配
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
        //SearchByProjection这个函数重载了四次！这个函数最终的结果是更新当前帧的地图点匹配
        // F.mvpMapPoints[bestIdx]=pMP
    }
}


//// --line---
void Tracking::SearchLocalLines()
{
    // step1：遍历在当前帧的mvpMapLines，标记这些MapLines不参与之后的搜索，因为当前的mvpMapLines一定在当前帧的视野中
    for(vector<MapLine*>::iterator vit=mCurrentFrame.mvpMapLines.begin(), vend=mCurrentFrame.mvpMapLines.end(); vit!=vend; vit++)
    {
        MapLine* pML = *vit;
        if(pML)
        {
            if(pML->isBad())
            {
                *vit = static_cast<MapLine*>(NULL);
            }
            else
            {
                // 更新能观测到该线段的帧数加1
                pML->IncreaseVisible();
                // 标记该点被当前帧观测到
                pML->mnLastFrameSeen = mCurrentFrame.mnId;
                // 标记该线段将来不被投影，因为已经匹配过
                pML->mbTrackInView = false;
            }
        }
    }

    int nToMatch = 0;

    // step2：将所有局部MapLines投影到当前帧，判断是否在视野范围内，然后进行投影匹配
    for(vector<MapLine*>::iterator vit=mvpLocalMapLines.begin(), vend=mvpLocalMapLines.end(); vit!=vend; vit++)
    {
        MapLine* pML = *vit;

        // 已经被当前帧观测到MapLine，不再判断是否能被当前帧观测到
        if(pML->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pML->isBad())
            continue;

        // step2.1：判断LocalMapLine是否在视野内
        if(mCurrentFrame.isInFrustum(pML, 0.5))   //todo 这个函数有可能有问题要做反复检查
        {
            // 观察到该点的帧数加1，该MapLine在某些帧的视野范围内
            pML->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        LSDmatcher matcher;
        int th = 1;

        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;  //todo_ 这里的阈值和ORB中一样，但是这是线啊，这些参数可以调节为其他值试试

        matcher.SearchByProjection(mCurrentFrame, mvpLocalMapLines, th);
        //目的：更新当前帧的地图点匹配，将其与局部地图点建立关系
    }
}


/**
 * @brief 更新LocalMap
 *
 * 局部地图包括： \n
 * - K1个关键帧、K2个临近关键帧和参考关键帧
 * - 由这些关键帧观测到的MapPoints
 */
void Tracking::UpdateLocalMap()
{
    // This is for visualization
    // 这行程序放在UpdateLocalPoints函数后面是不是好一些
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    // --line--
    mpMap->SetReferenceMapLines(mvpLocalMapLines);

    // Update
    // 更新局部关键帧和局部MapPoints
    UpdateLocalKeyFrames();

    UpdateLocalPoints(); //目的是update mvLocalMapPoints
    // --line--
    UpdateLocalLines();
}

/**
 * @brief 更新局部关键点，called by UpdateLocalMap()
 * 
 * 局部关键帧mvpLocalKeyFrames的MapPoints，更新mvpLocalMapPoints
 */
void Tracking::UpdateLocalPoints()
{
    // 步骤1：清空局部MapPoints
    mvpLocalMapPoints.clear();  //这个成员变量是Tracking类的

    // 步骤2：遍历局部关键帧mvpLocalKeyFrames
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();  //注意这里获取该关键帧的MapPoints

        // 步骤2：将局部关键帧的MapPoints添加到mvpLocalMapPoints
        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            // mnTrackReferenceForFrame防止重复添加局部MapPoint
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

/// --line-- 与上同
void Tracking::UpdateLocalLines()
{
    // step1：清空局部MapLines
    mvpLocalMapLines.clear();

    // step2：遍历局部关键帧mvpLocalKeyFrames
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapLine*> vpMLs = pKF->GetMapLineMatches();

        //step3：将局部关键帧的MapLines添加到mvpLocalMapLines
        for(vector<MapLine*>::const_iterator itML=vpMLs.begin(), itEndML=vpMLs.end(); itML!=itEndML; itML++)
        {
            MapLine* pML = *itML;
            if(!pML)
                continue;
            if(pML->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pML->isBad())
            {
                mvpLocalMapLines.push_back(pML);
                pML->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}



/**
 * @brief 更新局部关键帧，called by UpdateLocalMap()
 *
 * 遍历当前帧的MapPoints，将观测到这些MapPoints的关键帧和相邻的关键帧取出，更新mvpLocalKeyFrames
 *
 */

 // todo
 // 这个函数是遍历当前帧的所匹配的MapPoints，从而根据四种策略找到局部关键帧。如果加入线的话也不是不可
 // 猜想加入线之后，局部关键帧的数量可能会变多，所用时间可能增加，暂时先不加入线，但是应该考虑线的
 // 对mvpMapPoints基本上只有读操作，也没有写操作，我不加入线的内容影响不大。 稍微加入一点线的内容
void Tracking::UpdateLocalKeyFrames()  //没有变动
{
    // Each map point vote for the keyframes in which it has been observed
    // 步骤1：遍历当前帧的MapPoints，记录所有能观测到当前帧MapPoints的关键帧
    map<KeyFrame*,int> keyframeCounter;   //map数据结构用的好啊，第一个元素为关键帧，第二个元素为此关键帧看到当前帧地图点的数目
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                // 对于当前帧上的每一个地图点，把所观测到的关键帧都加入进来
                // 能观测到当前帧MapPoints的关键帧
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();  //@ 看GetObservations函数
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;  //即完成了添加关键帧 又更新了此关键帧对应的int
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    /// --line-- 该该函数中 只是这一部分增加了线相关的内容
    map<KeyFrame*, int> keyframeCounterLine;
    for(int i=0; i<mCurrentFrame.NL; i++)
    {
        if(mCurrentFrame.mvpMapLines[i])
        {
            MapLine* pML = mCurrentFrame.mvpMapLines[i];
            if(!pML->isBad())
            {
                const map<KeyFrame*, size_t> observations = pML->GetObservations();

                for(map<KeyFrame*,size_t >::const_iterator it=observations.begin(), itend=observations.end(); it != itend; it++)
                    keyframeCounterLine[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapLines[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    // 步骤2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有三个策略
    // 先清空局部关键帧
    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    // V-D K1: shares the map points with current frame
    // 策略1：能观测到当前帧MapPoints的关键帧作为局部关键帧
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;  //某个关键帧最多观测点的数量
            pKFmax=pKF;  //得到观测最多的关键帧
        }

        mvpLocalKeyFrames.push_back(it->first);  //只要能观测到当前帧的MapPoint就加入进来
        // mnTrackReferenceForFrame防止重复添加局部关键帧
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId; //更新这个关键帧的属性
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    // V-D K2: neighbors to K1 in the covisibility graph
    // 策略2：与策略1得到的局部关键帧共视程度很高的关键帧作为局部关键帧
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)  //局部关键帧数量最多80
            break;

        KeyFrame* pKF = *itKF;

        // 策略2.1:K1中每个共视关键帧的最佳共视的10帧 加进mvpLocalKeyFrames
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                // mnTrackReferenceForFrame防止重复添加局部关键帧
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId) //避免重复
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // 策略2.2：K1关键帧的子关键帧
        const set<KeyFrame*> spChilds = pKF->GetChilds();  //todo_ 检查这个函数
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // 策略2.3: K1关键帧的父关键帧
        KeyFrame* pParent = pKF->GetParent(); //父关键帧只有一个子关键帧有很多个
        if(pParent)
        {
            // mnTrackReferenceForFrame防止重复添加局部关键帧
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    // V-D Kref： shares the most map points with current frame
    // 步骤3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
    if(pKFmax)  //pKFmax： 看到当前帧上地图点的数量最多
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;   //与当前帧共视关系最高的帧作为当前帧的参考关键帧
    }
}


// 没有变动，这里面没有关于线的内容，暂时不考虑线了，心真累！
// 重定位部分，也是完成由点来决定…………………………
// todo_ 理论上加入线应该也可以，不知道有没有提高
bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    // 步骤1：计算当前帧特征点的Bow映射
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    // 步骤2：找到与当前帧相似的候选关键帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);  //先取出候选关键帧

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;  //这个i关键帧pass
        else
        {
            // 步骤3：通过BoW进行匹配
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);  //第三个参数为vector,存的是匹配好的当前帧特征点所对应的地图点
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                // 初始化PnPsolver
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC  /// P4P是在什么地方用的？？？　
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            // 步骤4：通过EPnP算法估计姿态
            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                // 步骤5：通过PoseOptimization对姿态进行优化求解
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                // 步骤6：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);  //SearchByProjection第三个重载函数了

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

//没有变动
void Tracking::Reset()
{

    if(mpViewer)  //显示器指针不为空
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())  //睡眠直到显示器停止
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    cout << "System Reseting" << endl;

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;  //这两个属于类的静态数据成员
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;


    //注意以下四个链表，是属于Tracking的随着时间不断维护的状态，最重要的目的是记录每一帧的位姿和关键帧，最终保存轨迹的时候用
    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();  //注意Viewer中RequesStop和Release的区别
}


// 这个函数没用到，略过
void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);  //修改Tracking的K矩阵（数据成员）

    cv::Mat DistCoef(4,1,CV_32F);  //4行1列的矩阵
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;  //修改Frame类的静态成员
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;  //定位模式的标识
}


} //namespace ORB_SLAM

// 现在只差单目初始化的部分代码我还没有掌握！有时间再来看！
