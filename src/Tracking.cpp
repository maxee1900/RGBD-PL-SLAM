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
#include"Initializer.h"
//

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
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
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

    // --line-- 在这里先得到映射值  todo 检查
    initUndistortRectifyMap(mK, mDistCoef, Mat_<double>::eye(3,3), mK, Size(img_width, img_height), CV_32F, mUndistX, mUndistY);

    cout << "mUndistX size = " << mUndistX.size << endl;
    cout << "mUndistY size = " << mUndistY.size << endl;
    // ---up---


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

/// PASS(因为我是用RGBD，暂时跳过)
// 输入左右目图像，可以为RGB、BGR、RGBA、GRAY
// 1、将图像转为mImGray和imGrayRight并初始化mCurrentFrame
// 2、进行tracking过程
// 输出世界坐标系到该帧相机坐标系的变换矩阵
cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    // 步骤1：将RGB或RGBA图像转为灰度图像
    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    // 步骤2：构造Frame
    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    /// 步骤3：跟踪   主函数吶！
    Track();

    return mCurrentFrame.mTcw.clone();
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


// 输入左目RGB或RGBA图像
// 1、将图像转为mImGray并初始化mCurrentFrame
// 2、进行tracking过程
// 输出世界坐标系到该帧相机坐标系的变换矩阵
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    // 步骤1：将RGB或RGBA图像转为灰度图像
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
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

    /// --line-- 校正图片
    cv::remap(mImGray, mImGray, mUndistX, mUndistY, cv::INTER_LINEAR);
    // todo 详细查看这个函数


    // Kitti数据集
//     mImGray = mImGray(Rect(15, 15, 1211, 346));  //这里是为了剪切掉修正后图片的弧形边缘，针对不同数据集，image的尺寸不一样，所以这里应该改成动态的
    // TUM数据集
    mImGray = mImGray(Rect(15, 15, 610, 450));  //这样的语法对不对，opencv中不是一般赋值用clone吗

    // 步骤2：构造Frame
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)// 没有成功初始化的前一个状态就是NO_IMAGES_YET
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);  //单目未初始化时ORB提取器不同
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

//    static int count=0;
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
    {
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
//        count++;
//        cout << "initial frame num = " << count << endl;  //统计了初始化帧的个数
    //注释符号//放在行首位置一般都是将本行代码注释掉，调试的时候再操作
    }
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);


    // 步骤3：跟踪
    Track();

    return mCurrentFrame.mTcw.clone();
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
        else
            // 统计单目初始化的时间
            // app方式打开如果没有文件则创建，如果有则追加在结尾；ate方式如果有文件则清空

//            ofstream file("InitialPoseEstimationTime.txt", ios::app);
//            chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

            MonocularInitialization();

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
        bool bOK;

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
        else  /// 定位模式：这一部分代码可以跳过了!
        {
            // Localization Mode: Local Mapping is deactivated
            // 只进行跟踪tracking，局部地图不工作
 
            // 步骤2.1：跟踪上一帧或者参考帧或者重定位

            // tracking跟丢了
            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                // mbVO是mbOnlyTracking为true时的才有的一个变量
                // mbVO为false表示此帧匹配了很多的MapPoints，跟踪很正常，
                // mbVO为true表明此帧匹配了很少的MapPoints，少于10个，要跪的节奏

                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map
                    // mbVO为0则表明此帧匹配了很多的3D map点，非常好

                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                        // 这个地方是不是应该加上：
                        // if(!bOK)
                        //    bOK = TrackReferenceKeyFrame();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else  //mbVO=true
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.
                    /// 这一步有点像章宏的论文，当前帧上匹配的3D点过少，则同时计算两种位姿 一种是由运动模型而来，一种是重定位而来，如果重定位成功了就选择重定位方法的位姿

                    // mbVO为1，则表明此帧匹配了很少的3D map点，少于10个，要跪的节奏，既做跟踪又做定位

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;

                    if(!mVelocity.empty())  //如果恒速模型不为空
                    {
                        bOKMM = TrackWithMotionModel();
                        //
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    // 重定位没有成功，但是如果跟踪成功
                    if(bOKMM && !bOKReloc)
                    {
                        // 这三行没啥用？  好像是的 赋值后又赋了回来，还是原始值
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)  //mbVO在这个大语句块中本来就是true的，不用if也可以
                        {
                            // 这段代码是不是有点多余？应该放到TrackLocalMap函数中统一做
                            // 更新当前帧的MapPoints被观测程度
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)// 只要重定位成功整个跟踪过程正常进行（重定位与恒速跟踪，更相信重定位）
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        //好像orbslam中参考帧都是关键帧吧，没有将普通帧叫做参考帧的
        // 将最新的关键帧作为当前帧的参考关键帧reference keyframe
        //当前帧是Tracking类的数据成员
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

//-----------------上面的步骤只是得到了当当前帧的初始估计位姿--------------------------------------


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
        else  //定位模式
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.

            // 重定位成功
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
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

            /// --line-- 考虑这里是不是也要mlpTemporalLines lan版本中没有  todo


            //todo 有个重要的问题：追踪过程中最终有没有生成地图点加入到地图中

            // Check if we need to insert a new keyframe
            // 步骤2.6：检测并插入关键帧，对于双目会产生新的MapPoints
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

            // --line--
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
 */
void Tracking::StereoInitialization()  //RGBD初始化的时候也要相应加入线的一些操作！
{
    if(mCurrentFrame.N>500)  //当前帧提取的关键点大于500
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
        mpMap->AddKeyFrame(pKFini);   //mpMap是指针传递，所有可以对mpMap进行操作。 mpMap应该只有一个实例，是不是符合单例模式

        // Create MapPoints and asscoiate to KeyFrame
        // 步骤4：为每个特征点构造MapPoint
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                // 步骤4.1：通过反投影得到该特征点的3D坐标
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
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

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;  //初始化时第一帧上的关键点数量

        // 步骤4：在局部地图中添加该初始关键帧
        mpLocalMapper->InsertKeyFrame(pKFini);

        //这三步操作是更新上一帧的状态包括：上一帧、上一关键帧id、上一关键帧
        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);   //属于Tracking类的数据成员，整个系统Tracking类只调用了一次，因此这里会随着运行一直进行操作
        mvpLocalMapPoints=mpMap->GetAllMapPoints();  //函数的返回是vector<MapPoint* >
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;  //对于第一帧，当前帧的参考关键帧就是第一个关键帧，而第一个关键帧又是当前帧构建来的

        // 把当前（最新的）局部MapPoints作为ReferenceMapPoints
        // ReferenceMapPoints是DrawMapPoints函数画图的时候用的
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);  //@ 关键帧的起源这个函数有什么用呢

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

/**
 * @brief 单目的地图初始化
 *
 * 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
 * 得到初始两帧的匹配、相对运动、初始MapPoints
 */
void Tracking::MonocularInitialization()
{
    // 如果单目初始器还没有被创建，则创建单目初始器
    if(!mpInitializer)   ///单目会用到单目的初始化器，对于RGBD使用到不到initializer.h这个库的
    {
        // Set Reference Frame
        // 单目初始帧的特征点数必须大于100
        if(mCurrentFrame.mvKeys.size()>100)
        {
            // 步骤1：得到用于初始化的第一帧，初始化需要两帧
            mInitialFrame = Frame(mCurrentFrame);
            // 记录最近的一帧
            mLastFrame = Frame(mCurrentFrame);
            // mvbPrevMatched最大的情况就是所有特征点都被跟踪上
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            // 这两句是多余的
            if(mpInitializer)
                delete mpInitializer;

            // 由当前帧构造初始器 sigma:1.0 iterations:200
            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize
        // 步骤2：如果当前帧特征点数大于100，则得到用于单目初始化的第二帧
        // 如果当前帧特征点太少，重新构造初始器
        // 因此只有连续两帧的特征点个数都大于100时，才能继续进行初始化过程
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        // 步骤3：在mInitialFrame与mCurrentFrame中找匹配的特征点对
        // mvbPrevMatched为前一帧的特征点，存储了mInitialFrame中哪些点将进行接下来的匹配
        // mvIniMatches存储mInitialFrame,mCurrentFrame之间匹配的特征点
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        // 步骤4：如果初始化的两帧之间的匹配点太少，重新初始化
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        // 步骤5：通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints
        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            // 步骤6：删除那些无法进行三角化的匹配点
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            // 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            // 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到该帧的变换矩阵
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            // 步骤6：将三角化得到的3D点包装成MapPoints
            // Initialize函数会得到mvIniP3D，
            // mvIniP3D是cv::Point3f类型的一个容器，是个存放3D点的临时变量，
            // CreateInitialMapMonocular将3D点包装成MapPoint类型存入KeyFrame和Map中
            CreateInitialMapMonocular();
        }
    }
}

/**
 * @brief CreateInitialMapMonocular
 *
 * 为单目摄像头三角化生成MapPoints
 */
void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    // 步骤1：将初始关键帧的描述子转为BoW
    pKFini->ComputeBoW();
    // 步骤2：将当前关键帧的描述子转为BoW
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    // 步骤3：将关键帧插入到地图
    // 凡是关键帧，都要插入地图
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    // 步骤4：将3D点包装成MapPoints
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        // 步骤4.1：用3D点构造MapPoint
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        // 步骤4.2：为该MapPoint添加属性：
        // a.观测到该MapPoint的关键帧
        // b.该MapPoint的描述子
        // c.该MapPoint的平均观测方向和深度范围

        // 步骤4.3：表示该KeyFrame的哪个特征点可以观测到哪个3D点
        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        // a.表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到
        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        // b.从众多观测到该MapPoint的特征点中挑选区分读最高的描述子
        pMP->ComputeDistinctiveDescriptors();
        // c.更新该MapPoint平均观测方向以及观测距离的范围
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        // 步骤4.4：在地图中添加该MapPoint
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    // 步骤5：更新关键帧间的连接关系
    // 在3D点和关键帧之间建立边，每个边有一个权重，边的权重是该关键帧与当前帧公共3D点的个数
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    // 步骤5：BA优化
    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    // 步骤6：!!!将MapPoints的中值深度归一化到1，并归一化两帧之间变换
    // 评估关键帧场景深度，q=2表示中值
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    // x/z y/z 将z归一化到1 
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    // 把3D点的尺度也归一化到1
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    // 这部分和SteroInitialization()相似
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;// 初始化成功，至此，初始化过程完成
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
 //todo lan版本中和ORB一样，但是考虑到线，是不是在这个函数中加入类似线的操作
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

    // 步骤2：通过特征点的BoW加快当前帧与参考帧之间的特征点匹配
    // 特征点的匹配关系由MapPoints进行维护
    // todo 是不是先匹配然后把参考关键帧上的匹配点对应的地图点赋给vpMapPointMatches
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
    ///追踪关键帧的最重要的还是在这一步，搞清楚当前帧和参考关键帧怎么做2D与3D的匹配！

    if(nmatches<15)
        return false;

    // 步骤3:将上一帧的位姿态作为当前帧位姿的初始值
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    //vector存每个特征点对应的MapPoint, vpMapPointMatches的维度是和mCurrentFrame.N一致的。

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

    return nmatchesMap>=10; //匹配数大于10 说明追踪关键帧成功
}


/**
 * @brief 双目或rgbd摄像头根据深度值为上一帧产生新的MapPoints
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

    // todo 这里也一样对于线特征，是不是也要仿照上面的操作，生成一些临时的地图线


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
    LSDmatcher lmatcher;

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
    // --line-- todo 这里是不是也需要更新一下mvpMapLines lan版本中没做

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;

    // 步骤2：根据匀速度模型进行对上一帧的MapPoints进行跟踪
    // 根据上一帧特征点对应的3D点投影的位置缩小特征点匹配范围
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);  ///@ 看这个函数的具体实现

    // --line-- todo 检查
    int lmatches = lmatcher.SearchByProjection(mCurrentFrame, mLastFrame);

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
    double lmatch_ratio = lmatches*1.0/mCurrentFrame.mvKeylinesUn.size();
//    cout << "line match ratio in current frame: " << lmatch_ratio << endl;

    // --line--
    if(nmatches<20 || lmatch_ratio<0.5)
        return false;


    // Optimize frame pose with all matches
    // 步骤3：优化位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    // 步骤4：优化位姿后剔除outlier的mvpMapPoints
    int nmatchesMap = 0;
    //todo 思考如果不剔除的话影响其他操作吗，可能会吧比如当前帧被判断为关键帧后会生成地图点？
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

    // --line-- todo 同样滴 lan版本中这里没有去除误匹配线的操作 考虑是否增加


    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;   //mbVO为真说明当前帧上所找的匹配点过少，作为track函数会用到的一个标识。
        return nmatches>20;     //当前帧上的匹配点大于20 说明TrackWithMotionModel成功
    }

    return nmatchesMap>=10;   //位姿优化之后，当前帧的特征点与地图点的正确匹配数量
}



/**
 * @brief 对Local Map的MapPoints进行跟踪
 * 
 * 1. 更新局部地图，包括局部关键帧和关键点
 * 2. 对局部MapPoints进行投影匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return true if success
 * @see V-D track Local Map
 */
/// 该函数没有修改，在后面重写TrackLocalMapWithLine()
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    // Update Local KeyFrames and Local Points
    // 步骤1：更新局部关键帧mvpLocalKeyFrames和局部地图点mvpLocalMapPoints
    UpdateLocalMap();

    /// 步骤2：在局部地图中查找与当前帧匹配的MapPoints，这一步很重要！
    SearchLocalPoints();

    // Optimize Pose
    // 在这个函数之前，在Relocalization、TrackReferenceKeyFrame、TrackWithMotionModel中都有位姿优化，
    // 步骤3：更新局部所有MapPoints后对位姿再次优化
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    // 步骤3：更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            // 由于当前帧的MapPoints可以被当前帧观测到，其被观测统计量加1
            if(!mCurrentFrame.mvbOutlier[i])  //不是外点
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();  ///@ 这个地图点是属于当前帧的还是属于地图的?
                if(!mbOnlyTracking)
                {
                    // 该MapPoint被其它关键帧观测到过！
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;   //这是Tracking的数据成员，表示当前帧上的匹配数
                }
                else
                    // 记录当前帧跟踪到的MapPoints，用于统计跟踪效果。
                    //定位模式下，不管该地图点有没有被其他关键帧观测，不是外点，就算作内点匹配；slam模式中只有地图点不是外点且被其他关键帧观测过才算做内点匹配
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);  //啥意思，双目下该店是地图点且不是外点这里会被置空？

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    // 步骤4：决定是否跟踪成功
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)  //匹配数过少而且当前帧在重定位后的30帧以内追踪不成功
        return false;

    if(mnMatchesInliers<30) //匹配数非常少，追踪局部地图也不成功
        return false;
    else
        return true;
}



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
            else if(mSensor==System::STEREO)  //todo 这个双目是真的双目还是RGBD都算
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                //如果这个匹配点是外点而且是双目相机，就把当前帧上的点匹配置空
                // todo 这里是不是有问题：如果是RGBD相机的话，如果这个点是外点当前帧的点匹配不置空吗
                // 难道说，这里即使匹配的是外点，不置空也没有关系？

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
            else if(mSensor==System::STEREO)  //要不要这里把RGBD也加上
                mCurrentFrame.mvpMapLines[i] = static_cast<MapLine*>(NULL);
                //同上，也要存疑
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    // step7：决定是否跟踪成功。 距离重定位帧比较近的时候要求匹配足够大才认为是追踪成功
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    // todo 这里不需要写关于线的操作吗，不然上面算线的匹配数量干嘛  lan版本中没有写
    // 可以考虑将上面的判断条件改为 mnMatchesInliers+mnLineMatchesInliers < 50

    if(mnMatchesInliers<30)  //todo 同样的这里的判断条件也可以修改
        return false;
    else
        return true;
}


#if 0
/**
 * @brief 判断当前帧是否为关键帧
 * @return true if needed
 */

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


    //// wubo版本中下面这段代码根跟人家ORBSLAM中的代码不一样啊，坑！我们改用原版ORBSLAM2的代码

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
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        nMap++;// 这些都是临时生成的地图点
            }
        }
    }
    else
    {
        // There are no visual odometry matches in the monocular case
        nMap=1;
        nTotal=1;
    }

    const float ratioMap = (float)nMap/(float)(std::max(1,nTotal));  //

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
#endif



///********
bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth) //都是近点
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    // 建立2D-3D匹配的环节：
                    // 1. TrackLocalMap中SearchLocalPoints()
                    // 2. TrackWithMotionModel中SearchByProjection().其中先为上一帧产生了一些临时地图点
                    // 3. TrackRefrenceKeyFrame中SearchByBoW()

                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry(指nNonTrackedClose) compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
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
 * 对于非单目的情况，同时创建新的MapPoints
 */
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
        ///这里的目的是因为当前帧的点反投影得到3D点的时候需要知道当前帧的绝对位姿Tcw, 在下面的UnprojectStere函数中会用到

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


        //todo 同样的根据深度信息生成真正的地图线这里是不是也要写上！
        // 因为lan版本是针对单目所有这里没有写


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
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);  //SearchByProjection这个函数重载了四次！之前也调用过这个函数（RGBD情况），那是当前帧和上一帧追踪生成一些临时地图点
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
            } else{
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
        if(mCurrentFrame.isInFrustum(pML, 0.5))
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
            th=5;  //todo 这里的阈值和ORB中一样，但是这是线啊，这些参数可以调节为其他值试试

        matcher.SearchByProjection(mCurrentFrame, mvpLocalMapLines, th);
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
 */
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

        // 策略2.2:自己的子关键帧，关键帧的子关键帧是？
        const set<KeyFrame*> spChilds = pKF->GetChilds();  //todo 检查这个函数
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

        // 策略2.3:自己的父关键帧
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


// 没有变动，这里面没有关于线的内容
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

    if(mpInitializer)  //单目的初始化器　
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    //注意以下四个链表，是属于Tracking的随着时间不断维护的状态，最重要的目的是记录每一帧的位姿和关键帧，最终保存轨迹的时候用
    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();  //注意Viewer中RequesStop和Release的区别
}

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
