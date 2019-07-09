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



#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>		// std::setprecision

// 内联函数，检查是否有带后缀的文件  
bool has_suffix(const std::string &str, const std::string &suffix) {
  std::size_t index = str.find(suffix, str.size() - suffix.size());  //string自带的find函数
  return (index != std::string::npos);  // 如果找到了,find不返回npos，index != npos, TRUE     
}

namespace ORB_SLAM2
{

    //构造system构造函数的实现
System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const bool bUseViewer): mSensor(sensor), mbReset(false),mbActivateLocalizationMode(false), mbDeactivateLocalizationMode(false)
{
    // Output welcome message
    cout << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";

    if(mSensor==MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);  // 读取设置文件
    if(!fsSettings.isOpened())  // 如果没有打开，fs表示fstream文件流 
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }


    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    mpVocabulary = new ORBVocabulary();
    bool bVocLoad = false; // chose loading method based on file extension
    
    /// wubo修改部分
    if (has_suffix(strVocFile, ".txt"))  // 检查字典文件路径是否含有txt后缀  
	  bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);   
	else if(has_suffix(strVocFile, ".bin"))
	  bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);  //这里是DBoW自带的成员函数直接调用即可
	else
	  bVocLoad = false;
    if(!bVocLoad)  //如果bVocLoad为假
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Failed to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;
    

    ///Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);  //传入参数为字典类的对象

    ///Create the Map
    mpMap = new Map();

    ///Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);  //参数为指针和string

    ///Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);
    //Tracking类的构造含有8个参数,TODO_检查这里this指针的含义

    ///Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);  // 第二个参数为bool
    mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run,mpLocalMapper);
    ///其中的Run函数是不带参数的，解释：
    ///std::thread t5(&A::f, &a, 8, 'w'); 调用的是 a.f(). 第二个参数为指针

    ///Initialize the Loop Closing thread and launch
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);
    mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);

    ///Initialize the Viewer thread and launch
    mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
    if(bUseViewer)
    {
        mptViewer = new thread(&Viewer::RunWithLine, mpViewer);
        //todo 这里改为了RunWithLine函数，如果程序不成功要检查这个函数实现的是不是有问题
    }

    mpTracker->SetViewer(mpViewer);

    ///Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);  //设置指针，也就是传入实参，三个主线程互相设置下
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);

}  //System构造函数


/**
 * RGBD模式的追踪函数
 */
cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp)
{
    if(mSensor!=RGBD)
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }    

    // Check mode change.
    {
        /*
         * 解释unique_lock：
         * std::lock_guard在构造函数中加锁，在析构函数中解锁，因此完成了自动解锁的功能不会出现使用mutex lock时忘了解锁而导致死锁
         * 类 unique_lock 是通用互斥包装器，允许延迟锁定、锁定的有时限尝试、递归锁定、所有权转移与条件变量一同使用。unique_lock比lock_guard使用更加灵活，功能更加强大。
         * 使用unique_lock需要付出更多的时间、性能成本。这里可以考虑把unique_lock改为lock_guard测试一下时间
         */
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())  //调用while循环一直到localmapping关闭
            {
                //usleep(1000);
				std::this_thread::sleep_for(std::chrono::milliseconds(1));  //当前线程也就是主线程睡眠1毫秒
			}

            mpTracker->InformOnlyTracking(true);  //会设置tracking中的很多bool标识
            mbActivateLocalizationMode = false;   //如果开启了定位模式，则把相应的很多标识改为true，很多函数就不会执行了。然后把定位模式的标识置为0
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)   //紫色说明为类System的数据成员
    {
        mpTracker->Reset();  //System类中reset为真 之后调用Tracking类中的Reset函数
        mbReset = false;
    }
    }

    return mpTracker->GrabImageRGBD(im,depthmap,timestamp);
    //获取当前帧的Tcw, 世界到相机的位姿。返回类型为 return cv::Mat

#if 0  //这部分代码先跳过，暂时没用，调试的时候可用来看点线追踪的情况
    cv::Mat Tcw = mpTracker->GrabImageRGBD(im,depthmap,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;  //可以得到追踪的特征点
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    mTrackedMapLines = mpTracker->mCurrentFrame.mvpMapLines;  //仿照KeyPoint，自己添加的
    mTrackedKeyLines = mpTracker->mCurrentFrame.mvKeylinesUn;  //仿照KeyPoint，自己添加的
    return Tcw;
#endif

}


//-----------以下三个函数实现类似：完成加锁操作 @加强理解--------
void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

//-------------------------------------------------------

void System::Shutdown()
{
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        //usleep(5000); 可见usleep参数的单位为微妙，最小单位
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}

    if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");  //@@
}

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);  //第三个参数为比较函数，根据keyframe的id来排序

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();  //vpKFs[0]存储第一个关键帧的位姿 Trw

    ofstream f;  //输出流，从程序中输出
    f.open(filename.c_str());  //如果没有该文件会创建
    f << fixed;  //以一般的小数方式输出浮点数而不是科学计数法

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();

    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)   //列表中的pose和列表中参考关键帧都是对应着的
    {
        if(*lbL)  //tracking failure
            continue;

        KeyFrame* pKF = *lRit;  //pKF是关键帧类指针，表示一个关键帧

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);  //4*4单位阵 32位浮点型

        // If the reference keyframe was culled（剔除）, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())  //
        {
            Trw = Trw*pKF->mTcp;  //Tcp是相对于父节点的位姿，所以这里相当于Trp
            pKF = pKF->GetParent();  //相当于Tpw
        }

        Trw = Trw*pKF->GetPose()*Two;  //Two为世界原点到第一个关键帧,现在要求世界原点到当前关键帧(ref)的位姿，前两者相乘就是Trw
                                       //相当于 Tro = Trw * Two

        cv::Mat Tcw = (*lit)*Trw;     //相当于 Tcw = Tcr * Trw
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);  //求逆之后平移变为 -Rt

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}


void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;  //fixed为文件流f设置属性 禁止使用科学计数法表示浮点数

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        //GetRotation返回的是Tcw, Mat类型，所以可以调用.t()表示转置
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        //TUM数据集中的 qx qy qz qw 是相对于世界的旋转. Twc
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}


//------点云部分，这部分需要加强理解！最终得到点云图，暂时跳过-----------------------------

/*

void System::SavePointCloud(const string &filename)
{
    cout << endl << "Saving PointCloud Map to " << filename << " ..." << endl;

    vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();
    vector<MapLine*> vpMLs = mpMap->GetAllMapLines();

    ofstream f;
    f.open(filename.c_str());

    f << "ply"
      << endl << "format ascii 1.0"
      << endl << "element vertex " << vpMPs.size()
      << endl << "property float x"
      << endl << "property float y"
      << endl << "property float z"
      << endl << "property uchar red"
      << endl << "property uchar green"
      << endl << "property uchar blue"
      << endl << "end_header" << endl;

    f << fixed << setprecision(std::numeric_limits<double>::digits10+1);

    for(size_t i=0; i<vpMPs.size(); i++)
    {
        if(vpMPs[i]->isBad())
            continue;
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        f << pos.at<float>(0) << " "
          << pos.at<float>(1) << " "
          << pos.at<float>(2) << " "
          << "255 255 255" << "\n";
    }

    f.close();
    cout << endl << "PointCloud Map saved! " << endl;
}

void System::ShowPointCloud()
{
    typedef pcl::PointXYZ PointT;
    typedef pcl::PointCloud<PointT> pointCloud;

    vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();
    vector<MapLine*> vpMLs = mpMap->GetAllMapLines();

    pointCloud::Ptr cloud(new pointCloud);
    cloud->points.resize(vpMPs.size());
    for(size_t i=0; i<cloud->size(); i++)
    {
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        cloud->points[i].x = pos.at<float>(0);
        cloud->points[i].y = pos.at<float>(1);
        cloud->points[i].z = pos.at<float>(2);
    }

    // 显示点云
    pcl::visualization::PCLVisualizer *p(new pcl::visualization::PCLVisualizer("display the 3Dpoints"));
    p->addPointCloud(cloud);
    p->addCoordinateSystem();
    p->setBackgroundColor(0.19, 0.19, 0.19);

    //-- 将线段的3D端点定义为点,然后划线
    PointT p0, p1;
    stringstream s1;
    for(size_t i=0; i<vpMLs.size(); i++)
    {
        Vector6d pos = vpMLs[i]->GetWorldPos();
        s1 << i;
        p0.x = pos(0);
        p0.y = pos(1);
        p0.z = pos(2);
        p1.x = pos(3);
        p1.y = pos(4);
        p1.z = pos(5);
        p->addLine(p0, p1, s1.str());
    }

    p->spin();
}
*/

} //namespace ORB_SLAM
