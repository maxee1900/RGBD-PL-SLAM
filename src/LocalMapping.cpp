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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const bool &bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()  // 这是主函数
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        // 告诉Tracking，LocalMapping正处于繁忙状态，
        // LocalMapping线程处理的关键帧都是Tracking线程发过的
        // 在LocalMapping线程还没有处理完关键帧之前Tracking线程最好不要发送太快
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        // 等待处理的关键帧列表不为空
        if(CheckNewKeyFrames())  //为true，说明队列中有关键帧
        {
            // BoW conversion and insertion in Map
            // VI-A keyframe insertion

            // 计算关键帧特征点的BoW映射，将关键帧插入地图
//            ofstream file1("KeyFrameInsertionTime.txt", ios::app);
//            chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
            ProcessNewKeyFrame();
//            chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//            chrono::duration<double> time_used1 = chrono::duration_cast<chrono::duration<double>>(t2-t1);
//            cout << "insertKF time: " << time_used1.count() << endl;
//            file1 << time_used1.count() << endl;
//            file1.close();

            // Check recent MapPoints
            // VI-B recent map points culling
            // 剔除ProcessNewKeyFrame函数中引入的不合格MapPoints
            //MapPointCulling();

//            ofstream file2("FeatureCullingTime.txt", ios::app);
//            chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
//            MapPointCulling();
//            MapLineCulling();
            thread threadCullingPoint(&LocalMapping::MapPointCulling, this);
            thread threadCullingLine(&LocalMapping::MapLineCulling, this);
            threadCullingPoint.join();
            threadCullingLine.join();
//            chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
//            chrono::duration<double> time_used2 = chrono::duration_cast<chrono::duration<double>>(t4-t3);
//            cout << "CullMapPL time: " << time_used2.count() << endl;
//            file2 << time_used2.count() << endl;
//            file2.close();


            // Triangulate new MapPoints
            // VI-C new map points creation
            // 相机运动过程中与相邻关键帧通过三角化恢复出一些MapPoints
            //CreateNewMapPoints();

//            ofstream file3("FeaturesCreationTime.txt", ios::app);
//            chrono::steady_clock::time_point t5 = chrono::steady_clock::now();

//            CreateNewMapPoints();
//            CreateNewMapLines2();   //采用极平面方式

            thread threadCreateP(&LocalMapping::CreateNewMapPoints, this);
            thread threadCreateL(&LocalMapping::CreateNewMapLines2, this);
            threadCreateP.join();
            threadCreateL.join();

//            chrono::steady_clock::time_point t6 = chrono::steady_clock::now();
//            chrono::duration<double> time_used3 = chrono::duration_cast<chrono::duration<double>>(t6-t5);
//            cout << "CreateMapPL time: " << time_used3.count() << endl;
//            file3 << time_used3.count() << endl;
//            file3.close();



            // 已经处理完队列中的最后的一个关键帧
            if(!CheckNewKeyFrames())  //队列为空
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                // 检查并融合当前关键帧与相邻帧（两级相邻）重复的MapPoints   和MapLines?
                //@@ 注意这里当前关键帧和两级相邻的概念！　
                SearchInNeighbors();  //todo 这个地方要不要加上SearchLineInNeighbors
            }

            mbAbortBA = false;  //中止BA

            // 已经处理完队列中的最后的一个关键帧，并且闭环检测没有请求停止LocalMapping
            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // VI-D Local BA
                if(mpMap->KeyFramesInMap()>2)
                {
//                    ofstream file4("LocalBATime.txt", ios::app);
//                    chrono::steady_clock::time_point t7 = chrono::steady_clock::now();
                    Optimizer::LocalBundleAdjustmentWithLine(mpCurrentKeyFrame, &mbAbortBA, mpMap);   ///包含了线特征的BA
//                    chrono::steady_clock::time_point t8 = chrono::steady_clock::now();
//                    chrono::duration<double> time_used4 = chrono::duration_cast<chrono::duration<double>>(t8-t7);
//                    cout << "LocalBA time: " << time_used4.count() << endl;
//                    file4 << time_used4.count() << endl;
//                    file4.close();

                    // Check redundant local Keyframes
                    // VI-E local keyframes culling
                    // 检测并剔除当前帧相邻的关键帧中冗余的关键帧
                    // 剔除的标准是：该关键帧的90%的MapPoints可以被其它关键帧观测到
                    // trick!
                    // Tracking中先把关键帧交给LocalMapping线程
                    // 并且在Tracking中InsertKeyFrame函数的条件比较松，交给LocalMapping线程的关键帧会比较密
                    // 在这里再删除冗余的关键帧
//                ofstream file5("KFCullingTime.txt",ios::app);
//                chrono::steady_clock::time_point t9 = chrono::steady_clock::now();
                    KeyFrameCulling();
//                chrono::steady_clock::time_point t10 = chrono::steady_clock::now();
//                chrono::duration<double> time_used5 = chrono::duration_cast<chrono::duration<double>>(t10-t9);
//                cout << "CullKF time: " << time_used5.count() << endl;
//                file5 << time_used5.count() << endl;
//                file5.close();

                }
            }

            // 将当前帧加入到闭环检测队列中,说明当前帧已经是地图中的关键帧了
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())  //如果局部地图停止，也就是空闲
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())  //一直睡眠直到CheckFinish为真
            {
                // usleep(3000); //3毫秒即3000微妙，调用这两个函数都可以
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            }
            if(CheckFinish())
                //如果结束了break退出大while循环，while是死循环，只有在break的地方退出
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is not busy
        SetAcceptKeyFrames(true);  //告诉Tracking线程可以接受关键帧

        if(CheckFinish())
            break;

        //usleep(3000);
        std::this_thread::sleep_for(std::chrono::milliseconds(3)); //3毫秒就是3000微秒

    }

    SetFinish();
    //将该局部地图类的数据成员mbStopped、mbFinished置为true
}


/**
 * @brief 插入关键帧
 *
 * 将关键帧插入到地图中，以便将来进行局部地图优化
 * 这里仅仅是将关键帧插入到列表中进行等待
 * @param pKF KeyFrame
 */
void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    // 将关键帧插入到列表中
    mlNewKeyFrames.push_back(pKF);  //注意局部地图中很关键的一个链表(列表)： mlNewKeyFrames
    mbAbortBA=true;  //插入关键帧的时候停止BA
}

/**
 * @brief 查看列表中是否有等待被插入的关键帧
 * @return 如果存在，返回true
 */
bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty()); //不为空函数返回真
}

/**
 * @brief 处理列表中的关键帧
 * 
 * - 计算Bow，加速三角化新的MapPoints
 * - 关联当前关键帧至MapPoints，并更新MapPoints的平均观测方向和观测距离范围
 * - 插入关键帧，更新Covisibility图和Essential图
 * @see VI-A keyframe insertion
 */
void LocalMapping::ProcessNewKeyFrame()
{
    // 步骤1：从缓冲队列中取出一帧关键帧
    // Tracking线程向LocalMapping中插入关键帧存在该队列中
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        // 从列表中获得一个等待被插入的关键帧
        mpCurrentKeyFrame = mlNewKeyFrames.front();  //链表作为队列来用 ，当前关键帧的写入
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    // 步骤2：计算该关键帧特征点的Bow映射关系
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    // 步骤3：跟踪局部地图过程中新匹配上的MapPoints和当前关键帧绑定
    // 在TrackLocalMap函数中将局部地图中的MapPoints与当前帧进行了匹配，
    // 但没有对这些匹配上的MapPoints与当前帧进行关联! 这句话很好！
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad()) //todo_ 详细思考这段代码的逻辑，很重要！暂时先按ORB中来仿写
            {
                // 非当前帧生成的MapPoints
				// 为当前帧在tracking过程跟踪到的MapPoints更新属性
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                    //该地图点的观测mObservations中没有当前关键帧。这些点是不会经历Culling的
                    ///该地图点的观测中没有当前关键帧，说明该地图点只是和当前关键帧上的特征点进行了匹配，接下来，要进行关联，好更新该地图点的属性。这些地图点是已有的不需要进行Culling
                {
                    // 添加观测
                    pMP->AddObservation(mpCurrentKeyFrame, i); //i为该地图点在当前帧上的索引
                    // 获得该点的平均观测方向和观测距离范围
                    pMP->UpdateNormalAndDepth();  //更新地图点的属性：方向和距离
                    // 加入关键帧后，更新3d点的最佳描述子
                    pMP->ComputeDistinctiveDescriptors();  //更新地图点的描述子
                }
                else /// this can only happen for new stereo points inserted by the Tracking. 在Tracking::CreateNewKeyFrame()创建了地图点并更新了地图点的观测，这里如果地图点的观测中有当前关键帧，说明这个地图点是在CreateNewKeyFrame()中生成的，这是双目的特点，因此这些点是新生成的需要进行Culling
                {
                    // 当前帧生成的MapPoints
                    // 将双目或RGBD跟踪过程中新插入的MapPoints放入mlpRecentAddedMapPoints，等待检查//todo_ 双目追踪插入的关键点为什么满足上述条件
                    // CreateNewMapPoints函数中通过三角化也会生成MapPoints
                    // 这些MapPoints都会经过MapPointCulling函数的检验
                    mlpRecentAddedMapPoints.push_back(pMP);

                }
            }
        }
    }



    //  ---line---
    /// 跟踪局部地图过程中新匹配上的MapLines,和当前关键帧进行绑定
    const vector<MapLine*> vpMapLineMatches = mpCurrentKeyFrame->GetMapLineMatches();

    for(size_t i=0; i<vpMapLineMatches.size(); i++)
    {
        MapLine* pML = vpMapLineMatches[i];
        if(pML)
        {
            if(!pML->isBad())
            {
                if(!pML->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pML->AddObservation(mpCurrentKeyFrame, i);  //添加观测
                    pML->UpdateAverageDir();    //更新观测方向
                    pML->ComputeDistinctiveDescriptors();
                }
                else
                {
                    mlpRecentAddedMapLines.push_back(pML);
                }
            }
        }
    }


    // Update links in the Covisibility Graph
    // 步骤4：更新关键帧间的连接关系，Covisibility图和Essential图(tree)
    mpCurrentKeyFrame->UpdateConnections(); //此处只与MapPoint有关，暂时不修改。（lan）todo

    // Insert Keyframe in Map
    // 步骤5：将该关键帧插入到地图中
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}


/**
 * @brief 剔除ProcessNewKeyFrame和CreateNewMapPoints函数中引入的质量不好的MapPoints
 *
 *
 * @see VI-B recent map points culling
 */
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;  //RGBD这个参数为3，地图点的观测小于等于3就剔除
    const int cnThObs = nThObs;
	
	// 遍历等待检查的MapPoints,遍历最近添加的点
    while(lit!=mlpRecentAddedMapPoints.end())  //剔除的都是最近新增的点
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            // 步骤1：已经是坏点的MapPoints直接从检查链表中删除
            lit = mlpRecentAddedMapPoints.erase(lit);  //删除后指针指向下一位置，没有++
        }
        else if(pMP->GetFoundRatio()<0.25f)
        {
            // 步骤2：将不满足VI-B条件的MapPoint剔除
            // VI-B 条件1：
            // 跟踪到该MapPoint的Frame数相比预计可观测到该MapPoint的Frame数的比例需大于25%
            // IncreaseFound / IncreaseVisible < 25%，注意不一定是关键帧。
            //这里就要注意Found和Visible是怎么来的

            //注意：IncreaseFound和IncreaseVisible是在怎么增加的
            pMP->SetBadFlag();  //会把该点设置为坏点
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            // 步骤3：将不满足VI-B条件的MapPoint剔除。简单的说这个地图点观测的关键帧过少
            // VI-B 条件2：从该点建立开始，到现在已经过了不小于2个关键帧
            // 但是观测到该点的关键帧数却不超过cnThObs帧，那么该点检验不合格
            pMP->SetBadFlag();  //该地图点设置为坏点
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            // 步骤4：从建立该点开始，已经过了3个关键帧而没有被剔除，则认为是质量高的点
            // 因此没有SetBadFlag()，仅从队列中删除，放弃继续对该MapPoint的检测
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}


//// --line--  完全按照剔除点的框架
void LocalMapping::MapLineCulling()
{
    // Check Recent Added MapLines
    list<MapLine*>::iterator lit = mlpRecentAddedMapLines.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    // 遍历等待检查的MapLines
    while(lit!=mlpRecentAddedMapLines.end())  //todo 这里一定要check最近新增加的线是怎么来的
    {
        MapLine* pML = *lit;
        if(pML->isBad())
        {
            // step1: 将已经是坏的MapLine从检查链中删除
            lit = mlpRecentAddedMapLines.erase(lit);
        }
        else if(pML->GetFoundRatio()<0.25f)  //这个参数也没改
        {
            pML->SetBadFlag();
            lit = mlpRecentAddedMapLines.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pML->mnFirstKFid)>=2 && pML->Observations()<=cnThObs)
        {
            pML->SetBadFlag();
            lit = mlpRecentAddedMapLines.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pML->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapLines.erase(lit);
        else
            lit++;
    }
}


/**
 * 相机运动过程中和共视程度比较高的关键帧通过三角化恢复出一些MapPoints
 */
void LocalMapping::CreateNewMapPoints()  ///该函数相当重要，思考RGBD生成地图点的方式
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;   //RGBD参数
    if(mbMonocular)
        nn=20;

    // 步骤1：在当前关键帧的共视关键帧中找到共视程度最高的nn帧相邻帧vpNeighKFs
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);  //得到最佳共视关键帧

    ORBmatcher matcher(0.6,false);  //todo_ check第二个参数 这里不检查旋转一致性
    // :第二个参数为检查方向吗，所谓的旋转一致性，检查的话会看旋转角度是否为前三大主要的旋转角区间

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F); //注意这里是三行四列，不是严格的转换矩阵
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));

    // 得到当前关键帧在世界坐标系中的坐标
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter(); //tcw

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    // 步骤2：遍历相邻关键帧vpNeighKFs
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())  //如果有关键帧待处理函数返回
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        // 邻接的关键帧在世界坐标系中的坐标
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        // 基线向量，两个关键帧间的相机位移
        cv::Mat vBaseline = Ow2-Ow1;
        // 基线长度
        const float baseline = cv::norm(vBaseline);

        // 步骤3：判断相机运动的位移是不是足够长（相邻关键帧与当前关键帧相机光心位移）
        if(!mbMonocular)
        {
            /// 如果是立体相机，关键帧间距太小时不生成3D点
            if(baseline<pKF2->mb) //todo_ 两关键帧之间的相机位移小于关键帧的基线baseline时跳过，这里这个参数可以修改
            continue;
        }
        else  //单目相机
        {
            // 邻接关键帧的场景深度中值 //可以理解为该关键帧中地图点的平均深度
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            // baseline与景深的比例
            const float ratioBaselineDepth = baseline/medianDepthKF2;
            // 如果地图点距离帧特别远(比例特别小)，那么不考虑该帧，不生成3D点，也可理解为baseline太小不生成3D点
            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        // 步骤4：根据两个关键帧的位姿计算它们之间的基本矩阵
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        // 步骤5：通过极线约束限制匹配时的搜索范围，进行特征点匹配
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false); //两个关键帧之间找匹配

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        // 步骤6：对每对匹配通过三角化生成3D点,he Triangulate函数差不多
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            // 步骤6.1：取出匹配特征点

            // 当前匹配对在当前关键帧中的索引
            const int &idx1 = vMatchedIndices[ikp].first;
            
            // 当前匹配对在邻接关键帧中的索引
            const int &idx2 = vMatchedIndices[ikp].second;

            // 当前匹配在当前关键帧中的特征点
            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            // mvuRight中存放着双目的深度值，如果不是双目，其值将为-1
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            // 当前匹配在邻接关键帧中的特征点
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            // mvuRight中存放着双目的深度值，如果不是双目，其值将为-1
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            // 步骤6.2：利用匹配点反投影得到视差角
            // 特征点反投影 这两个是归一化坐标
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            // 由相机坐标系转到世界坐标系，得到视差角余弦值
            cv::Mat ray1 = Rwc1*xn1;  //注意这里没有twc1
            cv::Mat ray2 = Rwc2*xn2;
            assert(cv::norm(ray1)*cv::norm(ray2) != 0);
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            // 加1是为了让cosParallaxStereo随便初始化为一个很大的值
            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            // 步骤6.3：对于双目，利用双目得到视差角
            if(bStereo1)//双目，且有深度
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
                 //这里通过画图可以得到公式，有一点近似的意味，中线不是严格的角平分线
            else if(bStereo2)//双目，且有深度
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            // 得到双目观测的视差角
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);
            /// 这个视差角指的是：两次观测中，较小的那个视差角，对应一个帧的视差角；单目则是两个帧构成的视差角

            // 步骤6.4：三角化恢复3D点
            cv::Mat x3D;
            // cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998)表明视差角正常
            // cosParallaxRays<cosParallaxStereo表明视差角很小
            /// 视差角度小时用三角法恢复3D点，视差角大时用双目恢复3D点（双目以及深度有效）
            // 单目视差角cos较小，单目视差角较大，是不是理解为平移越大，用三角法恢复3D点；单目视差角较小的时候用深度相机恢复。这个有一个参数选择的问题 todo 因为上一个todo位置将平移过小的情况跳过了
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                // 见Initializer.cpp的Triangulate函数
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();  //vt的第三行向量的转置

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2) //cos越小，说明夹角越大，说明点越近，深度越有效
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1); //这里用的是有畸变的rgb坐标
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            // 步骤6.5：检测生成的3D点是否在相机前方（两个相机都检查）
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);  //常规操作
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            // 步骤6.6：计算3D点在当前关键帧下的重投影误差
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);  //没毛病
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;  //是双目点同时检查右目坐标
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            // 计算3D点在另一个关键帧下的重投影误差
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave]; //所在的层数可能不一样
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                // 基于卡方检验计算出的阈值（假设测量有一个一个像素的偏差）
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            // 步骤6.7：检查尺度连续性  牛逼，三角化生成地图点很严格

            // 世界坐标系下，3D点与相机间的向量，方向由相机指向3D点
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            // ratioDist是不考虑金字塔尺度下的距离比例
            const float ratioDist = dist2/dist1;
            // 金字塔尺度因子的比例
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            // ratioDist*ratioFactor < ratioOctave 或 ratioDist/ratioOctave > ratioFactor表明尺度变化是连续的
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;  //这里为什么有待理解

            // Triangulation is succesfull
            // 步骤6.8：三角化生成3D点成功，构造成MapPoint
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

            // 步骤6.9：为该MapPoint添加属性：
            // a.1 该地图点增加对两个关键帧的观测
            // a.2 两个关键帧增加对该地图点的观测
            // b.该MapPoint的描述子
            // c.该MapPoint的平均观测方向和深度范围
            // d. 地图中增加该地图点
            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);

            // 步骤6.8：将新产生的点放入检测队列
            // 这些MapPoints都会经过MapPointCulling函数的检验
            mlpRecentAddedMapPoints.push_back(pMP);
            //原来这个数据成员在这里写入的，地图点的添加一共两处还有一处是在关键帧处理函数中

            nnew++;
        }
    }
}


//// ********************第一种：三角化端点生成线，误差较大**********************
//// ********************这个函数是最难写的***********************************

void LocalMapping::CreateNewMapLines1()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn=10;
    if(mbMonocular)
        nn=20;
    //step1：在当前关键帧的共视关键帧中找到共视成都最高的nn帧相邻帧vpNeighKFs
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    LSDmatcher lmatcher;

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));

    //得到当前关键帧在世界坐标系中的坐标
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew = 0;

    // Search matches with epipolar restriction and triangulate
    // step2: 遍历相邻关键帧vpNeighKFs
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        // 邻接的关键帧在世界坐标系中的坐标
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        // 基线向量，两个关键帧间的相机位移
        cv::Mat vBaseline = Ow2 - Ow1;
        // 基线长度
        const float baseline = cv::norm(vBaseline);

        // step3：判断相机运动的基线是不是足够长
        if(!mbMonocular)
        {
            // 如果是立体相机，关键帧间距太小时不生成3D点
            if(baseline<pKF2->mb)
                continue;
        }
        else
        {
            // 邻接关键帧的场景深度中值
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            // baseline 与景深的比例
            const float ratioBaselineDepth = baseline/medianDepthKF2;
            // 如果特别远（比例特别小），那么不考虑当前邻接的关键帧，不生成3D点
            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        // step4：根据两个关键帧的位姿计算它们之间的基本矩阵
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

        // Search matches that fulfill epipolar constraint
        // step5：通过极线约束限制匹配时的搜索单位，进行特征点匹配
        vector<pair<size_t, size_t>> vMatchedIndices;
        lmatcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, vMatchedIndices, false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3, 4, CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        // step6：对每对匹配通过三角化生成3D点
        const int nmatches = vMatchedIndices.size();
        for(int ikl=0; ikl<nmatches; ikl++)
        {
            // step6.1：取出匹配的特征线
            const int &idx1 = vMatchedIndices[ikl].first;
            const int &idx2 = vMatchedIndices[ikl].second;

            const KeyLine &keyline1 = mpCurrentKeyFrame->mvKeyLines[idx1];
            const KeyLine &keyline2 = pKF2->mvKeyLines[idx2];
            const Vector3d keyline2_function = pKF2->mvKeyLineFunctions[idx2];

            // todo 这里用的线段中点实际上我觉得用端点更合适
            // 特征线段的中点
            Point2f midP1 = keyline1.pt;
            Point2f midP2 = keyline2.pt;
            // step6.2:将两个线段的中点反投影得到视差角
            // 特征线段的中点反投影
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (midP1.x - cx1)*invfx1, (midP1.y - cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (midP2.x - cx2)*invfx2, (midP2.y - cy2)*invfy2, 1.0);

            // 由相机坐标系转到世界坐标系，得到视差角余弦值
            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));
//            cosParallaxRays = cosParallaxRays + 1;

            // todo 这里没有取检测是否有双目点，改成RGBD的话需要在这里做很大的修改

            // step6.3：线段端点在两帧图像中的坐标
            cv::Mat StartC1, EndC1, StartC2, EndC2;
            StartC1 = (cv::Mat_<float>(3,1) << (keyline1.startPointX-cx1)*invfx1, (keyline1.startPointY-cy1)*invfy1, 1.0);
            EndC1 = (cv::Mat_<float>(3,1) << (keyline1.endPointX-cx1)*invfx1, (keyline1.endPointY-cy1)*invfy1, 1.0);
            StartC2 = (cv::Mat_<float>(3,1) << (keyline2.startPointX-cx2)*invfx2, (keyline2.startPointY-cy2)*invfy2, 1.0);
            EndC2 = (cv::Mat_<float>(3,1) << (keyline2.endPointX-cx2)*invfx2, (keyline2.endPointY-cy2)*invfy2, 1.0);

            // step6.4：三角化恢复线段的3D端点
            cv::Mat s3D, e3D;
            if(cosParallaxRays>0 && cosParallaxRays<0.9998)
            {
                cv::Mat A(4,4,CV_32F);
                A.row(0) = StartC1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = StartC1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = StartC2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = StartC2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w1, u1, vt1;
                cv::SVD::compute(A, w1, u1, vt1, cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                s3D = vt1.row(3).t();

                if(s3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                s3D = s3D.rowRange(0,3)/s3D.at<float>(3);

                cv::Mat B(4,4,CV_32F);
                B.row(0) = EndC1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                B.row(1) = EndC1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                B.row(2) = EndC2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                B.row(3) = EndC2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w2, u2, vt2;
                cv::SVD::compute(B, w2, u2, vt2, cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                e3D = vt2.row(3).t();

                if(e3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                e3D = e3D.rowRange(0,3)/e3D.at<float>(3);
            } else
                continue;

            cv::Mat s3Dt = s3D.t();
            cv::Mat e3Dt = e3D.t();

            // step6.5：检测生成的3D点是否在相机前方
            float SZC1 = Rcw1.row(2).dot(s3Dt)+tcw1.at<float>(2);   //起始点在C1下的Z坐标值
            if(SZC1<=0)
                continue;

            float SZC2 = Rcw2.row(2).dot(s3Dt)+tcw2.at<float>(2);   //起始点在C2下的Z坐标值
            if(SZC2<=0)
                continue;

            float EZC1 = Rcw1.row(2).dot(e3Dt)+tcw1.at<float>(2);   //终止点在C1下的Z坐标值
            if(EZC1<=0)
                continue;

            float EZC2 = Rcw2.row(2).dot(e3Dt)+tcw2.at<float>(2);   //终止点在C2下的Z坐标值
            if(EZC2<=0)
                continue;

            // step6.6：计算3D点在当前关键帧下的重投影误差
            const float &sigmaSquare2 = mpCurrentKeyFrame->mvLevelSigma2[keyline1.octave];
            // -1.该keyline在当前帧中所在直线系数
            Vector3d lC1 = mpCurrentKeyFrame->mvKeyLineFunctions[idx1];

            // -2.起始点在当前帧的重投影误差
            const float x1 = Rcw1.row(0).dot(s3Dt) + tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(s3Dt) + tcw1.at<float>(1);
            float e1 = lC1(0)*x1 + lC1(1)*y1 + lC1(2);

            // -3.终止点在当前帧的重投影误差
            const float x2 = Rcw1.row(0).dot(e3Dt) + tcw1.at<float>(0);
            const float y2 = Rcw1.row(1).dot(e3Dt) + tcw1.at<float>(1);
            float e2 = lC1(0)*x2 + lC1(1)*y2 + lC1(2);

            // -4.判断线段在当前帧的重投影误差是否符合阈值
            float eC1 = e1 + e2;
            if(eC1>7.8*sigmaSquare2)    ///Q:7.8是仿照CreateMapPoints()函数中的双目来的，这里需要重新计算
                continue;

            // step6.7：计算3D点在另一个关键帧下的重投影去查
            // -1.该keyline在pKF2中所在直线系数
            Vector3d lC2 = pKF2->mvKeyLineFunctions[idx2];

            // -2.起始点在当前帧的重投影误差
            const float x3 = Rcw2.row(0).dot(s3Dt) + tcw2.at<float>(0);
            const float y3 = Rcw2.row(1).dot(s3Dt) + tcw2.at<float>(1);
            float e3 = lC2(0)*x3 + lC2(1)*y3 + lC2(2);  //算的是点线之间的距离

            // -3.终止点在当前帧的重投影误差
            const float x4 = Rcw2.row(0).dot(e3Dt) + tcw2.at<float>(0);
            const float y4 = Rcw2.row(1).dot(e3Dt) + tcw2.at<float>(1);
            float e4 = lC2(0)*x4 + lC2(1)*y3 + lC2(2);

            // -4.判断线段在当前帧的重投影误差是否符合阈值
            float eC2 = e3 + e4;
            if(eC1>7.8*sigmaSquare2)
                continue;

            // step6.8:检测尺度连续性
            cv::Mat middle3D = 0.5*(s3D+e3D);
            // 世界坐标系下，线段3D中点与相机间的向量，方向由相机指向3D点
            cv::Mat normal1 = middle3D - Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = middle3D - Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            // ratioDist是不考虑金字塔尺度下的距离比例
            const float ratioDist = dist2/dist1;
            // 金字塔尺度因子的比例
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[keyline1.octave]/pKF2->mvScaleFactors[keyline2.octave];
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // step6.9: 三角化成功，构造MapLine
//            cout << "三角化的线段的两个端点： " << "\n \t" << s3Dt << "\n \t" << e3Dt << endl;
//            cout << s3D.at<float>(0) << ", " << s3D.at<float>(1) << ", " << s3D.at<float>(2) << endl;
            Vector6d line3D;
            line3D << s3D.at<float>(0), s3D.at<float>(1), s3D.at<float>(2), e3D.at<float>(0), e3D.at<float>(1), e3D.at<float>(2);
            MapLine* pML = new MapLine(line3D, mpCurrentKeyFrame, mpMap);

            // step6.10：为该MapLine添加属性
            pML->AddObservation(mpCurrentKeyFrame, idx1);
            pML->AddObservation(pKF2, idx2);

            mpCurrentKeyFrame->AddMapLine(pML, idx1);
            pKF2->AddMapLine(pML, idx2);

            pML->ComputeDistinctiveDescriptors();
            pML->UpdateAverageDir();
            mpMap->AddMapLine(pML);

            // step6.11：将新产生的线特征放入检测队列，这些MapLines都会经过MapLineCulling函数的检验
            mlpRecentAddedMapLines.push_back(pML);

            nnew++;
        }
    }

}


//// ********************第二种：通过极平面生成线**********************
void LocalMapping::CreateNewMapLines2()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn=10;
    if(mbMonocular)
        nn=20;
    //step1：在当前关键帧的共视关键帧中找到共视成都最高的nn帧相邻帧vpNeighKFs
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    LSDmatcher lmatcher;    //建立线特征匹配

    // 获取当前帧的转换矩阵
    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));

    //得到当前关键帧在世界坐标系中的坐标
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    //获取当前帧的相机内参
    const Mat &K1 = mpCurrentKeyFrame->mK;
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew = 0;

    // Search matches with epipolar restriction and triangulate
    // step2: 遍历相邻关键帧vpNeighKFs
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        // 邻接的关键帧在世界坐标系中的坐标
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        // 基线向量，两个关键帧间的相机位移
        cv::Mat vBaseline = Ow2 - Ow1;
        // 基线长度
        const float baseline = cv::norm(vBaseline);

        // step3：判断相机运动的基线是不是足够长
        if(!mbMonocular)
        {
            // 如果是立体相机，关键帧间距太小时不生成3D点
            if(baseline<pKF2->mb)
                continue;
        }
        else
        {
            // 邻接关键帧的场景深度中值
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            // baseline 与景深的比例
            assert(medianDepthKF2 != 0);
            const float ratioBaselineDepth = baseline/medianDepthKF2;
            // 如果特别远（比例特别小），那么不考虑当前邻接的关键帧，不生成3D点
            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        // step4：根据两个关键帧的位姿计算它们之间的基本矩阵
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

        // Search matches that fulfill epipolar constraint
        // step5：通过极线约束限制匹配时的搜索单位，进行特征点匹配
        vector<pair<size_t, size_t>> vMatchedIndices;
        lmatcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, vMatchedIndices, false);
        // todo 要check上面这个函数实现的有没有问题

        // TODO RGBD相机：一种方式是通过深度直接得到点 另一种方式是通过极平面得到点
        // 不知道哪种方式更准确，或者说用怎样的机制来判断使用什么方式生成3D点

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3, 4, CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const Mat &K2 = pKF2->mK;
        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each matched line Segment
        const int nmatches = vMatchedIndices.size();
        for(int ikl=0; ikl<nmatches; ikl++)
        {
            // step6.1：取出匹配的特征线.
            const int &idx1 = vMatchedIndices[ikl].first;
            const int &idx2 = vMatchedIndices[ikl].second;

            const KeyLine &keyline1 = mpCurrentKeyFrame->mvKeyLines[idx1];
            const KeyLine &keyline2 = pKF2->mvKeyLines[idx2];
            const Vector3d keyline1_function = mpCurrentKeyFrame->mvKeyLineFunctions[idx1];
            const Vector3d keyline2_function = pKF2->mvKeyLineFunctions[idx2];
            const Mat klF1 = (Mat_<float>(3,1) << keyline1_function(0),
                    keyline1_function(1),
                    keyline1_function(2));
            const Mat klF2 = (Mat_<float>(3,1) << keyline2_function(0),
                    keyline2_function(1),
                    keyline2_function(2));

            // step6.2：线段在第一帧图像中的端点坐标(归一化坐标)
            cv::Mat StartC1, EndC1;
            StartC1 = (cv::Mat_<float>(3,1) << (keyline1.startPointX-cx1)*invfx1, (keyline1.startPointY-cy1)*invfy1, 1.0);
            EndC1 = (cv::Mat_<float>(3,1) << (keyline1.endPointX-cx1)*invfx1, (keyline1.endPointY-cy1)*invfy1, 1.0);

            // step6.3：两帧图像的投影矩阵
            Mat M1 = K1 * Tcw1;
            Mat M2 = K2 * Tcw2;

            /// 疑问：这里还是用端点恢复空间线段，但是用的第一帧上的端点还是第二帧上的呢?

            // step6.4：三角化恢复线段的3D端点
            cv::Mat s3D, e3D;
            // 起始点
            cv::Mat A(4,4,CV_32F);

            ///关键点！!! todo 理论上check这样计算对不对
            A.row(0) = klF1.t()*M1;
            A.row(1) = klF2.t()*M2;
            A.row(2) = StartC1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
            A.row(3) = StartC1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);

            cv::Mat w1, u1, vt1;
            cv::SVD::compute(A, w1, u1, vt1, cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

            s3D = vt1.row(3).t();

            if(s3D.at<float>(3)==0)
                continue;

            // Euclidean coordinates
            if(s3D.at<float>(3)==0)
                cerr << "error: LocalMapping::CreateNewMapLines2(): s3D.at<float>(3)==0 " << endl;
            s3D = s3D.rowRange(0,3)/s3D.at<float>(3);

            // 终止点
            cv::Mat B(4,4,CV_32F);
            B.row(0) = klF1.t()*M1;
            B.row(1) = klF2.t()*M2;
            B.row(2) = EndC1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
            B.row(3) = EndC1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);

            cv::Mat w2, u2, vt2;
            cv::SVD::compute(B, w2, u2, vt2, cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

            e3D = vt2.row(3).t();

            if(e3D.at<float>(3)==0)
                continue;

            // Euclidean coordinates
            e3D = e3D.rowRange(0,3)/e3D.at<float>(3);

            cv::Mat s3Dt = s3D.t();
            cv::Mat e3Dt = e3D.t();

            // 判断起始点是否离两个相机中心太近
            // 邻接关键帧的场景深度中值
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            cv::Mat v1 = s3D - Ow1;
            float distance1 = cv::norm(v1);
            assert(medianDepthKF2 != 0);
            const float ratio1 = distance1/medianDepthKF2;
            if(ratio1 < 0.3)
                continue;

            cv::Mat v2 = s3D - Ow2;
            float distance2 = cv::norm(v2);
            assert(medianDepthKF2 != 0);
            const float ratio2 = distance2/medianDepthKF2;
            if(ratio2 < 0.3)
                continue;

            // 判断线段是否太长
            cv::Mat v3 = e3D - s3D;
            float distance3 = cv::norm(v3);
            const float ratio3 = distance3/medianDepthKF2;
            if(ratio3 > 1)
                continue;
//            cout << "ratio1 = " << ratio1 << endl << "ratio2 = " << ratio2 << endl;
//            cout << "distance1 = " << distance1 << endl;
//            cout << "distance2 = " << distance2 << endl;

//            // 判断两个终止点离相机中心是否太近
//            cv::Mat v4 = e3D - Ow1;
//            float distance4 = cv::norm(v4);
//            const float ratio4 = distance4/medianDepthKF2;
//            if(ratio4 < 0.3)
//                continue;
//            cv::Mat v5 = e3D - Ow2;
//            float distance5 = cv::norm(v5);
//            const float ratio5 = distance5/medianDepthKF2;
//            if(ratio5 < 0.3)
//                continue;

            // step6.5：检测生成的3D点是否在相机前方
            float SZC1 = Rcw1.row(2).dot(s3Dt)+tcw1.at<float>(2);   //起始点在C1下的Z坐标值
            if(SZC1<=0)
                continue;

            float SZC2 = Rcw2.row(2).dot(s3Dt)+tcw2.at<float>(2);   //起始点在C2下的Z坐标值
            if(SZC2<=0)
                continue;

            float EZC1 = Rcw1.row(2).dot(e3Dt)+tcw1.at<float>(2);   //终止点在C1下的Z坐标值
            if(EZC1<=0)
                continue;

            float EZC2 = Rcw2.row(2).dot(e3Dt)+tcw2.at<float>(2);   //终止点在C2下的Z坐标值
            if(EZC2<=0)
                continue;

            // step6.9: 三角化成功，构造MapLine
            Vector6d line3D;
            line3D << s3D.at<float>(0), s3D.at<float>(1), s3D.at<float>(2), e3D.at<float>(0), e3D.at<float>(1), e3D.at<float>(2);
            MapLine* pML = new MapLine(line3D, mpCurrentKeyFrame, mpMap);

            // step6.10：为该MapLine添加属性
            pML->AddObservation(mpCurrentKeyFrame, idx1);
            pML->AddObservation(pKF2, idx2);

            mpCurrentKeyFrame->AddMapLine(pML, idx1);
            pKF2->AddMapLine(pML, idx2);

            pML->ComputeDistinctiveDescriptors();
            pML->UpdateAverageDir();
            mpMap->AddMapLine(pML);

            // step6.11：将新产生的线特征放入检测队列，这些MapLines都会经过MapLineCulling函数的检验
            mlpRecentAddedMapLines.push_back(pML);

            nnew++;
        }

    }
}


/**
 * 检查并融合当前关键帧与相邻帧（两级相邻）重复的MapPoints，共视和共视的共视
 */
void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    // 步骤1：获得当前关键帧在covisibility图中权重排名前nn的邻接关键帧
    // 找到当前帧一级相邻与二级相邻关键帧
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);  //至少共视10的相连关键帧

    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);// 加入一级相邻帧
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;// 并标记已经加入，避免重复

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5); //二级相连，至少共视5的相连关键帧
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);// 存入二级相邻帧
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;

    // 步骤2：将当前帧的MapPoints分别与一级二级相邻帧(的MapPoints)进行融合
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        // 投影当前帧的MapPoints到相邻关键帧pKFi中，并判断是否有重复的MapPoints
        // 1.如果MapPoint能匹配关键帧的特征点，并且该点有对应的MapPoint，那么将两个MapPoint合并（选择观测数多的）
        // 2.如果MapPoint能匹配关键帧的特征点，并且该点没有对应的MapPoint，那么为该点添加MapPoint
        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    // 用于存储一级邻接和二级邻接关键帧所有MapPoints的集合
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    // 步骤3：将一级二级相邻帧的MapPoints分别与当前帧（的MapPoints）进行融合
    // 遍历每一个一级邻接和二级邻接关键帧
    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches(); //这里取出了邻接关键帧的所有地图点

        // 遍历当前一级邻接和二级邻接关键帧中所有的MapPoints
        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            
            // 判断MapPoints是否为坏点，或者是否已经加进集合vpFuseCandidates
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;

            // 加入集合，并标记已经加入
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);  //和上面一样，是相对的你对我检查融合反过来我对你也检查融合

    // Update points
    // 步骤4：更新当前帧MapPoints的描述子，深度，观测主方向等属性
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                // 在所有找到pMP的关键帧中，获得最佳的描述子
                pMP->ComputeDistinctiveDescriptors();

                // 更新平均观测方向和观测距离
                pMP->UpdateNormalAndDepth();
            }
        }
    }


#if 1  // 下面的代码，其实也可以穿插到上面和点一起处理，而不是单独再用一个代码块

    //=====================MapLine=========仿照上面的思路对线进行相同操作

    LSDmatcher lineMatcher;
    vector<MapLine*> vpMapLineMatches = mpCurrentKeyFrame->GetMapLineMatches();     //也就是当前帧的mvpMapLines
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        lineMatcher.Fuse(pKFi, vpMapLineMatches);
    }

    vector<MapLine*> vpLineFuseCandidates;
    vpLineFuseCandidates.reserve(vpTargetKFs.size()*vpMapLineMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapLine*> vpMapLinesKFi = pKFi->GetMapLineMatches();

        // 遍历当前一级邻接和二级邻接关键帧中所有的MapLines
        for(vector<MapLine*>::iterator vitML=vpMapLinesKFi.begin(), vendML=vpMapLinesKFi.end(); vitML!=vendML; vitML++)
        {
            MapLine* pML = *vitML;
            if(!pML)
                continue;

            if(pML->isBad() || pML->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;

            pML->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpLineFuseCandidates.push_back(pML);
        }
    }

    lineMatcher.Fuse(mpCurrentKeyFrame, vpLineFuseCandidates);

    // Update Lines
    vpMapLineMatches = mpCurrentKeyFrame->GetMapLineMatches();
    for(size_t i=0, iend=vpMapLineMatches.size(); i<iend; i++)
    {
        MapLine* pML=vpMapLineMatches[i];
        if(pML)
        {
            if(!pML->isBad())
            {
                pML->ComputeDistinctiveDescriptors();
                pML->UpdateAverageDir();
            }
        }
    }

    //=================MapLine Done==================
#endif


    // Update connections in covisibility graph

    // 步骤5：更新当前帧的MapPoints后更新与其它帧的连接关系
    // 更新covisibility图
    mpCurrentKeyFrame->UpdateConnections();
}


//// --line--
/// 重写SerachInNeighbors函数，专门检查并融合当前帧与相邻关键帧的重复MapLines
/// 这个函数虽然写了，但是没有调用
/// lan程序中是把这个函数中的操作都加在了原有的SearchInNeighbors函数，因为有很多重复的操作这样避免浪费时间
/// 也可以写这个函数，然后在调用的时候用多线程并行点和线的融合！！
void LocalMapping::SearchLineInNeighbors()
{
    // step1:获得当前关键帧在Covisibility图中权重排名前nn的邻接关键帧，再寻找与一级关键帧相连的二级关键帧
    int nn=10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    // todo check这个函数中得到最多观测是只用了点还是点和线的观测都算？

    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF==mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vit2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId  || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

    cout << "vpTargetKFs.size() = " << vpTargetKFs.size() << endl;

    LSDmatcher matcher;

    // step2:将当前帧的MapLines分别与一级和二级相邻帧的MapLines进行融合
    vector<MapLine*> vpMapLineMatches = mpCurrentKeyFrame->GetMapLineMatches();     //也就是当前帧的mvpMapLines

    cout << "vpMapLineMatches.size() = " << vpMapLineMatches.size() << endl;
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        matcher.Fuse(pKFi, vpMapLineMatches);
    }

    vector<MapLine*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapLineMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapLine*> vpMapLinesKFi = pKFi->GetMapLineMatches();

        // 遍历当前一级邻接和二级邻接关键帧中所有的MapLines
        for(vector<MapLine*>::iterator vitML=vpMapLinesKFi.begin(), vendML=vpMapLinesKFi.end(); vitML!=vendML; vitML++)
        {
            MapLine* pML = *vitML;
            if(!pML)
                continue;

            if(pML->isBad() || pML->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;

            pML->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pML);
        }
    }

    cout << "vpFuseCandidates.size() = " << vpFuseCandidates.size() << endl;
    matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

    // Update Lines
    vpMapLineMatches = mpCurrentKeyFrame->GetMapLineMatches();
    for(size_t i=0, iend=vpMapLineMatches.size(); i<iend; i++)
    {
        MapLine* pML=vpMapLineMatches[i];
        if(pML)
        {
            if(!pML->isBad())
            {
                pML->ComputeDistinctiveDescriptors();
                pML->UpdateAverageDir();
            }
        }
    }

    mpCurrentKeyFrame->UpdateConnections();
}


/**
 * 根据两关键帧的姿态计算两个关键帧之间的基本矩阵
 * @param  pKF1 关键帧1
 * @param  pKF2 关键帧2
 * @return      基本矩阵
 */
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    // Essential Matrix: t12叉乘R12
    // Fundamental Matrix: inv(K1)*E*inv(K2)

    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w; //注意t12的计算

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

/**
 * @brief 关键帧剔除
 * 
 * 在Covisibility Graph中的关键帧，其90%以上的MapPoints能被其他关键帧（至少3个）观测到，则认为该关键帧为冗余关键帧。
 * @see VI-E Local Keyframe Culling
 */
 //todo 这里的观测只有MapPoints 是不是也把线的观测给算进去呢
 // 这个函数仅和线有关，暂时未改后期可以考虑将线的内容加入，把直线也作为一种观测算入90%内
void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points ?

    // 步骤1：根据Covisibility Graph提取当前帧的共视关键帧，在这些共视关键帧中检查是否有需要剔除的关键帧
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    // 对所有的局部关键帧进行遍历
    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        // 步骤2：提取每个共视关键帧的MapPoints
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;

        // 步骤3：遍历该局部关键帧的MapPoints，判断是否90%以上的MapPoints能被其它关键帧（至少3个）观测到
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        // 对于双目，仅考虑近处的MapPoints，不超过mbf * 35 / fx
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;  //所有的点数（双目即所有近处的点数），双目为什么特殊呢
                    // MapPoints至少被三个关键帧观测到
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();

                        // 判断该MapPoint是否同时被三个关键帧观测到
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            // Scale Condition 
                            // 尺度约束，要求MapPoint在该局部关键帧的特征尺度大于（或近似于）其它关键帧的特征尺度
                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                // 已经找到三个同尺度的关键帧可以观测到该MapPoint，不用继续找了
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        // 该MapPoint至少被三个关键帧观测到
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        // 步骤4：该局部关键帧90%以上的MapPoints能被其它关键帧（至少3个）观测到，则认为是冗余关键帧
        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();  //剔除
    }
}


cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        //usleep(3000);
        std::this_thread::sleep_for(std::chrono::milliseconds(3));

    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mlpRecentAddedMapLines.clear();   // --line--
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
