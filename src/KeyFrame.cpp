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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include  <mutex>

#include "auxiliar.h"

namespace ORB_SLAM2
{

long unsigned int KeyFrame::nNextId=0;


KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
        mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
        mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
        mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
        mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
        fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
        mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), NL(F.NL), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn), mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),

        mvKeyLines(F.mvKeylines), mvKeyLinesUn(F.mvKeylinesUn), mvuRightLineStart(F.mvuRightLineStart), mvuRightLineEnd(F.mvuRightLineEnd), mvDepthLineStart(F.mvDepthLineStart), mvDepthLineEnd(F.mvDepthLineEnd), mvKeyLineFunctions(F.mvKeyLineFunctions), mLineDescriptors(F.mLdesc.clone()),

        mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
        mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
        mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
        mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), mvpMapLines(F.mvpMapLines), mpKeyFrameDB(pKFDB), mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
        mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap)
{
    mnId=nNextId++;

    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];  //mGrid[][]所存放的元素是一个vector啊
    }

    SetPose(F.mTcw);
}


/**
 * @brief Bag of Words Representation
 *
 * 计算mBowVec，并且将描述子分散在第4层上，即mFeatVec记录了属于第i个node的ni个描述子!
 * @see ProcessNewKeyFrame()
 */
void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    unique_lock<mutex> lock(mMutexPose);  //加锁
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc*tcw;  //twc = -Rwc*tcw

    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));
    // center为相机坐标系（左目）下，立体相机中心的坐标
    // 立体相机中心点坐标与左目相机坐标之间只是在x轴上相差mHalfBaseline,
    // 因此可以看出，立体相机中两个摄像头的连线为x轴，正方向为左目相机指向右目相机
    cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
    // 世界坐标系下，左目相机中心到立体相机中心的向量，方向由左目相机指向立体相机中心
    Cw = Twc*center;  //获得双目相机中心的位置（世界坐标系下）
}

cv::Mat KeyFrame::GetPose()  //关键帧的绝对位姿
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();  //没有直接返回Tcw
}

cv::Mat KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}

cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}

cv::Mat KeyFrame::GetStereoCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}


cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();  //Tcw当前前三行前三列
}

cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).col(3).clone();
}

/**
 * @brief 为关键帧之间添加连接
 * 
 * 更新了mConnectedKeyFrameWeights
 * @param pKF    关键帧
 * @param weight 权重，该关键帧与pKF共同观测到的3d点数量
 */
void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)  //未
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        // std::map::count函数只可能返回0或1两种情况
        if(!mConnectedKeyFrameWeights.count(pKF)) // count函数返回0，mConnectedKeyFrameWeights中没有pKF，之前没有连接
            mConnectedKeyFrameWeights[pKF]=weight;  //map<KeyFrame*,int> 与该关键帧连接的关键帧和对应的权重，在map中加入元素并带属性，map是可以自动排序的但是排序函数是在哪里定义的？
        else if(mConnectedKeyFrameWeights[pKF]!=weight) // 之前连接的权重不一样，更新权重
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

/**
 * @brief 按照权重对连接的关键帧进行排序
 * 
 * 更新后的变量存储在mvpOrderedConnectedKeyFrames和mvOrderedWeights中
 */
void KeyFrame::UpdateBestCovisibles() //未改
{
    unique_lock<mutex> lock(mMutexConnections);
    // http://stackoverflow.com/questions/3389648/difference-between-stdliststdpair-and-stdmap-in-c-stl
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());  //map的size
    // 取出所有连接的关键帧，mConnectedKeyFrameWeights的类型为std::map<KeyFrame*,int>，而vPairs变量将共视的3D点数放在前面，利于排序
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));  //pair的第一个参数为权重

    // 按照权重进行排序
    sort(vPairs.begin(),vPairs.end());  //
    list<KeyFrame*> lKFs; // keyframe
    list<int> lWs; // weight
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        lKFs.push_front(vPairs[i].second);  ///注意这里是push_front所以最终权重是从大到小排的
        lWs.push_front(vPairs[i].first);
    }

    // 权重从大到小
    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    // list转换为vector
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}

/**
 * @brief 得到与该关键帧连接的关键帧
 * @return 连接的关键帧
 */
set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    // 注意下面这个mConnectedKeyFrameWeughts map类型
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(); mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);  //s是会自动排序的，排序函数是啥，排序规则呢？
    return s;
}

/**
 * @brief 得到与该关键帧连接的关键帧(已按权值排序)
 * @return 连接的关键帧
 */
vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

/**
 * @brief 得到与该关键帧连接的前N个关键帧(已按权值排序)
 * 
 * 如果连接的关键帧少于N，则返回所有连接的关键帧
 * @param N 前N个
 * @return 连接的关键帧
 */
vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)

{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);  //看来权重是按从大到小排列的
}

/**
 * @brief 得到与该关键帧连接的权重大于等于w的关键帧
 * @param w 权重
 * @return 连接的关键帧
 */
vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    // http://www.cplusplus.com/reference/algorithm/upper_bound/
    // 从mvOrderedWeights找出第一个大于w的那个迭代器
    // 这里应该使用lower_bound，因为lower_bound是返回第一个大于等于的位置，而upper_bound返回第一个大于的位置
    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);
    // lower_bound和upper_bound本来是用于递增序列的，lower_bound是返回第一个大于等于的位置，而upper_bound返回第一个大于的位置
    // 传入第四个参数，递减的比较函数，则用于递减序列，lower_bound是返回第一个小于等于的位置，而upper_bound返回第一个小于的位置
    if(it==mvOrderedWeights.end() && *mvOrderedWeights.rbegin()<w)  //最小值都比w小，说明所有的都比w小，没有合适的
        return vector<KeyFrame*>();
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n); //所以最终返回了大于等于w的数
    }
}

/**
 * @brief 得到该关键帧与输入参数pKF的权重
 * @param  pKF 关键帧
 * @return     权重
 */
int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

/**
 * @brief Add MapPoint to KeyFrame
 * @param pMP MapPoint
 * @param idx MapPoint在KeyFrame中的索引
 */
void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;  //修改关键帧的地图点列表即可
}

//// line
void KeyFrame::AddMapLine(MapLine *pML, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapLines[idx]=pML;
}


void KeyFrame::EraseMapPointMatch(const size_t &idx)  //传入参数只有该地图点在关键帧上的索引
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)  //删除关键帧上的对应的地图点索引
{
    int idx = pMP->GetIndexInKeyFrame(this);  //this是当前关键帧的指针，调用地图点的成员函数GetIndexInKeyFrame可以得到该地图点在this关键帧上的索引
    if(idx>=0)
        mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}


//// line

void KeyFrame::EraseMapLineMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapLines[idx]= static_cast<MapLine*>(NULL);
}
//// line
void KeyFrame::EraseMapLineMatch(MapLine *pML)
{
    int idx = pML->GetIndexInKeyFrame(this);
    if(idx>=0)
        mvpMapLines[idx]= static_cast<MapLine*>(NULL);
}

void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

//// line
void KeyFrame::ReplaceMapLineMatch(const size_t &idx, MapLine *pML)
{
    mvpMapLines[idx]=pML;
}

set<MapPoint*> KeyFrame::GetMapPoints()  //得到关键帧所观测到的地图点
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);  //s的size和mvpMapPonts的size不一致了，s只插入匹配的特征点
    }
    return s;
}

//// line
set<MapLine*> KeyFrame::GetMapLines()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapLine*> s;
    for(size_t i=0, iend=mvpMapLines.size(); i<iend; i++)
    {
        if(!mvpMapLines[i])
            continue;
        MapLine* pML = mvpMapLines[i];
        if(!pML->isBad())
            s.insert(pML);
    }
    return s;
}


/**
 * @brief 关键帧中，大于等于minObs的MapPoints的数量
 * minObs就是一个阈值，大于minObs就表示该MapPoint是一个高质量的MapPoint
 * 一个高质量的MapPoint会被多个KeyFrame观测到，
 * @param  minObs 最小观测
 */
int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)
                {
                    // 该MapPoint是一个高质量的MapPoint
                    if(mvpMapPoints[i]->Observations()>=minObs)  //该地图点至少被minObs个关键帧看到
                        nPoints++;
                }
                else
                    nPoints++;  //minObs<=0的时候返回所有的地图点
            }
        }
    }

    return nPoints;
}

//// line
//todo 这个函数写了，但是没有用，因此在TrackedMapPoints用到的地方看是否相应做添加
int KeyFrame::TrackedMapLines(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nLines = 0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<NL; i++)
    {
        MapLine* pML = mvpMapLines[i];
        if(pML)
        {
            if(!pML->isBad())
            {
                if(bCheckObs)
                {
                    //该MapLine是一个高质量的MapLine
                    if(mvpMapLines[i]->Observations()>=minObs)
                        nLines++;
                } else
                    nLines++;
            }
        }
    }
    return nLines;
}


/**
 * @brief Get MapPoint Matches
 *
 * 获取该关键帧的MapPoints
 */
vector<MapPoint*> KeyFrame::GetMapPointMatches()  //即该该帧上的地图点匹配列表
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

//// line
vector<MapLine*> KeyFrame::GetMapLineMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapLines;
}


MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

//// line
MapLine* KeyFrame::GetMapLine(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapLines[idx];
}


/**
 * @brief 更新图的连接， 这个函数很重要！
 * 
 * 1. 首先获得该关键帧的所有MapPoint点，统计观测到这些3d点的每个关键与其它所有关键帧之间的共视程度
 *    对每一个找到的关键帧，建立一条边，边的权重是该关键帧与当前关键帧公共3d点的个数。
 * 2. 并且该权重必须大于一个阈值，如果没有超过该阈值的权重，那么就只保留权重最大的边（与其它关键帧的共视程度比较高）
 * 3. 对这些连接按照权重从大到小进行排序，以方便将来的处理
 *    更新完covisibility图之后，如果没有初始化过，则初始化为连接权重最大的边（与其它关键帧共视程度最高的那个关键帧），类似于最大生成树
 */
 // 这个更新连接的函数跟ORBSLAM中一致，但是考虑到线也属于观测，
 // todo 我们是不是有必要把线的观测也加进去一起来更新关键帧之间的连接，我觉得尤其是针对弱纹理场景下这要做是非常有必要的， 暂时我未改
void KeyFrame::UpdateConnections()
{
    // 在没有执行这个函数前，关键帧只和MapPoints之间有连接关系，这个函数可以更新关键帧之间的连接关系

    //===============1==================================
    map<KeyFrame*,int> KFcounter; // 关键帧-权重，权重为其它关键帧与当前关键帧共视3d点的个数

    vector<MapPoint*> vpMP;

    {
        // 获得该关键帧的所有3D点
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    // 通过3D点间接统计可以观测到这些3D点的所有关键帧之间的共视程度
    // 即统计每一个关键帧都有多少关键帧与它存在共视关系，统计结果放在KFcounter
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;

        // 对于每一个MapPoint点，observations记录了可以观测到该MapPoint的所有关键帧
        map<KeyFrame*,size_t> observations = pMP->GetObservations();  //第二个参数为在关键镇上地图点的索引

        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            // 除去自身，自己与自己不算共视
            if(mit->first->mnId==mnId)
                continue;
            KFcounter[mit->first]++;
        }
    }

    // This should not happen
    if(KFcounter.empty())
        return;

    //===============2==================================
    // If the counter is greater than threshold add connection
    // In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax=NULL;
    int th = 15;

    // vPairs记录与其它关键帧共视帧数大于th的关键帧
    // pair<int,KeyFrame*>将关键帧的权重写在前面，关键帧写在后面方便后面排序
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(mit->second>nmax)
        {
            nmax=mit->second;  //共视关系最高的关键帧的共视值
            // 找到对应权重最大的关键帧（共视程度最高的关键帧）
            pKFmax=mit->first;
        }
        if(mit->second>=th)
        {
            // 对应权重需要大于阈值，对这些关键帧建立连接
            vPairs.push_back(make_pair(mit->second,mit->first)); //第一个权重，第二个关键帧
            // AddConnection主要完成：
            // 更新KFcounter中该关键帧的mConnectedKeyFrameWeights
            // 更新其它KeyFrame的mConnectedKeyFrameWeights，更新其它关键帧与当前帧的连接权重
            (mit->first)->AddConnection(this,mit->second);  //更新this关键帧在mit->first关键帧下的连接
        }
    }

    // 如果没有超过阈值的权重，则对权重最大的关键帧建立连接
    if(vPairs.empty())
    {
	    // 如果每个关键帧与它共视的关键帧的个数都少于th，
        // 那就只更新与其它关键帧共视程度最高的关键帧的mConnectedKeyFrameWeights
        // 这是对之前th这个阈值可能过高的一个补丁
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

    // vPairs里存的都是相互共视程度比较高的关键帧和共视权重，由大到小
    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    //===============3==================================
    {
        unique_lock<mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        // 更新图的连接(权重)
        mConnectedKeyFrameWeights = KFcounter;//更新该KeyFrame的mConnectedKeyFrameWeights，更新当前帧与其它关键帧的连接权重
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        /// 更新生成树的连接
        if(mbFirstConnection && mnId!=0)
        {
            // 初始化该关键帧的父关键帧为共视程度最高的那个关键帧
            mpParent = mvpOrderedConnectedKeyFrames.front();
            // 建立双向连接关系
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }
    }
}



void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);  //儿子是set，说明有很多个儿子啊
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;  //set
}

void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    // 这个地方是不是应该：(!mbToBeErased)，(wubo???) 是的我也这么认为
    // SetBadFlag函数就是将mbToBeErased置为true，mbToBeErased就表示该KeyFrame被擦除了
    if(mbToBeErased)
    {
        SetBadFlag();
    }
}


// todo_！ 这个函数中涉及到了地图点的操作，因此要把对应的地图线的操作加进去！
void KeyFrame::SetBadFlag() //未改
{   
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==0)
            return;
        else if(mbNotErase)// mbNotErase表示不应该擦除该KeyFrame，于是把mbToBeErased置为true，表示已经擦除了，其实没有擦除
        {
            mbToBeErased = true;
            return;
        }
    }

    for(map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        mit->first->EraseConnection(this);// 让其它的KeyFrame删除与自己的联系

    for(size_t i=0; i<mvpMapPoints.size(); i++)
        if(mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this);// 让与自己有联系的MapPoint删除与自己的联系
    for(size_t i=0; i<mvpMapLines.size(); ++i)
        if(mvpMapLines[i])
            mvpMapLines[i]->EraseObservation(this);


    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        //清空自己与其它关键帧之间的联系
        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree   @@
        set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        // 如果这个关键帧有自己的孩子关键帧，告诉这些子关键帧，它们的父关键帧不行了，赶紧找新的父关键帧
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;

            // 遍历每一个子关键帧，让它们更新它们指向的父关键帧
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
                KeyFrame* pKF = *sit;
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                // 子关键帧遍历每一个与它相连的关键帧（共视关键帧）
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                    // 如果该帧的子节点和父节点（祖孙节点）之间存在连接关系（共视）
                    // 举例：B-->A（B的父节点是A） C-->B（C的父节点是B） D--C（D与C相连） E--C（E与C相连） F--C（F与C相连） D-->A（D的父节点是A） E-->A（E的父节点是A）
                    //      现在B挂了，于是C在与自己相连的D、E、F节点中找到父节点指向A的D
                    //      此过程就是为了找到可以替换B的那个节点。
                    // 上面例子中，B为当前要设置为SetBadFlag的关键帧
                    //           A为spcit，也即sParentCandidates
                    //           C为pKF,pC，也即mspChildrens中的一个
                    //           D、E、F为vpConnected中的变量，由于C与D间的权重 比 C与E间的权重大，因此D为pP
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                // 因为父节点死了，并且子节点找到了新的父节点，子节点更新自己的父节点
                pC->ChangeParent(pP);
                // 因为子节点找到了新的父节点并更新了父节点，那么该子节点升级，作为其它子节点的备选父节点
                sParentCandidates.insert(pC);
                // 该子节点处理完毕
                mspChildrens.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        // 如果还有子节点没有找到新的父节点
        if(!mspChildrens.empty())
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                // 直接把父节点的父节点作为自己的父节点
                (*sit)->ChangeParent(mpParent);
            }

        mpParent->EraseChild(this);
        mTcp = Tcw*mpParent->GetPoseInverse();
        mbBad = true;
    }

    mpMap->EraseKeyFrame(this);  //地图和DataBase中删除该关键帧
    mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}

// r为边长（半径）
// 这个函数是得到关键帧上某一位置附近领域的特征点，返回特征点的索引。
/// 针对线特征，暂时没有增加类似的函数，不知道有没有影响  todo_  检查
vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const  //@@
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    // floor向下取整，mfGridElementWidthInv为每个像素占多少个格子
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    // ceil向上取整
    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}


//// --line-- 类比上面的函数，这里写一个得到某一个领域候选特征线的函数，这里和Frame类的中这个函数同
// x y是像素坐标
// todo! 我只是简单的计算点到直线的距离从而找出点附近的线特征，想想要不要修改
// 这个函数写的还行 不改好像也行的
vector<size_t> KeyFrame::GetLineFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;  //存放特征线的索引号

    vIndices.reserve(NL);

//    vector<KeyLine> vkl = mvKeyLinesUn;
    vector<Vector3d> vlineFun = mvKeyLineFunctions;

    for(size_t i=0; i<vlineFun.size(); i++)
    {
        Vector3d p(x, y, 1);
        double dist = p.dot(vlineFun[i]);  //点到直线的距离

        if(dist > r)
            continue;
        else
            vIndices.push_back(i);
    }

    return vIndices;
}



// 像素坐标是否在关键帧的图像内，加入线同样适用
bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

/**
 * @brief Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
 * @param  i 第i个keypoint
 * @return   3D点（相对于世界坐标系）
 */
cv::Mat KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        // 由2维图像反投影到相机坐标系
        // mvDepth是在ComputeStereoMatches函数中求取的
        // mvDepth对应的校正前的特征点，因此这里对校正前特征点反投影
        // 可在Frame::UnprojectStereo中却是对校正后的特征点mvKeysUn反投影
        // 在ComputeStereoMatches函数中应该对校正后的特征点求深度？？ (wubo???)
        // TODO 我也存在这样的疑问 测试 这里我觉得应该用去畸变后的坐标mvKeysUn
        const float u = mvKeys[i].pt.x; //有畸变的坐标
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        // 由相机坐标系转换到世界坐标系
        // Twc为相机坐标系到世界坐标系的变换矩阵
        // Twc.rosRange(0,3).colRange(0,3)取Twc矩阵的前3行与前3列
        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
    }
    else
        return cv::Mat();
}


/**
* @brief Backprojects a keyLineStartPoint (if stereo/depth info available) into 3D world coordinates.
* @param  i 第i个keyLine
* @return   3D点（相对于世界坐标系）
*/
cv::Mat KeyFrame::UnprojectStereoLineStart(int i)
{
    const float z = mvDepthLineStart[i];
    if(z>0)
    {
        // 由2维图像反投影到相机坐标系
        // mvDepth是在ComputeStereoMatches函数中求取的
        // mvDepth对应的校正前的特征点，因此这里对校正前特征点反投影
        // 可在Frame::UnprojectStereo中却是对校正后的特征点mvKeysUn反投影
        // 在ComputeStereoMatches函数中应该对校正后的特征点求深度？？ (wubo???)
        const float u = mvKeyLinesUn[i].startPointX;
        const float v = mvKeyLinesUn[i].startPointY;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        // 由相机坐标系转换到世界坐标系
        // Twc为相机坐标系到世界坐标系的变换矩阵
        // Twc.rosRange(0,3).colRange(0,3)取Twc矩阵的前3行与前3列
        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
    }
    else
        return cv::Mat();
}

/**
* @brief Backprojects a keyLineEndPoint (if stereo/depth info available) into 3D world coordinates.
* @param  i 第i个keyLine
* @return   3D点（相对于世界坐标系）
*/
cv::Mat KeyFrame::UnprojectStereoLineEnd(int i)
{
    const float z = mvDepthLineEnd[i];
    if(z>0)
    {
        // 由2维图像反投影到相机坐标系
        // mvDepth是在ComputeStereoMatches函数中求取的
        // mvDepth对应的校正前的特征点，因此这里对校正前特征点反投影
        // 可在Frame::UnprojectStereo中却是对校正后的特征点mvKeysUn反投影
        // 在ComputeStereoMatches函数中应该对校正后的特征点求深度？？ (wubo???)
        const float u = mvKeyLinesUn[i].endPointX;
        const float v = mvKeyLinesUn[i].endPointY;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        // 由相机坐标系转换到世界坐标系
        // Twc为相机坐标系到世界坐标系的变换矩阵
        // Twc.rosRange(0,3).colRange(0,3)取Twc矩阵的前3行与前3列
        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
    }
    else
        return cv::Mat();
}


/**
 * @brief 评估当前关键帧场景深度，q=2表示中值
 * @param q q=2
 * @return Median Depth
 */
// 这里的场景深度是经过点特征计算出来的，是不是需要把线特征也考虑进去呢

float KeyFrame::ComputeSceneMedianDepth(const int q)  // Compute Scene Depth (q=2 median). Used in monocular.
{
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2,3);
    for(int i=0; i<N; i++)
    {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw)+zcw; // (R*x3Dw+t)的第三行，即z
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];
}


// 和其他地方的实现是一样的
void KeyFrame::lineDescriptorMAD(vector<vector<DMatch>> line_matches, double &nn_mad, double &nn12_mad) const
{
    vector<vector<DMatch>> matches_nn, matches_12;
    matches_nn = line_matches;
    matches_12 = line_matches;
//    cout << "Frame::lineDescriptorMAD——matches_nn = "<<matches_nn.size() << endl;

    // estimate the NN's distance standard deviation
    double nn_dist_median;
    sort( matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
    nn_dist_median = matches_nn[int(matches_nn.size()/2)][0].distance;

    for(unsigned int i=0; i<matches_nn.size(); i++)
        matches_nn[i][0].distance = fabsf(matches_nn[i][0].distance - nn_dist_median);
    sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
    nn_mad = 1.4826 * matches_nn[int(matches_nn.size()/2)][0].distance;

    // estimate the NN's 12 distance standard deviation
    double nn12_dist_median;
    sort( matches_12.begin(), matches_12.end(), compare_descriptor_by_NN12_dist());
    nn12_dist_median = matches_12[int(matches_12.size()/2)][1].distance - matches_12[int(matches_12.size()/2)][0].distance;
    for (unsigned int j=0; j<matches_12.size(); j++)
        matches_12[j][0].distance = fabsf( matches_12[j][1].distance - matches_12[j][0].distance - nn12_dist_median);
    sort(matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist());
    nn12_mad = 1.4826 * matches_12[int(matches_12.size()/2)][0].distance;
}



} //namespace ORB_SLAM
