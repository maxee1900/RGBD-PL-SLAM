//
// Created by max on 19-6-24.
//

#include "MapLine.h"
#include "LSDmatcher.h"

#include <iostream>
#include <mutex>
#include <map>

using namespace std;

namespace ORB_SLAM2
{

//先处理两个静态数据成员，前面可以不写static，类的声明中static要加上
mutex MapLine::mGlobalMutex;  //静态数据成员要初始化
long unsigned int MapLine::nNextId = 0;

/**
 * @brief 给定坐标与keyframe构造MapPoint (这里是MapLine)
 *
 * 双目：StereoInitialization()，CreateNewKeyFrame()，LocalMapping::CreateNewMapPoints()
 * 单目：CreateInitialMapMonocular()，LocalMapping::CreateNewMapPoints()
 * @param Pos    MapPoint的坐标（wrt世界坐标系）
 * @param pRefKF KeyFrame
 * @param pMap   Map
 */
// todo 思考MapLine中的mWorldPos用什么来表示呢，是两个端点的坐标，还是一系列线上点的坐标
MapLine::MapLine(const Vector6d &Pos, KeyFrame* pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0),
    mnTrackReferenceForFrame(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopLineForKF(0),
    mnCorrectedByKF(0), mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF),
    mnVisible(1), mnFound(1), mbBad(false), mpReplaced(static_cast<MapLine*>(NULL)), mpMap(pMap)
{
    mWorldPos = Pos;

    mNormalVector << 0,0,0;

    // MapLine can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexLineCreation);
    mnId = nNextId++;
}

/**
 * @brief 给定坐标与frame构造MapPoint
 *
 * 双目：UpdateLastFrame()
 * @param Pos    MapPoint的坐标（wrt世界坐标系）
 * @param pMap   Map
 * @param pFrame Frame
 * @param idxF   MapPoint在Frame中的索引，即对应的特征点的编号
 */
MapLine::MapLine(const Vector6d &Pos, Map *pMap, Frame *pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopLineForKF(0),
    mnCorrectedByKF(0), mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1), mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
{
    mWorldPos = Pos;
    Mat Ow = pFrame->GetCameraCenter();
    Vector3d OW;
    OW << Ow.at<double>(0), Ow.at<double>(1), Ow.at<double>(2);
    mStart3D = Pos.head(3);
    mEnd3D = Pos.tail(3);
    Vector3d midPoint = 0.5 * (mStart3D + mEnd3D);
    mNormalVector = midPoint - OW;  //相机到地图线中点的向量
    mNormalVector.normalize();  //单位化。变成单位向量

    Vector3d PC = midPoint - OW;
    const float dist = PC.norm();  //二范数，向量的长度

    //todo 检查下面的参数是否合理，后两个都是点特征的参数啊
    const int level = pFrame->mvKeylinesUn[idxF].octave;
    //todo 测试以下两个cout
//    cout << "level = pFrame->mvKeylinesUn[idxF].octave : " << level << endl;
    const float levelScaleFactor = pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;  // 这个值是不是一直等于8
//    cout << "nLevels = pFrame->mnScaleLevels : " << nLevels << endl;


    //这里和ORB中同
    mfMaxDistance = dist * levelScaleFactor;
    mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels-1];

    pFrame->mLdesc.row(idxF).copyTo(mLDescriptor);  //将帧上的特征线的描述子赋给地图线
    // idxF是当前帧上地图线对应的特征索引，mLdesc是整幅图像的描述子矩阵，这里是单个特征的描述子

    unique_lock<mutex> lock(mpMap->mMutexLineCreation);
    mnId=nNextId++;

}

////
void MapLine::SetWorldPos(const Vector6d &Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    mWorldPos = Pos;
    //这里Pos类型为Eigen::Vector6d，可以直接赋值，如果是Mat类型用Pos.copy(mWorldPos)
}

////
Vector6d MapLine::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos;
}

////
Vector3d MapLine::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector;
}

////
KeyFrame* MapLine::GetReferenceKeyFrame() //GlobalBA中才用
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

////
void MapLine::AddObservation(KeyFrame *pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return;
    mObservations[pKF] = idx; // map数据结构的操作

    nObs++;

    if(pKF->mvuRightLineStart[idx]>=0 && pKF->mvDepthLineEnd[idx]>=0)  //todo_ 这个需要我加上mvuLineRight成员,这样才能和下一个函数保持一致
        nObs+=2; // 如果直线的两个端点都有深度，加2
    else
        nObs++; // 单目
}

////
void MapLine::EraseObservation(KeyFrame *pKF)
{
    bool bBad = false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))  //如果观测map中有pKF
        {
            int idx = mObservations[pKF];  //pKF中对应的特征线的索引,mObservation是map类型
            if(pKF->mvuRightLineStart[idx]>=0 && pKF->mvDepthLineEnd[idx]>0)
                //双目情况， todo_ 这里要将mvuRight改为mvuLineRight
                nObs -= 2;
            else
                nObs--;

            mObservations.erase(pKF);

            if(pKF == mpRefKF)  // 如果该关键帧为参考帧
                mpRefKF = mObservations.begin()->first;
            //注意：map是自动排序的。todo_ 找到mObservation这个map结构排序函数的定义在哪里？是按照键值KeyFrame的ID来排？

            // If only 2 observations or less, discard point
            // 当观测到该点的相机数目少于2时，丢弃该点
            if(nObs<=2)
                bBad=true;
        }

    }

    if(bBad)
        SetBadFlag();
}


////得到该地图线的观测信息 map<KeyFrame*, size_t>
map<KeyFrame*, size_t> MapLine::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}


////返回观测数
int MapLine::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

////告知可以观测到该MapLine的Frame,该MapLine已经被删除
void MapLine::SetBadFlag()
{
    map<KeyFrame*, size_t> obs;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;// 把mObservations转存到obs，obs和mObservations里存的是指针，赋值过程为浅拷贝
        mObservations.clear();// 把mObservations指向的内存释放，obs作为局部变量之后自动删除
    }

    for(map<KeyFrame*, size_t>::iterator mit=obs.begin(), mend=obs.end(); mit != mend; mit++)  //遍历观测中的所有关键帧
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapLineMatch(mit->second);  //关键帧上该地图线的索引也更新为空
    }

    mpMap->EraseMapLine(this);
}


////
MapLine* MapLine::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}


//// 在形成闭环的时候，会更新KeyFrame与MapLine之间的关系
void MapLine::Replace(MapLine *pML)
{
    if(pML->mnId == this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*, size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);

        obs = mObservations;
        mObservations.clear();
        mbBad = true; //该点替换了当然是坏点
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pML; //原来这个数据成员是专门存放要替换成的地图点
    }

    // 所有观测到该mapline的keyframe都要被替换
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit != mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if( !pML->IsInKeyFrame(pKF))  //pML观测中没有pKF的话
        {
            pKF->ReplaceMapLineMatch(mit->second, pML); //让KeyFrame用pML替换原来的mapline
            pML->AddObservation(pKF, mit->second); //把原来对KF的观测转到新的地图线上

        }
        else //pML本来就观测到pKF
        {
            // 产生冲突，即pKF中有两个特征点a,b（这两个特征点的描述子是近似相同的），这两个特征点对应两个MapLine为this,pMP
            // 然而在fuse的过程中pML的观测更多，需要替换this，因此保留b与pML的联系，去掉a与this的联系
            pKF->EraseMapLineMatch(mit->second);
            //todo 这里是不是应该加上下面一句. 搞清楚地图点或者地图线建立观测的时候都做了哪些修改！！

            // pKF->mvpMapLines[mit->second] = pML;  //感觉不用加，ORB中没加
        }
    }

    pML->IncreaseFound(nfound);
    pML->IncreaseVisible(nvisible);
    pML->ComputeDistinctiveDescriptors();

    mpMap->EraseMapLine(this);
}


//// 没有经过MapLineCulling检测的MapLines
bool MapLine::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}


/**
 * @brief Increase Visible
 *
 * Visible表示：
 * 1. 该MapLine在某些帧的视野范围内，通过Frame::isInFrustum()函数判断
 * 2. 该MapLine被这些帧观测到，但并不一定能和这些帧的特征点匹配上
 *    例如：有一个MapPoint（记为M），在某一帧F的视野范围内，
 *    但并不表明该点M可以和F这一帧的某个特征点能匹配上
 */
void MapLine::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible += n;
}



/**
 * @brief Increase Found
 *
 * 能找到该线的帧数+n，n默认为1
 * @see Tracking::TrackLocalMap()
 */
void MapLine::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

////
float MapLine::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mnVisible==0)
        cerr << "error: MapLine::GetFoundRatio(), mnVisible=0" << endl;
    return static_cast<float>(mnFound/mnVisible);
}


////先获得当前点的所有描述子，然后计算描述子之间的两两距离，最好的描述子与其他描述子应该具有最小的距离中值
void MapLine::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<Mat> vDescriptors;

    map<KeyFrame*, size_t> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return ;
        observations = mObservations;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad())
            vDescriptors.push_back(pKF->mLineDescriptors.row(mit->second));
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them
    // 获得这些描述子两两之间的距离
    const size_t N = vDescriptors.size();

    //float Distances[N][N];
    std::vector<std::vector<float> > Distances;
    Distances.resize(N, vector<float>(N, 0));  //设定维度, 并使得二维数组中的元素都为0
    for (size_t i = 0; i<N; i++)
    {
        Distances[i][i] = 0;
        for(size_t j=i+1; j<N; j++)
        {
            int distij = LSDmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
            Distances[i][j] = distij;
            Distances[j][i] = distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        // 第i个描述子到其它所有所有描述子之间的距离
        //vector<int> vDists(Distances[i],Distances[i]+N);
        vector<int> vDists(Distances[i].begin(), Distances[i].end());
        sort(vDists.begin(), vDists.end());

        // 获得中值 第i个描述子与其它所有所有描述子之间的距离的中值
        int median = vDists[0.5*(N-1)];

        // 寻找最小的中值
        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        // 最好的描述子，该描述子相对于其他描述子有最小的距离中值
        // 简化来讲，中值代表了这个描述子到其它描述子的平均距离
        // 最好的描述子就是和其它描述子的平均距离最小
        mLDescriptor = vDescriptors[BestIdx].clone();
    }

}


////
Mat MapLine::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mLDescriptor.clone();
}


////得到地图线在关键帧上的索引，即匹配的特征线的索引，如果没有匹配返回-1
int MapLine::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;  //
}


////理解为是否观测到关键帧
bool MapLine::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return ( mObservations.count(pKF) );
}



// todo 如果我在直线上采样了很多个点那么这个观测方向又变了，还要改。。shit
////
void MapLine::UpdateAverageDir()
{
    map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    Vector6d Pos; //地图线的坐标
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;

        observations=mObservations; // 获得观测到该3d点的所有关键帧
        pRefKF=mpRefKF;             // 观测到该点的参考关键帧?
        Pos = mWorldPos;
    }

    if(observations.empty())
        return;

    Vector3d normal(0, 0, 0);
    int n = 0;
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        Mat Owi = pKF->GetCameraCenter();
        Vector3d Ow(Owi.at<float>(0), Owi.at<float>(1), Owi.at<float>(2));
        Vector3d middlePos = 0.5*(mWorldPos.head(3)+mWorldPos.tail(3));
        Vector3d normali = middlePos - Ow;
        assert(normali.norm() != 0);
        normal = normal + normali/normali.norm();
        n++;
    }

    cv::Mat SP = (Mat_<float>(3,1) << Pos(0), Pos(1), Pos(2));
    cv::Mat EP = (Mat_<float>(3,1) << Pos(3), Pos(4), Pos(5));
    cv::Mat MP = 0.5*(SP+EP);

    cv::Mat CM = MP - pRefKF->GetCameraCenter();  // 参考关键帧相机指向3Dline的向量（在世界坐标系下的表示）
    const float dist = cv::norm(CM);

    const int level = pRefKF->mvKeyLines[observations[pRefKF]].octave;
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels; // 金字塔层数

    {
        unique_lock<mutex> lock3(mMutexPos);
        mfMaxDistance = dist*levelScaleFactor;                           // 观测到该点的距离下限
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1]; // 观测到该点的距离上限
        assert(n!=0);
        mNormalVector = normal/n;                                        // 获得平均的观测方向
    }

    //todo 有个疑问，直线不同与特征点，完全按照ORB的思路为线建立观测到该线的距离上下限合理吗？有没有其他的方法代替
}



////
float MapLine::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
}

////
float MapLine::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
}


// wubo画的图    ————
// Nearer      /____\     level:n-1 --> dmin
//            /______\                       d/dmin = 1.2^(n-1-m)
//           /________\   level:m   --> d
//          /__________\                     dmax/d = 1.2^m
// Farther /____________\ level:0   --> dmax
//
//           log(dmax/d)
// m = ceil(------------)
//            log(1.2)

// 上图中要搞清楚level和远近的关系 level低的时候表示远还是近？
//// 其中的第二个参数为Frame类的数据成员，这里是默认所有frame这个参数都是一样的？？ check
int MapLine::PredictScale(const float &currentDist, const float &logScaleFactor)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        assert(currentDist!=0);
        ratio = mfMaxDistance/currentDist;
    }

    assert(logScaleFactor!=0);
    return ceil(log(ratio)/logScaleFactor);  //ceil返回大于等于它的最小整数，向上取整函数

/* todo 7.8
 *   这里要不要改写为
 *
 *  int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
 */

}


#if 0
////**********以下三个函数是仿照orbslam中的我写的**********************************

// Nearer      /____\     level:n-1 --> dmin
//            /______\                       d/dmin = 1.2^(n-1-m)
//           /________\   level:m   --> d
//          /__________\                     dmax/d = 1.2^m
// Farther /____________\ level:0   --> dmax
//
//           log(dmax/d)
// m = ceil(------------)
//            log(1.2)
int MapLine::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        // mfMaxDistance = ref_dist*levelScaleFactor为参考帧考虑上尺度后的距离
        // ratio = mfMaxDistance/currentDist = ref_dist/cur_dist
        ratio = mfMaxDistance/currentDist;
    }

    // 同时取log线性化
    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

int MapLine::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}
#endif


} //ORB_SLAM2