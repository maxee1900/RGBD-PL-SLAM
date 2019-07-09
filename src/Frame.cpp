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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>
#include "LocalMapping.h"  //is it need?

namespace ORB_SLAM2
{

//在类的外部定义静态数据成员,静态数据成员必须要有类外的初始化
long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;  //没有赋值，应该会初始化为随机值，然后在构造函数中会修改
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}


/**
 * @brief Copy constructor
 *
 * 复制构造函数, mLastFrame = Frame(mCurrentFrame)
 */
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N),  NL(frame.NL), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier),

     mvKeylines(frame.mvKeylines), mvKeylinesUn(frame.mvKeylinesUn), mvuRightLineStart(frame.mvuRightLineStart), mvuRightLineEnd(frame.mvuRightLineEnd), mvDepthLineStart(frame.mvDepthLineStart), mvDepthLineEnd(frame.mvDepthLineEnd), mLdesc(frame.mLdesc), mvKeyLineFunctions(frame.mvKeyLineFunctions), mvpMapLines(frame.mvpMapLines), mvbLineOutlier(frame.mvbLineOutlier),

     mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];  //元素为vector类型

    if(!frame.mTcw.empty())  //这个直接在初始化列表中进行也可以啊
        SetPose(frame.mTcw);
}


// todo 思考一个问题：这个函数参数中传入了ORB的提取器，为什么不传入直线的提取器，不传入有什么影响吗
///  RGBD初始化，重点关注！
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)  //共9个参数
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(nullptr)),  //右目ORB提取器初始化为NULL指针
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)   //注意Mat的拷贝都要用clone函数。初始化列表中对imGray和imDepth不操作
{
    // Frame ID
    mnId=nNextId++;  //这里就可以看出：为什么mnId不是静态成员，nNextId是静态成员。 mnId从0开始计数

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    //ExtractORB(0,imGray);  //第一个参数为0，提取左目特征点

#if 1
        // 自己添加修改，同时对两种特征进行提取
//    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        thread threadPoint(&Frame::ExtractORB, this, 0, imGray);
        thread threadLine(&Frame::ExtractLSD, this, imGray);
        threadPoint.join();
        threadLine.join();
//    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
//    cout << "featureEx time: " << time_used.count() << endl;
//    ofstream file("extractFeatureTime.txt", ios::app);
//    file << time_used.count() << endl;
//    file.close();
#endif


    N = mvKeys.size();  //上一个函数已经更新了mvKeys
    NL = mvKeylines.size();

    if(mvKeys.empty() && mvKeylines.empty())
        return;

    // 调用OpenCV的矫正函数矫正orb提取的特征点，在前面加了修正图片的函数，这里就不需要了
    // check 上述调用opencv矫正函数在什么地方，其次其他的地方更新过来了吗
    // todo_ 下面这个函数在lan版本中没有用，这里肯定需要改。这里如果采用下面这个函数，那么对于直线也要去畸变
    UndistortKeyPoints();
    // --line--
    UndistortKeyLines();


    // 深度存放在mvuRight 和 mvDepth中。
    // TODO_ 改写这个函数，考虑引入两个数据成员mvuLineRight mvLineDepth
    ComputeStereoFromRGBD(imDepth);


    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // --line--
    mvpMapLines = vector<MapLine*>(NL,static_cast<MapLine*>(NULL));
    mvbLineOutlier = vector<bool>(NL,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)  //默认值为true
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);  //@ 这里暂时还没理解，先跳过不影响总体
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();  //这个函数：对于每一帧图像一共48*64个格子，将相应的特征点放入格子中
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    // 在mGrid中记录了各特征点
    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);  //如果kp特征点在该格子内加入
    }
}

////
void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);   //ORB提取器的初始化. 这里会调用ORBextractor类的()运算符重载函数
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight); //flag=1提取右目
}


//// --line--
void Frame::ExtractLSD(const cv::Mat &im)  //这里提取的特征存入了mvKeylines（未校正的）
{
    mpLineSegment->ExtractLineSegment(im, mvKeylines, mLdesc, mvKeyLineFunctions);
}


//// --line-- line descriptor MAD 计算两个线特征分布的中值绝对偏差
void Frame::lineDescriptorMAD( vector<vector<DMatch>> line_matches, double &nn_mad, double &nn12_mad) const {
    vector<vector<DMatch>> matches_nn, matches_12;
    matches_nn = line_matches;
    matches_12 = line_matches;
//    cout << "Frame::lineDescriptorMAD——matches_nn = "<<matches_nn.size() << endl;

    // 1. estimate the NN's distance standard deviation
    double nn_dist_median;
    sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
    //以每个特征所找到匹配的最小距离进行排序
    nn_dist_median = matches_nn[int(matches_nn.size() / 2)][0].distance;

    for (unsigned int i = 0; i < matches_nn.size(); i++)
        matches_nn[i][0].distance = fabsf(matches_nn[i][0].distance - nn_dist_median);

    sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
    nn_mad = 1.4826 * matches_nn[int(matches_nn.size() / 2)][0].distance;


    // 2. estimate the NN's 12 distance standard deviation
    double nn12_dist_median;
    sort(matches_12.begin(), matches_12.end(), compare_descriptor_by_NN12_dist());
    //以每个特征所找到匹配的最小距离和次小距离之间的差值进行排序
    nn12_dist_median = matches_12[int(matches_12.size() / 2)][1].distance - matches_12[int(matches_12.size() / 2)][0].distance;

    for (unsigned int j = 0; j < matches_12.size(); j++)
        matches_12[j][0].distance = fabsf(matches_12[j][1].distance - matches_12[j][0].distance - nn12_dist_median);

    sort(matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist());
    nn12_mad = 1.4826 * matches_12[int(matches_12.size() / 2)][0].distance;

}


/**
 * @brief Set the camera pose.
 * 
 * 设置相机姿态，随后会调用 UpdatePoseMatrices() 来改变mRcw,mRwc等变量的值
 * @param Tcw Transformation from world to camera
 */
void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

/**
 * @brief Computes rotation, translation and camera center matrices from the camera pose.
 *
 * 根据Tcw计算mRcw、mtcw和mRwc、mOw
 */
void Frame::UpdatePoseMatrices()
{
    // [x_camera 1] = [R|t]*[x_world 1]，坐标为齐次形式
    // x_camera = R*x_world + t
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();  //Mat的数据结构直接调用.t()即为转置
    mtcw = mTcw.rowRange(0,3).col(3);   //前三行　最后一列
    // mtcw, 即相机坐标系下相机坐标系到世界坐标系间的向量, 向量方向由相机坐标系指向世界坐标系
    // mOw, 即mtwc, 相机到世界的转换矩阵中的平移分量。 即世界坐标系下世界坐标系到相机坐标系间的向量, 向量方向由世界坐标系指向相机坐标系
    //也就是相机光心在世界坐标系中位置　
    mOw = -mRcw.t()*mtcw;  //twc = -Rcw.t() * tcw
}

/**
 * @brief 判断一个点是否在视野内
 *
 * 计算了重投影坐标，观测方向夹角，预测在当前帧的尺度
 * @param  pMP             MapPoint
 * @param  viewingCosLimit 视角和平均视角的方向阈值
 * @return                 true if is in view
 * @see SearchLocalPoints()
 */
 // 基于ORB，没有改动，在SearchLocalPoints函数中会用到
 // 同样滴我是不是应该仿写一个函数，在SearchLocalLines中调用？下面有
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    // 3D点P在相机坐标系下的坐标
    const cv::Mat Pc = mRcw*P+mtcw; // 这里的Rt是经过初步的优化后的
    const float &PcX = Pc.at<float>(0);
    const float &PcY = Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    // V-D 1) 将MapPoint投影到当前帧, 并判断是否在图像内
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    // V-D 3) 计算MapPoint到相机中心的距离, 并判断是否在尺度变化的距离内
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    // 世界坐标系下，PO为相机到3D点P的向量, 向量方向由相机指向3D点P
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);  //求该向量的二范数,也即是光心到3D点的直线距离

    if(dist<minDistance || dist>maxDistance)  //该距离有一个范围（满足尺度不变性的话）
        return false;

    // Check viewing angle
    // V-D 2) 计算当前视角和平均视角夹角的余弦值, 若小于cos(60), 即夹角大于60度则返回false
    cv::Mat Pn = pMP->GetNormal();
    //pMP为地图点（类）。这个函数应该返回的是地图点自身的属性，可能是在地图点建立的时候赋予了相机和地图点的观测角度向量

    const float viewCos = PO.dot(Pn)/dist;  //作为Mat类 PO有.dot()点乘函数

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    // V-D 4) 根据深度预测尺度（对应特征点在一层）
    const int nPredictedLevel = pMP->PredictScale(dist,this);  //这里的dist不是Z轴的深度 而是光心到点的距离

    // Data used by the tracking
    // 标记该点将来要被投影
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz; //该3D点投影到双目右侧相机上的横坐标
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel = nPredictedLevel;
    pMP->mTrackViewCos = viewCos;
    //可见判断点是否在图像上的函数，最后如果满足条件也赋予了地图点的很多属性，所有第一个参数地图点是指针传递

    return true;
}


/**
 * @brief 判断MapLine的两个端点是否在视野内
 *
 * @param pML               MapLine
 * @param viewingCosLimit   视角和平均视角的方向阈值
 * @return                  true if the MapLine is in view
 */
// todo！！ 这里没有考虑一个端点在视角时候怎么办
bool Frame::isInFrustum(MapLine *pML, float viewingCosLimit)
{
    pML->mbTrackInView = false;

    Vector6d P = pML->GetWorldPos();

    cv::Mat SP = (Mat_<float>(3,1) << P(0), P(1), P(2));
    cv::Mat EP = (Mat_<float>(3,1) << P(3), P(4), P(5));

    // 两个端点在相机坐标系下的坐标
    const cv::Mat SPc = mRcw*SP + mtcw;
    const float &SPcX = SPc.at<float>(0);
    const float &SPcY = SPc.at<float>(1);
    const float &SPcZ = SPc.at<float>(2);

    const cv::Mat EPc = mRcw*EP + mtcw;
    const float &EPcX = EPc.at<float>(0);
    const float &EPcY = EPc.at<float>(1);
    const float &EPcZ = EPc.at<float>(2);

    // 检测两个端点的Z值是否为正
    if(SPcZ<0.0f || EPcZ<0.0f)
        return false;

    // V-D 1) 将端点投影到当前帧上，并判断是否在图像内
    const float invz1 = 1.0f/SPcZ;
    const float u1 = fx * SPcX * invz1 + cx;
    const float v1 = fy * SPcY * invz1 + cy;

    if(u1<mnMinX || u1>mnMaxX)
        return false;
    if(v1<mnMinY || v1>mnMaxY)
        return false;

    const float invz2 = 1.0f/EPcZ;
    const float u2 = fx*EPcX*invz2 + cx;
    const float v2 = fy*EPcY*invz2 + cy;

    if(u2<mnMinX || u2>mnMaxX)
        return false;
    if(v2<mnMinY || v2>mnMaxY)
        return false;

    // V-D 3)计算MapLine到相机中心的距离，并判断是否在尺度变化的距离内
    const float maxDistance = pML->GetMaxDistanceInvariance();
    const float minDistance = pML->GetMinDistanceInvariance();
    // 世界坐标系下，相机到线段中点的向量，向量方向由相机指向中点
    const cv::Mat OM = 0.5*(SP+EP) - mOw;
    const float dist = cv::norm(OM);

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    // V-D 2)计算当前视角和平均视角夹角的余弦值，若小于cos(60°),即夹角大于60°则返回
    Vector3d Pn = pML->GetNormal();
    cv::Mat pn = (Mat_<float>(3,1) << Pn(0), Pn(1), Pn(2));
    const float viewCos = OM.dot(pn)/dist;

    if(viewCos<viewingCosLimit)  //todo 这个参数需要改啊夹角太大就不在视野了
        return false;

    // Predict scale in the image
    // V-D 4) 根据深度预测尺度（对应特征在一层）
    const int nPredictedLevel = pML->PredictScale(dist, mfLogScaleFactor);
    // orb中是这样操作的
    // const int nPredictedLevel = pMP->PredictScale(dist,this); //也就是说第二个参数为当前帧
    // TODO 这里我们可以把这个函数改成类似与ORB中的样子；或者就修改PredictScale(dist, mfLogScaleFactor)的函数实现，使得和ORB中保持一致

    // Data used by the tracking
    // 标记该特征将来要被投影
    pML->mbTrackInView = true;
    pML->mTrackProjX1 = u1;
    pML->mTrackProjY1 = v1;
    pML->mTrackProjX2 = u2;
    pML->mTrackProjY2 = v2;
    pML->mnTrackScaleLevel = nPredictedLevel;
    pML->mTrackViewCos = viewCos;

    return true;
}




/**
 * @brief 找到在 以x,y为中心,边长为2r的方形内且在[minLevel, maxLevel]的特征点
 * @param x        图像坐标u
 * @param y        图像坐标v
 * @param r        边长
 * @param minLevel 最小尺度
 * @param maxLevel 最大尺度
 * @return         满足条件的特征点的序号
 */
 // 这个函数是在ORBmatchers用到，SearchByProjection中会用到
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}


//// --line-- 类比上面的函数，这里写一个得到某一个领域候选特征线的函数
// 传入参数和上面差不多，变为了两个端点的坐标
// 这个函数实际上就是三步： 1.中点距离 2.角度 3.金字塔层数
// 但是如果两条直线不是严格端点匹配，对比中点显得不合理；其次金字塔层数，他美的不是提取了一层，还比较什么
// todo!  要改啊！并测试看看直线的端点是不是会严格匹配，直线的长度差异大不大，其次层数到底是怎么回事
// 这个函数写的还行 不改好像也行的
vector<size_t> Frame::GetLinesInArea(const float &x1, const float &y1, const float &x2, const float &y2, const float &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;  //存放特征线的索引号

    vector<KeyLine> vkl = this->mvKeylinesUn;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>0);

    for(size_t i=0; i<vkl.size(); i++)
    {

        KeyLine keyline = vkl[i];

        // 1.对比中点距离
        // todo 有时候两个直线的端点不是很严格匹配的时候，比较中点距离显得不是很好，可以考虑这一步策略取消试试
        double distance = (0.5*(x1+x2)-keyline.pt.x) * (0.5*(x1+x2)-keyline.pt.x)+(0.5*(y1+y2)-keyline.pt.y) * (0.5*(y1+y2)-keyline.pt.y);
        if(distance > r*r)
            continue;

        // 2.比较斜率，KeyLine的angle就是代表斜率 todo are you sure? 没毛病
        float slope = (y1-y2)/(x1-x2)-keyline.angle;  ///这里说明opencv中求出的特征线可以得到角度
        if(slope > r*0.01)  //这里的参数也可以调节
            continue;

        // 3.比较金字塔层数
        if(bCheckLevels)
        {
            if(keyline.octave<minLevel)  ///可见opencv中线的提取也是有尺度的！注意凡是涉及到尺度层的一些操作要保证和ORB中类似，切勿出错
//            cout << "线特征的octave： " << keyline.octave << endl;
                continue;
            if(maxLevel>=0 && keyline.octave>maxLevel)
                continue;
        }

        vIndices.push_back(i);

    }

    return vIndices;
}




bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

/**
 * @brief Bag of Words Representation。  对于每一个帧的词典表示，都会有这两个量  mBowVec 和 mFeatVec
 *
 * 计算词包mBowVec和mFeatVec，其中mFeatVec记录了属于第i个node（在第4层）的ni个描述子
 * @see CreateInitialMapMonocular() TrackReferenceKeyFrame() Relocalization()
 */
 //计算了当前帧的BowVec向量，以及它的第4层正向索引值FeatVec ///BowVec为BoW特征向量，FeatVec为正向索引；
void Frame::ComputeBoW()  ///@@ 找时间详细看一下博:DBow库的介绍
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);  //这个函数是把帧上的描述子矩阵转化为
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);  //这个是调用了Dbow库的函数，第一个参数为描述子的列表
    }
}

// 调用OpenCV的矫正函数矫正orb提取的特征点  check!
void Frame::UndistortKeyPoints()
{
    // 如果没有图像是矫正过的，没有失真. 怎么样矫正图像？？
    if(mDistCoef.at<float>(0)==0.0) ///
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    // N为提取的特征点数量，将N个特征点保存在N*2的mat中
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    // 调整mat的通道为2，矩阵的行列形状不变
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK); // 用cv的函数进行失真校正. 调用cv中现成的函数
    // 这里注意mDistCoef是Mat类型，该函数直接修改了这个变量
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    // 存储校正后的特征点
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::UndistortKeyLines()
{
    // 如果没有图像是矫正过的，没有失真
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeylinesUn=mvKeylines;
        return;
    }

    // NL为提取的特征线数量，起点和终点分别保存在NL*2的mat中
    cv::Mat matS(NL,2,CV_32F);
    cv::Mat matE(NL,2,CV_32F);
    for(int i=0; i<NL; i++)
    {
        matS.at<float>(i,0)=mvKeylines[i].startPointX;
        matS.at<float>(i,1)=mvKeylines[i].startPointY;
        matE.at<float>(i,0)=mvKeylines[i].endPointX;
        matE.at<float>(i,1)=mvKeylines[i].endPointY;
    }

    matS = matS.reshape(2);
    cv::undistortPoints(matS,matS,mK,mDistCoef,cv::Mat(),mK);
    matS = matS.reshape(1);

    matE = matE.reshape(2);
    cv::undistortPoints(matE,matE,mK,mDistCoef,cv::Mat(),mK);
    matE = matE.reshape(1);

    mvKeylinesUn.resize(NL);
    for(int i=0; i<NL; i++)
    {
        KeyLine kl = mvKeylines[i];
        kl.startPointX = matS.at<float>(i,0);
        kl.startPointY = matS.at<float>(i,1);
        kl.endPointX = matE.at<float>(i,0);
        kl.endPointY = matE.at<float>(i,1);
        mvKeylinesUn[i] = kl;
    }

}

//// 没有改动，计算了去畸变后图像的四个边缘点位置
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)  //说明mDistCoef矩阵不为空，有畸变
    {
        // 矫正前四个边界点：(0,0) (cols,0) (0,rows) (cols,rows)
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0;         //左上
		mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; //右上
		mat.at<float>(1,1)=0.0;
		mat.at<float>(2,0)=0.0;         //左下
		mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; //右下
		mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));//左上和左下横坐标最小的
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));//右上和右下横坐标最大的
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));//左上和右上纵坐标最小的
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));//左下和右下纵坐标最小的
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}


///*****RGBD******融入了线的部分
void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    // mvDepth直接由depth图像读取
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvuRightLineStart = vector<float>(NL,-1);
    mvuRightLineEnd = vector<float>(NL,-1);
    mvDepthLineStart = vector<float>(NL,-1);
    mvDepthLineEnd = vector<float>(NL,-1);

    for(int i=0; i<N; i++)  //点
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;  //TODO 测试！用去畸变坐标来取深度会怎么样
        const float &u = kp.pt.x;  //可以测试一下！

        const float d = imDepth.at<float>(v,u);

        if(d>0)  ///注意！可见深度图上有对应的深度，那么这两个地方存放的就不是-1，这里不管是不是近点
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d; //这里指的是像素坐标，所以一定是大于0的
             ///注意： 这里按照slam书上得到公式uR=uL - fb/d. 实际中UR的坐标应该为 uR* = uR + cx.
             ///为什么按照上面的公式计算出来没问题呢，因为上面公式中的第一个数为像素点的横坐标 实际上等价于 uR = uL+cx-fb/d。因此mvuRight中都是正数，单目的话都为-1.
        }
    }

    for(int i=0; i<NL; i++)  //线
    {
        const KeyLine &kl = mvKeylines[i];
        const KeyLine &klU = mvKeylinesUn[i];

        const float &vS = kl.startPointY;
        const float &uS = kl.startPointX;
        const float &vE = kl.endPointY;
        const float &uE = kl.endPointX;

        const float dS = imDepth.at<float>(vS,uS);
        const float dE = imDepth.at<float>(vE,uE);

        if(dS>0)
        {
            mvDepthLineStart[i] = dS;
            mvuRightLineStart[i] = klU.startPointX - mbf/dS;
        }

        if(dE>0)
        {
            mvDepthLineEnd[i] = dE;
            mvuRightLineEnd[i] = klU.endPointX - mbf/dE;
        }
    }

}

/**
 * @brief Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
 * @param  i 第i个keypoint
 * @return   3D点（相对于世界坐标系）
 */
 // 没有改动,主要用在Tracking::CreatNewKeyframe函数中
cv::Mat Frame::UnprojectStereo(const int &i)
{
    // KeyFrame::UnprojectStereo
    // mvDepth是在ComputeStereoMatches函数中求取的
    // mvDepth对应的校正前的特征点，可这里却是对校正后特征点反投影
    // KeyFrame::UnprojectStereo中是对校正前的特征点mvKeys反投影
    // 在ComputeStereoMatches函数中应该对校正后的特征点求深度？？ (wubo???)
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;   // 返回的是3D的 Pw = Twc * Pc
    }
    else
        return cv::Mat();

}

/// 仿写上面的函数  这里是两个端点
cv::Mat Frame::UnprojectStereoLine(const int &i)
{
    const float zs = mvDepthLineStart[i];
    cv::Mat xs3Dw = (cv::Mat_<float>(3,1) << 0,0,0);
    cv::Mat xe3Dw = (cv::Mat_<float>(3,1) << 0,0,0);
    if(zs>0)
    {
        const float us = mvKeylinesUn[i].startPointX;
        const float vs = mvKeylinesUn[i].startPointY;
        const float xs = (us-cx)*zs*invfx;
        const float ys = (vs-cy)*zs*invfx;
        cv::Mat xs3Dc = (cv::Mat_<float>(3,1) << xs,ys,zs );
        xs3Dw = mRwc*xs3Dc + mOw;
    }

    const float ze = mvDepthLineEnd[i];
    if(ze>0)
    {
        const float ue = mvKeylinesUn[i].endPointX;
        const float ve = mvKeylinesUn[i].endPointY;
        const float xe = (ue-cx)*zs*invfx;
        const float ye = (ve-cy)*zs*invfx;
        cv::Mat xe3Dc = (cv::Mat_<float>(3,1) << xe,ye,ze );
        xe3Dw = mRwc*xe3Dc + mOw;
    }

    cv::Mat line3Dw = (cv::Mat_<float>(6,1) <<
                       xs3Dw.at<float>(0), xs3Dw.at<float>(1), xs3Dw.at<float>(2),
                       xe3Dw.at<float>(0), xe3Dw.at<float>(1), xe3Dw.at<float>(2));

    if(zs>0 && ze>0)
        return line3Dw;
    else
        return cv::Mat();

}

cv::Mat Frame::UnprojectStereoLineStart(const int &i)
{
    const float zs = mvDepthLineStart[i];

    if(zs>0)
    {
        const float us = mvKeylinesUn[i].startPointX;
        const float vs = mvKeylinesUn[i].startPointY;
        const float xs = (us-cx)*zs*invfx;
        const float ys = (vs-cy)*zs*invfx;
        cv::Mat xs3Dc = (cv::Mat_<float>(3,1) << xs,ys,zs );
        return mRwc*xs3Dc + mOw;
    }
    else
        return cv::Mat();
}

cv::Mat Frame::UnprojectStereoLineEnd(const int &i)
{
    const float ze = mvDepthLineEnd[i];

    if(ze>0)
    {
        const float ue = mvKeylinesUn[i].endPointX;
        const float ve = mvKeylinesUn[i].endPointY;
        const float xe = (ue-cx)*ze*invfx;
        const float ye = (ve-cy)*ze*invfx;
        cv::Mat xe3Dc = (cv::Mat_<float>(3,1) << xe,ye,ze );
        return mRwc*xe3Dc + mOw;
    }
    else
        return cv::Mat();
}


} //namespace ORB_SLAM
