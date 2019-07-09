//
// Created by max on 19-6-25.
// 

#include "LSDmatcher.h"

using namespace std;

namespace ORB_SLAM2
{
    const int LSDmatcher::TH_HIGH = 100;
    const int LSDmatcher::TH_LOW = 50;
    const int LSDmatcher::HISTO_LENGTH = 30;

    LSDmatcher::LSDmatcher(float nnratio, bool checkOri):mfNNratio(nnratio), mbCheckOrientation(checkOri)
    {
    }

/// 通过投影，对上一帧的特征线进行追踪  如果我在TODO后面加了！，说明是重点
//todo! 根据ORBSLAM，这个函数有很大问题，都没有更新当前帧的直线匹配
//todo 这个函数相当于两帧之间的直线匹配，非常重要，后续再来改！！ 主要是第一增加一些判断长度或角的策略，增加旋转一致性检测（类似ORB），最后要更新当前帧的匹配，总之依照ORB中这个函数的思想来改！
/**函数应该类似完成以下功能
 * @brief 通过投影，对上一帧的特征点进行跟踪
 *
 * 上一帧中包含了MapPoints，对这些MapPoints进行tracking，由此增加当前帧的MapPoints \n
 * 1. 将上一帧的MapPoints投影到当前帧(根据速度模型可以估计当前帧的Tcw)
 * 2. 在投影点附近根据描述子距离选取匹配，以及最终的方向投票机制进行剔除
 * @param  CurrentFrame 当前帧
 * @param  LastFrame    上一帧
 * @param  th           阈值
 * @param  bMono        是否为单目
 * @return              成功匹配的数量
 */

int LSDmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
{
    // 匹配特征线的数量，最终要返回该值
    int nmatches = 0;
    BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
    Mat ldesc1, ldesc2;
    vector<vector<DMatch>> lmatches;
    ldesc1 = LastFrame.mLdesc;
    ldesc2 = CurrentFrame.mLdesc;
    bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

    double nn_dist_th, nn12_dist_th;
    CurrentFrame.lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
    nn12_dist_th = nn12_dist_th*0.5;
    sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
    for(int i=0; i<lmatches.size(); i++)
    {
        int qdx = lmatches[i][0].queryIdx;
        int tdx = lmatches[i][0].trainIdx;
        double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
        if(dist_12>nn12_dist_th)
        {
            MapLine* mapLine = LastFrame.mvpMapLines[qdx];
            nmatches++;
        }
    }
    return nmatches;
}

/// 我自己新增的，用在Tracking::TrackRefrenceKeyframe函数中
int LSDmatcher::SearchByProjection(KeyFrame *pKF, Frame &F, std::vector<MapLine*> &vpMapLineMatches) //todo 写函数实现
{


}



/// 这个函数和ORB中函数符合度较高
int LSDmatcher::SearchByProjection(Frame &F, const std::vector<MapLine *> &vpMapLines, const float th)
{
    int nmatches = 0;

    const bool bFactor = th!=1.0;

    for(size_t iML=0; iML<vpMapLines.size(); iML++)
    {
        MapLine* pML = vpMapLines[iML];

        // 判断该线段是否要投影
        if(!pML->mbTrackInView)
            continue;

        if(pML->isBad())
            continue;

        // 通过距离预测的金字塔层数，该层数对应于当前的帧
        const int &nPredictLevel = pML->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction
        // 搜索窗口的大小取决于视角，若当前视角和平均视角接近0°时，r取一个较小的值
        float r = RadiusByViewingCos(pML->mTrackViewCos);

        // 如果需要进行跟粗糙的搜索，则增大范围
        if(bFactor)
            r*=th;

        // 通过投影线段以及搜索窗口和预测的尺度进行搜索，找出附近可能的匹配线段
        vector<size_t> vIndices =
                F.GetLinesInArea(pML->mTrackProjX1, pML->mTrackProjY1, pML->mTrackProjX2, pML->mTrackProjY2,
                                 r*F.mvScaleFactors[nPredictLevel], nPredictLevel-1, nPredictLevel);

        if(vIndices.empty())
            continue;

        const cv::Mat MLdescriptor = pML->GetDescriptor();

        int bestDist=256;
        int bestLevel= -1;
        int bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        // Get best and second matches with near keylines
        // 以下这段代码逻辑和ORB中同，没问题
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            // 如果Frame中的该兴趣点已经有对应的MapLine了,则退出该次循环
            if(F.mvpMapLines[idx])
                if(F.mvpMapLines[idx]->Observations()>0)
                    continue;

            /* 这里是ORB中的代码，表示如果特征的右目坐标和地图特征的右目投影差别过大也退出这次循环
               if(F.mvuRight[idx]>0)
           {
               const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
               if(er>r*F.mvScaleFactors[nPredictedLevel])
                   continue;
           }
            */

            const cv::Mat &d = F.mLdesc.row(idx);

            const int dist = DescriptorDistance(MLdescriptor, d);

            // 根据描述子寻找描述子距离最小和次小的特征点
            if(dist<bestDist)
            {
                bestDist2 = bestDist;
                bestDist = dist;

                bestLevel2 = bestLevel;  //和上述一样，更新最小和次小所对应的层
                bestLevel = F.mvKeylinesUn[idx].octave; //todo 测试.octave等于什么，按理说直线提取只取了一层，这里的层数应为0？？

                bestIdx = idx;
            }
            else if(dist < bestDist2)
            {
                bestLevel2 = F.mvKeylinesUn[idx].octave;
                bestDist2 = dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if(bestDist <= TH_HIGH)  //最小距离达到了阈值
        {
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                continue;
            // 当最小和次小在同一层的时候，考虑 besDist/bestDist2 > mfNNratio
            // ratio越大，说明最小和次小越接近，不是个好的匹配，跳过

            F.mvpMapLines[bestIdx]=pML; // 为Frame中的兴趣线添加对应的MapLine
            nmatches++;
        }
    }
    return nmatches;
}



/// 计算二进制描述子之间的距离，这里完全和ORB中一样.应该没问题吧
int LSDmatcher::DescriptorDistance(const Mat &a, const Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}


/// Matching to triangulate new MapPoints. Check Epipolar Constraint.
// todo 这个函数也需要改！类似于ORB，比如检查旋转一致性
// 函数返回的是两个关键帧之间特征匹配的索引号对
int LSDmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2,
                                       vector<pair<size_t, size_t>> &vMatchedPairs, const bool bOnlyStereo)
{
    vMatchedPairs.clear();
    int nmatches = 0;
    BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
    Mat ldesc1, ldesc2;
    vector<vector<DMatch>> lmatches;
    ldesc1 = pKF1->mLineDescriptors;
    ldesc2 = pKF2->mLineDescriptors;
    bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

    double nn_dist_th, nn12_dist_th;
    pKF1->lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
    nn12_dist_th = nn12_dist_th*0.1;
    sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
    for(int i=0; i<lmatches.size(); i++)
    {
        int qdx = lmatches[i][0].queryIdx;
        int tdx = lmatches[i][0].trainIdx;
        double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
        if(dist_12>nn12_dist_th)
        {
            vMatchedPairs.push_back(make_pair(qdx, tdx));
            nmatches++;
        }
    }
    //todo 又是这样的套路，我觉得这样肯定有误匹配，学习一下ORB中的匹配

    return nmatches;
}

/// Project MapPoints into KeyFrame and search for duplicated MapPoints.
// TODO 这个函数的实现和ORB中的点的FUSE相比太弱了……需要改
int LSDmatcher::Fuse(KeyFrame *pKF, const vector<MapLine *> &vpMapLines)
{
    cv::Mat Rcw = pKF->GetRotation();  //关键帧的旋转和平移
    cv::Mat tcw = pKF->GetTranslation();

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    cv::Mat Ow = pKF->GetCameraCenter(); //关键帧的相机光心位置

    int nFused=0;

    Mat lineDesc = pKF->mLineDescriptors;   //待Fuse的关键帧的描述子，这是关键帧上线特征的所有描述子？
    const int nMLs = vpMapLines.size();  // 所有地图线的数量

    //遍历所有的MapLines
    for(int i=0; i<nMLs; i++)
    {
        MapLine* pML = vpMapLines[i];

        if(!pML)  // 地图线为空跳过
            continue;

        if(pML->isBad() || pML->IsInKeyFrame(pKF))  //第二个条件：关键帧在地图线的Observation中，map类型
            continue;
#if 0
        Vector6d LineW = pML->GetWorldPos();
        cv::Mat LineSW = (Mat_<float>(3,1) << LineW(0), LineW(1), LineW(2));
        cv::Mat LineSC = Rcw*LineSW + tcw;
        cv::Mat LineEW = (Mat_<float>(3,1) << LineW(3), LineW(4), LineW(5));
        cv::Mat LineEC = Rcw*LineEW + tcw;

        //Depth must be positive
        if(LineSC.at<float>(2)<0.0f || LineEC.at<float>(2)<0.0f)
            continue;

        // 获取起始点在图像上的投影坐标
        const float invz1 = 1/LineSC.at<float>(2);
        const float x1 = LineSC.at<float>(0)*invz1;
        const float y1 = LineSC.at<float>(1)*invz1;

        const float u1 = fx*x1 + cx;
        const float v1 = fy*y1 + cy;

        // 获取终止点在图像上的投影坐标
        const float invz2 = 1/LineEC.at<float>(2);
        const float x2 = LineEC.at<float>(0)*invz2;
        const float y2 = LineEC.at<float>(1)*invz2;

        const float u2 = fx*x2 + cx;
        const float v2 = fy*y2 + cy;
#endif
        Mat CurrentLineDesc = pML->mLDescriptor;        //MapLine[i]对应的线特征描述子,这应该是线特征的最优描述子

#if 0
        // 采用暴力匹配法,knnMatch
        BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
        vector<vector<DMatch>> lmatches;
        bfm->knnMatch(CurrentLineDesc, lineDesc, lmatches, 2);  //当前地图线的描述子和关键帧上线的所有描述子进行匹配
        double nn_dist_th, nn12_dist_th;
        pKF->lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
        nn12_dist_th = nn12_dist_th*0.1;
        sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
        for(int i=0; i<lmatches.size(); i++)
        {
            int tdx = lmatches[i][0].trainIdx;
            double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
            if(dist_12>nn12_dist_th)    //找到了pKF中对应ML
            {
                MapLine* pMLinKF = pKF->GetMapLine(tdx);
                if(pMLinKF)
                {
                    if(!pMLinKF->isBad())
                    {
                        if(pMLinKF->Observations()>pML->Observations())
                            pML->Replace(pMLinKF);
                        else
                            pMLinKF->Replace(pML);
                    }
                }
                nFused++;
            }
        }
#elif 1

        //todo 下面这部分代码似乎有问题呢。。参考下ORB中的代码
        // 使用暴力匹配法
        Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
        vector<DMatch> lmatches;
        matcher->match ( CurrentLineDesc, lineDesc, lmatches ); // 一个描述子和很多个描述子进行匹配，输出的是一维数组？

        double max_dist = 0;
        double min_dist = 100;

        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < CurrentLineDesc.rows; i++ )  //todo CurrentLineDesc.row?这应该是一个特征线的描述子啊？ 修改为lmatches.size()看看
        {
            double dist = lmatches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        // "good" matches (i.e. whose distance is less than 2*min_dist ) todo 这样不就变成了一个地图线特征对应很多个关键帧上的线特征了吗？
        std::vector< DMatch > good_matches;
        for( int i = 0; i < CurrentLineDesc.rows; i++ )  // 与上同修改
        {
            if( lmatches[i].distance < 1.5*min_dist )
            {
                int tdx = lmatches[i].trainIdx;
                MapLine* pMLinKF = pKF->GetMapLine(tdx);
                if(pMLinKF)
                {
                    if(!pMLinKF->isBad())
                    {
                        if(pMLinKF->Observations()>pML->Observations())
                            pML->Replace(pMLinKF);
                        else
                            pMLinKF->Replace(pML);
                    }
                }
                nFused++;
            }
        }

#else
            cout << "CurrentLineDesc.empty() = " << CurrentLineDesc.empty() << endl;
            cout << "lineDesc.empty() = " << lineDesc.empty() << endl;
            cout << CurrentLineDesc << endl;
            if(CurrentLineDesc.empty() || lineDesc.empty())
                continue;

            // 采用Flann方法
            FlannBasedMatcher flm;
            vector<DMatch> lmatches;
            flm.match(CurrentLineDesc, lineDesc, lmatches);

            double max_dist = 0;
            double min_dist = 100;

            //-- Quick calculation of max and min distances between keypoints
            cout << "CurrentLineDesc.rows = " << CurrentLineDesc.rows << endl;
            for( int i = 0; i < CurrentLineDesc.rows; i++ )
            { double dist = lmatches[i].distance;
                if( dist < min_dist ) min_dist = dist;
                if( dist > max_dist ) max_dist = dist;
            }

            // "good" matches (i.e. whose distance is less than 2*min_dist )
            std::vector< DMatch > good_matches;
            for( int i = 0; i < CurrentLineDesc.rows; i++ )
            {
                if( lmatches[i].distance < 2*min_dist )
                {
                    int tdx = lmatches[i].trainIdx;
                    MapLine* pMLinKF = pKF->GetMapLine(tdx);
                    if(pMLinKF)
                    {
                        if(!pMLinKF->isBad())
                        {
                            if(pMLinKF->Observations()>pML->Observations())
                                pML->Replace(pMLinKF);
                            else
                                pMLinKF->Replace(pML);
                        }
                    }
                    nFused++;
                }
            }
#endif
        }
        return nFused;
    }


float LSDmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 5.0;
    else
        return 8.0;
}


}  // namespace ORB_SLAM2
