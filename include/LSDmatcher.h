//
// Created by max on 19-6-24.
//

#ifndef ORB_SLAM2_LSDMATCHER_H
#define ORB_SLAM2_LSDMATCHER_H


//#include <line_descriptor/descriptor_custom.hpp>
//#include <line_descriptor_custom.hpp>


#include "MapLine.h"
#include "KeyFrame.h"
#include "Frame.h"

#include "auxiliar.h"

using namespace cv;
using namespace cv::line_descriptor;


namespace ORB_SLAM2
{
    class LSDmatcher
    {
    public:
        //construct
        LSDmatcher(float nnratio=0.6, bool checkOri=true);  //参数对应两个数据成员

        // 对上一帧的特征线进行追踪，和ORB中对比少了一个参数 const float th
        int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th=3, const bool bMono=false ); //输出是修改当前帧的数据成员，上一帧不改动

        // 对关键帧和当前帧做线匹配
        int SearchByProjection(KeyFrame *pKF, Frame &F, std::vector<MapLine*> &vpMapLineMatches);



        // 通过投影，对Local MapLine进行跟踪
        int SearchByProjection(Frame &F, const std::vector<MapLine*> &vpMapLines, const float th=3);
        // 该函数只有两个参数 第二个参数为引用传递 应该是把所有的地图线取出

        static int DescriptorDistance(const Mat &a, const Mat &b);

        // 严格比对ORB的话，这里应该写一个当前帧与关键帧之间的匹配函数，用于重定位，暂就不写了
        // Project MapPoints seen in KeyFrame into the Frame and search matches.
        // Used in relocalisation (Tracking)
//        int SearchByProjection(Frame &CurrentFrame, KeyFrame* pKF, const std::set<MapPoint*> &sAlreadyFound, const float th, const int ORBdist);

        // 这个函数应该是单目初始化的时候才会用到
//        int SerachForInitialize(Frame &InitialFrame, Frame &CurrentFrame, vector<pair<int,int>> &LineMatches);

        // 这个函数在生成地图线的时候会用到
        int SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, vector<pair<size_t, size_t>> &vMatchedPairs, const bool bOnlyStereo);
        // 我加了第三个参数const bool bOnlyStereo

        // Project MapLines into KeyFrame and search for duplicated MapLines
        int Fuse(KeyFrame* pKF, const vector<MapLine *> &vpMapLines);

        // ORB中写法如下，会多一个参数const float th
        // Project MapPoints into KeyFrame and search for duplicated MapPoints.
        // int Fuse(KeyFrame* pKF, const vector<MapPoint *> &vpMapPoints, const float th=3.0);

        // SearchBySim3 以及涉及闭环带尺度的Fuse函数等，也就是说闭环策略都是沿用ORBSLAM

    public:
        static const int TH_LOW;
        static const int TH_HIGH;
        static const int HISTO_LENGTH;


    protected:
        float mfNNratio;
        bool mbCheckOrientation;

        float RadiusByViewingCos(const float &viewCos); //通过观测视角确定半径 为了搜索用

    };

}






#endif //ORB_SLAM2_LSDMATCHER_H
