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

        // 对上一帧的特征线进行追踪
        int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame); //输出是修改当前帧的数据成员，上一帧不改动

        // 通过投影，对Local MapLine进行跟踪
        int SearchByProjection(Frame &F, const std::vector<MapLine*> &vpMapLines, const float th=3);
        // 该函数只有两个参数 第二个参数为引用传递 应该是把所有的地图线取出

        static int DescriptorDistance(const Mat &a, const Mat &b);

        // todo 这个是不是只有单目才会有
        int SerachForInitialize(Frame &InitialFrame, Frame &CurrentFrame, vector<pair<int,int>> &LineMatches);

        //
        int SerachForInitialize(Frame &InitialFrame, Frame &CurrentFrame, vector<pair<int,int>> &LineMatches);

        // Project MapLines into KeyFrame and search for duplicated MapLines
        int Fuse(KeyFrame* pKF, const vector<MapLine *> &vpMapLines);

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
