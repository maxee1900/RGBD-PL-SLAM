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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Frame.h"

#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM2
{

class LoopClosing;

class Optimizer   //Optimizer中的成员函数原来这么少，而且不需要数据成员
{
public:
    //该函数在全局BA中会调用, 注意这些函数带有static，静态成员函数
    void static BundleAdjustment(const std::vector<KeyFrame*> &vpKF, const std::vector<MapPoint*> &vpMP,
                                 int nIterations = 5, bool *pbStopFlag=NULL, const unsigned long nLoopKF=0,   //迭代数、停止标识、闭环帧数量？、是否鲁棒四个参数
                                 const bool bRobust = true);
    void static GlobalBundleAdjustemnt(Map* pMap, int nIterations=5, bool *pbStopFlag=NULL,      //第一个参数为地图，后四个参数和BundleAdjustment函数一致，而且类型也是一样的
                                       const unsigned long nLoopKF=0, const bool bRobust = true);
    void static LocalBundleAdjustment(KeyFrame* pKF, bool *pbStopFlag, Map *pMap);  //参数为关键帧，停止标识（即mbAbortBA），地图
    int static PoseOptimization(Frame* pFrame);   //因为需要改变输入参数，所以一般都是指针传递

    // if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)    //@ 这个函数在RGBD相机时也会调用吗？ correctLoop函数中会调用
    void static OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,    //参数为地图、闭环关键帧、当前关键帧， 剩下的都是常量引用传递
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       //注意这里KeyFrameAndPose是类LoopClosing中为现有类型定义的别名，这应该不算是变量，故不算做类的数据成员，是一种新的类型，可以用类名调用LoopClosing::
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections,
                                       const bool &bFixScale);

    // if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono)  ComputeSim3函数中会调用
    static int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches1,   //这里的返回值是静态int类型
                            g2o::Sim3 &g2oS12, const float th2, const bool bFixScale);


    //=============上面是原来的ORBSLAM中函数，下面改写为包含线特征的=======================

    // 包含了线特征的BA
    void static BundleAdjustmentWithLine(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP, const vector<MapLine *> &vpML,
                                 int nIterations = 5, bool* pbStopFlag=NULL, const unsigned long nLoopKF=0,
                                 const bool bRobust = true);

    // 包含线的全局BA
    void static GlobalBundleAdjustmentWithLine(Map* pMap, int nIterations=5, const bool bWithLine=false,
                                 bool *pbStopFlag=NULL, const unsigned long nLoopKF=0, const bool bRobust = true);

    // 包含线的局部BA
    void static LocalBundleAdjustmentWithLine(KeyFrame* pKF, bool *pbStopFlag, Map *pMap);

    int static PoseOptimizationWithLine(Frame* pFrame); //todo 这个函数待定，不知道有没有加入线


};

} //namespace ORB_SLAM

#endif // OPTIMIZER_H
