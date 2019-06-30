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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"    //可见这里的g2o是用的是程序中自带的第三方库中的版本
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include "lineEdge.h"  //lan版本中在头文件中添加，这里我加在源文件中

#include<mutex>

namespace ORB_SLAM2
{

// pMap中所有的MapPoints和关键帧做bundle adjustment优化
// 这个全局BA优化在本程序中有两个地方使用：
// a.单目初始化：CreateInitialMapMonocular函数
// b.闭环优化：RunGlobalBundleAdjustment函数
//这个函数做了较好的封装，输入主要参数为地图，在内部得到地图的关键帧列表和地图点列表，然后调用核心函数BundleAdjustment
void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
}

/// --line--
void Optimizer::GlobalBundleAdjustmentWithLine(Map* pMap, int nIterations, const bool bWithLine,
                                           bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust)  //函数实现的时候一般不写参数的默认值，第三个参数怎么是指针呢
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();

    if(bWithLine)
    {
        vector<MapLine*> vpML = pMap->GetAllMapLines();
        cout << "Global BA points and lines **** " << endl;
        BundleAdjustmentWithLine(vpKFs, vpMP, vpML, nIterations, pbStopFlag, nLoopKF, bRobust);
    }
    else
    {
        cout << "Global BA points ****" << endl;
        BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
    }
}


// 闭环校正的时候才会GlobalBA，然后调用BA函数，所以这个函数暂时先不看的
/**
 * @brief bundle adjustment Optimization
 * 
 * 3D-2D 最小化重投影误差 e = (u,v) - project(Tcw*Pw)
 * 误差 = 实际观测 - 预测（其中包含了优化的两个顶点）
 * 
 * 1. Vertex: g2o::VertexSE3Expmap()，即当前帧的Tcw
 *            g2o::VertexSBAPointXYZ()，MapPoint的mWorldPos
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：待优化当前帧的Tcw
 *         + Vertex：待优化MapPoint的mWorldPos
 *         + measurement：MapPoint在当前帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *         
 * @param   vpKFs    关键帧 
 *          vpMP     MapPoints
 *          nIterations 迭代次数（20次）
 *          pbStopFlag  是否强制暂停
 *          nLoopKF  关键帧的个数
 *          bRobust  是否使用核函数
 */
void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    // 步骤1：初始化g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;  //6维pose 3维landmark

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    /* *******************注意啦！！！！！***********************
     *
     * 优化器三部曲之:
     *  1. g2o::BlockSolver_6_3::LinearSolverType *     (linearSolver)
     *  2. g2o::BlockSolver_6_3 *                       (solver_ptr)
     *  3. g2o::OptimizationAlgorithmLevenberg *        (solver)
     *
     */


    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // 步骤2：向优化器添加顶点

    // Set KeyFrame vertices
    // 步骤2.1：向优化器添加关键帧位姿顶点
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];  //指针之间的赋值，相当于指针传递，pKF指向了地图中的i关键帧
        if(pKF->isBad())
            continue;

        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();  //在堆上申请动态变量，动态变量不会自动释放除非手动释放
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));  //g2o中有现成的g2o::SE3Quat(R,t)函数，完成转换矩阵到四元数的转换。 setEstimate的参数为四元素
        vSE3->setId(pKF->mnId);  //位姿优优化顶点的id就是关键帧的id属性  @ 这个id应该是连续的吧，从0开始……
        vSE3->setFixed(pKF->mnId==0);   //第0个关键帧的位置是不动的，不优化
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;   //保存关键帧列表中关键帧的最大id号，也就是最末尾的关键帧
    }

    const float thHuber2D = sqrt(5.99);  //huber核函数就一个参数，二次函数与一次函数的分界线
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    // 步骤2.2：向优化器添加MapPoints顶点
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;

        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));  //设定初始值
        const int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);  //这个id应该也是连续的吧，要看GetAllMapPoints函数，id不一定连续吧
        vPoint->setMarginalized(true);  ///地图点边缘化，意思先求解位姿？？因为位姿维度少 check
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();
        //观测到该MapPoint的KF和该MapPoint在KF中的索引

        int nEdges = 0;
        //SET EDGES
        // 步骤3：向优化器添加投影边边
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++) //i地图点的map中，观测到的每一个关键帧
        {

            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;

            nEdges++; //确实观测到关键帧了可构造误差边

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];
            //观测到的2D特征点，这里用去畸变坐标！！

            // 单目或RGBD相机 //这里修改吴博注释，应为单目相机
            if(pKF->mvuRight[mit->second]<0)  //该特征点的右目坐标为负值，单目
            {
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));  //地图点pos
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));  //关键帧se3
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave]; //根据关键点提取的金字塔层数来确定不确定度
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2); //2*2的信息矩阵

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);  //sqrt(5.99)
                }

                e->fx = pKF->fx;  //e是误差边，为误差变设置相机参数
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            }
            else// 双目或RGBD
            {
                Eigen::Matrix<double,3,1> obs;  //双目的测量有3维
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;  //具体不用管，反正知道多传入一个双目的参数即可

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                //todo check: 信息矩阵是3*3，invSigma2为常数，这样的话深度的观测不确定度和uv是一样的，思考对于RGBD是不是可以在这里更改！
                //todo check: 即使对于RGBD相机，观测也只取uv两维，看看最终的结果一样吗。因为观测如果取单位的话不就跟ICP差不多了？
                e->setInformation(Info);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;  //多设置了这一项

                optimizer.addEdge(e);
            }
        }

        if(nEdges==0)  //该地图点对应的可以构建误差的关键帧的数目
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;  //第i地图点不参与优化
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    // 步骤4：开始优化
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);  //默认迭代5次，在LoopClosing::RunGlobalBundleAdjustment函数中迭代了10次

    // Recover optimized data
    // 步骤5：得到优化的结果

    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==0)  //没有回环帧
        {
            pKF->SetPose(Converter::toCvMat(SE3quat));  // T
        }
        else  //有回环帧的情况下，更新关键帧的位姿mTcwGBA
        {
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;  //globalBA时所用到的回环帧的数量
        }
    }

    //Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])  //该地图点没有关键帧观测跳过了
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==0)
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));  //vPoint->estimate()是Eigen::Matrix3d类型
            pMP->UpdateNormalAndDepth(); //更新地图点位置后要更新一下属性
        }
        else //有回环帧的情况下
        {
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

}

//************************line**********************************
// 主要是针对单目的, 如果是RGBD相机这个函数可能还需要改！
/**
 * @brief 包含线特征的BA
 * @param vpKFs 关键帧
 * @param vpMP MapPoints
 * @param vpML MapLines
 * @param nIterations 迭代次数，20次  原版这里是迭代10次吧？ 20次对时间有影响吗
 * @param pbStopFlag 是否强制暂停
 * @param nLoopKF 关键帧的个数
 * @param bRobust 是否使用核函数
 *
 * 3D-2D 最小化重投影误差 e = (u,v)-project(Tcw*Pw)
 * 修改为：e = points的重投影误差 + lines的重投影误差
 *          = (u, v)-project(Tcw*Pw) + line * project(Tcw*LineStartPointW) + line * project(Tcw*LineEndPointW)
 * 1.Vertex:  g2o::VertexSE3Expmap(),即当前帧的Tcw
 *            g2o::VertexSBAPointXYZ(),MapPoint的mWorldPos
 *            g2o::VertexLinePointXYZ(), MapLine的端点世界坐标  新增
 * 2.Edge:
 *
 *      -g2o::EdgeSE3ProjectXYZ(): public  BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexSE3Expmap>
 *          + Vertex：待优化当前帧的Tcw
 *          + measurement: MapPoint在当前帧中的二维位置(u, v)
 *          + InfoMatrix：invSigma2（与特征点所在的尺度有关）
 *
 *      -EdgeLineProjectXYZ : public BaseBinaryEdge<3, Vector3d, g2o::VertexSE3Expmap, g2o::VertexLinePointXYZ>
 *          + Vertex1:待优化当前帧的Tcw
 *          + Vertex2:待优化的MapLine的一个端点
 *          + measurement: MapLine的一个端点在当前帧中的二维位置(u,v)
 */
void Optimizer::BundleAdjustmentWithLine(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP, const vector<MapLine *> &vpML,
                                     int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    double invSigma = 0.01;  //add

    vector<bool> vbNotIncludedMP;  //构不成观测的点
    vbNotIncludedMP.resize(vpMP.size());

    // 1.构造求解器,
    /// 增加了线特征后求解器维度啥的没有变是吗？
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    cout << "======= Optimizer::BundleAdjustment with lines ======="<< endl;
    // Set KeyFrame vertices
    // 2.添加关键帧顶点
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
//        cout << "KeyFrame Id = " << pKF->mnId << endl;
        vSE3->setFixed(pKF->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);  //和点特征一样，未改
    const float thHuber3D = sqrt(7.815);

    vector<int> MapPointID;  //add


    // **************************Set MapPoint vertices******************
//    cout << "set MapPoint vertices......."  << endl;

    // 3.设置MapPoint的顶点
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
//        cout << "MapPoint Id = " << id << endl;
        MapPointID.push_back(id);

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        // 4.设置MapPoint和位姿之间的误差边
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {

            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second]; //取出关键帧上的特征点

            if(pKF->mvuRight[mit->second]<0)  //单目相机
            {
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;  //这里用到的都是去畸变坐标

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            }
            else    //双目
            {
                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;  //观测是三个坐标

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }

        if(nEdges==0) //该地图点可以构造的误差边为0
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }
    /// 以上这部分和ORB中保持不变


    sort(MapPointID.begin(), MapPointID.end());
    int maxMapPointID = MapPointID[MapPointID.size()-1];
//    cout << "maxMapPointID = " << maxMapPointID << endl;
//    for(int i=0; i<MapPointID.size(); ++i)
//    {
//        cout << "MapPointVertex ID:" << MapPointID[i] << endl;
//    }


    // ***********************Set MapLine vertices**************************

    //wojia
    vector<bool> vbNotIncludedLineSP;  //构不成观测的线的起始点
    vbNotIncludedLineSP.resize(vpML.size());

    // 5.设置MapLine的顶点，线段有两个端点，因此顶点数量和对应的边的数量为MapLine数量的2倍
    // 5.1 先设置线段的起始点
    for(size_t i=0; i<vpML.size(); i++)
    {
        MapLine* pML = vpML[i];
        if(pML->isBad())
            continue;
//            VertexLinePointXYZ* vStartP = new VertexLinePointXYZ();   //用自定义的顶点
        g2o::VertexSBAPointXYZ* vStartP = new g2o::VertexSBAPointXYZ(); //g2o自带
        vStartP->setEstimate(pML->GetWorldPos().head(3));
        const int ids = pML->mnId + maxKFid + maxMapPointID + 1;
        //TODO 我觉得这里应该+2
        vStartP->setId(ids);
        vStartP->setMarginalized(true);
        optimizer.addVertex(vStartP);
//            cout << "MapLine StartPoint Id = " << ids << endl;

        const map<KeyFrame*,size_t> observations = pML->GetObservations();
        //这里我修改了，因为mObservation是protected成员

        int nLineSPointEdges = 0;

        // 设置线段起始点和相机位姿之间的边
        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId > maxKFid)
                continue;

            nLineSPointEdges++;

            Eigen::Vector3d line_obs;
            line_obs = pKF->mvKeyLineFunctions[mit->second];  ///

            EdgeLineProjectXYZ* e = new EdgeLineProjectXYZ();

            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(ids)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
            e->setMeasurement(line_obs); //观测是三维的
            e->setInformation(Eigen::Matrix3d::Identity()*invSigma);
            /// 原来直线直线的invSigma不像点一样是跟层数有关的，invSigma直接为常熟，那这样的话所有得边都是一个协方差，就没有什么意思了！ 所以这里可能需要改进！ 除非都是设置成单位阵效果更好

            //todo 如果是深度相机，在设置信息矩阵的时候我是不是可以改进！把观测的协方差矩阵加进去

            if(bRobust)
            {
                g2o::RobustKernelHuber* rk_line_s = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk_line_s);
                rk_line_s->setDelta(thHuber2D);
                //todo 对于点的鲁棒核参数这么设置的，但是对于线是不是可以改一改
            }

            e->fx = pKF->fx;
            e->fy = pKF->fy;
            e->cx = pKF->cx;
            e->cy = pKF->cy;

            e->Xw = pML->mWorldPos.head(3); //add

            optimizer.addEdge(e);
        }



        //wojia todo
        if(nLineSPointEdges==0) //该地图点可以构造的误差边为0
        {
            optimizer.removeVertex(vStartP);
            vbNotIncludedLineSP[i]=true;
        }
        else
        {
            vbNotIncludedLineSP[i]=false;
        }

    }

    // todo！ 注意上述这部分代码类似于单目，但是对于双目是不是也要类似于g2o中双目点构造双目边，这里是不是也有必要构造直线端点的双目边
    // 可以在原版ORB上测试，如果将双目边改成单目边看对结果影响大不大，如果可以的话那我们可以都是用单目边

// -----------------------------------------------------------------
    //wojia
    vector<bool> vbNotIncludedLineEP;  //构不成观测的线的起始点
    vbNotIncludedLineEP.resize(vpML.size());

    // 5.2 设置线段终止点顶点
    for(size_t i=0; i<vpML.size(); i++)
    {
        MapLine* pML = vpML[i];
        if(pML->isBad())
            continue;
//        VertexLinePointXYZ* vEndP = new VertexLinePointXYZ();     //自定义的顶点
        g2o::VertexSBAPointXYZ* vEndP = new g2o::VertexSBAPointXYZ();
        vEndP->setEstimate(pML->GetWorldPos().tail(3));
        const int ide = pML->mnId + maxKFid + maxMapPointID + vpML.size() + 1;
        // todo check 这里是不是应该+3， 另外设置顶点的时候顶点的id应该不需要是连续的吧?

        vEndP->setId(ide);
        vEndP->setMarginalized(true);
        optimizer.addVertex(vEndP);
//        cout << "MapLine EndP Id = " << ide << endl;

        const map<KeyFrame*,size_t> observations = pML->GetObservations();

        int nLineEPointEdges = 0;

        // 设置线段终止点和相机位姿之间的边
        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId > maxKFid)
                continue;

            nLineEPointEdges++;

            Eigen::Vector3d line_obs;
            line_obs = pKF->mvKeyLineFunctions[mit->second];

            EdgeLineProjectXYZ* e = new EdgeLineProjectXYZ();

            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(ide)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
            e->setMeasurement(line_obs);
            e->setInformation(Eigen::Matrix3d::Identity()*invSigma);

            if(bRobust)
            {
                g2o::RobustKernelHuber* rk_line_s = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk_line_s);
                rk_line_s->setDelta(thHuber2D);
            }

            e->fx = pKF->fx;
            e->fy = pKF->fy;
            e->cx = pKF->cx;
            e->cy = pKF->cy;

            e->Xw = pML->mWorldPos.head(3); //todo check这一句到底是什么含义

            optimizer.addEdge(e);
        }

        //wojia todo
        if(nLineEPointEdges==0) //该地图点可以构造的误差边为0
        {
            optimizer.removeVertex(vEndP);
            vbNotIncludedLineEP[i]=true;
        }
        else
        {
            vbNotIncludedLineEP[i]=false;
        }


    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data
    // 6.得到优化的结果

    // *********************Rocover optimized data*******************************

    // Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==0)
        {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        else
        {
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    //Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==0)
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

    //上述和ORB中一致

    //LineSegment Points
    for(size_t i=0; i<vpML.size(); i++)
    {
        if(vbNotIncludedMP[i])  //todo 这里需要改。改的话我可以把线段的起始点和终点代码分成两块分别恢复点的位姿
            continue;

        MapLine* pML = vpML[i];

        if(pML->isBad())
            continue;

//        VertexLinePointXYZ* vStartP = static_cast<VertexLinePointXYZ*>(optimizer.vertex(pML->mnId + maxKFid + vpMP.size() + 1));  //自定义的顶点
//        VertexLinePointXYZ* vEndP = static_cast<VertexLinePointXYZ*>(optimizer.vertex(2*pML->mnId + maxKFid + vpMP.size() + 1));  //自定义的顶点

        g2o::VertexSBAPointXYZ* vStartP = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pML->mnId + maxKFid + maxMapPointID + 1));
        g2o::VertexSBAPointXYZ* vEndP = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pML->mnId + maxKFid + maxMapPointID + vpML.size() + 1));
        //todo 如果更改了，序号也要对应起来

        if(nLoopKF==0)
        {
            Vector6d LinePos;
            LinePos << Converter::toVector3d(Converter::toCvMat(vStartP->estimate())), Converter::toVector3d(Converter::toCvMat(vEndP->estimate())); //todo check这是对吗
            pML->SetWorldPos(LinePos);
            pML->UpdateAverageDir();
        } else
        {
            pML->mPosGBA.create(6, 1, CV_32F);
            Converter::toCvMat(vStartP->estimate()).copyTo(pML->mPosGBA.rowRange(0,3));
            Converter::toCvMat(vEndP->estimate()).copyTo(pML->mPosGBA.rowRange(3,6));
        }
    }

}

//************************done**********************************



/**
 * @brief Pose Only Optimization
 * 
 * 3D-2D 最小化重投影误差 e = (u,v) - project(Tcw*Pw) \n
 * 只优化Frame的Tcw，不优化MapPoints的坐标
 * 
 * 1. Vertex: g2o::VertexSE3Expmap()，即当前帧的Tcw
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZOnlyPose()，BaseUnaryEdge
 *         + Vertex：待优化当前帧的Tcw
 *         + measurement：MapPoint在当前帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *     - g2o::EdgeStereoSE3ProjectXYZOnlyPose()，BaseUnaryEdge
 *         + Vertex：待优化当前帧的Tcw
 *         + measurement：MapPoint在当前帧中的二维位置(ul,v,ur)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *
 * @param   pFrame Frame
 * @return  inliers数量
 */
int Optimizer::PoseOptimization(Frame *pFrame)
{
    // 该优化函数主要用于Tracking线程中：运动跟踪、参考帧跟踪、地图跟踪、重定位

    // 步骤1：构造g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    // 步骤2：添加顶点：待优化当前帧的Tcw
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw)); //传入参数
    vSE3->setId(0);  //只有这一个顶点，id设置为0
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;  //帧上总的特征点数

    // for Monocular
    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono; //误差边列表
    vector<size_t> vnIndexEdgeMono; //误差边指数
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    /// for Stereo
    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo; //双目误差边总是与单目不同吗
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    // 步骤3：添加一元边：相机投影模型
    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex); //锁了

    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            // Monocular observation
            // 单目情况, 也有可能在双目下, 当前帧的左兴趣点找不到匹配的右兴趣点！好的
            if(pFrame->mvuRight[i]<0)
            {
                nInitialCorrespondences++; //该特征点有对应的地图点
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose(); //动态申请一个误差边变量，是一个一元边

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];

                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;  //一律采用核函数
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                //相当于点的Pose是固定的，输入误差边上另一固定的顶点
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i); //特征点的编号
            }
            else  // Stereo observation 双目
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                //SET EDGE
                Eigen::Matrix<double,3,1> obs;// 这里和单目不同
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                const float &kp_ur = pFrame->mvuRight[i];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;// 这里和单目不同

                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();// 这里和单目不同

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];  //这些都是通用操作
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }

    }
    }


    if(nInitialCorrespondences<3)   //可用的地图点
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    // 步骤4：开始优化，总共优化四次，每次优化后，将观测分为outlier和inlier，outlier不参与下次优化
    // 由于每次优化后是对所有的观测进行outlier和inlier判别，因此之前被判别为outlier有可能变成inlier，反之亦然
    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};// 四次迭代，每次迭代的次数

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);// 对level为0的边进行优化，@@ level为0啥意思？
        optimizer.optimize(its[it]);   //优化函数

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];  //取出第i条边

            const size_t idx = vnIndexEdgeMono[i];  //这条边的id

            if(pFrame->mvbOutlier[idx])  //如果是外点就计算误差，这一步有什么含义呢？
            {
                e->computeError(); // NOTE g2o只会计算active edge的误差
            }

            const float chi2 = e->chi2();  //等价于error的大小

            if(chi2>chi2Mono[it])
            {                
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);                 // 设置为outlier，会影响到下一次迭代的优化
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;  ///之前判定为外点的点这一步有可能又变成了内点，注意nBad是在每次迭代的的局部变量，所以最终的nBad是最后一次迭代得到的
                e->setLevel(0);                 // 设置为inlier
            }

            if(it==2)
                e->setRobustKernel(0); // 除了前两次优化需要RobustKernel以外, 其余的优化都不需要
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)  //TODO 这里单目还是双目没有用if判断，如果只是双目的话，可以跳过上面的代码块提高速度
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])  //这一步的用处
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {                
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        if(optimizer.edges().size()<10)  //一共四次迭代，因为总是在设置外点，边数会减少
            break;
    }    

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences-nBad;  //内点的数目
}

//*****************************line*****************************

/**
* 该优化函数主要用于Tracking线程中
*
* 3D-2D 最小化重投影误差 e = (u, v) - project(Tcw*Pw)
* 修改为： e = points的重投影误差 + lines的重投影误差
*          = (u, v) - project(Tcw*Pw) + line * project(Tcw * LineStartPointw) + line * project(Tcw * LineEndPointW)
* 只优化pFrame的Tcw，不优化MapPoints和MapLines的坐标
*
* 1.Vertex: g2o::VertexSE3Expmap，即当前帧的Tcw
* 2.Edge:
*      - g2o::EdgeSE3ProjectXYZOnlyPose: public  BaseUnaryEdge<2, Vector2d, VertexSE3Expmap>
*          + Vertex：待优化当前帧的Tcw
*          + measurement：MapPoint在当前帧中的二维位置(u,v)
*          + InfoMatrix：invSigma2（与特征点所在的尺度有关）
*
*      添加相机位姿与线特征之间的误差边：
*      - g2o::
*
* @param pFrame 图像帧
* @return
*/
int Optimizer::PoseOptimizationWithLine(Frame *pFrame)
{
    double invSigma = 10;  //todo 这个参数有什么意义呢
    // 1.构造求解器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();  //这里用的是Dense，可能会影响速度，前面用的是LinearSolverEigen

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false); //这个顶点当然是不固定的
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);


    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for(int i=0; i<N; i++)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if(pMP)
            {
                // Monocular observation
                if(pFrame->mvuRight[i]<0)
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double,2,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else  // Stereo observation
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    //SET EDGE
                    Eigen::Matrix<double,3,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    const float &kp_ur = pFrame->mvuRight[i];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    e->bf = pFrame->mbf;
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }
            }

        }
    }

    ///添加相机位姿和特征线段之间的误差边
    // Set MapLine vertices
    const int NL = pFrame->NL;
    int nLineInitalCorrespondences=0;

    // 起始点
    vector<EdgeLineProjectXYZOnlyPose*> vpEdgesLineSp;
    vector<size_t> vnIndexLineEdgeSp;
    vpEdgesLineSp.reserve(NL);
    vnIndexLineEdgeSp.reserve(NL);   //me add

    // 终止点
    vector<EdgeLineProjectXYZOnlyPose*> vpEdgesLineEp;
    vector<size_t> vnIndexLineEdgeEp;
    vpEdgesLineEp.reserve(NL);
    vnIndexLineEdgeEp.reserve(NL);

    {
        unique_lock<mutex> lock(MapLine::mGlobalMutex);  //类比于MapPoint

        for(int i=0; i<NL; i++)
        {
            MapLine* pML = pFrame->mvpMapLines[i];
            if(pML)
            {
//                cout << "******** PoseOptimization using line edges ********" << endl;
                //todo 这里没有判断是否为单目，不像ORB，那如果是深度相机这里是不是应该判断下
                //RGBD相机的话，观测可以考虑还是直线系数（三维的）

                nLineInitalCorrespondences++;
                pFrame->mvbLineOutlier[i] = false;

                Eigen::Vector3d line_obs;
                line_obs = pFrame->mvKeyLineFunctions[i]; //如果是

                // 特征线段的起始点
                EdgeLineProjectXYZOnlyPose* els = new EdgeLineProjectXYZOnlyPose();

                els->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                els->setMeasurement(line_obs);
                els->setInformation(Eigen::Matrix3d::Identity()*invSigma);
                //invSigma是一个超参数，可以考虑调节

                g2o::RobustKernelHuber* rk_line_s = new g2o::RobustKernelHuber;
                els->setRobustKernel(rk_line_s);
                rk_line_s->setDelta(deltaMono);  //重写了Line误差边之后关于核函数的部分应该没变の

                els->fx = pFrame->fx;
                els->fy = pFrame->fy;
                els->cx = pFrame->cx;
                els->cy = pFrame->cy;

//                cv::Mat SP = pML->mStart3D;
//                els->Xw[0] = SP.at<float>(0);
//                els->Xw[1] = SP.at<float>(1);
//                els->Xw[2] = SP.at<float>(2);
                els->Xw = pML->mWorldPos.head(3);
                //todo 这里是不是很有问题，应该设置的是直线的端点不变，这里如果是中点不变，是不是有区别。 TEST！

                optimizer.addEdge(els);

                vpEdgesLineSp.push_back(els);
                vnIndexLineEdgeSp.push_back(i);

                // ----------每个地图线，添加两个顶点，如果是很多个点，就要用for结构了--------

                // 特征点的终止点
                EdgeLineProjectXYZOnlyPose* ele = new EdgeLineProjectXYZOnlyPose();

                ele->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                ele->setMeasurement(line_obs); //这两个顶点所对应的观测是一样的
                ele->setInformation(Eigen::Matrix3d::Identity()*invSigma);

                g2o::RobustKernelHuber* rk_line_e = new g2o::RobustKernelHuber;
                ele->setRobustKernel(rk_line_e);
                rk_line_e->setDelta(deltaMono);  //RGBD相机这里也要改

                ele->fx = pFrame->fx;
                ele->fy = pFrame->fy;
                ele->cx = pFrame->cx;
                ele->cy = pFrame->cy;

//                cv::Mat EP = pML->mEnd3D;
//                ele->Xw[0] = EP.at<float>(0);
//                ele->Xw[1] = EP.at<float>(1);
//                ele->Xw[2] = EP.at<float>(2);
                ele->Xw = pML->mWorldPos.tail(3);

                optimizer.addEdge(ele);

                vpEdgesLineEp.push_back(ele);
                vnIndexLineEdgeEp.push_back(i);
            }
        }
    }

    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};

    int nBad=0;     //点特征
    int nLineBad=0; //线特征
    for(size_t it=0; it<4; it++)
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]); //总的迭代4次，每次小迭代10次

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }  //这一部分和ORB中一样，但是RGBD相机的话这段可以删除

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)  //双目情况下找外点
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        ///--------------对于线，找外点，这里有个问题如果线上一点为外点，一点为内点怎么办？----------------------------------
        nLineBad=0;
        for(size_t i=0, iend=vpEdgesLineSp.size(); i<iend; i++)
        {
            EdgeLineProjectXYZOnlyPose* e1 = vpEdgesLineSp[i];  //线段起始点误差边
            EdgeLineProjectXYZOnlyPose* e2 = vpEdgesLineEp[i];  //线段终止点误差边

            const size_t idx = vnIndexLineEdgeSp[i];    //线段起始点和终止点的误差边的index一样

            if(pFrame->mvbLineOutlier[idx])
            {
                e1->computeError();  //重写线误差边的时候这个函数也要重写吧
                e2->computeError();
            }

            const float chi2_s = e1->chi2();
            const float chi2_e = e2->chi2();

            if(chi2_s > 2*chi2Mono[it] || chi2_e > 2*chi2Mono[it]) //这里只要有一个端点的误差边误差过大，就认为这条线是外点了 TODO 如果我在直线上用很多个点来构造很多误差边，那么少数点误差过大不一定要把整条边都看作是外点
            {
                pFrame->mvbLineOutlier[idx]=true;
                e1->setLevel(1);
                e2->setLevel(1);
                nLineBad++;
            } else
            {
                pFrame->mvbLineOutlier[idx]=false;
                e1->setLevel(0);
                e2->setLevel(0);
            }

            if(it==2)
            {
                e1->setRobustKernel(0);
                e2->setRobustKernel(0);
            }
        }

        if(optimizer.edges().size()<10) //总共四次迭代，如果哪次迭代有效的误差边数小于10就中断。我猜的
            break;
    }

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences-nBad;
}

//*****************************done*****************************


/**
 * @brief Local Bundle Adjustment
 *
 * 1. Vertex:
 *     - g2o::VertexSE3Expmap()，LocalKeyFrames，即当前关键帧的位姿、与当前关键帧相连的关键帧的位姿
 *     - g2o::VertexSE3Expmap()，FixedCameras，即能观测到LocalMapPoints的关键帧但是和当前关键帧不相连（不在localKeyFrames中），在优化中这些关键帧的位姿不变 || 这里我还是不是很理解？
 *     - g2o::VertexSBAPointXYZ()，LocalMapPoints，即LocalKeyFrames能观测到的所有MapPoints的位置
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *     - g2o::EdgeStereoSE3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(ul,v,ur)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *         
 * @param pKF        KeyFrame
 * @param pbStopFlag 是否停止优化的标志
 * @param pMap       在优化后，更新状态时需要用到Map的互斥量mMutexMapUpdate
 */
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{
    // 该优化函数用于LocalMapping线程的局部BA优化

    // Local KeyFrames: First Breadth Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    // 步骤1：将当前关键帧加入lLocalKeyFrames
    lLocalKeyFrames.push_back(pKF);  //当前关键帧是list的第一帧
    pKF->mnBALocalForKF = pKF->mnId;

    // 步骤2：找到关键帧连接的关键帧（一级相连），加入lLocalKeyFrames中. @@ 搞清楚关键帧连接的概念！
    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();   //关键函数pKF->GetVectorCovisibleKeyFrames()
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;  //更新关键帧的数据成员（localBA也就是局部地图中的编号mnBALocalForKF），这些关键帧参与局部优化时那个当前关键帧的id
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    /// 步骤3：遍历lLocalKeyFrames中关键帧，将它们观测的MapPoints加入到lLocalMapPoints
    // 一级相连关键帧所观测道德地图点一起组成了局部地图点
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)  //一般遍历列表中的成员都是这么操作的
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();  //调用关键帧的GetMapPointMatches()函数，即获取关键帧的mvpMapPoints
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
            {
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;// 防止重复添加，所以定义了地图点在BALocal中的编号，该编号与当前关键帧编号一致
                        //比如在同一关键帧下两个特征点对应一个地图点的情况以及两个关键帧下看到同一个地图点的情况
                    }
            }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    // 步骤4：得到能被局部MapPoints观测到，但不属于局部关键帧的关键帧，这些关键帧在局部BA优化时不优化，待优化的关键帧是当前关键帧和其连接关键帧
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            // pKFi->mnBALocalForKF!=pKF->mnId表示局部关键帧，注意这里的pKF是当前关键帧，函数的输入参数
            // 其它的关键帧虽然能观测到，但不属于局部关键帧
            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)  //厉害了
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;// 防止重复添加
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    // 步骤5：构造g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)  //pbStopFlag是指针，指针不为空，优化强制停止
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    //----------------------------添加顶点和边----------------------------------------

    // Set Local KeyFrame vertices
    // 步骤6：添加顶点：Pose of Local KeyFrame
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId); //可见顶点的id并不需要从0开始或者连续
        vSE3->setFixed(pKFi->mnId==0);//第一帧位置固定
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;  //关键帧顶点中最大的关键帧编号
    }

    // Set Fixed KeyFrame vertices。 为什么会有固定的关键帧参与优化呢，因为可以用来构造误差边，优化这些关键帧看到的localMapPoints

    // 步骤7：添加顶点：Pose of Fixed KeyFrame，注意这里调用了vSE3->setFixed(true)。
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;  //是局部地图中所有关键帧的最大编号
    }

    // Set MapPoint vertices
    // 步骤7：添加3D顶点
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size(); //这里是最大size？是的可以简单画个图举个例子，见路游侠博客-优化

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;  //1.边的列表
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;  //2.关键帧的列表
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;  //3.地图点的列表
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)  //遍历所有局部地图中地图点
    {
        // 添加顶点：MapPoint
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId+maxKFid+1; //+1不能少，否则第0个地图点就会发生id冲突
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMP->GetObservations(); //该局部地图点看到的所有关键帧都会加入到优化问题中，只不过一些是固定的

        // Set edges
        // 步骤8：对每一对关联的MapPoint和KeyFrame构建边
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)  ///只要在该地图点中遍历所有观测到的关键帧即可
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())  //只要这个关键帧不是坏的，统统构造误差边加进去
            {                
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                if(pKFi->mvuRight[mit->second]<0)
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));  //这里只管加就行不管是不是固定的顶点
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));  //误差边的第0头是3D点坐标，在顶点中的id就是id
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));  //误差边的另一头1是位姿
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);  //误差边列表
                    vpEdgeKFStereo.push_back(pKFi);  //关键帧（包括固定的）列表
                    vpMapPointEdgeStereo.push_back(pMP);  //地图点列表，和上面是一一对应的，会有连续相同的地图点
                }
            }
        }
    }

    if(pbStopFlag)  //上述只是构建了优化问题，并没有开始优化
        if(*pbStopFlag)
            return;  //再次检查，如果有停止请求就不优化了

    // 步骤9：开始优化
    optimizer.initializeOptimization();  //全部初始化
    optimizer.optimize(5); //迭代了五次

    bool bDoMore= true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;  //优化完了有停止请求就不要DoMore了

    if(bDoMore)  //正常情况下
    {

    // Check inlier observations
    // 步骤10：检测outlier，并设置下次不优化
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)   ///如果我只用RGBD相机这一部分应该可以注释掉节省时间
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];  //对应的误差边和对应的地图点
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
        if(e->chi2()>5.991 || !e->isDepthPositive())  //深度为负值说明也是外点
        {
            e->setLevel(1);// 不优化
        }

        e->setRobustKernel(0);// 不使用核函数
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            e->setLevel(1);  //这条边的误差过大，设置level为1
        }

        e->setRobustKernel(0);  //false ，不用核函数
    }

    // Optimize again without the outliers  大的迭代一共两次
    // 步骤11：排除误差较大的outlier后再次优化。
    optimizer.initializeOptimization(0);  //level为0的初始化，所以上一次误差大的边level为1，相当于上一次优化中忽略误差大的边
    optimizer.optimize(10);  //第二次优化

    }  //if bDoMore语句块中

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations
    // 步骤12：在优化后重新计算误差，剔除连接误差比较大的关键帧和MapPoint
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];  //第i个边
        MapPoint* pMP = vpMapPointEdgeMono[i];  //第i个边上对应的地图点

        if(pMP->isBad())
            continue;

        // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];  //第i个边上对应的关键帧
            vToErase.push_back(make_pair(pKFi,pMP));  //这条误差边连的两头
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    /// Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate); //局部变量，后面的代码都会锁

    // 连接偏差比较大，在关键帧中剔除对该MapPoint的观测. 也就是消除这次观测，而这次观测对应了两头记录都要更新
    // 连接偏差比较大，在MapPoint中剔除对该关键帧的观测
    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);  ///
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data
    // 步骤13：优化后更新关键帧位姿以及MapPoints的位置、平均观测方向等属性

    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();  //每次地图点的位置变化后，都要更新Normal平均观测方向和深度，即观测到该点的距离上下限
    }
}


//***********************************line*********************************
///包含有线特征的局部BA

void Optimizer::LocalBundleAdjustmentWithLine(KeyFrame *pKF, bool *pbStopFlag, Map *pMap)
{
    double invSigma = 0.01;
    // Local KeyFrames: First Breath Search from Current KeyFrame
    list<KeyFrame*> lLocalKeyFrames;

    // step1: 将当前关键帧加入到lLocalKeyFrames
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    // step2:找到关键帧连接的关键帧（一级相连），加入到lLocalKeyFrames中
    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // step3：将lLocalKeyFrames的MapPoints加入到lLocalMapPoints
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
                }
            }
        }
    }

    /// step4: 遍历lLocalKeyFrames，将每个关键帧所能观测到的MapLine提取出来，放到lLocalMapLines
    list<MapLine*> lLocalMapLines;
    vector<long unsigned int> mlID; //这个好像没啥用
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapLine*> vpMLs = (*lit)->GetMapLineMatches();
        for(vector<MapLine*>::iterator vit=vpMLs.begin(), vend=vpMLs.end(); vit!=vend; vit++)
        {
            MapLine* pML = *vit;
            if(pML)
            {
                if(!pML->isBad())
                {
                    if(pML->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapLines.push_back(pML);
                        pML->mnBALocalForKF = pKF->mnId; //防止重复添加
                        mlID.push_back(pML->mnId);
                    }
                }
            }
        }
    }

#if 0
    for(list<MapLine*>::iterator lit=lLocalMapLines.begin(), lend=lLocalMapLines.end(); lit!=lend; lit++)
{
    int i=0;
    if(count(mlID.begin(), mlID.end(), mlID[i])>1)
    {
        cout << "exist ===============" << endl;
        lit = lLocalMapLines.erase(lit);
    }
    i++;
}
#endif

    // step5: 得到能被局部MapPoints观测到，但不属于局部关键帧的关键帧，这些关键帧在局部BA优化时固定
    /// todo 这块代码只考虑了点的情况，是不是应该把相应的线的操作也加进去
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*, size_t > observations = (*lit)->GetObservations();
        for(map<KeyFrame*, size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // step6：构造g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // step7：添加顶点，Pose of Local KeyFrame 未改
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
//        cout << "KeyFrame Id = " << pKFi->mnId << endl;
        vSE3->setFixed(pKFi->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // step8:添加固定帧的顶点，Pose of Fixed KeyFrame
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
//        cout << "Fixed KeyFrame Id = " << pKFi->mnId << endl;
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;  //注意这里的maxKFid是所有帧中的
    }

    vector<int> MapPointID;

    //***********************Set MapPoint Vertices******************************

    // step9：添加MapPoint的3D顶点
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size(); //未改

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    //TODO 这一部分都是只写了单目情况，需要改成双目的

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
//        cout << "MapPoint Id = " << id << endl;
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        MapPointID.push_back(id);

        const map<KeyFrame*, size_t > observations = pMP->GetObservations();

        // Set Edges
        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
//                cout << "invSigma = " << invSigma2 << endl;
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberMono);

                e->fx = pKFi->fx;
                e->fy = pKFi->fy;
                e->cx = pKFi->cx;
                e->cy = pKFi->cy;

                optimizer.addEdge(e);
                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKFi);
                vpMapPointEdgeMono.push_back(pMP);
            }
        }
    }  //所有局部地图点顶点和相应的关键帧顶点设置完毕

    sort(MapPointID.begin(), MapPointID.end());
    int maxMapPointID = MapPointID[MapPointID.size()-1];
//    cout << "************ maxMapPointID = " << maxMapPointID << " ************" << endl;

    //***********************Set MapLine Vertices******************************
    // step10：添加MapLine的顶点
    const int nLineExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapLines.size();  //仿写

    vector<EdgeLineProjectXYZ*> vpLineEdgesSP;
    vpLineEdgesSP.reserve(nLineExpectedSize);

    vector<EdgeLineProjectXYZ*> vpLineEdgesEP;
    vpLineEdgesEP.reserve(nLineExpectedSize);

    vector<KeyFrame*> vpLineEdgeKF;
    vpLineEdgeKF.reserve(nLineExpectedSize);

    vector<MapLine*> vpMapLineEdge;
    vpMapLineEdge.reserve(nLineExpectedSize);

    //TODO 改成双目

    vector<int> MapLineSPID;
    // 起始点的顶点
    for(list<MapLine*>::iterator lit=lLocalMapLines.begin(), lend=lLocalMapLines.end(); lit!=lend; lit++)
    {
        //添加顶点：MapLine: StartPoint EndPoint
        MapLine* pML = *lit;
        g2o::VertexSBAPointXYZ* vStartP = new g2o::VertexSBAPointXYZ();
        vStartP->setEstimate(pML->GetWorldPos().head(3));
        int ids = pML->mnId + maxKFid + maxMapPointID + 1; //todo 改成+2
        vStartP->setId(ids);
//        cout << "MapLine StartP Id = " << ids << endl;
        vStartP->setMarginalized(true);
        optimizer.addVertex(vStartP);
        MapLineSPID.push_back(ids);

        const map<KeyFrame*, size_t > observations = pML->GetObservations();

        // =========设置线段起始点和相机位姿之间的边=========
        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {
                Eigen::Vector3d line_obs;
                line_obs = pKFi->mvKeyLineFunctions[mit->second];

                EdgeLineProjectXYZ* e = new EdgeLineProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(ids)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(line_obs);
                e->setInformation(Eigen::Matrix3d::Identity()*invSigma);

                g2o::RobustKernelHuber* rk_line_s = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk_line_s);
                rk_line_s->setDelta(thHuberMono);

                e->fx = pKFi->fx;
                e->fy = pKFi->fy;
                e->cx = pKFi->cx;
                e->cy = pKFi->cy;

                e->Xw = pML->mWorldPos.head(3);

                optimizer.addEdge(e);
                vpLineEdgesSP.push_back(e);
                vpLineEdgeKF.push_back(pKFi);
                vpMapLineEdge.push_back(pML);
            }
        }
    }

    sort(MapLineSPID.begin(), MapLineSPID.end());
    int maxMapLineSPID = MapLineSPID[MapLineSPID.size()-1];
    // 终止点的顶点
    for(list<MapLine*>::iterator lit=lLocalMapLines.begin(), lend=lLocalMapLines.end(); lit!=lend; lit++)
    {
        MapLine* pML = *lit;
        g2o::VertexSBAPointXYZ* vEndP = new g2o::VertexSBAPointXYZ();
        vEndP->setEstimate(pML->GetWorldPos().tail(3));
//        int ide = pML->mnId + maxKFid + maxMapPointID + lLocalMapLines.size() + 1;
        int ide = pML->mnId + maxKFid + maxMapPointID + maxMapLineSPID + 1; //todo 改成+3
        vEndP->setId(ide);
//        cout << "MapLine EndP Id = " << ide << endl;
        vEndP->setMarginalized(true);
        optimizer.addVertex(vEndP);

        const map<KeyFrame*, size_t > observations = pML->GetObservations();

        // =========设置线段终止点和相机位姿之间的边=========
        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {
                Eigen::Vector3d line_obs;
                line_obs = pKFi->mvKeyLineFunctions[mit->second];

                EdgeLineProjectXYZ* e = new EdgeLineProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(ide)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setMeasurement(line_obs);
                e->setInformation(Eigen::Matrix3d::Identity()*invSigma);

                g2o::RobustKernelHuber* rk_line_e = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk_line_e);
                rk_line_e->setDelta(thHuberMono);

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                e->Xw = pML->mWorldPos.head(3);

                optimizer.addEdge(e);
                vpLineEdgesEP.push_back(e);
                vpLineEdgeKF.push_back(pKFi);
//                vpMapLineEdge.push_back(pML);
            }
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore = true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    if(bDoMore)
    {
        // Check inlier observations
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }
            e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpLineEdgesSP.size(); i<iend; i++)
        {
            EdgeLineProjectXYZ* e = vpLineEdgesSP[i];
            MapLine* pML = vpMapLineEdge[i];

            if(pML->isBad())
                continue;

            if(e->chi2()>7.815)  //todo 这里能不能为e也增加函数isDepthPositive()
            {
                e->setLevel(1);
            }
            e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpLineEdgesEP.size(); i<iend; i++)
        {
            EdgeLineProjectXYZ* e = vpLineEdgesEP[i];
            MapLine* pML = vpMapLineEdge[i];

            if(pML->isBad())
                continue;

            if(e->chi2()>7.815)
            {
                e->setLevel(1);
            }
            e->setRobustKernel(0);
        }

        // Optimize again without the outliers
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }

    vector<pair<KeyFrame*, MapPoint*>> vToErase;
    vToErase.reserve(vpEdgesMono.size());

    // check inlier observations
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    vector<pair<KeyFrame*,MapLine*>> vLineToErase;
    vLineToErase.reserve(vpLineEdgesSP.size());
    for(size_t i=0, iend=vpLineEdgesSP.size(); i<iend; i++)
    {
        EdgeLineProjectXYZ* e = vpLineEdgesSP[i];
        MapLine* pML = vpMapLineEdge[i];

        if(pML->isBad())
            continue;

        if(e->chi2()>7.815)
        {
            KeyFrame* pKFi = vpLineEdgeKF[i];
            vLineToErase.push_back(make_pair(pKFi, pML));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);


    if(!vToErase.empty())
    {
        for(size_t i=0; i<vToErase.size(); i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    if(!vLineToErase.empty())  //这段代码应该没有问题
    {
        for(size_t i=0; i<vLineToErase.size(); i++)
        {
            KeyFrame* pKFi = vLineToErase[i].first;
            MapLine* pMLi = vLineToErase[i].second;
            pKFi->EraseMapLineMatch(pMLi);
            pMLi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data
    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }

    // Lines
    for(list<MapLine*>::iterator lit=lLocalMapLines.begin(), lend=lLocalMapLines.end(); lit!=lend; lit++)
    {
        MapLine* pML = *lit;
        g2o::VertexSBAPointXYZ* vStartP = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pML->mnId + maxKFid + maxMapPointID + 1));
        g2o::VertexSBAPointXYZ* vEndP = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pML->mnId + maxKFid + maxMapPointID + maxMapLineSPID + 1));

        Vector6d LinePos;
        LinePos << Converter::toVector3d(Converter::toCvMat(vStartP->estimate())), Converter::toVector3d(Converter::toCvMat(vEndP->estimate()));
        pML->SetWorldPos(LinePos);  //TODO pML的mWorldPos应该不是中点而是两个端点合起来吧！呼应了之前的问题，所以严格检查用到mWorldPos的地方使用是否正确

        pML->UpdateAverageDir();
    }
}


//*********************************done********************************

// PASS
// 本质图优化涉及到的都是关键帧，所有加入了线之后对这一函数没影响
//本质图优化函数需要加强看，里面包含了好几种边！！
// 闭环检测部分的函数，如果只关注VO的话这一函数我先暂时不看的
/**
 * @brief 闭环检测后，EssentialGraph优化
 *
 * 1. Vertex:
 *     - g2o::VertexSim3Expmap，Essential graph中关键帧的位姿
 * 2. Edge:
 *     - g2o::EdgeSim3()，BaseBinaryEdge
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：经过CorrectLoop函数步骤2，Sim3传播校正后的位姿
 *         + InfoMatrix: 单位矩阵     
 *
 * @param pMap               全局地图
 * @param pLoopKF            闭环匹配上的关键帧
 * @param pCurKF             当前关键帧
 * @param NonCorrectedSim3   未经过Sim3传播调整过的关键帧位姿
 * @param CorrectedSim3      经过Sim3传播调整过的关键帧位姿
 * @param LoopConnections    因闭环时MapPoints调整而新生成的边
 */
void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{
    // Setup optimizer
    // 步骤1：构造优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    // 指定线性方程求解器使用Eigen的块求解器
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    // 构造线性求解器
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    // 使用LM算法进行非线性迭代
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);  //新增
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    // 仅经过Sim3传播调整，未经过优化的keyframe的pose
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    // 经过Sim3传播调整，经过优化的keyframe的pose
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    // 这个变量没有用
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;

    // Set KeyFrame vertices
    // 步骤2：将地图中所有keyframe的pose作为顶点添加到优化器
    // 尽可能使用经过Sim3调整的位姿
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();// 一直new，不用释放？(wubo???)

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        // 如果该关键帧在闭环时通过Sim3传播调整过，用校正后的位姿
        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else// 如果该关键帧在闭环时没有通过Sim3传播调整过，用自身的位姿
        {
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0); //必须转为相似矩阵的格式R t s
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        // 闭环匹配上的帧不进行位姿优化
        if(pKF==pLoopKF)
            VSim3->setFixed(true);  //666

        VSim3->setId(nIDi);  //等于关键帧的id
        VSim3->setMarginalized(false);  //不不边缘化
        VSim3->_fix_scale = bFixScale;  //@@ 查函数调用时这个参数的值

        optimizer.addVertex(VSim3);

        // 优化前的pose顶点，后面代码中没有使用
        vpVertices[nIDi]=VSim3;
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;  //关联容器set结构，排序的，单值的

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges
    // 步骤3：添加边：LoopConnections是闭环时因为MapPoints调整而出现的新关键帧连接关系（不是当前帧与闭环匹配帧之间的连接关系）
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            // 得到两个pose间的Sim3变换
            const g2o::Sim3 Sji = Sjw * Swi;  //边的观测就是两帧之间的位姿变换Sim ji。

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            // 根据两个Pose顶点的位姿算出相对位姿作为边，那还存在误差？优化有用？因为闭环MapPoints调整新形成的边不优化？（wubo???）
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // Set normal edges
    // 步骤4：添加跟踪时形成的边、闭环匹配成功形成的边
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        // 尽可能得到未经过Sim3传播调整的位姿
        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        // 步骤4.1：只添加扩展树的边（有父关键帧）
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            // 尽可能得到未经过Sim3传播调整的位姿
            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        // 步骤4.2：添加在CorrectLoop函数中AddLoopEdge函数添加的闭环连接边（当前帧与闭环匹配帧之间的连接关系）
        // 使用经过Sim3调整前关键帧之间的相对关系作为边
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                // 尽可能得到未经过Sim3传播调整的位姿
                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                // 根据两个Pose顶点的位姿算出相对位姿作为边，那还存在误差？优化有用？（wubo???）
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        // 步骤4.3：最有很好共视关系的关键帧也作为边进行优化
        // 使用经过Sim3调整前关键帧之间的相对关系作为边
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    // 尽可能得到未经过Sim3传播调整的位姿
                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    // Optimize!
    // 步骤5：开始g2o优化
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    // 步骤6：设定优化后的位姿
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    // 步骤7：步骤5和步骤6优化得到关键帧的位姿后，MapPoints根据参考帧优化前后的相对关系调整自己的位置
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        // 该MapPoint经过Sim3调整过，(LoopClosing.cpp，CorrectLoop函数，步骤2.2_
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            // 通过情况下MapPoint的参考关键帧就是创建该MapPoint的那个关键帧
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }

        // 得到MapPoint参考关键帧步骤5优化前的位姿
        g2o::Sim3 Srw = vScw[nIDr];
        // 得到MapPoint参考关键帧优化后的位姿
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
}


// PASS
/**
 * @brief 形成闭环时进行Sim3优化
 *
 * 1. Vertex:
 *     - g2o::VertexSim3Expmap()，两个关键帧的位姿
 *     - g2o::VertexSBAPointXYZ()，两个关键帧共有的MapPoints
 * 2. Edge:
 *     - g2o::EdgeSim3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Sim3，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *     - g2o::EdgeInverseSim3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Sim3，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *         
 * @param pKF1        KeyFrame
 * @param pKF2        KeyFrame
 * @param vpMatches1  两个关键帧的匹配关系
 * @param g2oS12      两个关键帧间的Sim3变换
 * @param th2         核函数阈值
 * @param bFixScale   是否优化尺度，单目进行尺度优化，双目不进行尺度优化
 */
 //未改动，闭环校正的时候会用到，暂时先不看，
 // 这个函数中因为会对地图点进行修改，如果加入了线特征之后函数中理应加入相关的操作，lan版本中未改
int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    // 步骤1：初始化g2o优化器
    // 先构造求解器
    g2o::SparseOptimizer optimizer;

    // 构造线性方程求解器，Hx = -b的求解器
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    // 使用dense的求解器，（常见非dense求解器有cholmod线性求解器和shur补线性求解器）
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    // 使用L-M迭代
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    // 步骤2.1 添加Sim3顶点
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();    
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);  //初值
    vSim3->setId(0);
    vSim3->setFixed(false);// 优化Sim3顶点
    vSim3->_principle_point1[0] = K1.at<float>(0,2); // 光心横坐标cx
    vSim3->_principle_point1[1] = K1.at<float>(1,2); // 光心纵坐标cy
    vSim3->_focal_length1[0] = K1.at<float>(0,0); // 焦距 fx
    vSim3->_focal_length1[1] = K1.at<float>(1,1); // 焦距 fy
    vSim3->_principle_point2[0] = K2.at<float>(0,2);
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();

    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12; //pKF2对应的MapPoints到pKF1的投影
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21; //pKF1对应的MapPoints到pKF2的投影
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for(int i=0; i<N; i++)   //这个函数最关键的地方！如何定义误差边。
    {
        if(!vpMatches1[i])
            continue;

        // pMP1和pMP2是匹配的MapPoints
        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                // 步骤2.2 添加PointXYZ顶点
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w*P3D1w + t1w;
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        // 步骤2.3 添加两个顶点（3D点）到相机投影的边
        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    // 步骤3：g2o开始优化，先迭代5次
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // 步骤4：剔除一些误差大的边
    // Check inliers
    // 进行卡方检验，大于阈值的边剔除
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers
    // 步骤5：再次g2o优化剔除后剩下的边
    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    // 步骤6：得到优化后的结果
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;
}


} //namespace ORB_SLAM
