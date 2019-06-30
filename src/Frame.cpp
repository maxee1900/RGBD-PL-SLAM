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
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),
     NL(frame.NL), mvKeylinesUn(frame.mvKeylinesUn), mLdesc(frame.mLdesc), mvpMapLines(frame.mvpMapLines), mvbLineOutlier(frame.mvbLineOutlier), mvKeyLineFunctions(frame.mvKeyLineFunctions)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];  //元素为vector类型

    if(!frame.mTcw.empty())  //这个直接在初始化列表中进行也可以啊
        SetPose(frame.mTcw);
}


//*****PASS*****
//// 双目的初始化,这里和orbslam中同没有改动
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mb(0), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    // 同时对左右目提特征， 有意思啊！这里用到了两个线程操作
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();  //都是阻塞的方式加入到主线程
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;
    // Undistort特征点，这里没有对双目进行校正，因为要求输入的图像已经进行极线校正
    UndistortKeyPoints();

    // 计算双目间的匹配, 匹配成功的特征点会计算其深度
    // 深度存放在mvuRight 和 mvDepth 中
    ComputeStereoMatches();

    // 对应的mappoints
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));   
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

///  RGBD初始化，重点关注！ todo line版本中对这个函数没有做改动，我需要改动。
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)  //共9个参数
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),  //右目ORB提取器初始化为NULL指针
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
    ExtractORB(0,imGray);  //第一个参数为0，提取左目特征点

    N = mvKeys.size();  //上一个函数已经更新了mvKeys

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();  ///

    // 深度存放在mvuRight 和 mvDepth中。
    // TODO 这个函数需要加入线的部分，考虑引入两个数据成员mvuLineRight mvLineDepth
    ComputeStereoFromRGBD(imDepth);  ///

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

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

    AssignFeaturesToGrid();  /// 总共有三个地方我用///标记着，重点理解这三个函数
}

//*****PASS*****
//// 单目初始化
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();  // float
    mfLogScaleFactor = log(mfScaleFactor);  // float
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction. 原ORB中
    // ExtractORB(0,imGray);

#if 1
// 自己添加修改，同时对两种特征进行提取
//    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    thread threadPoint(&Frame::ExtractORB, this, 0, imGray);
    thread threadLine(&Frame::ExtractLSD, this, imGray);  /// 注意ExtractLSD函数的实现
    threadPoint.join();
    threadLine.join();
//    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
//    cout << "featureEx time: " << time_used.count() << endl;
//    ofstream file("extractFeatureTime.txt", ios::app);
//    file << time_used.count() << endl;
//    file.close();
#endif


#if 0
    // 此处是先提取点特征，后提取线特征
    // ORB extraction
    ExtractORB(0,imGray);

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        // line feature extraction, 自己添加的
    ExtractLSD(imGray);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double textract= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//        cout << "只提取线特征耗时： " << textract << " 秒。" << endl;
#endif



    N = mvKeys.size();
    NL = mvKeylinesUn.size();

    if(mvKeys.empty())
        return;

    // 调用OpenCV的矫正函数矫正orb提取的特征点
    // todo 下面这个函数在line版本中没有用，这里可能有问题。 同样的是不是也要写一个矫正关键线的函数
    //UndistortKeyPoints();

    //line版本
    mvKeysUn = mvKeys;

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);  //初始化的时候默认每个特征点都不是外点

    // --line--
    mvpMapLines = vector<MapLine*>(NL,static_cast<MapLine*>(NULL));
    mvbLineOutlier = vector<bool>(NL,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
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

    AssignFeaturesToGrid();
}  // todo 这个函数的主要问题在于去畸变操作是在哪里进行的 how, 解决了这个问题后再应用到RGBD的初始化中


////
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

            //todo line版本中这里没有操作，但是是不是对于线特征也可以这样将线特征划分到格子内，或者将线特征的终点划分到
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
void Frame::ExtractLSD(const cv::Mat &im)
{
    mpLineSegment->ExtractLineSegment(im, mvKeylinesUn, mLdesc, mvKeyLineFunctions);
}


//// --line-- 这个函数是最长的，有点耐心，仔细看看实现并思考有没有问题.
//// 这个函数在系统似乎没有用到！先写着
//根据两个匹配的特征线计算特征线的3D坐标, frame1是当前帧，frame2是前一帧
void Frame::ComputeLine3D(Frame &frame1, Frame &frame2) /// 是不是应该const传递
{
    //*******A. 计算两帧的匹配线段
    BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);  //todo找时间看一下这个函数的两个参数是什么含义，是不是还可以换成其他的参数

    Mat ldesc1, ldesc2;
    ldesc1 = frame1.mLdesc;
    ldesc2 = frame2.mLdesc;
    vector<vector<DMatch>> lmatches;
    vector<DMatch> good_matches;

    bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

    double nn_dist_th, nn12_dist_th;
    frame1.lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);

    //    nn12_dist_th = nn12_dist_th * 0.1;
    nn12_dist_th = nn12_dist_th * 0.5;  // 这个参数可以调
    sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
    vector<KeyLine> keylines1 = frame1.mvKeylinesUn;     //暂存mvKeylinesUn的集合
    frame1.mvKeylinesUn.clear();    //清空当前帧的mvKeylinesUn。 这里可以改进

    vector<KeyLine> keylines2;

    for(int i=0; i<lmatches.size(); i++)
    {
        double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
        if(dist_12 > nn12_dist_th)
        {
            //认为这个匹配比较好，应该更新该帧的的匹配线，也就是只保存了有匹配的直线特征 ？
            good_matches.push_back(lmatches[i][0]);
            frame1.mvKeylinesUn.push_back(keylines1[lmatches[i][0].queryIdx]);
            //更新当前帧的匹配线
            keylines2.push_back(frame2.mvKeylinesUn[lmatches[i][0].trainIdx]);
            //暂存前一帧的匹配线，用于计算3D端点
        }
    }


    //*******B. 计算当前帧mvKeylinesUn对应的3D端点
    // B-1: frame1的R,t，世界坐标系，相机内参
    Mat Rcw1 = frame1.mRcw;
    Mat Rwc1 = frame1.mRwc;
    Mat tcw1 = frame1.mtcw;
    Mat Tcw1(3, 4, CV_32F);  // ?
    Rcw1.copyTo(Tcw1.rowRange(0,3).colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));

    Mat Ow1 = frame1.mOw;  //当前帧世界系下的位置

    const float &fx1 = frame1.fx;
    const float &fy1 = frame1.fy;
    const float &cx1 = frame1.cx;
    const float &cy1 = frame1.cy;
    const float &invfx1 = frame1.invfx;
    const float &invfy1 = frame1.invfy;

    // B-2: frame2的R,t，世界坐标系，相机内参
    Mat Rcw2 = frame2.mRcw;
    Mat Rwc2 = frame2.mRwc;
    Mat tcw2 = frame2.mtcw;
    Mat Tcw2(3, 4, CV_32F);
    Rcw2.copyTo(Tcw2.rowRange(0,3).colRange(0,3));
    tcw2.copyTo(Tcw2.col(3));

    Mat Ow2 = frame2.mOw;

    const float &fx2 = frame2.fx;
    const float &fy2 = frame2.fy;
    const float &cx2 = frame2.cx;
    const float &cy2 = frame2.cy;
    const float &invfx2 = frame2.invfx;
    const float &invfy2 = frame2.invfy;

    // B-3: 对每对匹配通过三角化生成3D端点
    Mat BaseLine = Ow2 - Ow1;
    const float baseLine = norm( BaseLine );   //向量的长度，二范数

    // B-3.1: 根据两帧的姿态计算两帧之间的基本矩阵, Essential Matrix: t12叉乘R2
    const Mat &K1 = frame1.mK;
    const Mat &K2 = frame2.mK;
    Mat R12 = Rcw1*Rwc2;
    Mat t12 = -Rcw1*Rwc2*tcw2 + tcw1;
    Mat t12x = SkewSymmetricMatrix(t12);
    Mat essential_matrix = K1.t().inv()*t12x*R12*K2.inv();  //todo 验证是否正确

#if 0  // 以下是局部地图线程中的计算两关键帧之间F12的函数，对比一下
    cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
    {
        // Essential Matrix: t12叉乘R12
        // Fundamental Matrix: inv(K1)*E*inv(K2)

        cv::Mat R1w = pKF1->GetRotation();
        cv::Mat t1w = pKF1->GetTranslation();
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();

        cv::Mat R12 = R1w*R2w.t();
        cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

        cv::Mat t12x = SkewSymmetricMatrix(t12);

        const cv::Mat &K1 = pKF1->mK;
        const cv::Mat &K2 = pKF2->mK;


        return K1.t().inv()*t12x*R12*K2.inv();
    }

#endif


    // B-3.2: 三角化
    const int nlmatches = good_matches.size();
    for (int i = 0; i < nlmatches; ++i) {
        KeyLine &kl1 = frame1.mvKeylinesUn[i]; //TODO 这里修改我可以再定义一个keylines1，对frame1中的数据成员不做变动
        KeyLine &kl2 = keylines2[i];

        //------起始点,start points----- todo 检查这段代码，参考LocalMapping中的三角化实现

        // 得到特征线段的起始点在归一化平面上的坐标，没问题得到的归一化坐标(X/Z,Y/Z,1)
        Mat sn1 = (Mat_<float>(3, 1) << (kl1.startPointX - cx1) * invfx1, (kl1.startPointY - cy1) * invfy1, 1.0);
        Mat sn2 = (Mat_<float>(3, 1) << (kl2.startPointX - cx2) * invfx2, (kl2.startPointY - cy2) * invfy2, 1.0);

        // 把对应的起始点坐标转换到世界坐标系下
        Mat sray1 = Rwc1 * sn1;  //
        Mat sray2 = Rwc2 * sn2;
        // 计算在世界坐标系下，两个坐标向量间的余弦值
        const float cosParallax_sn = sray1.dot(sray2) / (norm(sray1) * norm(sray2));
        Mat s3D;
        if (cosParallax_sn > 0 && cosParallax_sn < 0.998) {  //有点多余 余弦值基本上都在这个范围啊
            // linear triangulation method
            Mat A(4, 4, CV_32F);
            A.row(0) = sn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
            A.row(1) = sn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
            A.row(2) = sn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
            A.row(3) = sn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

            Mat w1, u1, vt1;
            SVD::compute(A, w1, u1, vt1, SVD::MODIFY_A | SVD::FULL_UV);

            s3D = vt1.row(3).t();

            if (s3D.at<float>(3) == 0)
                continue;

            // Euclidean coordinates
            s3D = s3D.rowRange(0, 3) / s3D.at<float>(3);  //除以了第四维向量
        }
        Mat s3Dt = s3D.t();



        //-------结束点,end points-------

        // 得到特征线段的起始点在归一化平面上的坐标
        Mat en1 = (Mat_<float>(3, 1) << (kl1.endPointX - cx1) * invfx1, (kl1.endPointY - cy1) * invfy1, 1.0);
        Mat en2 = (Mat_<float>(3, 1) << (kl2.endPointX - cx2) * invfx2, (kl2.endPointY - cy2) * invfy2, 1.0);

        // 把对应的起始点坐标转换到世界坐标系下
        Mat eray1 = Rwc1 * en1;
        Mat eray2 = Rwc2 * en2;
        // 计算在世界坐标系下，两个坐标向量间的余弦值
        const float cosParallax_en = eray1.dot(eray2) / (norm(eray1) * norm(eray2));
        Mat e3D;
        if (cosParallax_en > 0 && cosParallax_en < 0.998) {
            // linear triangulation method
            Mat B(4, 4, CV_32F);
            B.row(0) = en1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
            B.row(1) = en1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
            B.row(2) = en2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
            B.row(3) = en2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

            Mat w2, u2, vt2;
            SVD::compute(B, w2, u2, vt2, SVD::MODIFY_A | SVD::FULL_UV);

            e3D = vt2.row(3).t();

            if (e3D.at<float>(3) == 0)
                continue;

            // Euclidean coordinates
            e3D = e3D.rowRange(0, 3) / e3D.at<float>(3);
        }
        Mat e3Dt = e3D.t();

        // B-3.3：检测生成的3D点是否在相机前方，两个帧都需要检测

        float sz1 = Rcw1.row(2).dot(s3Dt) + tcw1.at<float>(2);
        if(sz1<=0)
            continue;

        float sz2 = Rcw2.row(2).dot(s3Dt) + tcw2.at<float>(2);
        if(sz2<=0)
            continue;

        float ez1 = Rcw1.row(2).dot(e3Dt) + tcw1.at<float>(2);
        if(ez1<=0)
            continue;

        float ez2 = Rcw2.row(2).dot(e3Dt) + tcw2.at<float>(2);
        if(ez2<=0)
            continue;

        //生成特征点时还有检测重投影误差和检测尺度连续性两个步骤，但是考虑到线特征的特殊性，先不写这两步
        //MapLine(int idx_, Vector6d line3D_, Mat desc_, int kf_obs_, Vector3d obs_, Vector3d dir_, Vector4d pts_, double sigma2_=1.f);
        //MapLine* pML = new MapLine();
    }
}


//// --line-- line descriptor MAD 计算两个线特征分布的中值绝对偏差
void Frame::lineDescriptorMAD( vector<vector<DMatch>> line_matches, double &nn_mad, double &nn12_mad) const {
    vector<vector<DMatch>> matches_nn, matches_12;
    matches_nn = line_matches;
    matches_12 = line_matches;
    // cout << "Frame::lineDescriptorMAD——matches_nn = "<<matches_nn.size() << endl;

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
    sort(matches_12.begin(), matches_12.end(), conpare_descriptor_by_NN12_dist());
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
 // 基于ORB，没有改动
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


//// --line-- 判断MapLine的两个端点是否在视野内.todo 这里没有考虑一个端点在视角时候怎么办
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

    if(viewCos<viewingCosLimit)  //夹角太大就不在视野了
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
 // 这个函数是在ORBmatchers用到
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
        double distance = (0.5*(x1+x2)-keyline.pt.x)^2+(0.5*(y1+y2)-keyline.pt.y)^2;
        if(distance > r*r)
            continue;

        // 2.比较斜率，KeyLine的angle就是代表斜率 todo are you sure?
        float slope = (y1-y2)/(x1-x2)-keyline.angle;  ///这里说明opencv中求出的特征线可以得到角度
        if(slope > r*0.01)  //这里的参数也可以调节
            continue;

        // 3.比较金字塔层数
        if(bCheckLevels)
        {
            if(keyline.octave<minLevel)  ///可见opencv中线的提取也是有尺度的！注意凡是涉及到尺度层的一些操作要保证和ORB中类似，切勿出错
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

// 调用OpenCV的矫正函数矫正orb提取的特征点
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


//// 没有改动
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


//*****PASS*****
/**
 * @brief 双目匹配。 暂时不看，略过
 *
 * 为左图的每一个特征点在右图中找到匹配点 \n
 * 根据基线(有冗余范围)上描述子距离找到匹配, 再进行SAD精确定位 \n
 * 最后对所有SAD的值进行排序, 剔除SAD值较大的匹配对，然后利用抛物线拟合得到亚像素精度的匹配 \n
 * 匹配成功后会更新 mvuRight(ur) 和 mvDepth(Z)
 */
void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    // 步骤1：建立特征点搜索范围对应表，一个特征点在一个带状区域内搜索匹配特征点
    // 匹配搜索的时候，不仅仅是在一条横线上搜索，而是在一条横向搜索带上搜索,简而言之，原本每个特征点的纵坐标为1，这里把特征点体积放大，纵坐标占好几行
    // 例如左目图像某个特征点的纵坐标为20，那么在右侧图像上搜索时是在纵坐标为18到22这条带上搜索，搜索带宽度为正负2，搜索带的宽度和特征点所在金字塔层数有关
    // 简单来说，如果纵坐标是20，特征点在图像第20行，那么认为18 19 20 21 22行都有这个特征点
    // vRowIndices[18]、vRowIndices[19]、vRowIndices[20]、vRowIndices[21]、vRowIndices[22]都有这个特征点编号
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        // !!在这个函数中没有对双目进行校正，双目校正是在外层程序中实现的
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        // 计算匹配搜索的纵向宽度，尺度越大（层数越高，距离越近），搜索范围越大
        // 如果特征点在金字塔第一层，则搜索范围为:正负2
        // 尺度越大其位置不确定性越高，所以其搜索半径越大
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;        // NOTE bug mb没有初始化，mb的赋值在构造函数中放在ComputeStereoMatches函数的后面
    const float minD = 0;        // 最小视差, 设置为0即可
    const float maxD = mbf/minZ;  // 最大视差, 对应最小深度 mbf/minZ = mbf/mb = mbf/(mbf/fx) = fx (wubo???)

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    // 步骤2：对左目相机每个特征点，通过描述子在右目带状搜索区域找到匹配点, 再通过SAD做亚像素匹配
    // 注意：这里是校正前的mvKeys，而不是校正后的mvKeysUn
    // KeyFrame::UnprojectStereo和Frame::UnprojectStereo函数中不一致
    // 这里是不是应该对校正后特征点求深度呢？(wubo???)
    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        // 可能的匹配点
        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD; // 最小匹配范围
        const float maxU = uL-minD; // 最大匹配范围

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        // 每个特征点描述子占一行，建立一个指针指向iL特征点对应的描述子
        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        // 步骤2.1：遍历右目所有可能的匹配点，找出最佳匹配点（描述子距离最小）
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            // 仅对近邻尺度的特征点进行匹配
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }
        // 最好的匹配的匹配误差存在bestDist，匹配点位置存在bestIdxR中

        // Subpixel match by correlation
        // 步骤2.2：通过SAD匹配提高像素匹配修正量bestincR
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            // kpL.pt.x对应金字塔最底层坐标，将最佳匹配的特征点对尺度变换到尺度对应层 (scaleduL, scaledvL) (scaleduR0, )
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5; // 滑动窗口的大小11*11 注意该窗口取自resize后的图像
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) * cv::Mat::ones(IL.rows,IL.cols,CV_32F);//窗口中的每个元素减去正中心的那个元素，简单归一化，减小光照强度影响

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1); // 11

            // 滑动窗口的滑动范围为（-L, L）,提前判断滑动窗口滑动过程中是否会越界
            const float iniu = scaleduR0+L-w; //这个地方是否应该是scaleduR0-L-w (wubo???)
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                // 横向滑动窗口
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) * cv::Mat::ones(IR.rows,IR.cols,CV_32F);//窗口中的每个元素减去正中心的那个元素，简单归一化，减小光照强度影响

                float dist = cv::norm(IL,IR,cv::NORM_L1); // 一范数，计算差的绝对值
                if(dist<bestDist)
                {
                    bestDist =  dist;// SAD匹配目前最小匹配偏差
                    bestincR = incR; // SAD匹配目前最佳的修正量
                }

                vDists[L+incR] = dist; // 正常情况下，这里面的数据应该以抛物线形式变化
            }

            if(bestincR==-L || bestincR==L) // 整个滑动窗口过程中，SAD最小值不是以抛物线形式出现，SAD匹配失败，同时放弃求该特征点的深度
                continue;

            // Sub-pixel match (Parabola fitting)
            // 步骤2.3：做抛物线拟合找谷底得到亚像素匹配deltaR
            // (bestincR,dist) (bestincR-1,dist) (bestincR+1,dist)三个点拟合出抛物线
            // bestincR+deltaR就是抛物线谷底的位置，相对SAD匹配出的最小值bestincR的修正量为deltaR
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            // 抛物线拟合得到的修正量不能超过一个像素，否则放弃求该特征点的深度
            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            // 通过描述子匹配得到匹配点位置为scaleduR0
            // 通过SAD匹配找到修正量bestincR
            // 通过抛物线拟合找到亚像素修正量deltaR
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            // 这里是disparity，根据它算出depth
            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD) // 最后判断视差是否在范围内
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                // depth 是在这里计算的
                // depth=baseline*fx/disparity
                mvDepth[iL]=mbf/disparity;   // 深度
                mvuRight[iL] = bestuR;       // 匹配对在右图的横坐标
                vDistIdx.push_back(pair<int,int>(bestDist,iL)); // 该特征点SAD匹配最小匹配偏差
            }
        }
    }

    // 步骤3：剔除SAD匹配偏差较大的匹配特征点
    // 前面SAD匹配只判断滑动窗口中是否有局部最小值，这里通过对比剔除SAD匹配偏差比较大的特征点的深度
    sort(vDistIdx.begin(),vDistIdx.end()); // 根据所有匹配对的SAD偏差进行排序, 距离由小到大
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median; // 计算自适应距离, 大于此距离的匹配对将剔除

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


///*****RGBD******
void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    // mvDepth直接由depth图像读取
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;  //TODO 这里是不是应该用去畸变坐标？？对校正后的点求深度
        const float &u = kp.pt.x;  //TODO 可以测试一下！

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d; //这里指的是像素坐标，所以一定是大于0的
             ///注意： 这里按照slam书上得到公式uR=uL - fb/d. 实际中UR的坐标应该为 uR* = uR + cx.
             ///为什么按照上面的公式计算出来没问题呢，因为上面公式中的第一个数为像素点的横坐标 实际上等价于 uR = uL+cx-fb/d。因此mvuRight中都是正数，单目的话都为-1.
        }
    }
}

/**
 * @brief Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
 * @param  i 第i个keypoint
 * @return   3D点（相对于世界坐标系）
 */
 // 没有改动
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

} //namespace ORB_SLAM
