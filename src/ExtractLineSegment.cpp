//
// Created by max on 19-6-24.
//
// 写头文件中相应函数的实现

#include "ExtractLineSegment.h"

namespace ORB_SLAM2
{
    // construct fun
    LineSegment::LineSegment() {}

void ExtractLineSegment(const Mat& img, vector<KeyLine> &keylines, Mat &ldesc, vector<Vector3d> &keylineFunctions, int scale=1.2, int numOctaves=1)
{
    /*
    Ptr<BinaryDescriptor> lbd = BinaryDescriptor::createBinaryDescriptor();
    Ptr<line_descriptor::LSDDetectorC> lsd = line_descriptor::LSDDetectorC::createLSDDetectorC();

    line_descriptor::LSDDetectorC::LSDOptions opts;
    opts.refine = 2;
    opts.scale = 1.2;
    opts.sigma_scale = 0.6;
    opts.quant = 2.0;
    opts.ang_th = 22.5;
    opts.log_eps = 1.0;
    opts.density_th = 0.6;
    opts.n_bins = 1024;
    opts.min_length = 0.025;

    lsd->detect(img, keylines, scale, 1, opts);

     //这段代码用来设置LSD中的参数
     */

    Ptr<BinaryDescriptor> lbd = BinaryDescriptor::createBinaryDescriptor();
    Ptr<line_descriptor::LSDDetector> lsd = line_descriptor::LSDDetector::createLSDDetector();
    lsd->detect(img, keylines, scale, numOctaves);

    unsigned int lsdNFeatures = 40;

    //filter lines
    if(keylines.size() > lsdNFeatures)
    {
        sort(keylines.begin(), keylines.end(), sort_lines_by_response());
        keylines.resize(lsdNFeatures);
        for(unsigned int i=0; i < lsdNFeatures; i++)
            keylines[i].class_id = i;
    }

    lbd->compute(img, keylines, ldesc);

    // 计算线段所在直线系数
    for(vector<KeyLine>::iterator it=keylines.begin(); it!=keylines.end(); ++it)
    {
        Vector3d sp_l;
        sp_l << it->startPointX, it->startPointY, 1.0;
        Vector3d ep_l;
        ep_l << it->endPointX, it->endPointY, 1.0;

        Vector3d lineP;
        lineP << sp_l.cross(ep_l);  //这里是叉乘，结果即为ax+by+c=0的直线系数(a,b,c)
        lineP = lineP / sqrt(lineP[0]*lineP[0] + lineP[1]*lineP[1]);
        keylineFunctions.push_back(lineP);  //按照直线的response排序后的直线参数
    }

}

//
void LineSegment::LineSegmentMathch(Mat &ldesc1, Mat &ldesc2)
{
    BFMatcher bfm(NORM_HAMMING, false);
    bfm.knnMatch(ldesc1, ldesc2, mvlineMatches, 2);  //这个函数的输出就是类的数据成员line_matches
}

// 计算中位数绝对偏差，衡量样本数据的差异性
void LineSegment::LineDescriptorMAD()
{
    vector<vector<DMatch>> matches_nn, matches_12;
    matches_nn = mvlineMatches;  //把描述子矩阵看做是一个二维矩阵 行表示第i个特征的匹配 第一列表示最小匹配 第二列表示次小匹配
    matches_12 = mvlineMatches;


    // estimate the NN's distance standard deviation
    double nn_dist_median;
    sort( matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
    nn_dist_median = matches_nn[int(matches_nn.size()/2)][0].distance;
    for(unsigned int i=0; i<matches_nn.size(); i++)
        matches_nn[i][0].distance = fabsf(matches_nn[i][0].distance - nn_dist_median);
    sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
    mnnMad = 1.4826 * matches_nn[int(matches_nn.size()/2)][0].distance;

    // estimate the NN's 12 distance standard deviation
    double nn12_dist_median;
    sort( matches_12.begin(), matches_12.end(), compare_descriptor_by_NN12_dist());
    nn12_dist_median = matches_12[int(matches_12.size()/2)][1].distance - matches_12[int(matches_12.size()/2)][0].distance;
    for (unsigned int j=0; j<matches_12.size(); j++)
        matches_12[j][0].distance = fabsf( matches_12[j][1].distance - matches_12[j][0].distance - nn12_dist_median);
    sort(matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist());
    mnn12Mad = 1.4826 * matches_12[int(matches_12.size()/2)][0].distance;

}


double LineSegment::LineSegmentOverlap(double spl_obs, double epl_obs, double spl_proj, double epl_proj)
{
    double sO    = min(spl_obs,  epl_obs);     //线特征两个端点的观察值的最小值. 这里要明白端点观测值到底是啥玩意，线段的端点应该是一个二维这里是怎么到一维的？
    double eO    = max(spl_obs,  epl_obs);     //线特征两个端点的观察值的最大值
    double sP    = min(spl_proj, epl_proj);    //线特征两个端点的投影点的最小值
    double eP    = max(spl_proj, epl_proj);    //线特征两个端点的投影点的最大值

    double length = eP-sP;  //mod 表示投影直线的长度

    double overlap;
    if ( (eP < sO) || (sP > eO) )
        overlap = 0.0;
    else{
        overlap = min(eO,eP) - max(sO,sP); //其实只需要这一句就，overlap是正数
    }

    if(length>0.01f)  //TODO 这里是不是有问题 有可能算出来overlap=1
        overlap = overlap / length;  //表示重合的部分占投影直线的比例
        // cout << "overlap:" << overlap << endl;
    else
        overlap = 0.f;

    return overlap;

}


}