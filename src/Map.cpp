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

#include "Map.h"

#include<mutex>

namespace ORB_SLAM2
{

    //注释版中原来没有mnBigChangeIdx，现在加上
Map::Map():mnMaxKFid(0),mnBigChangeIdx(0)
{
}

/**
 * @brief Insert KeyFrame in the map
 * @param pKF KeyFrame
 */
void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    //添加新的关键帧更新两部分：一个是mspKeyFrames；一个是mnMaxKFid
    mspKeyFrames.insert(pKF);
    if(pKF->mnId>mnMaxKFid)
        mnMaxKFid=pKF->mnId;
}

/**
 * @brief Insert MapPoint in the map
 * @param pMP MapPoint
 */
void Map::AddMapPoint(MapPoint *pMP)  //向地图中添加点仅仅是把该地图点加入地图点set集中
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

/**
 * @brief Erase MapPoint from the map
 * @param pMP MapPoint
 */
void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.  啥意思，不改的根ORB中保持一致
    // Delete the MapPoint
}

/**
 * @brief Erase KeyFrame from the map
 * @param pKF KeyFrame
 */
void Map::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

/**
 * @brief 设置参考MapPoints，将用于DrawMapPoints函数画图
 * @param vpMPs Local MapPoints
 */
void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs; //传入地图类中的数据成员即可
}

//----在吴博注释版上增加的---------------

void Map::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx()  //返回BigChange的个数
{
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}
//--止------------------------------------



vector<KeyFrame*> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());  //把set数据结构转换为vector结构
}

vector<MapPoint*> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());  //set转换为vector
}

//函数返回：地图中点的数量
long unsigned int Map::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

//函数返回地图中关键帧的数量
long unsigned int Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

//函数返回地图中的参考地图点列表
vector<MapPoint*> Map::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

//函数返回：地图中关键帧的最大ID号
long unsigned int Map::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

void Map::clear()
{
    for( auto sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
        delete *sit;

    for(set<MapLine*>::iterator sit=mspMapLines.begin(), send=mspMapLines.end(); sit!=send; sit++)
        delete *sit;

    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
        delete *sit;

    mspMapPoints.clear();
    mspMapLines.clear();
    mspKeyFrames.clear();
    mnMaxKFid = 0;
    mvpReferenceMapPoints.clear();
    mvpReferenceMapLines.clear();
    mvpKeyFrameOrigins.clear();
}


//-----line 相关函数实现------------------

void Map::AddMapLine(MapLine *pML)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapLines.insert(pML);
}

void Map::EraseMapLine(MapLine *pML)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapLines.erase(pML);
}

void Map::SetReferenceMapLines(const std::vector<MapLine *> &vpMLs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapLines = vpMLs;
}

vector<MapLine*> Map::GetAllMapLines()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapLine*> ( mspMapLines.begin(), mspMapLines.end());
}

vector<MapLine*> Map::GetReferenceMapLines()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapLines;
}

long unsigned int Map::MapLinesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapLines.size();
}


} //namespace ORB_SLAM
