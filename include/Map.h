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

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "MapLine.h"
#include "KeyFrame.h"
#include <set>

#include <mutex>



namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;
class MapLine;

class Map
{
public:
    Map();

    void AddKeyFrame(KeyFrame* pKF);
    void EraseKeyFrame(KeyFrame* pKF);

    ///这两个函数是ORBSLAM中有的，吴博注释版中没有这里我们也加上
    void InformNewBigChange();
    int GetLastBigChangeIdx();


    void AddMapPoint(MapPoint* pMP);
    void EraseMapPoint(MapPoint* pMP);
    // 设置参考地图点？
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);

    //--line--
    void AddMapLine(MapLine* pML);
    void EraseMapLine(MapLine* pML);
    void SetReferenceMapLines(const std::vector<MapLine*> &vpMPs);


    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();
    //--line--
    std::vector<MapLine*> GetAllMapLines();
    std::vector<MapLine*> GetReferenceMapLines();


    long unsigned int MapPointsInMap(); //这个函数的返回值是什么？
    //--line--
    long unsigned int MapLinesInMap();
    long unsigned  KeyFramesInMap();

    long unsigned int GetMaxKFid();

    void clear();

    vector<KeyFrame*> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;

    //--line--
    std::mutex mMutexLineCreation;

protected:
    std::set<MapPoint*> mspMapPoints; ///< MapPoints
    //--line--
    std::set<MapLine*> mspMapLines;  ///< MapLines

    std::set<KeyFrame*> mspKeyFrames; ///< Keyframs

    std::vector<MapPoint*> mvpReferenceMapPoints;
    //--line--
    std::vector<MapLine*> mvpReferenceMapLines;

    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA) .
    int mnBigChangeIdx;
    //这一步是原ORBSLAM中有的，但是wubo注释版中竟然没有……这不是坑吗。
    //可能是因为这个成员在系统从来没用到过

    std::mutex mMutexMap;
};

} //namespace ORB_SLAM

#endif // MAP_H
