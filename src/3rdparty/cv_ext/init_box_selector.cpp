/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
// License Agreement
// For Open Source Computer Vision Library
// (3-clause BSD License)
//
// Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2015, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of the contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//M*/

/*
// Original file: https://github.com/Itseez/opencv_contrib/blob/292b8fa6aa403fb7ad6d2afadf4484e39d8ca2f1/modules/tracking/samples/tracker.cpp
// + Author: Klaus Haag
// * Refactor file: Move target selection to separate class/file
*/

#include "init_box_selector.hpp"

void InitBoxSelector::onMouse(int event, int x, int y, int, void*)
{
    if (!selectObject)
    {
        switch (event)
        {
        case cv::EVENT_LBUTTONDOWN:
            //set origin of the bounding box
            startSelection = true;
            initBox.x = x;
            initBox.y = y;
            break;
        case cv::EVENT_LBUTTONUP:
            //set width and height of the bounding box
            initBox.width = std::abs(x - initBox.x);
            initBox.height = std::abs(y - initBox.y);
            startSelection = false;
            selectObject = true;
            break;
        case cv::EVENT_MOUSEMOVE:
            if (startSelection && !selectObject)
            {
                //draw the bounding box
                cv::Mat currentFrame;
                image.copyTo(currentFrame);
                cv::rectangle(currentFrame, cv::Point((int)initBox.x, (int)initBox.y), cv::Point(x, y), cv::Scalar(255, 0, 0), 2, 1);
                cv::imshow(windowTitle.c_str(), currentFrame);
            }
            break;
        }
    }
}

bool InitBoxSelector::selectBox(cv::Mat& frame, cv::Rect& initBox)
{
    frame.copyTo(image);
    startSelection = false;
    selectObject = false;
    cv::imshow(windowTitle.c_str(), image);
    cv::setMouseCallback(windowTitle.c_str(), onMouse, 0);

    while (selectObject == false)
    {
        char c = (char)cv::waitKey(10);

        if (c == 27)
            return false;
    }

    initBox = InitBoxSelector::initBox;
    cv::setMouseCallback(windowTitle.c_str(), 0, 0);
    cv::destroyWindow(windowTitle.c_str());
    return true;
}

const std::string InitBoxSelector::windowTitle = "Draw Bounding Box";
bool InitBoxSelector::startSelection = false;
bool InitBoxSelector::selectObject = false;
cv::Mat InitBoxSelector::image;
cv::Rect InitBoxSelector::initBox;
