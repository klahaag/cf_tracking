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
// + Author: Klaus Haag
// * The PSR was proposed in [1]. This file is based on PSR calculation calculation from:
// * https://github.com/Itseez/opencv/blob/d279f127c1a0e28951bfdbba5516edc1bf0a0965/samples/python2/mosse.py
// * Set values below 0 to 0 in the response for the sidelobe calculation

// References:
// [1] D. Bolme, et al.,
// “Visual Object Tracking using Adaptive Correlation Filters,”
// in Proc. CVPR, 2010.
*/

#ifndef PSR_HPP_
#define PSR_HPP_

#include <limits>

template<typename T> inline
T calcPsr(const cv::Mat &response, const cv::Point2i &maxResponseIdx, const int deletionRange, T& peakValue)
{
    peakValue = response.at<T>(maxResponseIdx);
    double psrClamped = 0;

    cv::Mat sideLobe = response.clone();
    sideLobe.setTo(0, sideLobe < 0);

    cv::rectangle(sideLobe,
        cv::Point2i(maxResponseIdx.x - deletionRange, maxResponseIdx.y - deletionRange),
        cv::Point2i(maxResponseIdx.x + deletionRange, maxResponseIdx.y + deletionRange),
        cv::Scalar(0), cv::FILLED);

    cv::Scalar mean_;
    cv::Scalar std_;
    cv::meanStdDev(sideLobe, mean_, std_);
    psrClamped = (peakValue - mean_[0]) / (std_[0] + std::numeric_limits<T>::epsilon());
    return static_cast<T>(psrClamped);
}

#endif
