/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//M*/

// Author: Intel
// Original file: https://github.com/Itseez/opencv/blob/46a6f70d88bf816525a3ded80e69237d1960152f/modules/core/src/dxt.cpp
// + Klaus Haag:
// * Converted mulSpectrums to divSpectrums
// * Converted mulSpectrums to addRealToSpectrum
// * Converted mulSpectrums to sumRealOfSpectrum
//

#ifndef MATH_SPECTRUMS_HPP_
#define MATH_SPECTRUMS_HPP_

#include <opencv2/core/core.hpp>

void divSpectrums(cv::InputArray _numeratorA, cv::InputArray _denominatorB,
    cv::OutputArray _dst, int flags = 0, bool conjB = false);

template <typename T>
cv::Mat addRealToSpectrum(T summand, cv::InputArray _numeratorA, int flags = 0)
{
    cv::Mat srcA = _numeratorA.getMat();
    int cn = srcA.channels(), type = srcA.type();
    int rows = srcA.rows, cols = srcA.cols;
    int j, k;

    CV_Assert(type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2);

    cv::Mat dst;
    dst.create(srcA.rows, srcA.cols, type);

    bool is_1d = (flags & cv::DFT_ROWS) || (rows == 1 || (cols == 1 &&
        srcA.isContinuous() && dst.isContinuous()));

    if (is_1d && !(flags & cv::DFT_ROWS))
        cols = cols + rows - 1, rows = 1;

    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    const T* dataA = srcA.ptr<T>();
    T* dataC = dst.ptr<T>();

    size_t stepA = srcA.step / sizeof(dataA[0]);
    size_t stepC = dst.step / sizeof(dataC[0]);

    if (!is_1d && cn == 1)
    {
        for (k = 0; k < (cols % 2 ? 1 : 2); k++)
        {
            if (k == 1)
                dataA += cols - 1, dataC += cols - 1;

            dataC[0] = dataA[0] + summand;

            if (rows % 2 == 0)
                dataC[(rows - 1)*stepC] = dataA[(rows - 1)*stepA] + summand;

            for (j = 1; j <= rows - 2; j += 2)
            {
                dataC[j*stepC] = dataA[j*stepA] + summand;
                dataC[(j + 1)*stepC] = dataA[(j + 1)*stepA];
            }

            if (k == 1)
                dataA -= cols - 1, dataC -= cols - 1;
        }
    }

    for (; rows--; dataA += stepA, dataC += stepC)
    {
        if (is_1d && cn == 1)
        {
            dataC[0] = dataA[0] + summand;

            if (cols % 2 == 0)
                dataC[j1] = dataA[j1] + summand;
        }

        for (j = j0; j < j1; j += 2)
        {
            dataC[j] = dataA[j] + summand;
            dataC[j + 1] = dataA[j + 1];
        }
    }

    return dst;
}

template <typename T>
T sumRealOfSpectrum(cv::InputArray _numeratorA, int flags = 0)
{
    cv::Mat srcA = _numeratorA.getMat();
    T sum_ = 0;
    int cn = srcA.channels(), type = srcA.type();
    int rows = srcA.rows, cols = srcA.cols;
    int j, k;
    CV_Assert(type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2);

    bool is_1d = (flags & cv::DFT_ROWS) || (rows == 1 || (cols == 1 &&
        srcA.isContinuous()));

    if (is_1d && !(flags & cv::DFT_ROWS))
        cols = cols + rows - 1, rows = 1;

    T multiplier = 1;

    if (cn == 1)
        multiplier = 2;

    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    const T* dataA = srcA.ptr<T>();
    size_t stepA = srcA.step / sizeof(dataA[0]);

    if (!is_1d && cn == 1)
    {
        for (k = 0; k < (cols % 2 ? 1 : 2); k++)
        {
            if (k == 1)
                dataA += cols - 1;

            sum_ += dataA[0];

            if (rows % 2 == 0)
                sum_ += dataA[(rows - 1)*stepA];

            for (j = 1; j <= rows - 2; j += 2)
            {
                sum_ += multiplier * dataA[j*stepA];
            }

            if (k == 1)
                dataA -= cols - 1;
        }
    }

    for (; rows--; dataA += stepA)
    {
        if (is_1d && cn == 1)
        {
            sum_ += dataA[0];

            if (cols % 2 == 0)
                sum_ += dataA[j1];
        }

        for (j = j0; j < j1; j += 2)
        {
            sum_ += multiplier * dataA[j];
        }
    }

    return sum_;
}

#endif
