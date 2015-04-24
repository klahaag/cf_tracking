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
// + Klaus Haag (converted mulSpectrums to divSpectrums)

// TODO:
// * vectorize
// * check precision errors for float (casting to double
//      may not be necessary)
// * check inconsistency for float for real parts only
//      (not casted to double)
// * needs more testing (conjB == true) is untested

#include "math_spectrums.hpp"

void divSpectrums(cv::InputArray _numeratorA, cv::InputArray _denominatorB,
    cv::OutputArray _dst, int flags, bool conjB)
{
    cv::Mat srcA = _numeratorA.getMat(), srcB = _denominatorB.getMat();
    int depth = srcA.depth(), cn = srcA.channels(), type = srcA.type();
    int rows = srcA.rows, cols = srcA.cols;
    int j, k;

    CV_Assert(type == srcB.type() && srcA.size() == srcB.size());
    CV_Assert(type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2);

    _dst.create(srcA.rows, srcA.cols, type);
    cv::Mat dst = _dst.getMat();

    bool is_1d = (flags & cv::DFT_ROWS) || (rows == 1 || (cols == 1 &&
        srcA.isContinuous() && srcB.isContinuous() && dst.isContinuous()));

    if (is_1d && !(flags & cv::DFT_ROWS))
        cols = cols + rows - 1, rows = 1;

    // complex number representation
    // http://mathworld.wolfram.com/ComplexDivision.html
    // (a,b) / (c,d) = ((ac+bd)/v , (bc-ad)/v)
    // with v = (c^2 + d^2)
    double a = 0.0, b = 0.0, c = 0.0, d = 0.0, v = 0.0;

    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    if (depth == CV_32F)
    {
        const float* dataA = srcA.ptr<float>();
        const float* dataB = srcB.ptr<float>();
        float* dataC = dst.ptr<float>();

        size_t stepA = srcA.step / sizeof(dataA[0]);
        size_t stepB = srcB.step / sizeof(dataB[0]);
        size_t stepC = dst.step / sizeof(dataC[0]);

        if (!is_1d && cn == 1)
        {
            // even: one loop execution
            // odd: two loop executions
            for (k = 0; k < (cols % 2 ? 1 : 2); k++)
            {
                if (k == 1)
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;

                // the following 2 elements do not have an imaginary part
                // TODO: check precision
                dataC[0] = dataA[0] / dataB[0];

                if (rows % 2 == 0)
                    dataC[(rows - 1)*stepC] = dataA[(rows - 1)*stepA] / dataB[(rows - 1)*stepB];

                if (!conjB)
                {
                    for (j = 1; j <= rows - 2; j += 2)
                    {
                        a = (double)dataA[j*stepA];
                        b = (double)dataA[(j + 1)*stepA];
                        c = (double)dataB[j*stepB];
                        d = (double)dataB[(j + 1)*stepB];
                        v = (c*c) + (d*d);

                        dataC[j*stepC] = (float)((a * c + b * d) / v);
                        dataC[(j + 1)*stepC] = (float)((b * c - a * d) / v);
                    }
                }
                else
                {
                    for (j = 1; j <= rows - 2; j += 2)
                    {
                        a = (double)dataA[j*stepA];
                        b = -(double)dataA[(j + 1)*stepA];
                        c = (double)dataB[j*stepB];
                        d = -(double)dataB[(j + 1)*stepB];
                        v = (c*c) + (d*d);

                        dataC[j*stepC] = (float)((a * c + b * d) / v);
                        dataC[(j + 1)*stepC] = (float)((b * c - a * d) / v);
                    }
                }

                if (k == 1)
                    dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
            }
        }

        for (; rows--; dataA += stepA, dataB += stepB, dataC += stepC)
        {
            if (is_1d && cn == 1)
            {
                // first and last element in row -> no imaginary part
                dataC[0] = dataA[0] / dataB[0];

                if (cols % 2 == 0)
                    dataC[j1] = dataA[j1] / dataB[j1];
            }

            if (!conjB)
            {
                for (j = j0; j < j1; j += 2)
                {
                    a = (double)dataA[j];
                    b = (double)dataA[j + 1];
                    c = (double)dataB[j];
                    d = (double)dataB[j + 1];
                    v = (c*c) + (d*d);
                    dataC[j] = (float)((a * c + b * d) / v);
                    dataC[j + 1] = (float)((b * c - a * d) / v);
                }
            }
            else
            {
                for (j = j0; j < j1; j += 2)
                {
                    a = (double)dataA[j];
                    b = -(double)dataA[j + 1];
                    c = (double)dataB[j];
                    d = -(double)dataB[j + 1];
                    v = (c*c) + (d*d);
                    dataC[j] = (float)((a * c + b * d) / v);
                    dataC[j + 1] = (float)((b * c - a * d) / v);
                }
            }
        }
    }
    else
    {
        const double* dataA = srcA.ptr<double>();
        const double* dataB = srcB.ptr<double>();
        double* dataC = dst.ptr<double>();

        size_t stepA = srcA.step / sizeof(dataA[0]);
        size_t stepB = srcB.step / sizeof(dataB[0]);
        size_t stepC = dst.step / sizeof(dataC[0]);

        if (!is_1d && cn == 1)
        {
            for (k = 0; k < (cols % 2 ? 1 : 2); k++)
            {
                if (k == 1)
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;

                dataC[0] = dataA[0] / dataB[0];

                if (rows % 2 == 0)
                    dataC[(rows - 1)*stepC] = dataA[(rows - 1)*stepA] / dataB[(rows - 1)*stepB];

                if (!conjB)
                {
                    for (j = 1; j <= rows - 2; j += 2)
                    {
                        a = dataA[j*stepA];
                        b = dataA[(j + 1)*stepA];
                        c = dataB[j*stepB];
                        d = dataB[(j + 1)*stepB];
                        v = (c*c) + (d*d);

                        dataC[j*stepC] = (a * c + b * d) / v;
                        dataC[(j + 1)*stepC] = (b * c - a * d) / v;
                    }
                }
                else
                {
                    for (j = 1; j <= rows - 2; j += 2)
                    {
                        a = dataA[j*stepA];
                        b = -dataA[(j + 1)*stepA];
                        c = dataB[j*stepB];
                        d = -dataB[(j + 1)*stepB];
                        v = (c*c) + (d*d);

                        dataC[j*stepC] = (a * c + b * d) / v;
                        dataC[(j + 1)*stepC] = (b * c - a * d) / v;
                    }
                }

                if (k == 1)
                    dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
            }
        }

        for (; rows--; dataA += stepA, dataB += stepB, dataC += stepC)
        {
            if (is_1d && cn == 1)
            {
                dataC[0] = dataA[0] / dataB[0];

                if (cols % 2 == 0)
                    dataC[j1] = dataA[j1] / dataB[j1];
            }

            if (!conjB)
            {
                for (j = j0; j < j1; j += 2)
                {
                    a = dataA[j];
                    b = dataA[j + 1];
                    c = dataB[j];
                    d = dataB[j + 1];
                    v = (c*c) + (d*d);
                    dataC[j] = (a * c + b * d) / v;
                    dataC[j + 1] = (b * c - a * d) / v;
                }
            }
            else
            {
                for (j = j0; j < j1; j += 2)
                {
                    a = dataA[j];
                    b = -dataA[j + 1];
                    c = dataB[j];
                    d = -dataB[j + 1];
                    v = (c*c) + (d*d);
                    dataC[j] = (a * c + b * d) / v;
                    dataC[j + 1] = (b * c - a * d) / v;
                }
            }
        }
    }
}
