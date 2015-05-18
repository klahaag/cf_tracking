/*
// License Agreement (3-clause BSD License)
// Copyright (c) 2015, Klaus Haag, all rights reserved.
// Third party copyrights and patents are property of their respective owners.
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
*/

#ifndef HELPER_H_
#define HELPER_H_

#include <opencv2/core/core.hpp>

#include "cv_ext.hpp"
#include "mat_consts.hpp"

namespace cf_tracking
{
    void dftCcs(const cv::Mat& input, cv::Mat& out, int flags = 0);
    void dftNoCcs(const cv::Mat& input, cv::Mat& out, int flags = 0);
    int mod(int dividend, int divisor);
    void depResize(const cv::Mat& source, cv::Mat& dst, const cv::Size& dsize);

    template<typename T>
    cv::Size_<T> sizeFloor(cv::Size_<T> size)
    {
        return cv::Size_<T>(floor(size.width), floor(size.height));
    }

    template <typename T>
    cv::Mat numberToRowVector(int n)
    {
        cv::Mat_<T> rowVec(n, 1);

        for (int i = 0; i < n; ++i)
            rowVec.template at<T>(i, 0) = static_cast<T>(i + 1);

        return rowVec;
    }

    template <typename T>
    cv::Mat numberToColVector(int n)
    {
        cv::Mat_<T> colVec(1, n);

        for (int i = 0; i < n; ++i)
            colVec.template at<T>(0, i) = static_cast<T>(i + 1);

        return colVec;
    }

    // http://home.isr.uc.pt/~henriques/circulant/
    template <typename T>
    T subPixelPeak(T* p)
    {
        T delta = mat_consts::constants<T>::c0_5 * (p[2] - p[0]) / (2 * p[1] - p[2] - p[0]);

        if (!std::isfinite(delta))
            return 0;

        return delta;
    }

    // http://home.isr.uc.pt/~henriques/circulant/
    template <typename T>
    cv::Point_<T> subPixelDelta(const cv::Mat& response, const cv::Point2i& delta)
    {
        cv::Point_<T> subDelta(static_cast<float>(delta.x), static_cast<float>(delta.y));
        T vNeighbors[3] = {};
        T hNeighbors[3] = {};

        for (int i = -1; i < 2; ++i)
        {
            vNeighbors[i + 1] = response.template at<T>(mod(delta.y + i, response.rows), delta.x);
            hNeighbors[i + 1] = response.template at<T>(delta.y, mod(delta.x + i, response.cols));
        }

        subDelta.y += subPixelPeak(vNeighbors);
        subDelta.x += subPixelPeak(hNeighbors);

        return subDelta;
    }

    // http://home.isr.uc.pt/~henriques/circulant/
    template <typename T>
    cv::Mat gaussianShapedLabels2D(T sigma, const cv::Size_<T>& size)
    {
        int width = static_cast<int>(size.width);
        int height = static_cast<int>(size.height);

        cv::Mat_<T> rs(height, width);

        CV_Assert(rs.isContinuous());

        T lowerBoundX = static_cast<T>(-floor(width * 0.5) + 1);
        T lowerBoundY = static_cast<T>(-floor(height * 0.5) + 1);

        T* colValues = new T[width];
        T* rsd = rs.template ptr<T>(0, 0);
        T rowValue = 0;
        T sigmaMult = static_cast<T>(-0.5 / (sigma*sigma));

        for (int i = 0; i < width; ++i)
            colValues[i] = (i + lowerBoundX) * (i + lowerBoundX);

        for (int row = 0; row < height; ++row)
        {
            rowValue = (row + lowerBoundY) * (row + lowerBoundY);

            for (int col = 0; col < width; ++col)
            {
                rsd[row*width + col] = exp((colValues[col] + rowValue) * sigmaMult);
            }
        }

        delete[] colValues;

        return rs;
    }

    // http://home.isr.uc.pt/~henriques/circulant/
    template <typename T>
    cv::Mat gaussianShapedLabelsShifted2D(T sigma, const cv::Size_<T>& size)
    {
        cv::Mat y = gaussianShapedLabels2D(sigma, size);
        cv::Point2f delta(static_cast<float>(1 - floor(size.width * 0.5)),
            static_cast<float>(1 - floor(size.height * 0.5)));

        shift(y, y, delta, cv::BORDER_WRAP);

        CV_Assert(y.at<T>(0, 0) == 1.0);
        return y;
    }

    template <typename BT, typename ET>
    cv::Mat pow(BT base_, const cv::Mat_<ET>& exponent)
    {
        cv::Mat dst = cv::Mat(exponent.rows, exponent.cols, exponent.type());
        int widthChannels = exponent.cols * exponent.channels();
        int height = exponent.rows;

        // http://docs.opencv.org/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#the-efficient-way
        if (exponent.isContinuous())
        {
            widthChannels *= height;
            height = 1;
        }

        int row = 0, col = 0;
        const ET* exponentd = 0;
        ET* dstd = 0;

        for (row = 0; row < height; ++row)
        {
            exponentd = exponent.template ptr<ET>(row);
            dstd = dst.template ptr<ET>(row);

            for (col = 0; col < widthChannels; ++col)
            {
                dstd[col] = std::pow(base_, exponentd[col]);
            }
        }

        return dst;
    }

    // http://en.wikipedia.org/wiki/Hann_function
    template<typename T>
    cv::Mat hanningWindow(int n)
    {
        CV_Assert(n > 0);
        cv::Mat_<T> w = cv::Mat_<T>(n, 1);

        if (n == 1)
        {
            w.template at<T>(0, 0) = 1;
            return w;
        }

        for (int i = 0; i < n; ++i)
            w.template at<T>(i, 0) = static_cast<T>(0.5 * (1.0 - cos(2.0 * 3.14159265358979323846 * i / (n - 1))));

        return w;
    }

    template <typename T>
    void divideSpectrumsNoCcs(const cv::Mat& numerator, const cv::Mat& denominator, cv::Mat& dst)
    {
        // http://mathworld.wolfram.com/ComplexDivision.html
        // (a,b) / (c,d) = ((ac+bd)/v , (bc-ad)/v)
        // with v = (c^2 + d^2)
        // Performance wise implemented according to
        // http://docs.opencv.org/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#howtoscanimagesopencv
        // TODO: this is still very slow => vectorize (note that mulSpectrums is not vectorized either...)

        int type = numerator.type();
        int channels = numerator.channels();

        CV_Assert(type == denominator.type()
            && numerator.size() == denominator.size()
            && channels == denominator.channels() && channels == 2);
        CV_Assert(type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2);

        dst = cv::Mat(numerator.rows, numerator.cols, type);
        int widthChannels = numerator.cols * channels;
        int height = numerator.rows;

        if (numerator.isContinuous() && denominator.isContinuous())
        {
            widthChannels *= height;
            height = 1;
        }

        int row = 0, col = 0;
        const T* numd, *denomd;
        T* dstd;
        T a, b, c, d, v;

        for (row = 0; row < height; ++row)
        {
            numd = numerator.ptr<T>(row);
            denomd = denominator.ptr<T>(row);
            dstd = dst.ptr<T>(row);

            for (col = 0; col < widthChannels; col += 2)
            {
                a = numd[col];          // real part
                b = numd[col + 1];      // imag part
                c = denomd[col];       // real part
                d = denomd[col + 1];   // imag part

                v = (c * c) + (d * d);

                dstd[col] = (a * c + b * d) / v;
                dstd[col + 1] = (b * c - a * d) / v;
            }
        }
    }

    // http://home.isr.uc.pt/~henriques/circulant/
    template<typename T>
    bool getSubWindow(const cv::Mat& image, cv::Mat& patch, const cv::Size_<T>& size,
        const cv::Point_<T>& pos, cv::Point_<T>* posInSubWindow = 0)
    {
        int width = static_cast<int>(size.width);
        int height = static_cast<int>(size.height);

        int xs = static_cast<int>(std::floor(pos.x) - std::floor(width / 2.0)) + 1;
        int ys = static_cast<int>(std::floor(pos.y) - std::floor(height / 2.0)) + 1;
        T posInSubWindowX = pos.x - xs;
        T posInSubWindowY = pos.y - ys;

        int diffTopX = -xs;
        int diffTopY = -ys;
        int diffBottomX = image.cols - xs - width;
        int diffBottomY = image.rows - ys - height;

        cv::Rect imageRect(0, 0, image.cols, image.rows);
        cv::Rect subRect(xs, ys, width, height);
        subRect &= imageRect;
        cv::Mat subWindow = image(subRect);

        if (subWindow.cols == 0 || subWindow.rows == 0)
            return false;

        if (diffTopX > 0 || diffTopY > 0
            || diffBottomX < 0 || diffBottomY < 0)
        {
            diffTopX = std::max(0, diffTopX);
            diffTopY = std::max(0, diffTopY);
            diffBottomX = std::min(0, diffBottomX);
            diffBottomY = std::min(0, diffBottomY);

            copyMakeBorder(subWindow, subWindow, diffTopY, -diffBottomY,
                diffTopX, -diffBottomX, cv::BORDER_REPLICATE);
        }

        // this if can be true if the sub window
        // is completely outside the image
        if (width != subWindow.cols ||
            height != subWindow.rows)
            return false;

        if (posInSubWindow != 0)
        {
            posInSubWindow->x = posInSubWindowX;
            posInSubWindow->y = posInSubWindowY;
        }

        patch = subWindow;

        return true;
    }
}

#endif
