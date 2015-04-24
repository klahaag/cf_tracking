/*
// Copyright (c) 2014, Klaus Haag
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// The views and conclusions contained in the software and documentation are those
// of the authors and should not be interpreted as representing official policies,
// either expressed or implied, of the FreeBSD Project.
*/

/*******************************************************************************
* OpenCV interface to Piotr's Computer Vision Matlab Toolbox' FHOG implementation:
https://github.com/pdollar/toolbox/blob/612f9a0451a6abbe2a64768c9e6654692929102e/channels/private/gradientMex.cpp

TODO:
* Rewrite fhog col major (clustered) calculation to row major (interleaved) calculation
** Calculating FHOG on an implicitly transposed image is possible as well (see cvFhogT),
-- but Performance is slightly lower for an unknown reason.
* Remove code duplication
* Fix hackfixes properly

*******************************************************************************/

#ifndef _GRADIENT_MEX_HPP_
#define _GRADIENT_MEX_HPP_

#include <opencv2/opencv.hpp>
#include "feature_channels.hpp"
#include "wrappers.hpp"

namespace piotr {
    void fhog(float * const M, float * const O,
        float * const H, int h, int w, int binSize,
        int nOrients, int softBin, float clip,
        bool calcEnergy = true);

    void gradMag(float * const I, float * const M,
        float * const O, int h, int w, int d, bool full);

    template<typename PRIMITIVE_TYPE>
    void fhogToCol(const cv::Mat& img, cv::Mat& cvFeatures,
        int binSize, int colIdx, PRIMITIVE_TYPE cosFactor)
    {
        const int orientations = 9;
        // ensure array is continuous
        const cv::Mat& image = (img.isContinuous() ? img : img.clone());
        int channels = image.channels();
        int computeChannels = 32;
        int width = image.cols;
        int height = image.rows;
        int widthBin = width / binSize;
        int heightBin = height / binSize;

        CV_Assert(channels == 1 || channels == 3);
        CV_Assert(cvFeatures.channels() == 1 && cvFeatures.isContinuous());

        float* const H = (float*)wrCalloc(static_cast<size_t>(widthBin * heightBin * computeChannels), sizeof(float));
        float* const I = (float*)wrCalloc(static_cast<size_t>(width * height * channels), sizeof(float));
        float* const M = (float*)wrCalloc(static_cast<size_t>(width * height), sizeof(float));
        float* const O = (float*)wrCalloc(static_cast<size_t>(width * height), sizeof(float));

        // row major (interleaved) to col major (non-interleaved;clustered;block)
        float* imageData = reinterpret_cast<float*>(image.data);

        float* const redChannel = I;
        float* const greenChannel = I + width * height;
        float* const blueChannel = I + 2 * width * height;
        int colMajorPos = 0, rowMajorPos = 0;

        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                colMajorPos = col * height + row;
                rowMajorPos = row * channels * width + col * channels;

                blueChannel[colMajorPos] = imageData[rowMajorPos];
                greenChannel[colMajorPos] = imageData[rowMajorPos + 1];
                redChannel[colMajorPos] = imageData[rowMajorPos + 2];
            }
        }

        // calc fhog in col major
        gradMag(I, M, O, height, width, channels, true);
        fhog(M, O, H, height, width, binSize, orientations, -1, 0.2f);

        // the order of rows in cvFeatures does not matter
        // as long as it is the same for all columns;
        // zero channel is not copied as it is the last
        // channel in H and cvFeatures rows doesn't include it
        PRIMITIVE_TYPE* cdata = reinterpret_cast<PRIMITIVE_TYPE*>(cvFeatures.data);
        int outputWidth = cvFeatures.cols;

        for (int row = 0; row < cvFeatures.rows; ++row)
            cdata[outputWidth*row + colIdx] = H[row] * cosFactor;

        wrFree(H);
        wrFree(M);
        wrFree(O);
        wrFree(I);
    }

    template<typename PRIMITIVE_TYPE>
    void fhogToCvColT(const cv::Mat& img, cv::Mat& cvFeatures,
        int binSize, int colIdx, PRIMITIVE_TYPE cosFactor)
    {
        const int orientations = 9;
        // ensure array is continuous
        const cv::Mat& image = (img.isContinuous() ? img : img.clone());
        int channels = image.channels();
        int computeChannels = 32;
        int width = image.cols;
        int height = image.rows;
        int widthBin = width / binSize;
        int heightBin = height / binSize;

        CV_Assert(channels == 1 || channels == 3);
        CV_Assert(cvFeatures.channels() == 1 && cvFeatures.isContinuous());

        float* const H = (float*)wrCalloc(static_cast<size_t>(widthBin * heightBin * computeChannels), sizeof(float));
        float* const M = (float*)wrCalloc(static_cast<size_t>(width * height), sizeof(float));
        float* const O = (float*)wrCalloc(static_cast<size_t>(width * height), sizeof(float));

        float* I = NULL;

        if (channels == 1)
            I = reinterpret_cast<float*>(image.data);
        else
        {
            I = (float*)wrCalloc(static_cast<size_t>(width * height * channels), sizeof(float));
            float* imageData = reinterpret_cast<float*>(image.data);
            float* redChannel = I;
            float* greenChannel = I + width * height;
            float* blueChannel = I + 2 * width * height;

            for (int i = 0; i < height * width; ++i)
            {
                blueChannel[i] = imageData[i * 3];
                greenChannel[i] = imageData[i * 3 + 1];
                redChannel[i] = imageData[i * 3 + 2];
            }
        }

        // calc fhog in col major - switch width and height
        gradMag(I, M, O, width, height, channels, true);
        fhog(M, O, H, width, height, binSize, orientations, -1, 0.2f);

        // the order of rows in cvFeatures does not matter
        // as long as it is the same for all columns;
        // zero channel is not copied as it is the last
        // channel in H and cvFeatures rows doesn't include it
        PRIMITIVE_TYPE* cdata = reinterpret_cast<PRIMITIVE_TYPE*>(cvFeatures.data);
        int outputWidth = cvFeatures.cols;

        for (int row = 0; row < cvFeatures.rows; ++row)
            cdata[outputWidth*row + colIdx] = H[row] * cosFactor;

        wrFree(H);
        wrFree(M);
        wrFree(O);

        if (channels != 1)
            wrFree(I);
    }

    template<typename PRIMITIVE_TYPE, class OUT>
    void cvFhog(const cv::Mat& img, std::shared_ptr<OUT>& cvFeatures, int binSize, int fhogChannelsToCopy = 31)
    {
        const int orientations = 9;
        // ensure array is continuous
        const cv::Mat& image = (img.isContinuous() ? img : img.clone());
        int channels = image.channels();
        int computeChannels = 32;
        int width = image.cols;
        int height = image.rows;
        int widthBin = width / binSize;
        int heightBin = height / binSize;

        float* const I = (float*)wrCalloc(static_cast<size_t>(width * height * channels), sizeof(float));
        float* const H = (float*)wrCalloc(static_cast<size_t>(widthBin * heightBin * computeChannels), sizeof(float));
        float* const M = (float*)wrCalloc(static_cast<size_t>(width * height), sizeof(float));
        float* const O = (float*)wrCalloc(static_cast<size_t>(width * height), sizeof(float));

        // row major (interleaved) to col major (non interleaved;clustered)
        float* imageData = reinterpret_cast<float*>(image.data);

        float* const redChannel = I;
        float* const greenChannel = I + width * height;
        float* const blueChannel = I + 2 * width * height;
        int colMajorPos = 0, rowMajorPos = 0;

        for (int row = 0; row < height; ++row)
        {
            for (int col = 0; col < width; ++col)
            {
                colMajorPos = col * height + row;
                rowMajorPos = row * channels * width + col * channels;

                blueChannel[colMajorPos] = imageData[rowMajorPos];
                greenChannel[colMajorPos] = imageData[rowMajorPos + 1];
                redChannel[colMajorPos] = imageData[rowMajorPos + 2];
            }
        }

        // calc fhog in col major
        gradMag(I, M, O, height, width, channels, true);

        if (fhogChannelsToCopy == 27)
            fhog(M, O, H, height, width, binSize, orientations, -1, 0.2f, false);
        else
            fhog(M, O, H, height, width, binSize, orientations, -1, 0.2f);

        // only copy the amount of the channels the user wants
        // or the amount that fits into the output array
        int channelsToCopy = std::min(fhogChannelsToCopy, OUT::numberOfChannels());

        for (int c = 0; c < channelsToCopy; ++c)
        {
            cv::Mat_<PRIMITIVE_TYPE> m(heightBin, widthBin);
            cvFeatures->channels[c] = m;
        }

        PRIMITIVE_TYPE* cdata = 0;
        //col major to row major with separate channels
        for (int c = 0; c < channelsToCopy; ++c)
        {
            float* Hc = H + widthBin * heightBin * c;
            cdata = reinterpret_cast<PRIMITIVE_TYPE*>(cvFeatures->channels[c].data);

            for (int row = 0; row < heightBin; ++row)
                for (int col = 0; col < widthBin; ++col)

                    cdata[row * widthBin + col] = Hc[row + heightBin * col];
        }

        wrFree(M);
        wrFree(O);
        wrFree(I);
        wrFree(H);
    }

    template<typename PRIMITIVE_TYPE, class OUT>
    void cvFhogT(const cv::Mat& img, std::shared_ptr<OUT>& cvFeatures, int binSize, int fhogChannelsToCopy = 31)
    {
        const int orientations = 9;
        // ensure array is continuous
        const cv::Mat& image = (img.isContinuous() ? img : img.clone());

        int channels = image.channels();
        int computeChannels = 32;
        int width = image.cols;
        int height = image.rows;
        int widthBin = width / binSize;
        int heightBin = height / binSize;

        CV_Assert(channels == 1 || channels == 3);

        float* const H = (float*)wrCalloc(static_cast<size_t>(widthBin * heightBin * computeChannels), sizeof(float));
        float* const M = (float*)wrCalloc(static_cast<size_t>(width * height), sizeof(float));
        float* const O = (float*)wrCalloc(static_cast<size_t>(width * height), sizeof(float));

        float* I = NULL;

        if (channels == 1)
            I = reinterpret_cast<float*>(image.data);
        else
        {
            I = (float*)wrCalloc(static_cast<size_t>(width * height * channels), sizeof(float));
            float* imageData = reinterpret_cast<float*>(image.data);
            float* redChannel = I;
            float* greenChannel = I + width * height;
            float* blueChannel = I + 2 * width * height;

            for (int i = 0; i < height * width; ++i)
            {
                blueChannel[i] = imageData[i * 3];
                greenChannel[i] = imageData[i * 3 + 1];
                redChannel[i] = imageData[i * 3 + 2];
            }
        }

        // calc fhog in col major - switch width and height
        gradMag(I, M, O, width, height, channels, true);

        if (fhogChannelsToCopy == 27)
            fhog(M, O, H, width, height, binSize, orientations, -1, 0.2f, false);
        else
            fhog(M, O, H, width, height, binSize, orientations, -1, 0.2f);

        // only copy the amount of the channels the user wants
        // or the amount that fits into the output array
        int channelsToCopy = std::min(fhogChannelsToCopy, OUT::numberOfChannels());

        // init channels
        for (int c = 0; c < channelsToCopy; ++c)
        {
            cv::Mat_<PRIMITIVE_TYPE> m(heightBin, widthBin);
            cvFeatures->channels[c] = m;
        }

        PRIMITIVE_TYPE* cdata = 0;
        // implicit transpose on every channel due to col-major to row-major matrix
        for (int c = 0; c < channelsToCopy; ++c)
        {
            float* Hc = H + widthBin * heightBin * c;
            cdata = reinterpret_cast<PRIMITIVE_TYPE*>(cvFeatures->channels[c].data);

            for (int i = 0; i < heightBin * widthBin; ++i)
                cdata[i] = Hc[i];
        }

        wrFree(M);
        wrFree(O);

        if (channels != 1)
            wrFree(I);

        wrFree(H);
    }
}
#endif
