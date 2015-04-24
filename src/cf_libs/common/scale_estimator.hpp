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

/* This class represents the 1D correlation filter proposed in  [1]. It is used to estimate the
scale of a target.

It is implemented closely to the Matlab implementation by the original authors:
http://www.cvl.isy.liu.se/en/research/objrec/visualtracking/scalvistrack/index.html
However, some implementation details differ and some difference in performance
has to be expected.

Every complex matrix is as default in CCS packed form:
see : https://software.intel.com/en-us/node/504243
and http://docs.opencv.org/modules/core/doc/operations_on_arrays.html

References:
[1] M. Danelljan, et al.,
"Accurate Scale Estimation for Robust Visual Tracking,"
in Proc. BMVC, 2014.

*/

#ifndef SCALE_ESTIMATOR_HPP_
#define SCALE_ESTIMATOR_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/core/traits.hpp>
#include <algorithm>

#include "mat_consts.hpp"
#include "cv_ext.hpp"
#include "feature_channels.hpp"
#include "gradientMex.hpp"
#include "math_helper.hpp"

namespace cf_tracking
{
    template<typename T>
    struct ScaleEstimatorParas
    {
        int scaleCellSize = 4;
        T scaleModelMaxArea = static_cast<T>(512);
        T scaleStep = static_cast<T>(1.02);
        int numberOfScales = 33;
        T scaleSigmaFactor = static_cast<T>(1.0 / 4.0);

        T lambda = static_cast<T>(0.01);
        T learningRate = static_cast<T>(0.025);

        // testing
        bool useFhogTranspose = false;
        int resizeType = cv::INTER_LINEAR;
        bool debugOutput = true;
        bool originalVersion = false;
    };

    template<typename T>
    class ScaleEstimator
    {
    public:
        typedef FhogFeatureChannels<T> FFC;
        typedef cv::Size_<T> Size;
        typedef cv::Point_<T> Point;
        typedef mat_consts::constants<T> consts;

        ScaleEstimator(ScaleEstimatorParas<T> paras) :
            _frameIdx(0),
            _isInitialized(false),
            _SCALE_CELL_SIZE(paras.scaleCellSize),
            _SCALE_MODEL_MAX_AREA(paras.scaleModelMaxArea),
            _SCALE_STEP(paras.scaleStep),
            _N_SCALES(paras.numberOfScales),
            _SCALE_SIGMA_FACTOR(paras.scaleSigmaFactor),
            _LAMBDA(paras.lambda),
            _LEARNING_RATE(paras.learningRate),
            _TYPE(cv::DataType<T>::type),
            _RESIZE_TYPE(paras.resizeType),
            _DEBUG_OUTPUT(paras.debugOutput),
            _ORIGINAL_VERSION(paras.originalVersion)
        {
            // init dft
            cv::Mat initDft = (cv::Mat_<T>(1, 1) << 1);
            dft(initDft, initDft);

            if (_DEBUG_OUTPUT)
            {
                if (CV_MAJOR_VERSION < 3)
                {
                    std::cout << "ScaleEstimator: Using OpenCV Version: " << CV_MAJOR_VERSION << std::endl;
                    std::cout << "For more speed use 3.0 or higher!" << std::endl;
                }
            }

            if (paras.useFhogTranspose)
                fhogToCvCol = &piotr::fhogToCvColT;
            else
                fhogToCvCol = &piotr::fhogToCol;
        }

        bool reinit(const cv::Mat& image, const Point& pos,
            const Size& targetSize, const T& currentScaleFactor)
        {
            _targetSize = targetSize;
            // scale filter output target
            T scaleSigma = static_cast<T>(sqrt(_N_SCALES) * _SCALE_SIGMA_FACTOR);
            cv::Mat colScales = numberToColVector<T>(_N_SCALES);
            T scaleHalf = static_cast<T>(ceil(_N_SCALES / 2.0));

            cv::Mat ss = colScales - scaleHalf;
            cv::Mat ys;
            exp(-0.5 * ss.mul(ss) / (scaleSigma * scaleSigma), ys);

            cv::Mat ysf;
            // always use CCS here; regular COMPLEX_OUTPUT is bugged
            cv::dft(ys, ysf, cv::DFT_ROWS);

            // scale filter cos window
            if (_N_SCALES % 2 == 0)
            {
                _scaleWindow = hanningWindow<T>(_N_SCALES + 1);
                _scaleWindow = _scaleWindow.rowRange(1, _scaleWindow.rows);
            }
            else
            {
                _scaleWindow = hanningWindow<T>(_N_SCALES);
            }

            ss = scaleHalf - colScales;
            _scaleFactors = pow<T, T>(_SCALE_STEP, ss);
            _scaleModelFactor = sqrt(_SCALE_MODEL_MAX_AREA / targetSize.area());
            _scaleModelSz = sizeFloor(targetSize *  _scaleModelFactor);

            // expand ysf to have the number of rows of scale samples
            int ysfRow = static_cast<int>(floor(_scaleModelSz.width / _SCALE_CELL_SIZE)
                * floor(_scaleModelSz.height / _SCALE_CELL_SIZE) * FFC::numberOfChannels());

            _ysf = repeat(ysf, ysfRow, 1);

            cv::Mat sfNum, sfDen;

            if (getScaleTrainingData(image, pos,
                currentScaleFactor, sfNum, sfDen) == false)
                return false;

            _sfNumerator = sfNum;
            _sfDenominator = sfDen;

            _isInitialized = true;
            ++_frameIdx;
            return true;
        }

        virtual ~ScaleEstimator(){}

        bool detectScale(const cv::Mat& image, const Point& pos,
            T& currentScaleFactor) const
        {
            cv::Mat xs;
            if (getScaleFeatures(image, pos, xs, currentScaleFactor) == false)
                return false;

            cv::Mat xsf;
            dft(xs, xsf, cv::DFT_ROWS);

            mulSpectrums(_sfNumerator, xsf, xsf, cv::DFT_ROWS);
            reduce(xsf, xsf, 0, cv::REDUCE_SUM, -1);

            cv::Mat sfDenLambda;
            sfDenLambda = addRealToSpectrum<T>(_LAMBDA, _sfDenominator, cv::DFT_ROWS);

            cv::Mat responseSf;
            divSpectrums(xsf, sfDenLambda, responseSf, cv::DFT_ROWS, false);

            cv::Mat scaleResponse;
            idft(responseSf, scaleResponse, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE | cv::DFT_ROWS);

            cv::Point recoveredScale;
            double maxScaleResponse;
            minMaxLoc(scaleResponse, 0, &maxScaleResponse, 0, &recoveredScale);

            currentScaleFactor *= _scaleFactors.at<T>(recoveredScale);

            currentScaleFactor = std::max(currentScaleFactor, _MIN_SCALE_FACTOR);
            currentScaleFactor = std::min(currentScaleFactor, _MAX_SCALE_FACTOR);
            return true;
        }

        bool updateScale(const cv::Mat& image, const Point& pos,
            const T& currentScaleFactor)
        {
            ++_frameIdx;
            cv::Mat sfNum, sfDen;

            if (getScaleTrainingData(image, pos, currentScaleFactor,
                sfNum, sfDen) == false)
                return false;

            // both summands are in CCS packaged format; thus adding is OK
            _sfDenominator = (1 - _LEARNING_RATE) * _sfDenominator + _LEARNING_RATE * sfDen;
            _sfNumerator = (1 - _LEARNING_RATE) * _sfNumerator + _LEARNING_RATE * sfNum;
            return true;
        }

    private:
        bool getScaleTrainingData(const cv::Mat& image,
            const Point& pos,
            const T& currentScaleFactor,
            cv::Mat& sfNum, cv::Mat& sfDen) const
        {
            cv::Mat xs;
            if (getScaleFeatures(image, pos, xs, currentScaleFactor) == false)
                return false;

            cv::Mat xsf;
            dft(xs, xsf, cv::DFT_ROWS);
            mulSpectrums(_ysf, xsf, sfNum, cv::DFT_ROWS, true);
            cv::Mat mulTemp;
            mulSpectrums(xsf, xsf, mulTemp, cv::DFT_ROWS, true);
            reduce(mulTemp, sfDen, 0, cv::REDUCE_SUM, -1);
            return true;
        }

        bool getScaleFeatures(const cv::Mat& image, const Point& pos,
            cv::Mat& features, T scale) const
        {
            int colElems = _ysf.rows;
            features = cv::Mat::zeros(colElems, _N_SCALES, _TYPE);
            cv::Mat patch;
            cv::Mat patchResized;
            cv::Mat patchResizedFloat;
            cv::Mat firstPatch;
            T cosFactor = -1;

            // do not extract features for first and last scale,
            // since the scaleWindow will always multiply these with 0;
            // extract first required sub window separately; smaller scales are extracted
            // from this patch to avoid multiple border replicates on out of image patches
            int idxScale = 1;
            T patchScale = scale * _scaleFactors.at<T>(0, idxScale);
            Size firstPatchSize = sizeFloor(_targetSize * patchScale);
            Point posInFirstPatch(0, 0);
            cosFactor = _scaleWindow.at<T>(idxScale, 0);

            if (getSubWindow(image, firstPatch, firstPatchSize, pos, &posInFirstPatch) == false)
                return false;

            if (_ORIGINAL_VERSION)
                depResize(firstPatch, patchResized, _scaleModelSz);
            else
                cv::resize(firstPatch, patchResized, _scaleModelSz, 0, 0, _RESIZE_TYPE);

            patchResized.convertTo(patchResizedFloat, CV_32FC(3));
            fhogToCvCol(patchResizedFloat, features, _SCALE_CELL_SIZE, idxScale, cosFactor);

            for (idxScale = 2; idxScale < _N_SCALES - 1; ++idxScale)
            {
                T patchScale = scale *_scaleFactors.at<T>(0, idxScale);
                Size patchSize = sizeFloor(_targetSize * patchScale);
                cosFactor = _scaleWindow.at<T>(idxScale, 0);

                if (getSubWindow(firstPatch, patch, patchSize, posInFirstPatch) == false)
                    return false;

                if (_ORIGINAL_VERSION)
                    depResize(patch, patchResized, _scaleModelSz);
                else
                    cv::resize(patch, patchResized, _scaleModelSz, 0, 0, _RESIZE_TYPE);

                patchResized.convertTo(patchResizedFloat, CV_32FC(3));
                fhogToCvCol(patchResizedFloat, features, _SCALE_CELL_SIZE, idxScale, cosFactor);
            }

            return true;
        }

    private:
        typedef void(*fhogToCvRowPtr)
            (const cv::Mat& img, cv::Mat& cvFeatures, int binSize, int rowIdx, T cosFactor);
        fhogToCvRowPtr fhogToCvCol = 0;

        cv::Mat _scaleWindow;
        T _scaleModelFactor = 0;
        cv::Mat _sfNumerator;
        cv::Mat _sfDenominator;
        cv::Mat _scaleFactors;
        Size _scaleModelSz;
        Size _targetSize;
        cv::Mat _ysf;
        int _frameIdx;
        bool _isInitialized;

        const int _TYPE;
        const int _SCALE_CELL_SIZE;
        const T _SCALE_MODEL_MAX_AREA;
        const T _SCALE_STEP;
        const int _N_SCALES;
        const T _SCALE_SIGMA_FACTOR;
        const T _LAMBDA;
        const T _LEARNING_RATE;
        const int _RESIZE_TYPE;
        // it should be possible to find more reasonable values for min/max scale; application dependent
        T _MIN_SCALE_FACTOR = static_cast<T>(0.01);
        T _MAX_SCALE_FACTOR = static_cast<T>(40);

        const bool _DEBUG_OUTPUT;
        const bool _ORIGINAL_VERSION;
    };
}

#endif
