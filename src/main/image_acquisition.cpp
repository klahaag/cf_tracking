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

#include "image_acquisition.hpp"

ImageAcquisition::ImageAcquisition()
{
}

ImageAcquisition& ImageAcquisition::operator>>(CV_OUT cv::Mat& image)
{
    if (_paras.isMock)
        _mockCap >> image;
    else
        _cvCap >> image;

    return *this;
}

void ImageAcquisition::release()
{
    if (!_paras.isMock)
        _cvCap.release();
}

bool ImageAcquisition::isOpened()
{
    if (_paras.isMock)
        return _mockCap.isOpened();
    else
        return _cvCap.isOpened();
}

void ImageAcquisition::set(int key, int value)
{
    if (!_paras.isMock)
        _cvCap.set(key, value);
}

void ImageAcquisition::open(ImgAcqParas paras)
{
    _paras = paras;

    if (_paras.isMock)
    {
        _mockCap.open();
    }
    else
    {
        if (_paras.sequencePath.empty())
            _cvCap.open(_paras.device);
        else
        {
            std::string sequenceExpansion =
                _paras.sequencePath + _paras.expansionStr;

            _cvCap.open(sequenceExpansion);
        }
    }
}

ImageAcquisition::~ImageAcquisition()
{
}

double ImageAcquisition::get(int key)
{
    if (!_paras.isMock)
        return _cvCap.get(key);

    return 0.0;
}

void VideoCaptureMock::release()
{
}

bool VideoCaptureMock::isOpened()
{
    return isOpen;
}

VideoCaptureMock& VideoCaptureMock::operator>>(CV_OUT cv::Mat& image)
{
    image = _staticImage;
    return *this;
}

void VideoCaptureMock::open()
{
    isOpen = true;
}

VideoCaptureMock::~VideoCaptureMock()
{
}

VideoCaptureMock::VideoCaptureMock() : isOpen(false)
{
    _staticImage = cv::Mat(360, 640, CV_8UC3);
    cv::randu(_staticImage, cv::Scalar::all(0), cv::Scalar::all(255));
}
