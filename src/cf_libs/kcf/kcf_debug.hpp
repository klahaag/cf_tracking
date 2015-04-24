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

#ifndef KCF_DEBUG_HPP_
#define KCF_DEBUG_HPP_

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <fstream>

#include "tracker_debug.hpp"

namespace cf_tracking
{
    template<typename T>
    class KcfDebug : public TrackerDebug
    {
    public:
        KcfDebug() :
            _maxResponse(0),
            _psrClamped(0)
        {}

        virtual ~KcfDebug()
        {
            if (_outputFile.is_open())
                _outputFile.close();
        }

        virtual void init(std::string outputFilePath)
        {
            namedWindow(_SUB_WINDOW_TITLE, cv::WINDOW_NORMAL);
            namedWindow(_RESPONSE_TITLE, cv::WINDOW_NORMAL);
            _outputFile.open(outputFilePath.c_str());
        }

        virtual void printOnImage(cv::Mat& image)
        {
            _ss.str("");
            _ss.clear();
            _ss << "Max Response: " << _maxResponse;
            putText(image, _ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));

            _ss.str("");
            _ss.clear();
            _ss << "PSR Clamped: " << _psrClamped;
            putText(image, _ss.str(), cv::Point(20, 80), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));
        }

        virtual void printConsoleOutput()
        {
        }

        virtual void printToFile()
        {
            _outputFile << _maxResponse << "," << _psrClamped << std::endl;
        }

        void showPatch(const cv::Mat& patchResized)
        {
            imshow(_SUB_WINDOW_TITLE, patchResized);
        }

        void setPsr(T psrClamped)
        {
            _psrClamped = psrClamped;
            std::cout << "PSR: " << psrClamped << std::endl;
        }

        void showResponse(const cv::Mat& response, T maxResponse)
        {
            cv::Mat responseOutput = response.clone();
            _maxResponse = maxResponse;
            imshow(_RESPONSE_TITLE, responseOutput);
        }

    private:
        const std::string _SUB_WINDOW_TITLE = "Sub Window";
        const std::string _RESPONSE_TITLE = "Response";
        T _maxResponse;
        T _psrClamped;
        std::stringstream _ss;
        std::ofstream _outputFile;
    };
}

#endif
