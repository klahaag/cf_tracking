/*
//  License Agreement (3-clause BSD License)
//  Copyright (c) 2015, Klaus Haag, all rights reserved.
//  Third party copyrights and patents are property of their respective owners.
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
//  * Neither the names of the copyright holders nor the names of the contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
//  This software is provided by the copyright holders and contributors "as is" and
//  any express or implied warranties, including, but not limited to, the implied
//  warranties of merchantability and fitness for a particular purpose are disclaimed.
//  In no event shall copyright holders or contributors be liable for any direct,
//  indirect, incidental, special, exemplary, or consequential damages
//  (including, but not limited to, procurement of substitute goods or services;
//  loss of use, data, or profits; or business interruption) however caused
//  and on any theory of liability, whether in contract, strict liability,
//  or tort (including negligence or otherwise) arising in any way out of
//  the use of this software, even if advised of the possibility of such damage.
*/

#include "math_helper.hpp"
#include <opencv2/imgproc/imgproc.hpp>

namespace cf_tracking
{
    int mod(int dividend, int divisor)
    {
        // http://stackoverflow.com/questions/12276675/modulus-with-negative-numbers-in-c
        return ((dividend % divisor) + divisor) % divisor;
    }

    void dftCcs(const cv::Mat& input, cv::Mat& out, int flags)
    {
        cv::dft(input, out, flags);
    }

    void dftNoCcs(const cv::Mat& input, cv::Mat& out, int flags)
    {
        flags = flags | cv::DFT_COMPLEX_OUTPUT;
        cv::dft(input, out, flags);
    }

    // use bi-linear interpolation on zoom, area otherwise
    // similar to mexResize.cpp of DSST
    // http://www.cvl.isy.liu.se/en/research/objrec/visualtracking/scalvistrack/index.html
    void depResize(const cv::Mat& source, cv::Mat& dst, const cv::Size& dsize)
    {
        int interpolationType = cv::INTER_AREA;

        if (dsize.width > source.cols
            || dsize.height > source.rows)
            interpolationType = cv::INTER_LINEAR;

        cv::resize(source, dst, dsize, 0, 0, interpolationType);
    }
}
