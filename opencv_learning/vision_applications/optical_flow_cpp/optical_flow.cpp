#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;

int main()
{
    const double SCALE = 0.5; // Downsample factor (0.5 = half size, 0.25 = quarter size)

    auto capture = cv::VideoCapture(0);
    if (!capture.isOpened())
    {
        cerr << "Cannot open capture" << endl;
        return -1;
    }

    cv::Mat prevFrame, prevFrameGray, hsvMask;
    if (!capture.read(prevFrame))
    {
        cerr << "Camera not capturing" << endl;
        return -1;
    }
    if (prevFrame.empty())
    {
        cerr << "Camera not capturing" << endl;
        return -1;
    }

    // Downsample for faster processing
    cv::resize(prevFrame, prevFrame, cv::Size(), SCALE, SCALE);

    cv::cvtColor(prevFrame, prevFrameGray, cv::COLOR_BGR2GRAY);

    hsvMask = cv::Mat(prevFrame.size(), CV_8UC3);
    hsvMask.setTo(cv::Scalar(0, 255, 0));

    using clock = chrono::high_resolution_clock;
    auto lastTime = clock::now();
    double fps = 0;

    while (capture.isOpened())
    {
        cv::Mat nextFrame, nextFrameGray, flow, bgrFrame;
        if (!capture.read(nextFrame))
        {
            cerr << "Camera not capturing" << endl;
            return -1;
        }

        // downsample for faster processing
        cv::resize(nextFrame, nextFrame, cv::Size(), SCALE, SCALE);

        // convert to grayscale
        cv::cvtColor(nextFrame, nextFrameGray, cv::COLOR_BGR2GRAY);

        // dense optical flow
        cv::calcOpticalFlowFarneback(prevFrameGray, nextFrameGray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        cv::Mat magnitude, angle;
        vector<cv::Mat> flowCart;
        cv::split(flow, flowCart);
        cv::cartToPolar(flowCart[0], flowCart[1], magnitude, angle, true);

        vector<cv::Mat> hsvMaskChannels;
        cv::split(hsvMask, hsvMaskChannels);

        angle.convertTo(hsvMaskChannels[0], CV_8U, 0.5);
        hsvMaskChannels[1].setTo(255);
        cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);
        magnitude.convertTo(hsvMaskChannels[2], CV_8U);

        cv::merge(hsvMaskChannels, hsvMask);
        cv::cvtColor(hsvMask, bgrFrame, cv::COLOR_HSV2BGR);

        const auto currentTime = clock::now();
        const auto elapsed = chrono::duration<double>(currentTime - lastTime).count();
        fps = 1 / elapsed;
        lastTime = currentTime;

        const cv::String message = cv::format("FPS: %.1f", fps);
        cv::putText(bgrFrame, message, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 100, 200), 2);

        cv::imshow("Dense Optical Flow", bgrFrame);

        auto k = cv::waitKey(1);
        if (k == 27 || k == 'q')
        {
            break;
        }

        prevFrameGray = nextFrameGray.clone();
    }

    capture.release();
    cv::destroyAllWindows();

    return 0;
}
