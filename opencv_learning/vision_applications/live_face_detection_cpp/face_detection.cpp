#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

void detectFace(cv::CascadeClassifier &classifier, cv::Mat &image);

int main()
{
    auto capture = cv::VideoCapture(0);
    if (capture.isOpened())
    {
        cout << "Capture Opened" << endl;
    }
    else
    {
        cerr << "Capture Didn't Open" << endl;
        return -1;
    }

    auto faceClassifier = cv::CascadeClassifier("haarcascade_frontalface_default.xml");

    cout << "Classifier Loaded!" << endl;

    cv::Mat frame;

    while (capture.isOpened())
    {
        if (!capture.read(frame))
        {
            cerr << "End of video, read error" << endl;
            break;
        }

        detectFace(faceClassifier, frame);

        cv::imshow("Face Detection", frame);

        auto k = cv::waitKey(25);

        if (k == 27 || k == 'q')
        {
            break;
        }
    }

    capture.release();
    cv::destroyAllWindows();

    return 0;
}

void detectFace(cv::CascadeClassifier &classifier, cv::Mat &image)
{
    cv::Mat faceImageGray;
    cv::cvtColor(image, faceImageGray, cv::COLOR_BGR2GRAY);
    vector<cv::Rect> faces;
    classifier.detectMultiScale(faceImageGray, faces, 1.2, 5, 0, cv::Size(30, 30));

    for (const auto &face : faces)
    {
        rectangle(image, face, cv::Scalar(255, 255, 255), 5);
    }
}
