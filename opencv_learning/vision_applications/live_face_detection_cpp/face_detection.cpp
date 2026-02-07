#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;

void detectFace(cv::CascadeClassifier &faceClassifier, cv::CascadeClassifier &eyeClassifier, cv::Mat &image, cv::Mat &imageGray);

void drawFPS(double const fps, cv::Mat &image);

int main()
{
    auto capture = cv::VideoCapture(0);
    if (!capture.isOpened())
    {
        cerr << "Capture Didn't Open" << endl;
        return -1;
    }
    cout << "Capture Opened: " << capture.getBackendName() << endl;

    cv::CascadeClassifier faceClassifier, eyesClassifier;
    if (!faceClassifier.load("haarcascade_frontalface_default.xml"))
    {
        cerr << "Face Classifier not loaded" << endl;
        return -1;
    }
    if (!eyesClassifier.load("haarcascade_eye.xml"))
    {
        cerr << "Eyes Classifier not loaded" << endl;
        return -1;
    }

    cout << "Face and eyes Classifier Loaded!" << endl;

    cv::Mat frame, frameGray;
    using clock = chrono::high_resolution_clock;
    auto lastTime = clock::now();
    double fps = 0.0;

    const int flags = cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO;
    cv::namedWindow("Face Detection", flags);
    cv::namedWindow("Face Detection Grayscale", flags);

    while (capture.isOpened())
    {
        if (!capture.read(frame))
        {
            cerr << "End of video, read error" << endl;
            break;
        }

        const auto currentTime = clock::now();
        auto elapsed = chrono::duration<double>(currentTime - lastTime).count();
        lastTime = currentTime;
        fps = 1.0 / elapsed;

        // detect face
        detectFace(faceClassifier, eyesClassifier, frame, frameGray);

        // draw fps
        drawFPS(fps, frame);

        cv::imshow("Face Detection", frame);
        cv::imshow("Face Detection Grayscale", frameGray);

        auto k = cv::waitKey(1);

        if (k == 27 || k == 'q')
        {
            break;
        }
    }

    capture.release();
    cv::destroyAllWindows();

    return 0;
}

void drawFPS(double const fps, cv::Mat &image)
{
    const cv::String message = cv::format("FPS: %.1f", fps);
    cv::putText(image, message, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
}

void detectFace(cv::CascadeClassifier &faceClassifier, cv::CascadeClassifier &eyeClassifier, cv::Mat &image, cv::Mat &imageGray)
{
    cv::Mat faceImageGray;
    cv::cvtColor(image, faceImageGray, cv::COLOR_BGR2GRAY);
    const double scaleDown = 0.5;
    cv::resize(faceImageGray, faceImageGray, cv::Size(), scaleDown, scaleDown, cv::INTER_LINEAR);
    cv::equalizeHist(faceImageGray, faceImageGray);

    vector<cv::Rect> faces;
    faceClassifier.detectMultiScale(faceImageGray, faces, 1.2, 5, 0, cv::Size(30, 30));

    for (const auto &face : faces)
    {
        // Scale face coordinates back to original image size
        cv::Rect scaledFace = cv::Rect(
            face.x / scaleDown, face.y / scaleDown,
            face.width / scaleDown, face.height / scaleDown);

        // draw rect on face
        rectangle(image, scaledFace, cv::Scalar(255, 255, 255), 2);

        // draw circle on eyes
        cv::Mat faceROI = faceImageGray(cv::Rect(face.x, face.y, face.width, face.height / 2));
        vector<cv::Rect> eyes;

        eyeClassifier.detectMultiScale(faceROI, eyes, 1.1, 5, 0, cv::Size(15, 15));

        for (const auto &eye : eyes)
        {
            const cv::Point center = cv::Point(
                (face.x + eye.x + eye.width / 2) / scaleDown,
                (face.y + eye.y + eye.height / 2) / scaleDown);
            const int radius = (eye.width / 2) / scaleDown;
            cv::circle(image, center, radius, cv::Scalar(200, 100, 100), 2, cv::LINE_AA);
        }
    }
    faceImageGray.copyTo(imageGray);
}
