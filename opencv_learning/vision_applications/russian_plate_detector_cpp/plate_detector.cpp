#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

cv::Mat detectPlate(cv::CascadeClassifier &classifier, cv::Mat &image);

int main()
{
    auto image = cv::imread("car_plate.jpg");
    cout << "Image loaded with size: " << image.size() << endl;
    auto plateCascade = cv::CascadeClassifier("haarcascade_russian_plate_number.xml");
    cout << "Classifier Loaded" << endl;

    cv::Mat resultImage = detectPlate(plateCascade, image);

    while (true)
    {
        cv::imshow("Result Image", resultImage);

        auto k = cv::waitKey(25);

        if (k == 'q')
        {
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}

cv::Mat detectPlate(cv::CascadeClassifier &classifier, cv::Mat &image)
{
    auto plateImage = image.clone();
    vector<cv::Rect> rects;

    classifier.detectMultiScale(plateImage, rects, 1.3, 3);

    for (const auto &rect : rects)
    {
        cv::Mat roi = plateImage(rect);

        cv::Mat blurredRoi;
        cv::medianBlur(roi, blurredRoi, 7);

        blurredRoi.copyTo(plateImage(rect));
    }

    return plateImage;
}
