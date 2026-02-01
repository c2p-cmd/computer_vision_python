#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "state.h"

using namespace std;
using namespace cv;

Mat road, roadCopy, markerImage, segments;
State state;

void mouseCallback(int event, int x, int y, int flags, void *userdata);

int main()
{

    road = imread("road_image.jpg");
    if (road.empty())
    {
        cerr << "Image not found\n";
        return -1;
    }

    cout << "Image Loaded!" << endl;

    roadCopy = road.clone();
    markerImage = Mat::zeros(road.size(), CV_32S);
    segments = Mat::zeros(road.size(), CV_8UC3);

    namedWindow("Road Image");
    setMouseCallback("Road Image", mouseCallback);

    while (true)
    {
        imshow("Road Image", roadCopy);
        imshow("Watershed Segments", segments);

        int key = waitKey(1);

        if (key == 27 || key == 'q')
        {
            break;
        }
        else if (key == 'c')
        {
            roadCopy = road.clone();
            markerImage = Mat::zeros(road.size(), CV_32S);
            segments = Mat::zeros(road.size(), CV_8UC3);
        }
        else if (key >= '0' && key <= '9')
        {
            state.setCurrentMarker(key - '0');
        }

        if (state.getMarksUpdated())
        {
            Mat markerCopy = markerImage.clone();
            watershed(road, markerCopy);

            segments = Mat::zeros(road.size(), CV_8UC3);

            for (int i = 0; i < state.getNMarkers(); i++)
            {
                segments.setTo(state.getColorAt(i), markerCopy == i);
            }

            state.setMarkUpdated(false);
        }
    }

    destroyAllWindows();
    return 0;
}

void mouseCallback(int event, int x, int y, int flags, void *userdata)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        circle(markerImage, Point(x, y), 10, state.getCurrentMarker(), -1);
        circle(roadCopy, Point(x, y), 10, state.getColorAt(state.getCurrentMarker()), -1);
        state.setMarkUpdated(true);
    }
}
