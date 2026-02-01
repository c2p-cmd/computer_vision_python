#ifndef STATE_H
#define STATE_H

#include <vector>
#include <opencv2/opencv.hpp>

class State
{
public:
    State()
    {
        this->currentMarker = 1;
        this->marksUpdated = false;
        this->nMarkers = 10;
        this->colors = createColors();
    }

    int getNMarkers() const { return this->nMarkers; }
    int getCurrentMarker() const { return this->currentMarker; }
    bool getMarksUpdated() const { return this->marksUpdated; }
    cv::Vec3b getColorAt(int idx) const { return this->colors[idx]; }

    void setMarkUpdated(bool marksUpdated)
    {
        this->marksUpdated = marksUpdated;
    }

    void setCurrentMarker(int newMarker)
    {
        this->currentMarker = newMarker;
    }

private:
    std::vector<cv::Vec3b> colors;
    int currentMarker;
    bool marksUpdated;
    int nMarkers;

    std::vector<cv::Vec3b> createColors()
    {
        return {
            {0, 0, 0},     // background
            {255, 0, 0},   // blue
            {0, 255, 0},   // green
            {0, 0, 255},   // red
            {255, 255, 0}, // cyan
            {255, 0, 255}, // magenta
            {0, 255, 255}, // yellow
            {128, 0, 255},
            {255, 128, 0},
            {128, 255, 0},
        };
    }
};

#endif // STATE_H
