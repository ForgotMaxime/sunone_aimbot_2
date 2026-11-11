// Golden tests for detector post-processing (backend independent).
// - NMS() must match cv::dnn::NMSBoxes applied per class on a fixed fixture.
// - Overlapping boxes of different classes must both survive (per-class NMS).
// - postProcessYoloDML() decoders are pinned on fixed xyxy / feature-major inputs.
#include "detector/postProcess.h"

#include <opencv2/dnn.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace
{
void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

struct ScoredBox
{
    cv::Rect box;
    float confidence;
    int classId;
};

bool sameDetections(std::vector<Detection> a, std::vector<Detection> b)
{
    const auto order = [](const Detection& lhs, const Detection& rhs)
    {
        if (lhs.classId != rhs.classId)
            return lhs.classId < rhs.classId;
        if (lhs.box.x != rhs.box.x)
            return lhs.box.x < rhs.box.x;
        if (lhs.box.y != rhs.box.y)
            return lhs.box.y < rhs.box.y;
        if (lhs.box.width != rhs.box.width)
            return lhs.box.width < rhs.box.width;
        if (lhs.box.height != rhs.box.height)
            return lhs.box.height < rhs.box.height;
        return lhs.confidence < rhs.confidence;
    };
    std::sort(a.begin(), a.end(), order);
    std::sort(b.begin(), b.end(), order);
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (a[i].classId != b[i].classId || a[i].box != b[i].box ||
            std::fabs(a[i].confidence - b[i].confidence) > 1e-6f)
            return false;
    }
    return true;
}

// Reference: OpenCV per-class NMS over the same inputs.
std::vector<Detection> referencePerClassNms(const std::vector<Detection>& input, float nmsThreshold)
{
    std::vector<Detection> kept;
    std::vector<int> classes;
    for (const auto& det : input)
        if (std::find(classes.begin(), classes.end(), det.classId) == classes.end())
            classes.push_back(det.classId);
    for (int klass : classes)
    {
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<Detection> members;
        for (const auto& det : input)
            if (det.classId == klass)
            {
                boxes.push_back(det.box);
                scores.push_back(det.confidence);
                members.push_back(det);
            }
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, 0.0f, nmsThreshold, indices);
        for (int index : indices)
            kept.push_back(members[static_cast<size_t>(index)]);
    }
    return kept;
}

void testPerClassSurvival()
{
    std::vector<Detection> detections = {
        {cv::Rect(10, 10, 50, 50), 0.9f, 0},
        {cv::Rect(10, 10, 50, 50), 0.8f, 1}, // identical box, other class: must survive
        {cv::Rect(12, 12, 50, 50), 0.7f, 0}, // same class, IoU >> 0.5: suppressed
        {cv::Rect(200, 200, 30, 30), 0.6f, 0},
    };
    NMS(detections, 0.5f, nullptr);
    require(detections.size() == 3, "Per-class NMS must keep cross-class overlaps");
    bool keptOther = false, keptDupe = false;
    for (const auto& det : detections)
    {
        if (det.classId == 1 && det.box == cv::Rect(10, 10, 50, 50))
            keptOther = true;
        if (det.classId == 0 && det.box == cv::Rect(12, 12, 50, 50))
            keptDupe = true;
    }
    require(keptOther, "The other-class box was wrongly suppressed");
    require(!keptDupe, "The same-class duplicate was not suppressed");
}

void testGoldenAgainstOpenCv()
{
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_int_distribution<int> xy(0, 280);
    std::uniform_int_distribution<int> wh(10, 60);
    std::uniform_real_distribution<float> conf(0.05f, 0.99f);
    std::uniform_int_distribution<int> klass(0, 2);

    for (float threshold : {0.3f, 0.45f, 0.6f})
    {
        std::vector<Detection> fixture;
        fixture.reserve(150);
        for (int i = 0; i < 150; ++i)
        {
            const int w = wh(rng), h = wh(rng);
            fixture.push_back({cv::Rect(xy(rng), xy(rng), w, h), conf(rng), klass(rng)});
        }
        // A tight cluster stressing the quadratic path deterministically.
        for (int i = 0; i < 25; ++i)
            fixture.push_back({cv::Rect(100 + i, 100 + (i % 3), 40, 40), 0.95f - 0.01f * i, 0});

        std::vector<Detection> actual = fixture;
        NMS(actual, threshold, nullptr);
        const std::vector<Detection> expected = referencePerClassNms(fixture, threshold);
        require(sameDetections(actual, expected), "NMS diverged from cv::dnn::NMSBoxes golden");
    }
}

void testXyxyDecoderGolden()
{
    // Rows: [x1 y1 x2 y2 confidence classId].
    const float output[] = {
        10.0f, 10.0f, 60.0f, 60.0f, 0.9f, 0.0f,
        12.0f, 12.0f, 62.0f, 62.0f, 0.8f, 0.0f,
        200.0f, 200.0f, 260.0f, 260.0f, 0.7f, 1.0f,
        0.0f, 0.0f, 1.0f, 1.0f, 0.1f, 0.0f, // below conf threshold
    };
    const std::vector<int64_t> shape = {4, 6};
    const std::vector<Detection> detections =
        postProcessYoloDML(output, shape, 1, 0.5f, 0.45f, 100, nullptr);
    require(detections.size() == 2, "xyxy golden must keep 2 boxes");
    require(detections[0].classId == 0 && detections[0].confidence == 0.9f,
        "xyxy golden top box changed");
    require(detections[0].box == cv::Rect(10, 10, 50, 50), "xyxy golden box geometry changed");
}

void testFeatureMajorDecoderGolden()
{
    // [1, 4+numClasses, cols] without objectness, 2 classes, 3 predictions.
    // Prediction 1 and 2 overlap heavily (same class 0); prediction 3 is class 1.
    const float output[] = {
        // cx
        50.0f, 52.0f, 200.0f,
        // cy
        50.0f, 52.0f, 200.0f,
        // w
        40.0f, 40.0f, 30.0f,
        // h
        40.0f, 40.0f, 30.0f,
        // class 0 scores
        0.9f, 0.8f, 0.1f,
        // class 1 scores
        0.05f, 0.05f, 0.85f,
    };
    const std::vector<int64_t> shape = {1, 6, 3};
    const std::vector<Detection> detections =
        postProcessYoloDML(output, shape, 2, 0.5f, 0.45f, 100, nullptr);
    require(detections.size() == 2, "feature-major golden must keep 2 boxes");
    require(detections[0].classId == 0 && detections[1].classId == 1,
        "feature-major golden class assignment changed");
    require(detections[0].confidence > detections[1].confidence,
        "feature-major golden ordering changed");
}
} // namespace

int main()
{
    try
    {
        testPerClassSurvival();
        testGoldenAgainstOpenCv();
        testXyxyDecoderGolden();
        testFeatureMajorDecoderGolden();
        std::cout << "postprocess golden tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
