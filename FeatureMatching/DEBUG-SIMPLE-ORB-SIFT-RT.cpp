#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include <string>
#include <iostream>
#define NAME(x) x, #x
#define MAT_TYPE CV_32FC1
#define DATA_TYPE float
#define DEBUG
//#define Orb

//std::string file_name1 = "./feature_little1.jpg", file_name2 = "./feature_little2.jpg";
std::string file_name1 = "./feature_much1.jpg", file_name2 = "./feature_much2.jpg";
#ifndef DEBUG
DATA_TYPE ratio = 0.7;
#else
int ratio = 7;
#endif
std::vector<cv::DMatch> good_matches;
std::vector<std::vector<cv::DMatch>> knn_match;
cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
cv::Mat img1, img2, img, Final_match;
std::vector<cv::KeyPoint> keypoint1, keypoint2;

void Show(cv::Mat& image_x, const char* name) {
    cv::imshow(name, image_x);
#ifndef DEBUG
    cv::waitKey(0);
#endif
}

cv::Mat ReadImg(std::string path) {
    img = cv::imread(path);
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    //img.convertTo(img, MAT_TYPE, 1.0 / 255); // SIFT貌似不支持浮点类型
    return img;
}

std::vector<cv::KeyPoint> GetKeyPoint(cv::Mat &origin_img) {
#ifdef Orb
    cv::Ptr<cv::ORB> dtor = cv::ORB::create(3000);
    dtor->setFastThreshold(0);
#else
    cv::Ptr<cv::SiftFeatureDetector> dtor = cv::SiftFeatureDetector::create();
#endif
    std::vector<cv::KeyPoint> res;
    dtor->detect(origin_img, res);
    cv::Mat feature_output;
    cv::drawKeypoints(origin_img, res, feature_output);
    Show(NAME(feature_output));
    return res;
}

cv::Mat GetDescriptor(cv::Mat& img, std::vector<cv::KeyPoint>& keypoint) {
    cv::Mat descriptor;
#ifdef Orb
    cv::Ptr<cv::ORB> des_ex = cv::ORB::create(3000);
    des_ex->setFastThreshold(0);
#else
    cv::Ptr<cv::SiftDescriptorExtractor> des_ex = cv::SiftDescriptorExtractor::create();
#endif
    des_ex->compute(img, keypoint, descriptor);
    return descriptor;
}

void Feature_Get() {
    img1 = ReadImg(file_name1);
    keypoint1 = GetKeyPoint(img1);
    img2 = ReadImg(file_name2);
    keypoint2 = GetKeyPoint(img2);

    auto descrip1 = GetDescriptor(img1, keypoint1);
    auto descrip2 = GetDescriptor(img2, keypoint2);
    
#ifndef Orb
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2);
#else
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
#endif
    knn_match.clear();
    matcher->knnMatch(descrip1, descrip2, knn_match, 2);
}

static void Feature_Match(int, void*) {
    good_matches.clear();
    for (auto k_m : knn_match)
#ifndef DEBUG
        if (k_m[0].distance < k_m[1].distance * ratio)
#else
        if (k_m[0].distance < k_m[1].distance * ((float)ratio) / 100.0f)
#endif
            good_matches.push_back(k_m[0]);

    Final_match;
    cv::drawMatches(img1, keypoint1, img2, keypoint2, good_matches, Final_match, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    Show(NAME(Final_match));
}

int main()
{
    Feature_Get();
#ifndef DEBUG
    Feature_Matching_SIFT();
#else
    cv::namedWindow("Final_match", cv::WINDOW_AUTOSIZE);
    char s1[50];
    memset(s1, 0, sizeof s1);
    sprintf_s(s1, "ratio : %d %%", ratio);
    cv::createTrackbar(s1, "Final_match", &ratio, 200, Feature_Match);
    Feature_Match(0, 0);
    cv::waitKey(0);
#endif // !DEBUG

    return 0;
}