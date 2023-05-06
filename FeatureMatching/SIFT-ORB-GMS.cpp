#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include <string>
#include <iostream>

typedef std::pair<int, int> PII;
#define NAME(x) x, #x
#define MAT_TYPE CV_32FC1
#define DATA_TYPE float
//#define DEBUG
//#define Orb
#ifdef Orb
int THRESH_FACTOR = 40;
#else
int THRESH_FACTOR = 30;
#endif
const int RotationType[8][9] = {
    1,2,3,
    4,5,6,
    7,8,9,

    4,1,2,
    7,5,3,
    8,9,6,

    7,4,1,
    8,5,2,
    9,6,3,

    8,7,4,
    9,5,1,
    6,3,2,

    9,8,7,
    6,5,4,
    3,2,1,

    6,9,8,
    3,5,7,
    2,1,4,

    3,6,9,
    2,5,8,
    1,4,7,

    2,3,6,
    1,5,9,
    4,7,8
};

//std::string file_name1 = "./feature_little1.jpg", file_name2 = "./feature_little2.jpg";
std::string file_name1 = "./feature_much1.jpg", file_name2 = "./feature_much2.jpg";
#ifdef DEBUG
int ratio = 0;
#else
DATA_TYPE ratio = 0.9;
#endif
std::vector<cv::DMatch> good_matches;
std::vector<std::vector<cv::DMatch>> knn_match;
cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
cv::Size grid_size = cv::Size(20, 20), grid_match_size = cv::Size(0, 0);
cv::Mat img1, img2, img, Final_match, MotionStatistic;
std::vector<cv::KeyPoint> keypoint1, keypoint2;
std::vector<cv::DMatch> match_after_filtered;
std::vector<PII> match_mid;
std::vector<int> kp_num;
std::vector<bool> filter;
const double ScaleRatio[5] = {1.0, 0.5, 1.0 / sqrt(2.0), sqrt(2.0), 2.0};
std::vector<int> Grid2Grid;
int GRID_SIZE = 20;

void Show(cv::Mat& image_x, const char* name) {
    cv::imshow(name, image_x);
#ifndef DEBUG
    cv::waitKey(0);
#endif
}

cv::Mat ReadImg(std::string path) {
    img = cv::imread(path);
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    //img.convertTo(img, MAT_TYPE, 1.0 / 255); 
    return img;
}

std::vector<cv::KeyPoint> GetKeyPoint(cv::Mat& origin_img) {
#ifdef Orb
    cv::Ptr<cv::ORB> dtor = cv::ORB::create(10000);
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
    cv::Ptr<cv::ORB> des_ex = cv::ORB::create(10000);
    des_ex->setFastThreshold(0);
#else
    cv::Ptr<cv::SiftDescriptorExtractor> des_ex = cv::SiftDescriptorExtractor::create();
#endif
    des_ex->compute(img, keypoint, descriptor);
    return descriptor;
}

void Feature_Match() {
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
    good_matches.clear();
    for (auto k_m : knn_match)
        if (k_m[0].distance < k_m[1].distance * ratio)
            good_matches.push_back(k_m[0]);
}

void DMatch2Match(std::vector<cv::DMatch>& raw_match, std::vector<PII>& to_match) {
    to_match.resize(raw_match.size());
    for (size_t i = 0; i < raw_match.size(); i++)
        to_match[i] = std::make_pair(raw_match[i].queryIdx, raw_match[i].trainIdx);
}

int GetGridIdx(cv::Point2f& pt, int t) {
    if (!t) {
        int idx_x = pt.x * grid_size.width / img1.cols, idx_y = pt.y * grid_size.height / img1.rows;
        return idx_y * grid_size.width + idx_x;
    }
    else
    {
        int idx_x = pt.x * grid_match_size.width / img2.cols, idx_y = pt.y * grid_match_size.height / img2.rows;
        return idx_y * grid_match_size.width + idx_x;
    }
}

std::vector<int> GetNeighbir(int idx, int t) {
    std::vector<int> ans(9, -1);
    int idx_x, idx_y, Size_w, Size_h;
    if (!t) {
        idx_x = idx % grid_size.width;
        idx_y = idx / grid_size.width;
        Size_w = grid_size.width;
        Size_h = grid_size.height;
    }
    else
    {
        idx_x = idx % grid_match_size.width;
        idx_y = idx / grid_match_size.width;
        Size_w = grid_match_size.width;
        Size_h = grid_match_size.height;
    }

    for (int xi = -1; xi <= 1; xi ++ )
        for (int yi = -1; yi <= 1; yi++)
        {
            int idx_xx = idx_x + xi, idx_yy = idx_y + yi;
            if (idx_xx < 0 || idx_xx >= Size_w || idx_yy < 0 || idx_yy >= Size_h) continue;

            ans[xi + 1 + 3 * (yi + 1)] = idx_xx + idx_yy * Size_w;
        }

    return ans;
}

void gms_filter(cv::Mat &img1, std::vector<cv::KeyPoint> &kp1, cv::Mat &img2, std::vector<cv::KeyPoint> &kp2, std::vector<cv::DMatch> &raw_match, int ratio) {
    DMatch2Match(raw_match, match_mid);
    filter.assign(raw_match.size(), false);
    grid_size = cv::Size(GRID_SIZE, GRID_SIZE);
    grid_match_size.height = ScaleRatio[ratio] * grid_size.height;
    grid_match_size.width = ScaleRatio[ratio] * grid_size.width;
    MotionStatistic = cv::Mat::zeros(grid_size.height * grid_size.width, grid_match_size.width * grid_match_size.height, CV_32SC1);
    kp_num.assign(grid_size.height * grid_size.width, 0);
    Grid2Grid.assign(grid_size.width * grid_size.height, -1);
    for (auto mt : match_mid) {
        cv::Point2f lp, rp;
        lp.x = kp1[mt.first].pt.x, lp.y = kp1[mt.first].pt.y;
        rp.x = kp2[mt.second].pt.x, rp.y = kp2[mt.second].pt.y;
        MotionStatistic.at<int>(GetGridIdx(lp, 0), GetGridIdx(rp, 1))++;
        kp_num[GetGridIdx(lp, 0)]++;
    }

    for (int i = 0; i < grid_size.width * grid_size.height; i++) {

        int max_val = 0;
        for (int j = 0; j < grid_match_size.width * grid_match_size.height; j++) {
            if (MotionStatistic.at<int>(i, j) > max_val) {
                max_val = MotionStatistic.at<int>(i, j);
                Grid2Grid[i] = j;
            }
        }
        if (!max_val) continue;

        int score = 0, nump = 0;
        double thresh = 0.0;
        auto nb_left = GetNeighbir(i, 0), nb_right = GetNeighbir(Grid2Grid[i], 1);

        for (int j = 0; j < 8; j++) {
            for (int i = 0; i < 9; i++) {
                const int* Rt = RotationType[j];
                int lp = nb_left[i], rp = nb_right[Rt[i] - 1];
                if (lp == -1 || rp == -1) continue;

                score += MotionStatistic.at<int>(lp, rp);
                thresh += kp_num[lp];
                nump++;
            }

            thresh = (double)THRESH_FACTOR / 10.0 * sqrt(thresh / nump);

            if (score < thresh) Grid2Grid[i] = -1;
        }
    }

    for (int i = 0; i < match_mid.size(); i ++ ) {
        auto mt = match_mid[i];
        cv::Point2f lp, rp;
        lp.x = kp1[mt.first].pt.x, lp.y = kp1[mt.first].pt.y;
        rp.x = kp2[mt.second].pt.x, rp.y = kp2[mt.second].pt.y;
        if (Grid2Grid[GetGridIdx(lp, 0)] == GetGridIdx(rp, 1) && Grid2Grid[GetGridIdx(lp, 0)] != -1)
            filter[i] = true;
    }
}

static void Feature_Filter(int, void*) {
    match_after_filtered.clear();
    gms_filter(img1, keypoint1, img2, keypoint2, good_matches, 0);
    for (size_t i = 0; i < good_matches.size(); i++)
        if (filter[i])
            match_after_filtered.push_back(good_matches[i]);
    cv::drawMatches(img1, keypoint1, img2, keypoint2, match_after_filtered, Final_match, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    Show(NAME(Final_match));
}

int main()
{
    Feature_Match();
    Feature_Filter(0, 0);
#ifdef DEBUG
    cv::namedWindow("Final_match", cv::WINDOW_AUTOSIZE);
    char s1[50], s2[50];
    memset(s1, 0, sizeof s1);
    memset(s2, 0, sizeof s2);
    sprintf_s(s1, "THRESH_FACTOR : %d", THRESH_FACTOR);
    sprintf_s(s2, "GRID_SIZE : %d", GRID_SIZE);
    cv::createTrackbar(s1, "Final_match", &THRESH_FACTOR, 100, Feature_Filter);
    cv::createTrackbar(s2, "Final_match", &GRID_SIZE, 30, Feature_Filter);
    Feature_Filter(0, 0);
    //Feature_Filter(0, 0);
    cv::waitKey(0);
#endif // !DEBUG

    return 0;
}