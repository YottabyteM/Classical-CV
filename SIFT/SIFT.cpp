#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <set>

#define MAT_TYPE CV_64FC1
#define DATA_TYPE double
#define INT_MAX 2147483647
#define MAX_OCTAVES 8
#define EXP_SIGMA 0.5
#define GAUSS_SIZE_RATIO 3
#define HIST_NUMBER 36
#define IMG_IGNORE 2
#define MAX_STEP 5
#define RADIUS_NEIGHBOR 4.5
#define ORI_SIG_FAC 1.5
#define NAME(x) x, #x
#define SIFT_RATIO 0.8
#define DESCR_WIDTH 4
#define DESCR_HIST_BIN 8
#define DESCR_SCL_FACTOR 3.0
#define DESCR_MAG_THR 0.2
#define KNN_FACTOR 0.8
#define THRESH_FACTOR 4
typedef std::pair<int, int> PII;
typedef std::vector<std::vector<cv::Mat>> vector2d;
typedef std::vector<cv::KeyPoint> vectorKey;
std::string path1 = "./1.png", path2 = "2.png";
//std::string path1 = "./feature_much1.jpg", path2 = "feature_much2.jpg";
cv::Size grid_size = cv::Size(20, 20), grid_match_size = cv::Size(0, 0);
cv::Mat MotionStatistic, img1, img2;
std::vector<cv::DMatch> match_after_filtered;
std::vector<PII> match_mid;
std::vector<bool> filter;
int GRID_SIZE = 20;
const double ScaleRatio[5] = { 1.0, 0.5, 1.0 / sqrt(2.0), sqrt(2.0), 2.0 };
std::vector<int> Grid2Grid;
std::vector<int> kp_num;

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

void Show(cv::Mat& image_x, const char* name) {
    cv::imshow(name, image_x);
    cv::waitKey(0);
}

void ShowHist(std::vector<double> arr)
{
	cv::Mat hist = cv::Mat::zeros(cv::Size(512, 512), CV_8UC3);

	int _max = 0;
	for (int i = 0; i < arr.size(); i++)
	{
		if (arr[i] > _max)
		{
			_max = arr[i];
		}
	}

	for (int i = 1; i < hist.rows; i++)
	{
		int current_value = (int)(double(arr[(int)((double)i / (double)hist.cols * 256)]) / double(_max) * hist.rows);
		cv::line(hist, cv::Point(i, hist.rows - 1), cv::Point(i, hist.rows - 1 - current_value), cv::Scalar(255, 0, 255));
	}
	Show(NAME(hist));
}

class MySift {
private:
	int nfeatures;
	int nOctaveLayers;
	double contrastThreshold;
	double edgeThreshold;
	double sigma;
	bool double_size;
public:
	MySift(int nfeature = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04, double edgeThreshold = 10, double sigma = 1.6, bool double_size = true) :
	nfeatures(nfeature), nOctaveLayers(nOctaveLayers), contrastThreshold(contrastThreshold), edgeThreshold(edgeThreshold), sigma(sigma), double_size(double_size) {};

	int Get_Num_Octaves(cv::Mat& img);
	void Get_Gaussian_Pyramid(cv::Mat&, vector2d&);
	void Get_Dog_Pyramid(vector2d&, vector2d&);
	double Get_Main_Direction(cv::Mat&, cv::Point&, double, std::vector<double>&);
	bool GetPrecisePos(vector2d&, cv::KeyPoint&, int, int&, int&, int&);
	void Find_ExTra(vector2d&, vector2d&, vectorKey&);
	void detect(cv::Mat&, vector2d&, vector2d&, vectorKey&);
	void GetDescriptor(vector2d&, vectorKey&, cv::Mat&);
};

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

	for (int xi = -1; xi <= 1; xi++)
		for (int yi = -1; yi <= 1; yi++)
		{
			int idx_xx = idx_x + xi, idx_yy = idx_y + yi;
			if (idx_xx < 0 || idx_xx >= Size_w || idx_yy < 0 || idx_yy >= Size_h) continue;

			ans[xi + 1 + 3 * (yi + 1)] = idx_xx + idx_yy * Size_w;
		}

	return ans;
}

void gms_filter(cv::Mat& img1, std::vector<cv::KeyPoint>& kp1, cv::Mat& img2, std::vector<cv::KeyPoint>& kp2, std::vector<cv::DMatch>& raw_match, int ratio) {
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

			thresh = (double)THRESH_FACTOR * sqrt(thresh / nump);

			if (score < thresh) Grid2Grid[i] = -1;
		}
	}

	for (int i = 0; i < match_mid.size(); i++) {
		auto mt = match_mid[i];
		cv::Point2f lp, rp;
		lp.x = kp1[mt.first].pt.x, lp.y = kp1[mt.first].pt.y;
		rp.x = kp2[mt.second].pt.x, rp.y = kp2[mt.second].pt.y;
		if (Grid2Grid[GetGridIdx(lp, 0)] == GetGridIdx(rp, 1) && Grid2Grid[GetGridIdx(lp, 0)] != -1)
			filter[i] = true;
	}
}

int main()
{
	img1 = cv::imread(path1), img2 = cv::imread(path2);
	MySift SiftDetect1, SiftDetect2;
	std::vector<std::vector<cv::Mat>> gauss_pyr1, gauss_dog_pyr1, gauss_pyr2, gauss_dog_pyr2;
	cv::Mat descriptor1, descriptor2;
	vectorKey keypoints1, keypoints2;
	std::vector<std::vector<cv::DMatch>> knn_match;
	std::vector<cv::DMatch> good_match; 
	double st1 = (double)cv::getTickCount();
	SiftDetect1.detect(img1, gauss_pyr1, gauss_dog_pyr1, keypoints1);
	SiftDetect1.GetDescriptor(gauss_pyr1, keypoints1, descriptor1);
	double ed1 = ((double)cv::getTickCount() - st1) / cv::getTickFrequency();
	std::cout << "Time is " << ed1 << std::endl;
	SiftDetect2.detect(img2, gauss_pyr2, gauss_dog_pyr2, keypoints2);
	SiftDetect2.GetDescriptor(gauss_pyr2, keypoints2, descriptor2);
	cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2);
	knn_match.clear();
	matcher->knnMatch(descriptor1, descriptor2, knn_match, 2);
	for (auto k_m : knn_match)
		if (k_m[0].distance < k_m[1].distance * KNN_FACTOR)
			good_match.push_back(k_m[0]);
	cv::Mat final_match;
	//match_after_filtered.clear();
	//gms_filter(img1, keypoints1, img2, keypoints2, good_match, 0);
	//for (size_t i = 0; i < good_match.size(); i++)
	//	if (filter[i])
	//		match_after_filtered.push_back(good_match[i]);
	//cv::drawMatches(img1, keypoints1, img2, keypoints2, match_after_filtered, final_match, cv::Scalar::all(-1),
	//	cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::drawMatches(img1, keypoints1, img2, keypoints2, good_match, final_match, cv::Scalar::all(-1),
		cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::Mat feature_detect;
	cv::drawKeypoints(img2, keypoints2, feature_detect);
	for (auto &kp : keypoints2) {
		double scale = kp.size / (1 << (kp.octave & 255)) * DESCR_WIDTH;
		cv::circle(feature_detect, kp.pt, scale, cv::Scalar(255, 0, 255));
		cv::arrowedLine(feature_detect, kp.pt, cv::Point2f(kp.pt.x + scale * cos(kp.angle), kp.pt.y + scale * sin(kp.angle)), cv::Scalar(255, 0, 255));
	}
	Show(NAME(feature_detect));
	cv::Mat feature_detect2;
	cv::drawKeypoints(img1, keypoints1, feature_detect2);
	for (auto& kp : keypoints1) {
		double scale = kp.size / (1 << (kp.octave & 255)) * DESCR_WIDTH;
		cv::arrowedLine(feature_detect2, kp.pt, cv::Point2f(kp.pt.x + scale * cos(kp.angle), kp.pt.y + scale * sin(kp.angle)), cv::Scalar(255, 0, 255));
		cv::circle(feature_detect2, kp.pt, scale, cv::Scalar(255, 0, 255));
	}
	Show(NAME(feature_detect2));
	Show(NAME(final_match));
}

int MySift::Get_Num_Octaves(cv::Mat& img) {
	auto min_scale = std::min(img.cols, img.rows);
	int res = round(std::log((float)min_scale) / std::log(2.0)) - 3;
	if (double_size) res += 1;
	return std::min(res, MAX_OCTAVES);
}

void MySift::Get_Gaussian_Pyramid(cv::Mat& img, vector2d& res_pyramid) {
	cv::Mat temp_img;
	cv::cvtColor(img, temp_img, cv::COLOR_RGB2GRAY);
	int nOctaves = Get_Num_Octaves(img);
	temp_img.convertTo(temp_img, MAT_TYPE, 1.0 / 255.0, 0);
	double sigma_diff = 0.0;
	if (double_size) {
		cv::Mat tempimg;
		cv::resize(temp_img, tempimg, cv::Size(2 * temp_img.cols, 2 * temp_img.rows), 0.0, 0.0, cv::INTER_LINEAR);
		sigma_diff = sqrt(sigma * sigma - 4.0 * EXP_SIGMA * EXP_SIGMA);
		int kernel_width = 2 * round(GAUSS_SIZE_RATIO * sigma_diff) + 1;
		cv::Size kernel_size(kernel_width, kernel_width);
		GaussianBlur(tempimg, temp_img, kernel_size, sigma_diff, sigma_diff);
	}
	else {
		sigma_diff = sqrt(sigma * sigma - EXP_SIGMA * EXP_SIGMA);
		int kernel_size = 2 * round(sigma_diff * GAUSS_SIZE_RATIO) + 1;
		cv::GaussianBlur(temp_img, temp_img, cv::Size(kernel_size, kernel_size), sigma_diff, sigma_diff);
	}
	std::vector<double> sigma_vec({sigma});
	double k = std::pow(2.0, 1.0 / (double)nOctaveLayers);
	for (int i = 1; i < nOctaveLayers + 3; i++) {
		double pre_sig = pow(k, (double)(i - 1)) * sigma;
		double cur_sig = k * pre_sig;
		sigma_vec.push_back(sqrt(cur_sig * cur_sig - pre_sig * pre_sig));
	}

	res_pyramid.resize(nOctaves);

	for (int i = 0; i < nOctaves; i ++)
		res_pyramid[i].resize(nOctaveLayers + 3);
	res_pyramid[0][0] = temp_img;
	for (int i = 0; i < nOctaves; i ++ )
		for (int j = 0; j < nOctaveLayers + 3; j++)
			if (i || j)
				if (!j)
					cv::resize(res_pyramid[i - 1][3], 
						res_pyramid[i][0],
						cv::Size(res_pyramid[i - 1][3].cols / 2, res_pyramid[i - 1][3].rows / 2),
						0, 0, cv::INTER_LINEAR);
				else
				{
					int kernel_size = 2 * round(GAUSS_SIZE_RATIO * sigma_vec[j]) + 1;
					cv::GaussianBlur(res_pyramid[i][j - 1], res_pyramid[i][j],
						cv::Size(kernel_size, kernel_size),
						sigma_vec[j], sigma_vec[j]);
				}
}

void MySift::Get_Dog_Pyramid(vector2d& src, vector2d& dst) {
	dst.resize(src.size());
	for (int i = 0; i < src.size(); i++)
		for (int j = 0; j < nOctaveLayers + 2; j++)
			dst[i].push_back(src[i][j + 1] - src[i][j]);
}

double MySift::Get_Main_Direction(cv::Mat& img, cv::Point& pt, double scale, std::vector<double>&hist) {
	int r = round(RADIUS_NEIGHBOR * scale);
	double sigma_cur = ORI_SIG_FAC * scale, exp_scale = -1.0 / (2 * sigma_cur * sigma_cur);
	std::vector<double> dx, dy, Magnitude, Angle, WGaussian, temp_hist(hist.size(), 0);
	for (int i = -r; i < r; i++) {
		int y = pt.y + i;
		if (y <= 0 || y >= img.rows - 1) continue;
		for (int j = -r; j < r; j++) {
			int x = pt.x + j;
			if (x <= 0 || x >= img.cols - 1) continue;
			dx.push_back(img.at<DATA_TYPE>(y, x + 1) - img.at<DATA_TYPE>(y, x - 1));
			dy.push_back(img.at<DATA_TYPE>(y + 1, x) - img.at<DATA_TYPE>(y - 1, x));
			WGaussian.push_back(exp((double)(i * i + j * j) * exp_scale));
			Angle.push_back(cv::fastAtan2(*dy.rbegin(), *dx.rbegin()));
			Magnitude.push_back(sqrt((*dy.rbegin()) * (*dy.rbegin()) + (*dx.rbegin()) * (*dx.rbegin())));
		}
	}

	for (int i = 0; i < Magnitude.size(); i++) {
		int bin = round(hist.size() / 360.0 * Angle[i]);
		bin = (bin + hist.size()) % hist.size();
		temp_hist[bin] += Magnitude[i] * WGaussian[i];
	}
	
	double max_val = -1.0;
	for (int i = 0; i < hist.size(); i++) {
		int n = hist.size();
		int x1 = (i - 2 + n) % n, x2 = (i - 1 + n) % n, x3 = (i + n) % n, x4 = (i + 1 + n) % n, x5 = (i + 2 + n) % n;
		hist[i] = (temp_hist[x1] + temp_hist[x5]) * (1.0 / 16.0) + (temp_hist[x2] + temp_hist[x4]) * (4.0 / 16.0) + temp_hist[x3] * (6.0 / 16.0);
		max_val = std::max(max_val, hist[i]);
	}
	//ShowHist(hist);
	return max_val;
}

void MySift::Find_ExTra(vector2d& dog_Pyr, vector2d& gauss_pyr, vectorKey& res_key) {
	double threshold = (double)(contrastThreshold / (double)nOctaveLayers);
	int hist_num = HIST_NUMBER;
	std::vector<double> hist(hist_num, 0.0);
	res_key.clear();
	cv::KeyPoint tmp_kpt;

	for (int i = 0; i < dog_Pyr.size(); i++)
		for (int j = 1; j <= nOctaveLayers; j++) {
			//std::cout << i << " " << j << std::endl;
			for (int x = IMG_IGNORE; x < dog_Pyr[i][j].rows - IMG_IGNORE; x++)
				for (int y = IMG_IGNORE; y < dog_Pyr[i][j].cols - IMG_IGNORE; y++)
				{
					auto val = dog_Pyr[i][j].at<DATA_TYPE>(x, y);
					if (fabs(val) < threshold) continue;
					int flag = val > 0;
					for (int k = -1; k <= 1 && flag != -2; k++)
						for (int p = -1; p <= 1 && flag != -2; p++)
							for (int q = -1; q <= 1 && flag != -2; q++)
								if (k || p || q)
								{
									int t = flag ^ (val >= dog_Pyr[i][j + k].at<DATA_TYPE>(x + p, q + y));
									if (t) flag = -2;
								}
					if (flag == -2) continue;
					int cur_octave = i, cur_layer = j, cur_pos_x = x, cur_pos_y = y;
					if (!GetPrecisePos(dog_Pyr, tmp_kpt, cur_octave, cur_layer, cur_pos_x, cur_pos_y)) continue;
					cv::Point ptt = cv::Point(tmp_kpt.pt.x, tmp_kpt.pt.y);
					double max_hist = Get_Main_Direction(gauss_pyr[cur_octave][cur_layer], ptt, tmp_kpt.size / (double)(1 << cur_octave), hist);
					for (size_t i = 0; i < hist.size(); i++) {
						int left = (i - 1 + hist.size()) % hist.size(), right = (i + 1 + hist.size()) % hist.size();
						if (hist[i] > hist[left] && hist[i] > hist[right] && hist[i] > max_hist * SIFT_RATIO)
						{
							double bin = i + 0.5 * (hist[right] - hist[left]) / (hist[left] + hist[right] - 2.0 * hist[i]);
							if (bin < 0) bin += hist.size();
							if (bin >= hist.size()) bin -= hist.size();
							tmp_kpt.angle = (360.0 / hist.size()) * bin;
							res_key.push_back(tmp_kpt);
						}
					}
				}
		}
}

bool MySift::GetPrecisePos(vector2d& dog_pyr, cv::KeyPoint& kpt, int octave, int &layer, int &i, int &j) {
	int adjust_times = 0;
	double xi = 0.0, xj = 0.0, xl = 0.0;
	while (adjust_times < MAX_STEP) {
		adjust_times++;
		auto img_now = dog_pyr[octave][layer], img_prev = dog_pyr[octave][layer - 1], img_nxt = dog_pyr[octave][layer + 1];
		double dx = (img_now.at<DATA_TYPE>(i, j + 1) - img_now.at<DATA_TYPE>(i, j - 1)) * (1.0 / 2.0),
			dy = (img_now.at<DATA_TYPE>(i + 1, j) - img_now.at<DATA_TYPE>(i - 1, j)) * (1.0 / 2.0),
			dl = (img_nxt.at<DATA_TYPE>(i, j) - img_prev.at<DATA_TYPE>(i, j)) * (1.0 / 2.0),
			dxx = img_now.at<DATA_TYPE>(i, j + 1) + img_now.at<DATA_TYPE>(i, j - 1) - 2 * img_now.at<DATA_TYPE>(i, j),
			dyy = img_now.at<DATA_TYPE>(i + 1, j) + img_now.at<DATA_TYPE>(i - 1, j) - 2 * img_now.at<DATA_TYPE>(i, j),
			dll = img_nxt.at<DATA_TYPE>(i, j) + img_prev.at<DATA_TYPE>(i, j) - 2 * img_now.at<DATA_TYPE>(i, j),
			dxy = (img_now.at<DATA_TYPE>(i + 1, j + 1) + img_now.at<DATA_TYPE>(i - 1, j - 1) - img_now.at<DATA_TYPE>(i + 1, j - 1) - img_now.at<DATA_TYPE>(i - 1, j + 1)) * (1.0 / 4.0),
			dxl = (img_nxt.at<DATA_TYPE>(i, j + 1) + img_prev.at<DATA_TYPE>(i, j - 1) - img_nxt.at<DATA_TYPE>(i, j - 1) - img_prev.at<DATA_TYPE>(i, j + 1)) * (1.0 / 4.0),
			dyl = (img_nxt.at<DATA_TYPE>(i + 1, j) + img_prev.at<DATA_TYPE>(i - 1, j) - img_nxt.at<DATA_TYPE>(i - 1, j) - img_prev.at<DATA_TYPE>(i + 1, j)) * (1.0 / 4.0);
		cv::Matx33d H(dxx, dxy, dxl, 
					  dxy, dyy, dyl, 
					  dxl, dyl, dll);
		cv::Vec3d dD(dx, dy, dl);
		cv::Vec3d temp_res = H.solve(dD, cv::DECOMP_SVD);
		xj = -temp_res[0], xi = -temp_res[1], xl = -temp_res[2];
		if (fabs(xj) < 0.5 && fabs(xi) < 0.5 && fabs(xl) < 0.5) {
			double val = dD.dot(temp_res);
			double contr = val * 0.5 + img_now.at<DATA_TYPE>(i, j);
			if (fabs(contr) < contrastThreshold / (double)nOctaveLayers) return false;
			double trace = dxx + dyy, det = dxx * dyy - dxy * dxy;
			if (det < 0 || trace * trace * edgeThreshold >= det * (edgeThreshold + 1) * (edgeThreshold + 1)) return false;
			kpt.pt.x = ((double)j + xj) * (double)(1 << octave), kpt.pt.y = ((double)i + xi) * (double)(1 << octave), kpt.octave = octave + (layer << 8),
				kpt.size = sigma * pow(2.0, (layer + xl) / (double)nOctaveLayers) * (double)(1 << octave), kpt.response = fabs(contr);
			return true;
		}
		if (fabs(xi) > (double)(INT_MAX / 3) ||
			fabs(xj) > (double)(INT_MAX / 3) ||
			fabs(xl) > (double)(INT_MAX / 3)
			) 
			return false;
		layer += xl, i += xi, j += xj;
		if (layer < 1 || layer > nOctaveLayers || i < IMG_IGNORE || i > img_now.rows - IMG_IGNORE || j < IMG_IGNORE || j > img_now.cols - IMG_IGNORE) return false;
	}

	return false;
}

void MySift::detect(cv::Mat& img, vector2d& gauss_pyr, vector2d& dog_pyr, vectorKey& keypoints) {
	Get_Gaussian_Pyramid(img, gauss_pyr);
	Get_Dog_Pyramid(gauss_pyr, dog_pyr);
	//for (auto pp : dog_pyr)
	//	for (auto p : pp)
	//		Show(NAME(p));
	Find_ExTra(dog_pyr, gauss_pyr, keypoints);

	std::cout << keypoints.size() << std::endl;
	if (nfeatures && nfeatures < keypoints.size()) {
		std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
			return a.response > b.response;
			});
		keypoints.erase(keypoints.begin() + nfeatures, keypoints.end());
	}
}

void MySift::GetDescriptor(vector2d& src, vectorKey& kp, cv::Mat& descriptor) {
	int d = DESCR_WIDTH, n = DESCR_HIST_BIN;
	descriptor.create(kp.size(), d * d * n, MAT_TYPE);
	for (size_t i = 0; i < kp.size(); i++) {
		int octaves = kp[i].octave & 255, layer = (kp[i].octave >> 8) & 255;
		auto gauss_img = src[octaves][layer];
		cv::Point2d pt(round(kp[i].pt.x / (1 << octaves)), round(kp[i].pt.y / (1 << octaves)));
		double scale = kp[i].size / (1 << octaves), main_angle = kp[i].angle;
		double hist_width = DESCR_SCL_FACTOR * scale;
		double bin_rad = n / 360.0, exp_scale = -1.0 / (d * d * 0.5);
		int img_rows = gauss_img.rows, img_cols = gauss_img.cols;
		int radius = std::min(round(hist_width * (d + 1) * sqrt(2.0) * 0.5), sqrt((double)(img_rows * img_rows + img_cols * img_cols)));
		int len = (2 * radius + 1) * (2 * radius + 1), hist_len = (d + 2) * (d + 2) * (n + 2);
		std::vector<double> dx, dy, Magnitude, Angle, WGaussian, Rbin, Cbin, hist(hist_len, 0);
		auto cos_main = std::cos(-main_angle * (acos(-1) / 180.0)) / hist_width, sin_main = std::cos(-main_angle * (acos(-1) / 180.0)) / hist_width;
		for (int x = -radius; x <= radius; x ++ )
			for (int y = -radius; y <= radius; y++)
			{
				double c_rot = y * cos_main - x * sin_main, r_rot = y * sin_main + x * cos_main,
					   r_bin = r_rot + d / 2.0 - 0.5, c_bin = c_rot + d / 2.0 - 0.5;
				int r = pt.y + x, c = pt.x + y;
				if (r_bin > -1 && r_bin< d && c_bin > -1 && c_bin < d &&
					r>0 && r < img_rows - 1 && c > 0 && c < img_cols - 1)
				{
					auto ddx = gauss_img.at<DATA_TYPE>(r, c + 1) - gauss_img.at<DATA_TYPE>(r, c - 1),
					     ddy = gauss_img.at<DATA_TYPE>(r + 1, c) - gauss_img.at<DATA_TYPE>(r - 1, c);
					dx.push_back(ddx);
					dy.push_back(ddy);
					Angle.push_back(cv::fastAtan2(ddy, ddx));
					Rbin.push_back(r_bin);
					Cbin.push_back(c_bin);
					Magnitude.push_back(sqrt(ddx * ddx + ddy * ddy));
					WGaussian.push_back(exp((c_rot * c_rot + r_rot * r_rot) * exp_scale));
				}
			}
		
		for (size_t ii = 0; ii < dx.size(); ii++) {
			auto rbin = Rbin[ii], cbin = Cbin[ii], curbin = (Angle[ii] - main_angle) * bin_rad,
				magnitude = Magnitude[ii] * WGaussian[ii];
			int r0 = floor(rbin), c0 = floor(cbin), cur0 = floor(curbin);
			rbin -= r0, cbin -= c0, curbin -= cur0;
			if (cur0 < 0) cur0 += n;
			if (cur0 >= n) cur0 -= n;

			double v_1 = magnitude * rbin,
				   v_0 = magnitude - v_1,
				   v_11 = v_1 * cbin,
				   v_10 = v_1 - v_11,
				   v_01 = v_0 * cbin,
				   v_00 = v_0 - v_01,
				   v_111 = v_11 * curbin,
				   v_110 = v_11 - v_111,
				   v_101 = v_10 * curbin,
				   v_100 = v_10 - v_101,
				   v_011 = v_01 * curbin,
				   v_010 = v_01 - v_011,
                   v_001 = v_00 * curbin,
				   v_000 = v_00 - v_001;
			int idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + cur0;
			hist[idx] += v_000;
			hist[idx + 1] += v_001;
			hist[idx + n + 2] += v_010;
			hist[idx + n + 3] += v_011;
			hist[idx + (d + 2) * (n + 2)] += v_100;
			hist[idx + (d + 2) * (n + 2) + 1] += v_101;
			hist[idx + (d + 3) * (n + 2)] += v_110;
			hist[idx + (d + 3) * (n + 2) + 1] += v_111;
		}

		for (int ii = 0; ii < d; ii++) 
			for (int jj = 0; jj < d; jj++) {
				int idx = ((ii + 1) * (d + 2) + (jj + 1)) * (n + 2);
				hist[idx] += hist[idx + n];
				hist[idx + 1] += hist[idx + n + 1];
				for (int kk = 0; kk < n; kk++)
					descriptor.at<DATA_TYPE>(i, (ii * d + jj) * n + kk) = hist[idx + kk];
			}
		/*double sum_sq = 0.0;
		for (size_t j = 0; j < d * d * n; j++)
			sum_sq += descriptor.at<DATA_TYPE>(i, j) * descriptor.at<DATA_TYPE>(i, j);
		sum_sq = sqrt(sum_sq);
		double thr = sum_sq * DESCR_MAG_THR;
		sum_sq = 0.0;
		for (size_t j = 0; j < d * d * n; j++) {
			descriptor.at<DATA_TYPE>(i, j) = std::min(descriptor.at<DATA_TYPE>(i, j), DESCR_MAG_THR);
			sum_sq += descriptor.at<DATA_TYPE>(i, j) * descriptor.at<DATA_TYPE>(i, j);
		}
		sum_sq = 512.0 / std::max(sqrt(sum_sq), 1.19209290E-07);
		for (size_t j = 0; j < d * d * n; j++)
			descriptor.at<DATA_TYPE>(i, j) = std::min(std::max(0.0, descriptor.at<DATA_TYPE>(i, j) * sum_sq), 255.0);
			*/
		if (double_size)
		{
			kp[i].pt.x /= 2.0;
			kp[i].pt.y /= 2.0;
		}
		int length_of_des = d * d * n;
		double sum_sq = 0.0;
		for (int j = 0; j < length_of_des; ++j)
		{
			sum_sq = sum_sq + descriptor.at<DATA_TYPE>(i, j) * descriptor.at<DATA_TYPE>(i, j);
		}
		sum_sq = sqrt(sum_sq);
		for (int j = 0; j < length_of_des; ++j)
		{
			descriptor.at<DATA_TYPE>(i, j) = descriptor.at<DATA_TYPE>(i, j) / sum_sq;
		}

		for (int j = 0; j < length_of_des; ++j)
		{
			descriptor.at<DATA_TYPE>(i, j) = std::min(descriptor.at<DATA_TYPE>(i, j), DESCR_MAG_THR);
		}

		sum_sq = 0.0;
		for (int j = 0; j < length_of_des; ++j)
		{
			sum_sq = sum_sq + descriptor.at<DATA_TYPE>(i, j) * descriptor.at<DATA_TYPE>(i, j);
		}
		sum_sq = sqrt(sum_sq);
		for (int j = 0; j < length_of_des; ++j)
		{
			descriptor.at<DATA_TYPE>(i, j) = descriptor.at<DATA_TYPE>(i, j) / sum_sq;
		}
	}
	descriptor.convertTo(descriptor, CV_32FC1, 1.0 / 255.0, 0);
}