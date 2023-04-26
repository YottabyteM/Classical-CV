#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <set>
#define MAT_TYPE CV_64FC1
#define DATA_TYPE double
#define ORIGIN 0
#define GAUSS 1
#define SOBEL 2
#define NAME(x) x, #x
#define DEBUG
#define NMS_4
//#define SHOWFREQ
typedef std::pair <int, int> PII;

std::string PATH = "./Lenna.jpg", WINDOWNAME = "Canny";
const int N = 1024;
const DATA_TYPE PI = acos(-1);
cv::Mat origin_img;
int gauss_size = 5, gauss_sigma = 4, color[3] = {0, 0, 0}, s = 0, hh, tt;
double sobel_x[3][3] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 }, sobel_y[3][3] = {1, 2, 1, 0, 0, 0, -1, -2, -1}, Theta[N][N];
#ifdef DEBUG
int threshold = 3, ratio = 4;
#else
double threshold = 0.2, ratio = 2.3;
#endif
cv::Mat suppress, threshold_img, hsv, res_th;
std::vector<std::set<PII>> lines;
std::set<PII> l;
std::vector<cv::Mat> channels(3);
PII q[N * N];

struct dimension {
    int tot_x, tot_y;
    int tot_origin_x, tot_origin_y;
};
struct RGBC
{
    int R, G, B;
};
std::vector<dimension> Dim(3);
bool st[N][N];

int rev[N], bit;

void init(cv::Mat &Image_m, int order, int tot_w1 = 0, int tot_w2 = 0)
{
    Dim[order].tot_origin_x = Image_m.rows, Dim[order].tot_origin_y = Image_m.cols;
    if (!tot_w1) {
        bit = 0;
        while ((1 << bit) < Image_m.rows) bit++;
        Dim[order].tot_x = 1 << bit;
        bit = 0;
        while ((1 << bit) < Image_m.cols) bit++;
        Dim[order].tot_y = 1 << bit;
    }
    else
    {
        Dim[order].tot_x = tot_w1;
        Dim[order].tot_y = tot_w2;
    }
}

void Show(cv::Mat& image_x, const char* name) {
    cv::imshow(name, image_x);
#ifndef DEBUG
    cv::waitKey(0);
#endif
}

void ShowColor(double theta[][N], cv::Mat& imag_src)
{
    cv::Mat hsv = cv::Mat(imag_src.size(), CV_64FC3);
    cv::cvtColor(hsv, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    channels[2] = 255;
    channels[1] = 255;
    for (int i = 0; i < channels[0].rows; i++)
        for (int j = 0; j < channels[0].cols; j++) {
            if (imag_src.at<DATA_TYPE>(i, j) > 0.3)
                channels[0].at<uint8_t>(i, j) = (theta[i][j] + PI / 2) / (PI) * 255;
            if (imag_src.at<DATA_TYPE>(i, j) < 0.3)
            {
                channels[0].at<uint8_t>(i, j) = 0;
                channels[1].at<uint8_t>(i, j) = 0;
                channels[2].at<uint8_t>(i, j) = 0;
            }
        }
    cv::merge(channels, hsv);
    cv::cvtColor(hsv, hsv, cv::COLOR_HSV2BGR);
    Show(NAME(hsv));
}

void add_color(int c[], int step)
{
    int rest = 0;
    for (int i = 0; i < 3; i++) {
        c[i] += step + rest;
        if (c[i] <= 255) break;
        else
        {
            rest = 1;
            c[i] %= 255;
        }
    }
}

void ShowConnect(cv::Mat& Image_src)
{
    if (lines.empty()) return;
    hsv.setTo(cv::Scalar(0));
    cv::cvtColor(hsv, hsv, cv::COLOR_BGR2HSV);
    cv::split(hsv, channels);
    color[1] = 0, color[2] = 0, color[0] = 100, s = (255 * 255 * 255 - 100) / (lines.size() + 1);
    if (!s) s = 1;
    for (auto l : lines) {
        add_color(color, s);
        for (auto pos : l) {
            channels[0].at<uint8_t>(pos.first, pos.second) = color[2];
            channels[1].at<uint8_t>(pos.first, pos.second) = color[1];
            channels[2].at<uint8_t>(pos.first, pos.second) = color[0];
        }
    }
    cv::merge(channels, hsv);
    cv::cvtColor(hsv, hsv, cv::COLOR_HSV2BGR);
    Show(NAME(hsv));
}

cv::Mat normalize(cv::Mat& image_m, double H)
{
    cv::Mat res = cv::Mat(image_m.size(), MAT_TYPE);
    DATA_TYPE val_max = image_m.at<DATA_TYPE>(0, 0), val_min = image_m.at<DATA_TYPE>(0, 0);
    for (int i = 0; i < res.rows; i++)
        for (int j = 0; j < res.cols; j++)
            val_max = std::max(val_max, image_m.at<DATA_TYPE>(i, j)), val_min = std::min(val_min, image_m.at<DATA_TYPE>(i, j));
    for (int i = 0; i < res.rows; i++)
        for (int j = 0; j < res.cols; j++)
            res.at<DATA_TYPE>(i, j) = H * (image_m.at<DATA_TYPE>(i, j) - val_min) / (val_max - val_min);
    return res;
}

struct Complex
{
    DATA_TYPE x = 0.0, y = 0.0;
    Complex operator+ (const Complex& t) const
    {
        return { x + t.x, y + t.y };
    }
    Complex operator- (const Complex& t) const
    {
        return { x - t.x, y - t.y };
    }
    Complex operator* (const Complex& t) const
    {
        return { x * t.x - y * t.y, x * t.y + y * t.x };
    }
    Complex operator/ (const int& t) const
    {
        return { x / t, y / t };
    }
}origin_fft[N][N], gauss_fft[N][N], blur1[N][N], blur2[N][N], sobel_fft_x[N][N], sobel_fft_y[N][N], temp[N];

void Mat2Complex(Complex a[][N], cv::Mat& b) {
    for (int i = 0; i < b.rows; i++)
        for (int j = 0; j < b.cols; j++)
            a[i][j].x = 0.0;
    for (int i = 0; i < b.rows; i++)
        for (int j = 0; j < b.cols; j++)
            a[i][j].x = b.at<DATA_TYPE>(i, j);
}

void Complex2Mati(Complex a[][N], cv::Mat& b) {
    for (int i = 0; i < b.rows; i++)
        for (int j = 0; j < b.cols; j++)
            b.at<DATA_TYPE>(i, j) = (DATA_TYPE)(sqrt(a[i][j].x * a[i][j].x + a[i][j].y * a[i][j].y));
}

void Complex2Mat(Complex a[][N], cv::Mat& b) {
    for (int i = 0; i < b.rows; i++)
        for (int j = 0; j < b.cols; j++)
            b.at<DATA_TYPE>(i, j) = a[i][j].x;
}

cv::Mat reverse_img(cv::Mat& Image_src)
{
    cv::Mat res = cv::Mat(Image_src.size(), MAT_TYPE);
    for (int i = 0; i < res.rows; i++)
        for (int j = 0; j < res.cols; j++)
        {
            int center_x = i < (res.rows >> 1) ? (res.rows / 2 - 1) : ((res.rows / 2 - 1) + (res.rows / 4)) * 2;
            int center_y = j < (res.cols >> 1) ? (res.cols / 2 - 1) : ((res.cols / 2 - 1) + (res.cols / 4)) * 2;
            res.at<DATA_TYPE>(i, j) = Image_src.at<DATA_TYPE>(center_x - i, center_y - j);
        }
    return res;
}

void Showfreq(Complex a[][N], const char* name, int order, int H) {
    cv::Mat image_x = cv::Mat(cv::Size(Dim[order].tot_y, Dim[order].tot_x), MAT_TYPE);
    Complex2Mati(a, image_x);
    cv::Mat res = reverse_img(image_x);
    res = normalize(res, H);
    Show(res, name);
}

void fftcp(Complex a[], int inv, int tot)
{
    for (int i = 0; i < tot; i++)
        if (i < rev[i])
            std::swap(a[i], a[rev[i]]);
    for (int mid = 1; mid < tot; mid <<= 1)
    {
        auto w1 = Complex({ cos(PI / mid), inv * sin(PI / mid) });
        for (int i = 0; i < tot; i += (mid << 1))
        {
            auto wk = Complex({ 1, 0 });
            for (int j = 0; j < mid; j++, wk = wk * w1)
            {
                auto x = a[i + j], y = wk * a[i + j + mid];
                a[i + j] = x + y, a[i + j + mid] = x - y;
            }
        }
    }
    if (inv == -1)
        for (int i = 0; i < tot; i++)
            a[i] = a[i] / tot;
}

void fft(cv::Mat &Image_m, Complex a[][N], int order)
{
    for (int i = 0; i < std::max(Dim[order].tot_x, Dim[order].tot_y); i++)
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    Mat2Complex(a, Image_m);
    int tot_x = Dim[order].tot_x, tot_y = Dim[order].tot_y;
    for (int i = 0; i < tot_x; i++) {
        fftcp(a[i], 1, tot_y);
    }
    for (int i = 0; i < tot_y; i++) {
        for (int j = 0; j < tot_x; j++)
            temp[j] = a[j][i];
        fftcp(temp, 1, tot_x);
        for (int j = 0; j < tot_x; j++)
            a[j][i] = temp[j];
    }
}

cv::Mat ifft(Complex a[][N], int order)
{
    for (int i = 0; i < std::max(Dim[order].tot_x, Dim[order].tot_y); i++)
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    int tot_x = Dim[order].tot_x, tot_y = Dim[order].tot_y;
    for (int i = 0; i < tot_y; i++) {
        for (int j = 0; j < tot_x; j++)
            temp[j] = a[j][i];
        fftcp(temp, -1, tot_y);
        for (int j = 0; j < tot_x; j++)
            a[j][i] = temp[j];
    }

    for (int i = 0; i < tot_x; i++) {
        fftcp(a[i], -1, tot_y);
    }
    cv::Mat res = cv::Mat(cv::Size(tot_y, tot_x), MAT_TYPE);
    Complex2Mat(a, res);
    return res;
}

cv::Mat make_gauss_kernel(int x, int y, double sigmaX, double sigmaY) {
    x |= 1, y |= 1;
    cv::Mat gauss_kernel = cv::Mat(cv::Size(x, y), MAT_TYPE);
    int mid_X = x / 2, mid_Y = y / 2;
    for (int i = 0; i < x; i++)
        for (int j = 0; j < y; j++)
            gauss_kernel.at<DATA_TYPE>(i, j) = - ((double) ((i - mid_X) * (i - mid_X)) / (2.0 * sigmaX * sigmaX) + (double) ((j - mid_Y) * (j - mid_Y)) / (2.0 * sigmaY * sigmaY));

    cv::exp(gauss_kernel, gauss_kernel);
    gauss_kernel /= 2 * acos(-1) * sigmaX * sigmaY;
    return gauss_kernel;
}

cv::Mat make_sobel_kernel(int c)
{
    cv::Mat sobel_kernel = cv::Mat(cv::Size(3, 3), MAT_TYPE);
    if (!c)
    {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                sobel_kernel.at<DATA_TYPE>(i, j) = sobel_x[i][j];
    }
    else
    {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                sobel_kernel.at<DATA_TYPE>(i, j) = sobel_y[i][j];
    }
    return sobel_kernel;
}

void blur(Complex img[][N], Complex kernel[][N], Complex dst[][N])
{
    for (int i = 0; i < Dim[ORIGIN].tot_x; i++)
        for (int j = 0; j < Dim[ORIGIN].tot_y; j++)
            dst[i][j] = img[i][j] * kernel[i][j];
}

cv::Mat Square_root_of_sum_of_squares(cv::Mat img1, cv::Mat img2)
{
    cv::Mat res = cv::Mat(img1.size(), MAT_TYPE);
    for (int i = 0; i < res.rows; i++)
        for (int j = 0; j < res.cols; j++) {
            res.at<DATA_TYPE>(i, j) = sqrt(img1.at<DATA_TYPE>(i, j) * img1.at<DATA_TYPE>(i, j) + img2.at<DATA_TYPE>(i, j) * img2.at<DATA_TYPE>(i, j));
            Theta[i][j] = atan(img2.at<DATA_TYPE>(i, j) / img1.at<DATA_TYPE>(i, j));
        }
    return res;
}

bool isIn(int x, int y, cv::Mat& img)
{
    return x < img.rows && x >= 0 && y < img.cols && y >= 0;
}

double interpolation(double w_1, double w_2, double w_3, double w_4, 
#ifndef NMS_4
    double w_5, double w_6, double w_7, double w_8,
#endif 
    double w, double theta
)
{
#ifdef NMS_4
    double val1 = (w_1 + w_2) / 2 + (w_2 - w_1) / tan(theta) / 2, val2 = (w_3 + w_4) / 2 + (w_3 - w_4) / tan(theta) / 2;
    if (fabs(theta) < PI / 4) val1 = (w_4 + w_2) / 2 + (w_2 - w_4) * tan(theta) / 2, val2 = (w_1 + w_3) / 2 + (w_3 - w_1) * tan(theta) / 2;
    if (w > val1 && w > val2) 
        return w;
    else return 0;
#else
    int op = (theta + (PI / 2.0)) / (PI / 4.0);
    double val1 = 0.0, val2 = 0.0;
    theta = fabs(theta);
#define f(a, b, c) (a + (b - a) / tan(c))
    switch (op)
    {
    case 0:
        val1 = f(w_8, w_4, theta);
        val2 = f(w_5, w_1, theta);
        break;
    case 1:
        val1 = f(w_7, w_4, theta);
        val2 = f(w_6, w_1, theta);
        break;
    case 2:
        val1 = f(w_7, w_2, theta);
        val2 = f(w_6, w_3, theta);
        break;
    case 3:
        val1 = f(w_5, w_2, theta);
        val2 = f(w_8, w_3, theta);
        break;
    default:
        val1 = w_5;
        val2 = w_8;
        break;
    }
#undef f
    if ((w > val1 && w > val2)) 
        return w;
    else return 0;
#endif
}

cv::Mat NonMaxSupress(cv::Mat& Image_src)
{
#ifdef NMS_4
    cv::Mat res = cv::Mat(Image_src.size(), MAT_TYPE);
    for (int i = 0; i < Image_src.rows; i++)
        for (int j = 0; j < Image_src.cols; j++) {
            double w1 = isIn(i - 1, j - 1, Image_src) ? Image_src.at<DATA_TYPE>(i - 1, j - 1) : 0,
                w2 = isIn(i - 1, j + 1, Image_src) ? Image_src.at<DATA_TYPE>(i - 1, j + 1) : 0,
                w3 = isIn(i + 1, j - 1, Image_src) ? Image_src.at<DATA_TYPE>(i + 1, j - 1) : 0,
                w4 = isIn(i + 1, j + 1, Image_src) ? Image_src.at<DATA_TYPE>(i + 1, j + 1) : 0;
            res.at<DATA_TYPE>(i, j) = interpolation(w1, w2, w3, w4, Image_src.at<DATA_TYPE>(i, j), Theta[i][j]);
        }
    return res;
#else
    cv::Mat res = cv::Mat(Image_src.size(), MAT_TYPE);
    for (int i = 0; i < Image_src.rows; i ++ )
        for (int j = 0; j < Image_src.cols; j++) {
            double w1 = isIn(i - 1, j - 1, Image_src) ? Image_src.at<DATA_TYPE>(i - 1, j - 1) : 0,
                   w2 = isIn(i - 1, j + 1, Image_src) ? Image_src.at<DATA_TYPE>(i - 1, j + 1) : 0,
                   w3 = isIn(i + 1, j - 1, Image_src) ? Image_src.at<DATA_TYPE>(i + 1, j - 1) : 0,
                   w4 = isIn(i + 1, j + 1, Image_src) ? Image_src.at<DATA_TYPE>(i + 1, j + 1) : 0,
                   w5 = isIn(i - 1, j, Image_src) ? Image_src.at<DATA_TYPE>(i - 1, j) : 0,
                   w6 = isIn(i, j - 1, Image_src) ? Image_src.at<DATA_TYPE>(i, j - 1) : 0,
                   w7 = isIn(i, j + 1, Image_src) ? Image_src.at<DATA_TYPE>(i, j + 1) : 0,
                   w8 = isIn(i + 1, j, Image_src) ? Image_src.at<DATA_TYPE>(i + 1, j) : 0;
            res.at<DATA_TYPE>(i, j) = interpolation(w1, w2, w3, w4, w5, w6, w7, w8, Image_src.at<DATA_TYPE>(i, j), Theta[i][j]);
        }
    return res;
#endif
}

void Flood_Fill(int xx, int yy, double threshold1, cv::Mat& Image_src, cv::Mat& dst, std::set<PII>& line)
{
    if (st[xx][yy]) return;
    hh = 0, tt = -1;
    q[0] = { xx, yy };
    st[xx][yy] = true;
    tt++;
    while (hh <= tt) {
        auto t = q[hh++];
        for (int i = t.first - 1; i <= t.first + 1; i++)
            for (int j = t.second - 1; j <= t.second + 1; j++)
            {
                if (i == xx && yy == j) {
                    dst.at<DATA_TYPE>(i, j) = 5;
                    line.insert({ i, j });
                    continue; 
                }
                if (st[i][j] || i < 0 || i >= Image_src.rows || j < 0 || j >= Image_src.cols) continue;
                if (Image_src.at<DATA_TYPE>(i, j) < threshold1) continue;

                q[++tt] = { i, j };
                line.insert({ i, j });
                dst.at<DATA_TYPE>(i, j) = 5;
                st[i][j] = true;
            }
    }
}
cv::Mat Double_Threshold(double threshold1, double threshold2, cv::Mat &Image_src)
{
    res_th = cv::Mat(Image_src.size(), MAT_TYPE);
    memset(st, 0, sizeof st);
    for (int i = 0; i < Image_src.rows; i++)
        for (int j = 0; j < Image_src.cols; j++)
            if (Image_src.at<DATA_TYPE>(i, j) >= threshold2 && !st[i][j]) {
                std::set<PII>().swap(l);
                l = std::set<PII>{};
                l.insert({ i, j });
                Flood_Fill(i, j, threshold1, Image_src, res_th, l);
                lines.push_back(l);
            }
    return res_th;
}

void Canny()
{
    cv::Mat gauss_kernel = make_gauss_kernel(gauss_size, gauss_size, gauss_sigma, gauss_sigma), 
            sobel_kernel_x = make_sobel_kernel(0),
            sobel_kernel_y = make_sobel_kernel(1);
    origin_img = cv::imread(PATH);
    cv::cvtColor(origin_img, origin_img, cv::COLOR_BGR2GRAY);
    origin_img.convertTo(origin_img, MAT_TYPE, 1.0 / 255);
    Show(NAME(origin_img));
    init(origin_img, ORIGIN);
    init(gauss_kernel, GAUSS, Dim[ORIGIN].tot_x, Dim[ORIGIN].tot_y);
    init(sobel_kernel_x, SOBEL, Dim[ORIGIN].tot_x, Dim[ORIGIN].tot_y);
    init(sobel_kernel_y, SOBEL, Dim[ORIGIN].tot_x, Dim[ORIGIN].tot_y);
    fft(origin_img, origin_fft, ORIGIN);
    fft(gauss_kernel, gauss_fft, GAUSS);
    fft(sobel_kernel_x, sobel_fft_x, SOBEL);
    fft(sobel_kernel_y, sobel_fft_y, SOBEL);
#ifdef SHOWFREQ
    Showfreq(NAME(origin_fft), ORIGIN, 10.0);
    Showfreq(NAME(gauss_fft), GAUSS, 1.0);
    Showfreq(NAME(sobel_fft_x), SOBEL, 5.0);
    Showfreq(NAME(sobel_fft_y), SOBEL, 5.0);
#endif // SHOWFREQ

    blur(origin_fft, gauss_fft, blur1);
    blur(origin_fft, gauss_fft, blur2);
    blur(blur1, sobel_fft_x, blur1);
    blur(blur2, sobel_fft_y, blur2);
    cv::Mat after_img_x = ifft(blur1, ORIGIN);
    cv::Mat after_img_y = ifft(blur2, ORIGIN);
    cv::Mat after_img = Square_root_of_sum_of_squares(after_img_x, after_img_y);
    int mid_x = 1 + (gauss_size >> 1), mid_y = 1 + (gauss_size >> 1);
    cv::Rect rect(mid_x, mid_y, Dim[ORIGIN].tot_origin_y - mid_y * 2, Dim[ORIGIN].tot_origin_x - mid_x * 2);
    after_img = after_img(rect);
    after_img = normalize(after_img, 5.0);
    suppress = NonMaxSupress(after_img);
    suppress = normalize(suppress, 5.0);
    Show(NAME(suppress));
#ifndef DEBUG
    cv::Mat threshold_img = Double_Threshold(threshold, threshold * ratio, suppress);
    Show(NAME(after_img));
    ShowColor(Theta, suppress);
    Show(NAME(suppress));
    Show(NAME(threshold_img));
#endif
}

static void DEBUG_RES(int, void*)
{
    for (auto l : lines)
        std::set<PII>().swap(l);
    if (!lines.empty()) std::vector<std::set<PII>>().swap(lines);
    if (!lines.empty()) lines = std::vector<std::set<PII>>{};
    threshold_img = Double_Threshold(((double)threshold / 10.0), ((double)threshold * (double)(ratio + 10.0) / 100.0), suppress);
    Show(NAME(threshold_img));
    ShowConnect(threshold_img);
}

void ShowRes()
{
    Canny();
    char s1[50], s2[50];
    memset(s1, 0, sizeof s1);
    memset(s2, 0, sizeof s2);
#ifdef DEBUG
    hsv = cv::Mat(origin_img.size(), CV_8UC3);
    cv::namedWindow("threshold_img", cv::WINDOW_AUTOSIZE);
    sprintf_s(s1, "threshold : %d", threshold);
    cv::createTrackbar(s1, "threshold_img", &threshold, 50, DEBUG_RES);
    sprintf_s(s2, "ratio : %d", ratio);
    cv::createTrackbar(s2, "threshold_img", &ratio, 50, DEBUG_RES);
    DEBUG_RES(0, 0);
    DEBUG_RES(0, 0);
    cv::waitKey(0);
#endif
}

int main()
{
    ShowRes();
	return 0;
}