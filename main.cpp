#include <iostream>
#include <fstream>
#include <chrono>
#include <set>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>

#include "multilayer_stixel_world.h"
#include "semi_global_matching.h"

static cv::Scalar computeColor(float val)
{
	const float hscale = 6.f;
	float h = 0.6f * (1.f - val), s = 1.f, v = 1.f;
	float r, g, b;

	static const int sector_data[][3] =
	{ { 1,3,0 },{ 1,0,2 },{ 3,0,1 },{ 0,2,1 },{ 0,1,3 },{ 2,1,0 } };
	float tab[4];
	int sector;
	h *= hscale;
	if (h < 0)
		do h += 6; while (h < 0);
	else if (h >= 6)
		do h -= 6; while (h >= 6);
	sector = cvFloor(h);
	h -= sector;
	if ((unsigned)sector >= 6u)
	{
		sector = 0;
		h = 0.f;
	}

	tab[0] = v;
	tab[1] = v * (1.f - s);
	tab[2] = v * (1.f - s * h);
	tab[3] = v * (1.f - s * (1.f - h));

	b = tab[sector_data[sector][0]];
	g = tab[sector_data[sector][1]];
	r = tab[sector_data[sector][2]];
	return 255 * cv::Scalar(b, g, r);
}

static cv::Scalar dispToColor(float disp, float maxdisp = 64, float offset = 0)
{
	if (disp < 0)
		return cv::Scalar(128, 128, 128);
	return computeColor(std::min(disp + offset, maxdisp) / maxdisp);
}

static void drawStixel(cv::Mat& img, const Stixel& stixel, cv::Scalar color)
{
	const int radius = std::max(stixel.width / 2, 1);
	const cv::Point tl(stixel.u - radius, stixel.vT);
	const cv::Point br(stixel.u + radius, stixel.vB);
	cv::rectangle(img, cv::Rect(tl, br), color, -1);
	cv::rectangle(img, cv::Rect(tl, br), cv::Scalar(255, 255, 255), 1);
}

class SGMWrapper
{

public:

	SGMWrapper(int numDisparities)
	{
		SemiGlobalMatching::Parameters param;
		param.numDisparities = numDisparities / 2;
		param.max12Diff = -1;
		param.medianKernelSize = -1;

		param.P1 = 8 * 11 * 11;
		param.P2 = 32 * 11 * 11;
		param.medianKernelSize = 3;

		sgm_ = cv::Ptr<SemiGlobalMatching>(new SemiGlobalMatching(param));
	}

	void compute(const cv::Mat& I1, const cv::Mat& I2, cv::Mat& D1)
	{
		cv::pyrDown(I1, I1_);
		cv::pyrDown(I2, I2_);

		sgm_->compute(I1_, I2_, D1_, D2_);

		cv::resize(D1_, D1, I1.size(), 0, 0, cv::INTER_CUBIC);
		cv::resize(D2_, D2, I1.size(), 0, 0, cv::INTER_CUBIC);
		D1 *= 2;
		D2 *= 2;
		cv::medianBlur(D1, D1, 3);
		cv::medianBlur(D2, D2, 3);
		SemiGlobalMatching::LRConsistencyCheck(D1, D2, 5);
	}

private:
	cv::Mat I1_, I2_, D1_, D2_, D2;
	cv::Ptr<SemiGlobalMatching> sgm_;
};

void ComputeDepth(const cv::Mat& disparity) {
	std::vector<double> x_vec;
	std::vector<double> y_vec;
	std::vector<double> z_vec;

	std::ofstream out_file("pointcloud.txt");

	double f = 1050;
	double x_c = 0;
	double y_c = 0;
	double b = 0.37;

	for (int i=0; i<disparity.rows; ++i) {
		for (int j=0; j<disparity.cols; ++j) {
			double x=0, y=0, z=0;

			if (disparity.at<float>(i, j) > 10) {
				z = ( (f * b) / disparity.at<float>(i, j) );
				x = ( (z * (j - x_c)) / f );
				y = ( (z * (i - y_c)) / f );
			}

			x_vec.push_back(x);
			y_vec.push_back(y);
			z_vec.push_back(z);

			out_file << x << "," << y << "," << z << "\n";
		}
	}

	out_file.close();
}

struct SortStixels {
	inline bool operator() (const Stixel& stixel1, const Stixel& stixel2) {
		if (stixel1.u == stixel2.u) {
			return stixel1.vB > stixel2.vB;
		}
		else {
			return stixel1.u < stixel2.u;
		}
	}
};

int main(int argc, char* argv[])
{
	if (argc < 4)
	{
		std::cout << "usage: " << argv[0] << " left-image-format right-image-format camera.xml" << std::endl;
		return -1;
	}

	// stereo SGBM
	const int numDisparities = 256;
	SGMWrapper sgm(numDisparities);

	const int wsize = 13;
	// const int numDisparities = 64;
	const int P1 = 8 * wsize * wsize;
	const int P2 = 32 * wsize * wsize;
	cv::Ptr<cv::StereoSGBM> ssgbm = cv::StereoSGBM::create(0, numDisparities/2, wsize, P1, P2,
		0, 0, 0, 0, 0, cv::StereoSGBM::MODE_SGBM_3WAY);

	cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilter(ssgbm);
	wls_filter->setLambda(1000);
	wls_filter->setSigmaColor(1.2);
	cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(ssgbm);
	
	cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(numDisparities);

	// read camera parameters
	const cv::FileStorage fs(argv[3], cv::FileStorage::READ);
	CV_Assert(fs.isOpened());

	// input parameters
	MultiLayerStixelWorld::Parameters param;
	param.camera.fu = fs["FocalLengthX"];
	param.camera.fv = fs["FocalLengthY"];
	param.camera.u0 = fs["CenterX"];
	param.camera.v0 = fs["CenterY"];
	param.camera.baseline = fs["BaseLine"];
	param.camera.height = fs["Height"];
	param.camera.tilt = fs["Tilt"];
	param.dmax = numDisparities;
	param.roadEstimation = MultiLayerStixelWorld::ROAD_ESTIMATION_CAMERA;
	param.stixelWidth = 41;

	cv::Mat disparity, disparity_rl, filtered_disp;
	MultiLayerStixelWorld stixelWorld(param);

	cv::namedWindow("disparity", CV_WINDOW_NORMAL);
	cv::resizeWindow("disparity", 1280, 720);

	for (int frameno = 1;; frameno++)
	{
		cv::Mat I1_orig = cv::imread(cv::format(argv[1], frameno), 0);
		cv::Mat I2_orig = cv::imread(cv::format(argv[2], frameno), 0);

		if (I1_orig.empty() || I2_orig.empty())
		{
			std::cerr << "imread failed." << std::endl;
			break;
		}

		cv::Rect2d ROI1(0, 0, 1920, 1000);
		cv::Rect2d ROI2(0, 0, 1920, 1000);

		cv::Mat I1 = I1_orig(ROI1);
		cv::Mat I2 = I2_orig(ROI2);

		// cv::Mat I1 = I1_orig;
		// cv::Mat I2 = I2_orig;

		// cv::pyrDown(I1, I1, cv::Size(I1.cols/2, I1.rows/2));
		// cv::pyrDown(I2, I2, cv::Size(I2.cols/2, I2.rows/2));

		// CV_Assert(I1.size() == I2.size() && I1.type() == I2.type());
		// CV_Assert(I1.type() == CV_8U || I1.type() == CV_16U);

		if (I1.type() == CV_16U)
		{
			cv::normalize(I1, I1, 0, 255, cv::NORM_MINMAX);
			cv::normalize(I2, I2, 0, 255, cv::NORM_MINMAX);
			I1.convertTo(I1, CV_8U);
			I2.convertTo(I2, CV_8U);
		}

		const auto t1 = std::chrono::steady_clock::now();

		// compute dispaliry
		sgm.compute(I1, I2, disparity);
		// ssgbm->compute(I1, I2, disparity);
		// right_matcher->compute(I2, I1, disparity_rl);
		// wls_filter->filter(disparity, I1, filtered_disp, disparity_rl);
		// filtered_disp.convertTo(filtered_disp, CV_32F, 1. / SemiGlobalMatching::DISP_SCALE);
		// filtered_disp.copyTo(disparity);

		// sbm->compute(I1, I2, disparity);
		disparity.convertTo(disparity, CV_32F, 1. / SemiGlobalMatching::DISP_SCALE);

		// ComputeDepth(disparity);

		// compute stixels
		const auto t2 = std::chrono::steady_clock::now();

		std::vector<Stixel> stixels;
		stixelWorld.compute(disparity, stixels);
		sort(stixels.begin(), stixels.end(), SortStixels());

		const auto t3 = std::chrono::steady_clock::now();
		const auto duration12 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		const auto duration23 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

		// colorize disparity
		cv::Mat disparityColor;
		disparity.convertTo(disparityColor, CV_8U, 255. / numDisparities);
		cv::applyColorMap(disparityColor, disparityColor, cv::COLORMAP_JET);
		disparityColor.setTo(cv::Scalar::all(0), disparity < 0);

		// put processing time
		cv::putText(disparityColor, cv::format("dispaliry computation time: %4.1f [msec]", 1e-3 * duration12),
			cv::Point(100, 50), 2, 0.75, cv::Scalar(255, 255, 255));
		cv::putText(disparityColor, cv::format("stixel computation time: %4.1f [msec]", 1e-3 * duration23),
			cv::Point(100, 80), 2, 0.75, cv::Scalar(255, 255, 255));

		// draw stixels
		cv::Mat draw;
		cv::cvtColor(I1, draw, cv::COLOR_GRAY2BGR);

		cv::Mat stixelImg = cv::Mat::zeros(I1.size(), draw.type());
		std::set<float> unique_disparities;
		for (const auto& stixel : stixels) {
			drawStixel(stixelImg, stixel, dispToColor(stixel.disp));
			unique_disparities.insert(stixel.disp);
		}

		int prev_x = -1;
		int prev_y = -1;
		int curr_x = prev_x;
		int curr_y = prev_y;
		for (const auto& stixel : stixels) {
			// std::cout << stixel.u << "," << stixel.vB << " ";
			curr_x = stixel.u;
			curr_y = stixel.vB;

			if (curr_x != prev_x) {
				// std::cout << curr_x << ",";
				double depth = 100;
				if (stixel.disp > 5) {
					depth = (1050 * 0.37) / stixel.disp;
					// depth = (707 * 0.54) / stixel.disp;
				}
				// cv::putText(stixelImg,
				// 			std::to_string((int) depth),
				// 			cv::Point(stixel.u, (stixel.vT + stixel.vT)/2),
				// 			cv::FONT_HERSHEY_SIMPLEX, 1,
				// 			cv::Scalar(255,255,255), 3);
				cv::putText(stixelImg, 
							std::to_string((int) depth), 
							cv::Point(stixel.u, (stixel.vT + stixel.vT)/2), 
							cv::FONT_HERSHEY_SIMPLEX, 1, 
							cv::Scalar(0,0,0), 3);
			}

			prev_x = curr_x;
			prev_y = curr_y;
		}

		// std::cout << std::endl << std::endl;

		// std::cout << "Unique disparities: \n";
		// for (const auto& disp : unique_disparities) {
		// 	std::cout << disp << ",";
		// }
		// std::cout << std::endl << std::endl;

		cv::addWeighted(draw, 1, stixelImg, 0.5, 0, draw);

		cv::imshow("disparity", disparityColor);
		cv::imshow("stixels", draw);

		const char c = cv::waitKey(2000);
		if (c == 27)
			break;
		if (c == 'p')
			cv::waitKey(0);
	}
}
