#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/imgproc.hpp>
#include <android/log.h>
#include <string>
using namespace cv;
using namespace std;

// PARAMETRI KOJI SE PODESAVAJU
int heightAfterResize = 600; //visina slike nakon resizovanja
int blendingAngle = 3; // [1-360] ugao pod kojim su slikane 2 slike
int blendingAngleStep = blendingAngle; // Broj za koji se ugao povecava kad se slika preskace
int blendingWidthPercent = 50; // [0-100] sirina blendovanog dela u procentima
int imgNumb = 360; // Broj slika
string finalName = "PanoramaFinal.jpg"; // Ime finalne slike
string topName = "PanoramaTop.jpg";
string bottomName = "PanoramaBottom.jpg";

// Avoiding name mangling
extern "C" {
    // Attributes to prevent 'unused' function from being removed and to make it visible
    __attribute__((visibility("default"))) __attribute__((used))
    const char* version() {
        __android_log_print(ANDROID_LOG_VERBOSE,"OpencvAwesome","Dart ffi is setup");
        return CV_VERSION;
    }


__attribute__((visibility("default"))) __attribute__((used))

struct tokens: ctype<char>
{
    tokens(): std::ctype<char>(get_table()) {}

    static std::ctype_base::mask const* get_table()
    {
        typedef std::ctype<char> cctype;
        static const cctype::mask *const_rc= cctype::classic_table();

        static cctype::mask rc[cctype::table_size];
        std::memcpy(rc, const_rc, cctype::table_size * sizeof(cctype::mask));

        rc[','] =  ctype_base::space;
        rc[' '] =  ctype_base::space;
        return &rc[0];
    }
};
    vector<string> getpathlist(string path_string){
         string sub_string = path_string.substr(1,path_string.length()-2);
         stringstream ss(sub_string);
        ss.imbue( locale( locale(), new tokens()));
         istream_iterator<std::string> begin(ss);
         istream_iterator<std::string> end;
         vector<std::string> pathlist(begin, end);
        return pathlist;
    }

    //------------------------MOJE FUNKCIJE--------------------------------

    void blendHorizontal(Mat input1, Mat input2, Mat& output)
    {
    	int width1 = input1.cols;
    	int height1 = input1.rows;
    	int width2 = input2.cols;
    	int height2 = input2.rows;

    	float blendingAnglePixels = blendingAngle * 0.02731 * width2; //2.731% je konstanta koja otprilike predstavlja procenat sirine slike za 1 ugao rotacije
    	float blendingArea = blendingAnglePixels * blendingWidthPercent * 0.01; //preklop u procentima

    	int cropWidth = (width2 - blendingAnglePixels) / 2;

    	input1 = input1(Rect(0, 0, width1 - cropWidth, height1));
    	input2 = input2(Rect(width2 / 2 - blendingArea, 0, width2 - cropWidth, height2));

    	width1 = input1.cols;
    	width2 = input2.cols;

    	int blendWidth = width1 + width2 - blendingArea;
    	int blendHeight;

    	float alpha;

    	if (height1 > height2) blendHeight = height1;
    	if (height1 < height2) blendHeight = height2;
    	if (height1 == height2) blendHeight = height1;

    	Mat blend1 = Mat::zeros(blendHeight, blendWidth, input1.type());
    	Mat left(blend1, Rect(0, 0, width1, height1));
    	input1.copyTo(left);

    	Mat blend2 = Mat::zeros(blendHeight, blendWidth, input2.type());
    	Mat right(blend2, Rect(width1 - blendingArea, blendHeight/2- height2/2, width2, height2));
    	input2.copyTo(right);

    	output = Mat::zeros(blendHeight, blendWidth, input1.type());

    	cv::addWeighted(blend1, 1.0, blend2, 1.0, 0.0, output);

    	for (int i = width1 - blendingArea; i < width1; i++)
    	{
    		for (int j = 0; j < blendHeight; j++)
    		{
    			alpha = 1 - ((i - width1 + blendingArea) * (1 / blendingArea));

    			output.at<Vec3b>(j, i)[0] = (int)(blend1.at<Vec3b>(j, i)[0] * alpha + blend2.at<Vec3b>(j, i)[0] * (1 - alpha));
    			output.at<Vec3b>(j, i)[1] = (int)(blend1.at<Vec3b>(j, i)[1] * alpha + blend2.at<Vec3b>(j, i)[1] * (1 - alpha));
    			output.at<Vec3b>(j, i)[2] = (int)(blend1.at<Vec3b>(j, i)[2] * alpha + blend2.at<Vec3b>(j, i)[2] * (1 - alpha));
    		}
    	}
    }

    void blendVertical(Mat input1, Mat input2, Mat& output)
    {
    	double p = (double)heightAfterResize*0.33; //preklop u pikselima
    	float alpha;

    	int width1 = input1.cols;
    	int width2 = input2.cols;
    	int height1 = input1.rows;
    	int height2 = input2.rows;

    	int blendWidth;

    	int blendHeight = input1.rows + input2.rows - p;

    	if (width1 > width2) blendWidth = width1;
    	if (width1 < width2) blendWidth = width2;
    	if (width1 == width2) blendWidth = width1;

    	Mat blend1 = Mat::zeros(blendHeight, blendWidth, input1.type());
    	Mat left(blend1, Rect(0, 0, width1, height1));
    	input1.copyTo(left);

    	Mat blend2 = Mat::zeros(blendHeight, blendWidth, input2.type());
    	Mat right(blend2, Rect(0, height1 - p, width2, height2));
    	input2.copyTo(right);

    	output = cv::Mat::zeros(blendHeight, blendWidth, input1.type());

    	cv::addWeighted(blend1, 1.0, blend2, 1.0, 0.0, output);

    	for (int i = 0; i < blendWidth; i++)
    	{
    		for (int j = height1 - p; j < height1; j++)
    		{
    			alpha = 1 - ((j - (height1 - p)) * (1 / p));

    			output.at<Vec3b>(j, i)[0] = (int)(blend1.at<Vec3b>(j, i)[0] * alpha + blend2.at<Vec3b>(j, i)[0] * (1 - alpha));
    			output.at<Vec3b>(j, i)[1] = (int)(blend1.at<Vec3b>(j, i)[1] * alpha + blend2.at<Vec3b>(j, i)[1] * (1 - alpha));
    			output.at<Vec3b>(j, i)[2] = (int)(blend1.at<Vec3b>(j, i)[2] * alpha + blend2.at<Vec3b>(j, i)[2] * (1 - alpha));
    		}
    	}
    }

    Mat processImage(Mat img, string angleTilt, string angleLean)
    {
    	float ratio = (float)img.cols / (float)img.rows;
    	int scaleHeight = heightAfterResize;
    	int scaleWidth = scaleHeight * ratio;
    	resize(img, img, Size(scaleWidth, scaleHeight), INTER_CUBIC);

    	// rotacija
    	double tilt = stod(angleTilt) * -1.;
    	Point2f center(img.cols / 2., img.rows / 2.);
    	Mat rotat_mat = getRotationMatrix2D(center, tilt, 1.0);
    	warpAffine(img, img, rotat_mat, Size(scaleWidth, scaleHeight));

    	// translacija
    	double offset = stod(angleLean) * 10;
    	Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, 0, 0, 1, offset);
    	warpAffine(img, img, trans_mat, Size(scaleWidth, scaleHeight + abs(offset)));

    	return img;
    }


    void stitch_image(char* path, char* angleTilt, char* angleLean ) {

        //------------------------DEO SAMO ZA ANDROID---------------------
        string angleLean_string = angleLean;
        vector<string> anglesLean = getpathlist(angleLean_string);
        string angleTilt_string = angleTilt;
        vector<string> anglesTilt = getpathlist(angleTilt_string);
        //----------------------------------------------------------------


        string finalPath(path + finalName);
        	string topPath(path + topName);
        	string bottomPath(path + bottomName);

        	Mat blendTopLastFrame, blendTopCurrentFrame, blendTopLayer;
        	Mat blendBottomLastFrame, blendBottomCurrentFrame, blendBottomLayer;

        	for (int i = 1; i < imgNumb; i++)
        	{
        		if (i == 1)
        		{
        			blendTopLastFrame = processImage((imread(path + to_string(i - 1) + "-top.jpg")), anglesTilt[i - 1], anglesLean[i - 1]);
        		}

        		Mat blendTopCurrentFrame = imread(path + to_string(i) + "-top.jpg");

        		if (blendTopCurrentFrame.empty())
        		{
        			blendingAngle = blendingAngle + blendingAngleStep;
        		}

        		if (!blendTopCurrentFrame.empty())
        		{

        			blendingAngle = blendingAngleStep;

        			blendTopCurrentFrame = processImage(blendTopCurrentFrame, anglesTilt[i], anglesLean[i]);

        			blendHorizontal(blendTopLastFrame, blendTopCurrentFrame, blendTopLayer);

        			blendTopLastFrame = blendTopLayer.clone();

        			//imwrite(path + to_string(i) + "blend-top.jpg", blendTopLayer); //debug
        		}
        	}

        	blendingAngle = blendingAngleStep;

        	for (int i = 1; i < imgNumb; i++)
        	{
        		if (i == 1)
        		{
        			blendBottomLastFrame = processImage((imread(path + to_string(i - 1) + "-bottom.jpg")), anglesTilt[i - 1], anglesLean[i - 1]);
        		}

        		Mat blendBottomCurrentFrame = imread(path + to_string(i) + "-bottom.jpg");

        		if (blendBottomCurrentFrame.empty())
        		{
        			blendingAngle = blendingAngle + blendingAngleStep;
        		}

        		if (!blendBottomCurrentFrame.empty())
        		{

        			blendingAngle = blendingAngleStep;

        			blendBottomCurrentFrame = processImage(blendBottomCurrentFrame, anglesTilt[i], anglesLean[i]);

        			blendHorizontal(blendBottomLastFrame, blendBottomCurrentFrame, blendBottomLayer);

        			blendBottomLastFrame = blendBottomLayer.clone();

        			//imwrite(path + to_string(i) + "blend-bottom.jpg", blendBottomLayer); //debug
        		}
        	}

        	Mat finalTopCrop, finalBottomCrop, finalOutput;

        	finalTopCrop = blendTopLayer(Rect(350, 0, blendTopLayer.cols - 600, blendTopLayer.rows));
        	finalBottomCrop = blendBottomLayer(Rect(350, 0, blendBottomLayer.cols - 600, blendBottomLayer.rows));

        	imwrite(topPath, finalTopCrop);
        	imwrite(bottomPath, finalBottomCrop);

        	blendVertical(finalTopCrop, finalTopCrop, finalOutput);

        	imwrite(finalPath, finalOutput);

        //-----------------------------------------------------------------------------------------
    }
}