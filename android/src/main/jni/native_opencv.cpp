#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/imgproc.hpp>
#include <android/log.h>
#include <string>
using namespace cv;
using namespace std;


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

    Mat blendImages(Mat blendedImages, Mat middleImg, int blendHeight, int blendWidth) {

    	float alpha;

    	int width = middleImg.cols;
    	int height = middleImg.rows;
    	int heightBlended = blendedImages.rows;

    	int blend1Start = heightBlended/2 - height/2;
    	int blend2Start = blend1Start + height - blendHeight;

    	//Pravljenje prednje slike(cista kropovana panorama)
    	Mat frontImg = Mat::zeros(heightBlended, width, middleImg.type());
    	Mat temp(frontImg, Rect(0, blend1Start, width, height));
    	middleImg.copyTo(temp);

    	//Spajanje svega u jednu sliku koja sluzi kao pozadinska
    	Mat backImg = blendedImages.clone();
    	Mat temp4(backImg, Rect(0, blend1Start, width, height));
    	middleImg.copyTo(temp4);

    	//Pravljenje delica koji se kopira sa desne ivice na levu da bi se blendovao
    	Mat backImgTemp = backImg.clone();
    	backImgTemp = backImgTemp(Rect(width - blendWidth, 0, blendWidth, heightBlended));

    	Mat backImgHoriz = backImg.clone();
    	Mat temp2(backImgHoriz, Rect(0, 0, blendWidth, heightBlended));
    	backImgTemp.copyTo(temp2);

    	Mat finalImg = backImg.clone();

    	//Prolazak kroz horizontalne ivice
    	for (int i = 0; i < width; i++)
    	{
    		//Gornja ivica
    		for (int j = blend1Start; j < blend1Start + blendHeight; j++)
    		{
    			alpha = 1 - (j - blend1Start) * (static_cast<float>(1) / blendHeight);

    			finalImg.at<Vec3b>(j, i)[0] = (int)(blendedImages.at<Vec3b>(j, i)[0] * alpha) + backImg.at<Vec3b>(j, i)[0] * (1 - alpha);
    			finalImg.at<Vec3b>(j, i)[1] = (int)(blendedImages.at<Vec3b>(j, i)[1] * alpha) + backImg.at<Vec3b>(j, i)[1] * (1 - alpha);
    			finalImg.at<Vec3b>(j, i)[2] = (int)(blendedImages.at<Vec3b>(j, i)[2] * alpha) + backImg.at<Vec3b>(j, i)[2] * (1 - alpha);
    		}

    		//Donja ivica
    		for (int j = blend2Start; j < blend2Start + blendHeight; j++)
    		{
    			alpha = (j - blend2Start) * (static_cast<float>(1) / blendHeight);

    			finalImg.at<Vec3b>(j, i)[0] = (int)(blendedImages.at<Vec3b>(j, i)[0] * alpha) + backImg.at<Vec3b>(j, i)[0] * (1 - alpha);
    			finalImg.at<Vec3b>(j, i)[1] = (int)(blendedImages.at<Vec3b>(j, i)[1] * alpha) + backImg.at<Vec3b>(j, i)[1] * (1 - alpha);
    			finalImg.at<Vec3b>(j, i)[2] = (int)(blendedImages.at<Vec3b>(j, i)[2] * alpha) + backImg.at<Vec3b>(j, i)[2] * (1 - alpha);
    		}
    	}

    	Mat final2Img = finalImg.clone();

    	//Prolazak kroz jednu vertikalnu ivicu
    	for (int i = 0; i < blendWidth; i++)
    	{
    		for (int j = 0; j < heightBlended; j++)
    		{
    			alpha = i * (static_cast<float>(1) / blendWidth);

    			final2Img.at<Vec3b>(j, i)[0] = (int)(finalImg.at<Vec3b>(j, i)[0] * alpha + backImgHoriz.at<Vec3b>(j, i)[0] * (1 - alpha));
    			final2Img.at<Vec3b>(j, i)[1] = (int)(finalImg.at<Vec3b>(j, i)[1] * alpha + backImgHoriz.at<Vec3b>(j, i)[1] * (1 - alpha));
    			final2Img.at<Vec3b>(j, i)[2] = (int)(finalImg.at<Vec3b>(j, i)[2] * alpha + backImgHoriz.at<Vec3b>(j, i)[2] * (1 - alpha));
    		}
    	}




    	return final2Img;
    }


    void stitch_image(char* path) {

        // OPENCV KOD

        //Ucitavanje slike i dobijanje velicina
        string pathString =  path;

        Mat img = imread(pathString + ".jpg");
        double heightOriginal = img.rows;
        double widthOriginal = img.cols;
        int blendHeight = heightOriginal*0.15;
        int blendWidth = widthOriginal * 0.03;

        //Dobijanje pocetne i kranje koordinate za kropovanje(ovo se mora odraditi preko tensor flow-a da bi se dobile tacne koordinate)
        int cropStart = widthOriginal * 0.05;
        int cropEnd = (widthOriginal * 0.9125) - cropStart + blendWidth;

        //Izracunavanje velicina finalne slike na osnovu kropovanih koordinata
        int heightAfter = heightOriginal * 2.63;
        int widthAfter = cropEnd;

        //Izracunavanje visina za watermark i za blurovane delove
        int watermarkHeight = heightAfter * 0.15;
        int cropBlurredHeight = heightAfter / 2 - heightOriginal / 2 - watermarkHeight;

        //Kropovanje slike
        Mat imgCrop = img(Rect(cropStart, 0, cropEnd, heightOriginal));
        Mat imgCropTempTop = imgCrop.clone();
        Mat imgCropTempBottom = imgCrop.clone();

        //Gornji blurovani deo
        Mat imgCropTop = imgCropTempTop(Rect( 0, 0, widthAfter, heightOriginal *0.05));
        flip(imgCropTop, imgCropTop, 0);
        resize(imgCropTop, imgCropTop, Size(widthAfter, cropBlurredHeight + blendHeight));
        Mat imgCropTopBlurred;
        blur(imgCropTop, imgCropTopBlurred, Size(150, 150));

        //Donji blurovani deo
        Mat imgCropBottom = imgCropTempBottom(Rect(0, heightOriginal * 0.8, widthAfter, heightOriginal * 0.2));
        flip(imgCropBottom, imgCropBottom, 0);
        resize(imgCropBottom, imgCropBottom, Size(widthAfter, cropBlurredHeight + blendHeight));
        Mat imgCropBottomBlurred;
        blur(imgCropBottom, imgCropBottomBlurred, Size(150, 150));

        //Spajanje svih delova na glavnu veliku sliku
        Mat imgFullSize = Mat(heightAfter, widthAfter, CV_8UC3, Scalar(168, 228, 19));

        Mat topImage(imgFullSize, Rect(0, watermarkHeight, widthAfter, imgCropTopBlurred.rows));
        imgCropTopBlurred.copyTo(topImage);

        Mat bottomImage(imgFullSize, Rect(0, heightAfter/2 + heightOriginal/2 - blendHeight, widthAfter, imgCropBottomBlurred.rows));
        imgCropBottomBlurred.copyTo(bottomImage);

        Mat finalImage = blendImages(imgFullSize, imgCrop, blendHeight, blendWidth);
        Mat finalImageCropped = finalImage(Rect(0, 0, finalImage.cols - blendWidth, finalImage.rows));

        //Kraj i cuvanje slike
        imwrite( pathString + "-edited.jpg", finalImageCropped);

        //-----------------------------------------------------------------------------------------
    }
}