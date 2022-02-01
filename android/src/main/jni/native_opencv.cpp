#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <string>

#include <android/log.h>

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
    struct tokens: ctype<char>{
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

    //------------------------MOJE FUNKCIJE--------------------------------
    int calculateFinalWidth(Mat input);
    int calculateFinalHeight(Mat inputBlended);
    Mat blendSides(Mat input);
    Mat addLogos(Mat input, Mat logo);

    void stitch_image(char* path) {

        // OPENCV KOD
        __android_log_print(ANDROID_LOG_VERBOSE,"+++++++++++++++++OPENCV+++++++++++++++++", "OBRADA POCELA");

        //Ucitavanje slike i dobijanje velicina
        string pathString = path;

        __android_log_print(ANDROID_LOG_VERBOSE, "+++++++++++++++++OPENCV+++++++++++++++++", path);

        Mat img = imread(pathString + "/panoRaw.png");
        Mat logo = imread(pathString + "/logoRaw.png");

        if (logo.empty()) {
            __android_log_print(ANDROID_LOG_VERBOSE, "+++++++++++++++++OPENCV+++++++++++++++++", "SLIKA LOGOA NIJE UCITANA!");
        }
        if (img.empty()) {
            __android_log_print(ANDROID_LOG_VERBOSE, "+++++++++++++++++OPENCV+++++++++++++++++", "SLIKA PANORAME NIJE UCITANA!");
        }

        if (!img.empty() && !logo.empty()) {

            //Pozivanje funkcije za blendovanje leve i desne strane i funkcije za dodavanje logoa
            __android_log_print(ANDROID_LOG_VERBOSE, "+++++++++++++++++OPENCV+++++++++++++++++", "POCETAK BLENDA");
            Mat imgBlend = blendSides(img);
            __android_log_print(ANDROID_LOG_VERBOSE, "+++++++++++++++++OPENCV+++++++++++++++++", "KRAJ BLENDA I POCETAK DODAVANJA LOGOA");
            Mat imgFinal = addLogos(imgBlend, logo);
            __android_log_print(ANDROID_LOG_VERBOSE, "+++++++++++++++++OPENCV+++++++++++++++++", "KRAJ DODAVANJA LOGOA");

            //Kraj i cuvanje slike
            imwrite(pathString + "/panoFinal.jpg", imgFinal);
        }

        //-----------------------------------------------------------------------------------------
    }

    Mat blendSides(Mat input) {

        float alpha;

        int width = input.cols;
        int height = input.rows;

        int cropWidth = calculateFinalWidth(input);

        int blendWidth = cropWidth * 0.02;
        int leftMargin = (width - cropWidth) / 2;

        Mat finalImg(input, Rect(leftMargin, 0, cropWidth, height));
        Mat blendPart(input, Rect(leftMargin + cropWidth, 0, blendWidth, height));

        for (int i = 0; i < blendWidth; i++)
        {
            alpha = i * (static_cast<float>(1) / blendWidth);

            for (int j = 0; j < height; j++)
            {
                finalImg.at<Vec3b>(j, i)[0] = (int)(finalImg.at<Vec3b>(j, i)[0] * alpha + blendPart.at<Vec3b>(j, i)[0] * (1 - alpha));
                finalImg.at<Vec3b>(j, i)[1] = (int)(finalImg.at<Vec3b>(j, i)[1] * alpha + blendPart.at<Vec3b>(j, i)[1] * (1 - alpha));
                finalImg.at<Vec3b>(j, i)[2] = (int)(finalImg.at<Vec3b>(j, i)[2] * alpha + blendPart.at<Vec3b>(j, i)[2] * (1 - alpha));
            }
        }

        return finalImg;
    }

    Mat addLogos(Mat input, Mat logo) {

        int height = input.rows;
        int width = input.cols;

        int finalHeight = calculateFinalHeight(input);
        int finalWidth = width;

        int logoHeight = (finalHeight - height) / 2;
        int logoWidth = finalWidth;

        Mat logoBottomInput = logo.clone();
        Mat logoTopInput;
        resize(logoBottomInput, logoBottomInput, Size(logoWidth, logoHeight), INTER_CUBIC);
        rotate(logoBottomInput, logoTopInput, ROTATE_180);

        Mat imgFinal = Mat(finalHeight, finalWidth, CV_8UC3, Scalar(168, 228, 19));

        Mat mainImg(imgFinal, Rect(0, logoHeight, width, height));
        input.copyTo(mainImg);

        Mat logoBottom(imgFinal, Rect(0, finalHeight - logoHeight, logoWidth, logoHeight));
        logoBottomInput.copyTo(logoBottom);

        Mat logoTop(imgFinal, Rect(0, 0, logoWidth, logoHeight));
        logoTopInput.copyTo(logoTop);

        return imgFinal;

    }

    int calculateFinalHeight(Mat inputBlended) {

        int height = inputBlended.rows;
        int width = inputBlended.cols;

        //Formula za finalnu visinu glasi: Uzmi odnos ulazne sirine i visine, podeli ga sa 2 i taj koeficient pomnozi sa ulaznom visinom
        int heightAfter = static_cast<double>(width) / static_cast<double>(height) / 2.0 * static_cast<double>(height);

        return heightAfter;
    }

    int calculateFinalWidth(Mat input) {

        int width = input.cols;
        int height = input.rows;

        Mat leftImg(input, Rect(0, 0, width * 0.1, height));
        Mat rightImg(input, Rect(width * 0.9, 0, width - width * 0.9, height));

        Mat leftImgGray, rightImgGray;
        Mat imgForDrawing = input.clone();

        cvtColor(leftImg, leftImgGray, COLOR_BGR2GRAY);
        cvtColor(rightImg, rightImgGray, COLOR_BGR2GRAY);

        vector<Point2f> p0, p1;
        vector<Point2f> p0_filtered, p1_filtered;

        // Detekcija kornera na prvoj slici
        goodFeaturesToTrack(leftImgGray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);

        // Koriscenje optical flow-a za detekciju tacaka
        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 30, 0.01);
        calcOpticalFlowPyrLK(leftImgGray, rightImgGray, p0, p1, status, err, Size(12, 12), 2, criteria);


        for (uint i = 0; i < p0.size(); i++)
        {
            //Selektuj dobre tacke po statusu
            if (status[i] == 1) {

                //Dodaj ofset od 90% sirine posto se tacke pretrazuju samo na prvih 10% i poslednjih 10% ulazne slike
                p1[i].x += width * 0.9;

                int distanceX = p1[i].x - p0[i].x;
                int distanceY = p1[i].y - p0[i].y;

                if (abs(distanceY) < height * 0.02) {
                    //Dodaj filterovane tacke u nove nizove
                    p0_filtered.push_back(p0[i]);
                    p1_filtered.push_back(p1[i]);

                    //Ispisivanje i iscrtavanje tacaka i linija(samo za debug)
                    cout << distanceX << endl;
                    line(imgForDrawing, p1[i], p0[i], Scalar(0, 0, 255), 10);
                    circle(imgForDrawing, p0[i], 1, Scalar(255, 0, 0), 10);
                    circle(imgForDrawing, p1[i], 1, Scalar(255, 0, 0), 10);
                }
            }
        }

        double widthFinal = 0, p0_medium = 0, p1_medium = 0, p0_sum = 0, p1_sum = 0;

        //Uzima sve filterovane tacke i nalazi sredinu iz njih
        for (int i = 0; i < p0_filtered.size(); i++) {
            p0_sum += p0_filtered[i].x;
            p1_sum += p1_filtered[i].x;
        }

        p0_medium = p0_sum / p0_filtered.size();
        p1_medium = p1_sum / p1_filtered.size();

        widthFinal = p1_medium - p0_medium;

        //Ako finalna sirina spada van ovih okvira, dodeli neku defaultnu vrednost
        if (widthFinal < width * 0.88 || widthFinal > width * 0.93) {
            widthFinal = width * 0.9;
        }

        return widthFinal;
    }
}