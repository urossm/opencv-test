#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

const float inlier_threshold = 2.5f; // Distance threshold to identify inliers with homography check
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

#include <android/log.h>

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

    //------------------------MOJE FUNKCIJE----------------------------------------------------------------------------------------------------------------------------------------
    //-------------------------------------------------------------

    Mat distortImage(Mat img, double xCoord, double strength) //strength - jacina zakrivljenja, pozitivne vrednosti su fisheye efekat a negativne suprotno
    {
        int width = img.cols;
        int height = img.rows;

        cv::Mat distCoeff;
        distCoeff = cv::Mat::zeros(4, 1, CV_64FC1);

        // indices: k1, k2, p1, p2, k3, k4, k5, k6
        // TODO: add your coefficients here!

        double k1 = 1e-5 * strength;
        double k2 = 0;
        double p1 = 0;
        double p2 = 0;

        distCoeff.at<double>(0, 0) = k1;
        distCoeff.at<double>(1, 0) = k2;
        distCoeff.at<double>(2, 0) = p1;
        distCoeff.at<double>(3, 0) = p2;

        // assume unit matrix for camera, so no movement
        cv::Mat cam;
        cam = cv::Mat::eye(3, 3, CV_32FC1);

        cam.at<float>(0, 2) = width * xCoord;
        cam.at<float>(1, 2) = height * 0.5;
        cam.at<float>(0, 0) = 10;
        cam.at<float>(1, 1) = 10;

        Mat imgFinal;

        undistort(img, imgFinal, cam, distCoeff);

        Mat nonZeroes, imgFinalGray;
        cvtColor(imgFinal, imgFinalGray, COLOR_RGB2GRAY);
        findNonZero(imgFinalGray, nonZeroes);
        Rect rect = boundingRect(nonZeroes);

        Mat croppedImage = imgFinal(rect);

        return croppedImage;
    }

    Mat undistortImage(Mat img, double strength) //strength - jacina zakrivljenja, pozitivne vrednosti su fisheye efekat a negativne suprotno
    {
        int width = img.cols;
        int height = img.rows;

        cv::Mat distCoeff;
        distCoeff = cv::Mat::zeros(4, 1, CV_64FC1);

        // indices: k1, k2, p1, p2, k3, k4, k5, k6
        // TODO: add your coefficients here!

        double k1 = 1e-5 * strength;
        double k2 = 0;
        double p1 = 0;
        double p2 = 0;

        distCoeff.at<double>(0, 0) = k1;
        distCoeff.at<double>(1, 0) = k2;
        distCoeff.at<double>(2, 0) = p1;
        distCoeff.at<double>(3, 0) = p2;

        // assume unit matrix for camera, so no movement
        cv::Mat cam;
        cam = cv::Mat::eye(3, 3, CV_32FC1);

        cam.at<float>(0, 2) = width * 0.5;
        cam.at<float>(1, 2) = height * 0.5;
        cam.at<float>(0, 0) = 10;
        cam.at<float>(1, 1) = 10;

        Mat imgFinal;

        undistort(img, imgFinal, cam, distCoeff);

        Mat nonZeroes, imgFinalGray;
        cvtColor(imgFinal, imgFinalGray, COLOR_RGB2GRAY);
        findNonZero(imgFinalGray, nonZeroes);
        Rect rect = boundingRect(nonZeroes);

        Mat croppedImage = imgFinal(rect);

        return imgFinal;
    }

    array<string, 4> splitArguments(string argument) {

        string delimiter = ";";
        size_t pos = 0;
        array<string, 4> tokens;

        int i = 0;

        while ((pos = argument.find(delimiter)) != string::npos) {
            string token = argument.substr(0, pos);
            tokens[i] = token;
            argument.erase(0, pos + delimiter.length());
            i++;
        }

        tokens[3] = argument;

        return tokens;
    }

    Mat blendVertical(Mat img1, Mat img2, double xOffset) {

        vector<Mat> finalImages;
        finalImages.push_back(img1);
        finalImages.push_back(img2);

        double img1width = finalImages[0].cols;
        double img1height = finalImages[0].rows;
        double img2width = finalImages[1].cols;
        double img2height = finalImages[1].rows;

        double finalWidth = img2width + xOffset;
        double finalHeight = img2height;

        vector<UMat> sources(2);
        finalImages[0].convertTo(sources[0], CV_32F);
        finalImages[1].convertTo(sources[1], CV_32F);

        double scaleFactor = 4.0;
        double gap = 10.0;

        Size newSize = Size(finalWidth / scaleFactor, finalHeight / scaleFactor);

        resize(sources[0], sources[0], Size(img1width / scaleFactor, img1height / scaleFactor), INTER_LINEAR);
        resize(sources[1], sources[1], Size(img2width / scaleFactor, img2height / scaleFactor), INTER_LINEAR);

        vector<Point> cornersScaled;
        cornersScaled.push_back(Point(0, 0));
        cornersScaled.push_back(Point(xOffset / scaleFactor + gap, 0));

        vector<UMat> masks(2);
        Mat mask1(finalImages[0].size(), CV_8U, Scalar(255));
        Mat mask2(finalImages[1].size(), CV_8U, Scalar(255));
        mask1.convertTo(masks[0], CV_8U);
        mask2.convertTo(masks[1], CV_8U);

        resize(masks[0], masks[0], Size(img1width / scaleFactor, img1height / scaleFactor), INTER_LINEAR);
        resize(masks[1], masks[1], Size(img2width / scaleFactor, img2height / scaleFactor), INTER_LINEAR);

        Ptr<SeamFinder> seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
        seam_finder->find(sources, cornersScaled, masks);

        vector<Mat> finalMasks;

        finalMasks.push_back(masks[0].getMat(ACCESS_READ));
        finalMasks.push_back(masks[1].getMat(ACCESS_READ));

        //Povecanje rezolucije druge slike da bude ista kao canvas
        Mat translateMat1 = (Mat_<double>(2, 3) << 1, 0, 0, 0, 1, 0);
        warpAffine(finalImages[0], finalImages[0], translateMat1, Size(finalWidth, finalHeight));

        //Translacija druge slike
        Mat translateMat2 = (Mat_<double>(2, 3) << 1, 0, xOffset, 0, 1, 0);
        warpAffine(finalImages[1], finalImages[1], translateMat2, Size(finalWidth, finalHeight));

        resize(finalMasks[0], finalMasks[0], Size(img1width, img1height), INTER_NEAREST);
        resize(finalMasks[1], finalMasks[1], Size(img2width, img2height), INTER_NEAREST);

        warpAffine(finalMasks[0], finalMasks[0], translateMat1, Size(finalWidth, finalHeight));
        warpAffine(finalMasks[1], finalMasks[1], translateMat2, Size(finalWidth, finalHeight));

        Mat canvas(finalHeight, finalWidth, CV_8UC3);
        canvas.setTo(0);

        Mat blendMask(finalHeight, finalWidth, CV_8U);
        blendMask.setTo(0);

        bool isFirstCoordsDetected = false;
        vector<int> firstCoord;
        vector<int> lastCoord{ 0,0 };

        for (int i = 0; i < finalHeight; i++) {
            for (int j = 0; j < finalWidth; j++) {
                if (finalMasks[0].at<uchar>(i, j) != 0 && finalMasks[1].at<uchar>(i, j) != 0) {

                    //Sirina blendovanog dela
                    int blendWidth = 40;

                    if (j < xOffset + blendWidth) {
                        j = xOffset + blendWidth;
                    }

                    //Detektovanje prve koordinate da bi posle svi redovi pre prvog de
                    if (!isFirstCoordsDetected) {
                        firstCoord.push_back(i);
                        firstCoord.push_back(j);
                        isFirstCoordsDetected = true;
                    }

                    //Ovaj for iscrtava tu blendovanu sirinu
                    for (int k = 0; k < blendWidth; k++) {
                        float alpha = 1 - (static_cast<float>(k) * 1 / blendWidth);
                        blendMask.at<uchar>(i, j - k) = 255 * alpha;
                    }

                    //Ovaj for prolazi kroz ostatak reda i iscrtava ga belom bojom
                    for (int l = j; l < finalWidth; l++) {
                        blendMask.at<uchar>(i, l) = 255;
                    }

                    lastCoord[0] = i;
                    lastCoord[1] = j;

                    j = finalWidth;
                }
            }
        }

        //Prodji kroz prvih nekoliko redova koji su skroz crni i iskopiraj prvi red sa belim da bi se maska popunila
        for (int i = 0; i < firstCoord[0]; i++) {
            for (int j = 0; j < finalWidth; j++) {
                blendMask.at<uchar>(i, j) = blendMask.at<uchar>(firstCoord[0], j);
            }
        }

        //Prodji kroz poslednjih nekoliko redova koji su skroz crni i iskopiraj poslednji red sa belim da bi se maska popunila
        for (int i = lastCoord[0]; i < finalHeight; i++) {
            for (int j = 0; j < finalWidth; j++) {
                blendMask.at<uchar>(i, j) = blendMask.at<uchar>(lastCoord[0], j);
            }
        }

        GaussianBlur(blendMask, blendMask, Size(5, 5), 0);

        finalImages[0].convertTo(finalImages[0], CV_8UC3);
        finalImages[1].convertTo(finalImages[1], CV_8UC3);

        for (int i = 0; i < finalHeight; i++) {
            for (int j = 0; j < finalWidth; j++) {

                float alpha = blendMask.at<uchar>(i, j) / 256.0;

                canvas.at<Vec3b>(i, j)[0] = (int)(finalImages[1].at<Vec3b>(i, j)[0] * alpha + finalImages[0].at<Vec3b>(i, j)[0] * (1 - alpha));
                canvas.at<Vec3b>(i, j)[1] = (int)(finalImages[1].at<Vec3b>(i, j)[1] * alpha + finalImages[0].at<Vec3b>(i, j)[1] * (1 - alpha));
                canvas.at<Vec3b>(i, j)[2] = (int)(finalImages[1].at<Vec3b>(i, j)[2] * alpha + finalImages[0].at<Vec3b>(i, j)[2] * (1 - alpha));
            }
        }

        //imwrite("test2/MASK1.jpg", blendMask);

        return canvas;
    }

    Mat undistortImg2Rings(Mat canvas) {
        //Mat translateMatCanvas = (Mat_<double>(2, 3) << 1, 0, 150, 0, 1, 250);
        //warpAffine(canvas, canvas, translateMatCanvas, Size(canvas.cols + 300, canvas.rows + 500));

        Mat canvasUndistorted = undistortImage(canvas, -1.2);

        //Mat canvasCropped = canvasUndistorted(Rect(150, 100, canvasUndistorted.cols - 300, canvasUndistorted.rows - 200));

        return canvasUndistorted;
    }

    Mat undistortImg3Rings(Mat canvas) {
        Mat translateMatCanvas = (Mat_<double>(2, 3) << 1, 0, 0, 0, 1, 0);
        warpAffine(canvas, canvas, translateMatCanvas, Size(canvas.cols, canvas.rows));

        Mat canvasUndistorted = undistortImage(canvas, -0.6);

        //Mat canvasCropped = canvasUndistorted(Rect(150, 100, canvasUndistorted.cols - 300, canvasUndistorted.rows - 200));

        return canvasUndistorted;
    }

    Mat stitch(vector<Mat> inputImages) {

        double work_megapix = 0.6;
        double seam_megapix = 0.1;
        double compose_megapix = -1;
        float conf_thresh = 1.f;
        float match_conf = 0.3f;
        float blend_strength = 5;

        //Return this mat if there is some error
        Mat empty;

        // Check if have enough images
        int num_images = static_cast<int>(inputImages.size());
        if (num_images < 2)
        {
            std::cout << "Need more images\n";
            return empty;
        }

        double work_scale = 1, seam_scale = 1, compose_scale = 1;
        bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

        //(1) Create Feature Finder
        Ptr<Feature2D> finder = ORB::create();

        // (2) Read the image, scale it appropriately, and compute the feature description of the image
        Mat full_img, img;
        vector<ImageFeatures> features(num_images);
        vector<Mat> images(num_images);
        vector<Size> full_img_sizes(num_images);
        double seam_work_aspect = 1;

        for (int i = 0; i < num_images; ++i)
        {
            full_img = inputImages[i];
            full_img_sizes[i] = full_img.size();

            if (full_img.empty())
            {
                cout << "Can't open image " << i << std::endl;
                return empty;
            }
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale, INTER_CUBIC);

            if (!is_seam_scale_set)
            {
                seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
                seam_work_aspect = seam_scale / work_scale;
                is_seam_scale_set = true;
            }

            computeImageFeatures(finder, img, features[i]);
            features[i].img_idx = i;
            //std::cout << "Features in image #" << i + 1 << ": " << features[i].keypoints.size() << std::endl;

            resize(full_img, img, Size(), seam_scale, seam_scale, INTER_CUBIC);
            images[i] = img.clone();
        }

        full_img.release();
        img.release();


        // (3) Create an image feature matcher to calculate matching information
        vector<MatchesInfo> pairwise_matches;
        Ptr<FeaturesMatcher>  matcher = makePtr<BestOf2NearestMatcher>(false, match_conf);
        (*matcher)(features, pairwise_matches);
        matcher->collectGarbage();

        //! (4) Eliminate outliers and keep the most confident large ingredients
        // Leave only images we are sure are from the same panorama
        vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
        vector<Mat> img_subset;
        vector<String> img_names_subset;
        vector<Size> full_img_sizes_subset;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            //img_names_subset.push_back(img_names[indices[i]]);
            img_subset.push_back(images[indices[i]]);
            full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
        }

        images = img_subset;
        //img_names = img_names_subset;
        full_img_sizes = full_img_sizes_subset;

        // Check if we still have enough images
        num_images = static_cast<int>(images.size());
        if (num_images < 2)
        {
            std::cout << "Need more images\n";
            return empty;
        }

        //!(5) Estimate homography
        Ptr<Estimator> estimator = makePtr<HomographyBasedEstimator>();
        vector<CameraParams> cameras;
        if (!(*estimator)(features, pairwise_matches, cameras))
        {
            cout << "Homography estimation failed.\n";
            return empty;
        }

        for (size_t i = 0; i < cameras.size(); ++i)
        {
            Mat R;
            cameras[i].R.convertTo(R, CV_32F);
            cameras[i].R = R;
            //std::cout << "\nInitial camera intrinsics #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R << std::endl;
        }

        //(6) Create a Constraint Adjuster
        Ptr<detail::BundleAdjusterBase> adjuster = makePtr<detail::BundleAdjusterRay>();
        adjuster->setConfThresh(conf_thresh);
        Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
        refine_mask(0, 0) = 1;
        refine_mask(0, 1) = 1;
        refine_mask(0, 2) = 1;
        refine_mask(1, 1) = 1;
        refine_mask(1, 2) = 1;
        adjuster->setRefinementMask(refine_mask);
        if (!(*adjuster)(features, pairwise_matches, cameras))
        {
            cout << "Camera parameters adjusting failed.\n";
            return empty;
        }

        // Find median focal length
        vector<double> focals;
        for (size_t i = 0; i < cameras.size(); ++i)
        {
            focals.push_back(cameras[i].focal);
        }

        sort(focals.begin(), focals.end());
        float warped_image_scale;
        if (focals.size() % 2 == 1)
            warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
        else
            warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;


        //std::cout << "\nWarping images (auxiliary)... \n";

        vector<Point> corners(num_images);
        vector<UMat> masks_warped(num_images);
        vector<UMat> images_warped(num_images);
        vector<Size> sizes(num_images);
        vector<UMat> masks(num_images);

        // Preapre images masks
        for (int i = 0; i < num_images; ++i)
        {
            masks[i].create(images[i].size(), CV_8U);
            masks[i].setTo(Scalar::all(255));
        }

        // Warp images and their masks
        Ptr<WarperCreator> warper_creator = makePtr<cv::CylindricalWarper>();
        if (!warper_creator)
        {
            cout << "Can't create the warper \n";
            return empty;
        }

        //! Create RotationWarper
        Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

        //! Calculate warped corners/sizes/mask
        for (int i = 0; i < num_images; ++i)
        {
            Mat_<float> K;
            cameras[i].K().convertTo(K, CV_32F);
            float swa = (float)seam_work_aspect;
            K(0, 0) *= swa; K(0, 2) *= swa;
            K(1, 1) *= swa; K(1, 2) *= swa;
            corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_CUBIC, BORDER_REFLECT, images_warped[i]);
            sizes[i] = images_warped[i].size();
            warper->warp(masks[i], K, cameras[i].R, INTER_CUBIC, BORDER_CONSTANT, masks_warped[i]);
        }

        vector<UMat> images_warped_f(num_images);
        for (int i = 0; i < num_images; ++i)
            images_warped[i].convertTo(images_warped_f[i], CV_32F);

        //std::cout << "Compensating exposure... \n";

        //! Calculate exposure, adjust image exposure, reduce brightness differences
        Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
        if (dynamic_cast<BlocksCompensator*>(compensator.get()))
        {
            BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
            bcompensator->setNrFeeds(1);
            bcompensator->setNrGainsFilteringIterations(2);
            bcompensator->setBlockSize(32, 32);
        }

        compensator->feed(corners, images_warped, masks_warped);

        Ptr<SeamFinder> seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
        seam_finder->find(images_warped_f, corners, masks_warped);

        // Release unused memory
        images.clear();
        images_warped.clear();
        images_warped_f.clear();
        masks.clear();

        Mat img_warped, img_warped_s;
        Mat dilated_mask, seam_mask, mask, mask_warped;
        double compose_work_aspect = 1;

        vector<Mat> finalImages;
        vector<Mat> finalMasks;

        for (int img_idx = 0; img_idx < num_images; ++img_idx)
        {
            // Read image and resize it if necessary
            full_img = inputImages[img_idx];
            if (!is_compose_scale_set)
            {
                is_compose_scale_set = true;
                compose_work_aspect = compose_scale / work_scale;

                // Update warped image scale
                warped_image_scale *= static_cast<float>(compose_work_aspect);
                warper = warper_creator->create(warped_image_scale);

                // Update corners and sizes
                for (int i = 0; i < num_images; ++i)
                {
                    cameras[i].focal *= compose_work_aspect;
                    cameras[i].ppx *= compose_work_aspect;
                    cameras[i].ppy *= compose_work_aspect;

                    Size sz = full_img_sizes[i];
                    if (std::abs(compose_scale - 1) > 1e-1)
                    {
                        sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                        sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                    }

                    Mat K;
                    cameras[i].K().convertTo(K, CV_32F);
                    Rect roi = warper->warpRoi(sz, K, cameras[i].R);

                    corners[i] = roi.tl();
                    sizes[i] = roi.size();
                }
            }

            if (abs(compose_scale - 1) > 1e-1)
                resize(full_img, img, Size(), compose_scale, compose_scale, INTER_CUBIC);
            else
                img = full_img;
            full_img.release();
            Size img_size = img.size();

            Mat K, R;
            cameras[img_idx].K().convertTo(K, CV_32F);
            R = cameras[img_idx].R;

            // Warp the current image : img => img_warped
            warper->warp(img, K, cameras[img_idx].R, INTER_CUBIC, BORDER_REFLECT, img_warped);

            // Warp the current image mask
            mask.create(img_size, CV_8U);
            mask.setTo(Scalar::all(255));
            warper->warp(mask, K, cameras[img_idx].R, INTER_CUBIC, BORDER_CONSTANT, mask_warped);

            compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

            img_warped.convertTo(img_warped_s, CV_16S);
            img_warped.release();
            img.release();
            mask.release();

            dilate(masks_warped[img_idx], dilated_mask, Mat());
            resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_NEAREST);
            mask_warped = seam_mask & mask_warped;

            finalImages.push_back(img_warped_s);
            finalMasks.push_back(mask_warped);
        }

        /* ===========================================================================*/

        //cout << corners << endl;

        double xOffset = corners[1].x - corners[0].x;
        double yOffset = corners[1].y - corners[0].y;

        //cout << xOffset << "," << yOffset << endl;

        vector<Point> cornersNew;
        cornersNew.push_back(Point(0, 0));
        cornersNew.push_back(Point(xOffset, yOffset));

        double finalWidth = finalImages[1].cols + xOffset;
        double finalHeight = finalImages[1].rows + yOffset;

        //Povecanje rezolucije druge slike da bude ista kao canvas
        Mat translateMat1 = (Mat_<double>(2, 3) << 1, 0, 0, 0, 1, 0);
        warpAffine(finalImages[0], finalImages[0], translateMat1, Size(finalWidth, finalHeight));
        warpAffine(finalMasks[0], finalMasks[0], translateMat1, Size(finalWidth, finalHeight));

        //Translacija druge slike
        Mat translateMat2 = (Mat_<double>(2, 3) << 1, 0, xOffset, 0, 1, yOffset);
        warpAffine(finalImages[1], finalImages[1], translateMat2, Size(finalWidth, finalHeight));
        warpAffine(finalMasks[1], finalMasks[1], translateMat2, Size(finalWidth, finalHeight));

        Mat canvas(finalHeight, finalWidth, CV_8UC3);
        canvas.setTo(0);

        Mat blendMask(finalHeight, finalWidth, CV_8U);
        blendMask.setTo(0);

        bool isFirstCoordsDetected = false;
        vector<int> firstCoord;
        vector<int> lastCoord{ 0,0 };

        for (int i = 0; i < finalHeight; i++) {
            for (int j = 0; j < finalWidth; j++) {
                if (finalMasks[0].at<uchar>(i, j) != 0 && finalMasks[1].at<uchar>(i, j) != 0) {

                    //Sirina blendovanog dela
                    int blendWidth = 40;

                    if ( j < xOffset + blendWidth) {
                        j = xOffset + blendWidth;
                    }

                    //Detektovanje prve koordinate da bi posle svi redovi pre prvog de
                    if (!isFirstCoordsDetected) {
                        firstCoord.push_back(i);
                        firstCoord.push_back(j);
                        isFirstCoordsDetected = true;
                    }

                    //Ovaj for iscrtava tu blendovanu sirinu
                    for (int k = 0; k < blendWidth; k++) {
                        float alpha = 1 - (static_cast<float>(k) * 1/blendWidth);
                        blendMask.at<uchar>(i, j - k) = 255 * alpha;
                    }

                    //Ovaj for prolazi kroz ostatak reda i iscrtava ga belom bojom
                    for (int l = j ; l < finalWidth; l++) {
                        blendMask.at<uchar>(i, l) = 255;
                    }

                    lastCoord[0] = i;
                    lastCoord[1] = j;

                    j = finalWidth;
                }
            }
        }

        //Prodji kroz prvih nekoliko redova koji su skroz crni i iskopiraj prvi red sa belim da bi se maska popunila
        for (int i = 0; i < firstCoord[0]; i++) {
            for (int j = 0; j < finalWidth; j++) {
                blendMask.at<uchar>(i, j) = blendMask.at<uchar>(firstCoord[0], j);
            }
        }

        //Prodji kroz poslednjih nekoliko redova koji su skroz crni i iskopiraj poslednji red sa belim da bi se maska popunila
        for (int i = lastCoord[0]; i < finalHeight; i++) {
            for (int j = 0; j < finalWidth; j++) {
                blendMask.at<uchar>(i, j) = blendMask.at<uchar>(lastCoord[0], j);
            }
        }

        GaussianBlur(blendMask, blendMask, Size(3, 3), 0);

        finalImages[0].convertTo(finalImages[0], CV_8UC3);
        finalImages[1].convertTo(finalImages[1], CV_8UC3);

        for (int i = 0; i < finalHeight; i++) {
            for (int j = 0; j < finalWidth; j++) {

                float alpha = blendMask.at<uchar>(i, j) / 256.0;

                canvas.at<Vec3b>(i, j)[0] = (int)(finalImages[1].at<Vec3b>(i, j)[0] * alpha + finalImages[0].at<Vec3b>(i, j)[0] * (1 - alpha));
                canvas.at<Vec3b>(i, j)[1] = (int)(finalImages[1].at<Vec3b>(i, j)[1] * alpha + finalImages[0].at<Vec3b>(i, j)[1] * (1 - alpha));
                canvas.at<Vec3b>(i, j)[2] = (int)(finalImages[1].at<Vec3b>(i, j)[2] * alpha + finalImages[0].at<Vec3b>(i, j)[2] * (1 - alpha));
            }
        }

        return canvas;

        //finalWidth = canvas.cols;
        //finalHeight = canvas.rows;

        //double canvasExpandWidth = finalWidth * 0.2 * 0;
        //double canvasExpandHeight = finalHeight * 0.4 * 0;

        //Mat translateMatCanvas = (Mat_<double>(2, 3) << 1, 0, canvasExpandWidth/2, 0, 1, canvasExpandHeight/2);
        //warpAffine(canvas, canvas, translateMatCanvas, Size(finalWidth + canvasExpandWidth, finalHeight + canvasExpandHeight));

        //Mat canvasUndistorted = undistortImage(canvas, 0);

        //Mat canvasCropped = canvasUndistorted(Rect(150, 100, canvasUndistorted.cols - 300, canvasUndistorted.rows - 200));

        //return canvasUndistorted;

        //Ovo otkomentarisati ako hocu sliku sa alfom
        //
        //cvtColor(canvas, canvas, COLOR_RGB2RGBA);
        //Mat outputImageWithAlpha(finalHeight, finalWidth, CV_8UC4);
        //outputImageWithAlpha = cv::Scalar(255, 255, 255, 0);

        //Mat outputImageWithoutAlpha(finalHeight, finalWidth, CV_8UC3);
        //outputImageWithoutAlpha.setTo(0);

        //canvas.copyTo(outputImageWithoutAlpha, finalMask);

        //return canvas;

        //imwrite("test2/TEST1.png", finalImages[0]);
        //imwrite("test2/TEST2.png", finalImages[1]);
        //imwrite("test2/BLENDMASK.png", blendMask);
        //imwrite("test2/FINALMASK.png", finalMask);
    }

    //--------------------------------------------------------------


    int stitch_image(char* path) {

        string arguments(path);

        //Splitovanje argumenata i inicijalizacija promenljivih
        array<string, 4> tokens = splitArguments(arguments);

        string inputPath = tokens[0];
        int imagesPerRow = stoi(tokens[1]);
        int numberOfRows = stoi(tokens[2]);
        bool hasUltrawide;

        if (tokens[3] == "true") {
            hasUltrawide = true;
        }
        else {
            hasUltrawide = false;
        }

        //Definisanje enumeracija za putanje
        string pathInput = inputPath + "/src";
        string pathFinal = inputPath + "/panoFinal.jpg";
        string pathThumbnail = inputPath + "/thumbnail.jpg";

        //Definisanje finalne velicine jednog frejma(slike) unutar finalne panoramske slike
        int heightAfterResize = 1800;
        int widthAfterCrop = 1000;

        Mat finalPano(heightAfterResize, widthAfterCrop * imagesPerRow, CV_8UC3);
        finalPano.setTo(0);

        //Promenljiva koja govori da li je smer ucitavanja obrnut posto se slika u paternu gore-dole-dole-gore
        bool reversed = false;

        //Prolazak kroz sve slike gde prvo sticuje sve slike po vertikali i dodaje ih u niz
        for (int i = 0; i < imagesPerRow * numberOfRows; i += numberOfRows) {

            vector<Mat> imgsToStitch;

            //Proverava da li je smer ucitavanja obrnut posto se slika u paternu gore-dole-dole-gore
            //i onda na svake dve ucitane slike obrce smer ucitavanja pa se slike ucitavaju 1-2-4-3-5-6-8-7
            if (!reversed) {
                for (int j = 0; j < numberOfRows; j++) {
                    Mat img = imread(pathInput + "/" + to_string(i + j) + ".jpg");

                    if (img.rows > img.cols) {
                        rotate(img, img, ROTATE_90_COUNTERCLOCKWISE);
                    }

                    imgsToStitch.push_back(img);
                }
                reversed = true;
            }
            else {
                for (int j = numberOfRows - 1; j >= 0; j--) {
                    Mat img = imread(pathInput + "/" + to_string(i + j) + ".jpg");

                    if (img.rows > img.cols) {
                        rotate(img, img, ROTATE_90_COUNTERCLOCKWISE);
                    }

                    imgsToStitch.push_back(img);
                }
                reversed = false;
            }

            //Deo za sticovanje

            Mat stitchedImg;

            if (hasUltrawide) {
                stitchedImg = stitch(imgsToStitch);
            }
            else {

                vector<Mat> imgsToStitchTEMP;

                imgsToStitchTEMP.push_back(imgsToStitch[0]);
                imgsToStitchTEMP.push_back(imgsToStitch[1]);

                Mat stitchedImgTEMP = stitch(imgsToStitchTEMP);

                if (!stitchedImgTEMP.empty()) {
                    vector<Mat> imgsToStitchTEMP2;
                    imgsToStitchTEMP2.push_back(stitchedImgTEMP);
                    imgsToStitchTEMP2.push_back(imgsToStitch[2]);
                    stitchedImg = stitch(imgsToStitchTEMP2);
                }
            }

            //Ako ne moze da sticuje, nek proba da blenduje

            if (stitchedImg.empty()) {

                Mat imgBlend0, imgBlend1, imgBlend2;

                imgBlend0 = distortImage(imgsToStitch[0], 0.5, 5);
                imgBlend1 = distortImage(imgsToStitch[1], 0.5, 5);

                if (hasUltrawide) {
                    Mat stitchedImgTEMP = blendVertical(imgBlend0, imgBlend1, 370);
                    stitchedImg = undistortImg2Rings(stitchedImgTEMP);
                }
                else {
                    imgBlend2 = distortImage(imgsToStitch[2], 0.5, 5);
                    Mat stitchedImgTEMP = blendVertical(imgBlend0, imgBlend1, 420);
                    Mat stitchedImgTEMP2 = blendVertical(stitchedImgTEMP, imgBlend2, 800);
                    stitchedImg = undistortImg3Rings(stitchedImgTEMP2);
                }
                /*
                Mat curvedImgTop, curvedImgMiddle, curvedImgBottom;
                if (!reversed) {
                    curvedImgTop = distortImage(imgsToStitch[0], 0.5, 5);
                    curvedImgMiddle = distortImage(imgsToStitch[1], 0.5, 5);
                    if (!hasUltrawide) {
                        curvedImgBottom = distortImage(imgsToStitch[2], 0.5, 5);
                    }
                }
                else {
                    if (!hasUltrawide) {
                        curvedImgTop = distortImage(imgsToStitch[0], 0.5, 5);
                    }
                    curvedImgMiddle = distortImage(imgsToStitch[1], 0.5, 5);
                    curvedImgBottom = distortImage(imgsToStitch[2], 0.5, 5);
                }

                if (!reversed) {
                    if (hasUltrawide) {
                        Mat stitchedImgTEMP = blendVertical(imgsToStitch[0], imgsToStitch[1], 370);
                        stitchedImg = undistortImg2Rings(stitchedImgTEMP);
                    }
                    else {
                        Mat stitchedImgTEMP = blendVertical(imgsToStitch[0], imgsToStitch[1], 420);
                        Mat stitchedImgTEMP2 = blendVertical(stitchedImgTEMP, imgsToStitch[2], 800);
                        stitchedImg = undistortImg3Rings(stitchedImgTEMP2);
                    }
                }
                else {
                    if (hasUltrawide) {
                        Mat stitchedImgTEMP = blendVertical(imgsToStitch[0], imgsToStitch[1], 370);
                        stitchedImg = undistortImg2Rings(stitchedImgTEMP);
                    }
                    else {
                        Mat stitchedImgTEMP = blendVertical(curvedImgMiddle, curvedImgBottom, 420);
                        Mat stitchedImgTEMP2 = blendVertical(curvedImgTop, stitchedImgTEMP, 380);
                        stitchedImg = undistortImg3Rings(stitchedImgTEMP2);
                    }
                }*/
            }

            if (stitchedImg.cols > stitchedImg.rows) {
                rotate(stitchedImg, stitchedImg, ROTATE_90_CLOCKWISE);
            }

            int width = stitchedImg.cols;
            int height = stitchedImg.rows;


            if (i == 0) {
                imwrite(pathThumbnail, stitchedImg);
            }

            double scaleFactor = static_cast<double>(heightAfterResize) / height;

            resize(stitchedImg, stitchedImg, Size(width * scaleFactor, heightAfterResize), INTER_CUBIC);

            int newWidth = stitchedImg.cols;

            if (newWidth < widthAfterCrop) {
                if (hasUltrawide) {
                    resize(stitchedImg, stitchedImg, Size(widthAfterCrop, heightAfterResize), INTER_CUBIC);
                }
                else {
                    //TODO Dodati kod ovde ako nema ultrawide a slika je uza nego sto je predvidjeno
                    resize(stitchedImg, stitchedImg, Size(widthAfterCrop, heightAfterResize), INTER_CUBIC);
                }
            }
            else {
                int cropStart = (newWidth - widthAfterCrop) / 2;
                Mat stitchedImgTemp = stitchedImg(Rect(cropStart, 0, widthAfterCrop, heightAfterResize));
                stitchedImg = stitchedImgTemp.clone();
            }

            Mat stitchedImgROI(finalPano, Rect(i / numberOfRows * widthAfterCrop, 0, stitchedImg.cols, stitchedImg.rows));
            stitchedImg.copyTo(stitchedImgROI);
        }

        imwrite(pathFinal, finalPano, { IMWRITE_JPEG_QUALITY, 60 });

        __android_log_print(ANDROID_LOG_VERBOSE,"+++++++++++++++++OPENCV+++++++++++++++++", "ZAVRSENO!!!");

        return 0;

        //-----------------------------------------------------------------------------------------
    }
}