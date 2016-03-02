#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "IPUtils.h"
#include "Sherwood.h"
#include "DataPointCollection.h"
#include "Classification.h"
#include "Regression.h"

using namespace std;
using namespace MicrosoftResearch::Cambridge::Sherwood;

#define LOOP_DELAY 30
#define DEF_REG_LEVELS 20
#define DEF_CLASS_LEVELS 22

#ifdef _WIN32
const std::string FILE_PATH = "D:\\";
#endif
#ifdef __linux__
const std::string FILE_PATH = "/media/james/data_wd/";
#endif

int trainClassificationPar(ProgramParameters& progParams)
{
    if(progParams.TrainingImagesPath.back() != '/')
        progParams.TrainingImagesPath += "/";

    std::string filename = progParams.TrainingImagesPath + progParams.OutputFilename + "_classifier.frst";
     
    std::cout << "Searching for some IR and depth images in " << progParams.TrainingImagesPath << std::endl;
    
    //std::unique_ptr<DataPointCollection> training_data = DataPointCollection::LoadImagesClass(progParams);
    std::unique_ptr<DataPointCollection> training_data = DataPointCollection::LoadImages(progParams, true);
    
    int images = training_data->CountImages();
    std::cout << "Data loaded from images: " << std::to_string(images) << std::endl;
    std::cout << "of size: " << std::to_string(progParams.ImgHeight * progParams.ImgWidth) << std::endl;
    std::cout << "Total points:" << std::to_string(training_data->Count()) << std::endl;
    std::cout << (training_data->low_memory? "Low Memory Implementation" : "Inefficient Memory Implementation") << std::endl;
 
    std::cout << "\nAttempting training" << std::endl;
    try
    {
        std::unique_ptr<Forest<PixelSubtractionResponse, HistogramAggregator> > forest = 
            Classifier<PixelSubtractionResponse>::TrainPar(*training_data, progParams.Tpc);

        forest->Serialize(filename);
        std::cout << "Training complete, forest saved in :" << filename << std::endl;
    }
    catch (const std::runtime_error& e)
    {
        std::cout << "Training Failed" << std::endl;
        std::cerr << e.what() << std::endl;
    }

    return 0;
}

int trainRegressionPar(ProgramParameters& progParams, int class_expert_no = -1)
{
    std::string file_suffix;
    
    if(class_expert_no != -1)
        file_suffix = "_expert" + std::to_string(class_expert_no) + ".frst";
    else
        file_suffix = "_regressor.frst";

    if(progParams.TrainingImagesPath.back() != '/')
        progParams.TrainingImagesPath += "/";

    std::string filename = progParams.TrainingImagesPath + progParams.OutputFilename + file_suffix;

    std::cout << "Searching for some IR and depth images in " << progParams.TrainingImagesPath << std::endl;

    
    // create a DataPointCollection in the regression format
    //std::unique_ptr<DataPointCollection> training_data = DataPointCollection::LoadImagesRegression(progParams, class_expert_no); 
    std::unique_ptr<DataPointCollection> training_data = DataPointCollection::LoadImages(progParams, false, class_expert_no); 

    int images = training_data->CountImages();
    std::cout << "Data loaded from images: " << std::to_string(images) << std::endl;
    std::cout << "of size: " << std::to_string(progParams.ImgHeight * progParams.ImgWidth) << std::endl;
    std::cout << "Total points:" << std::to_string(training_data->Count()) << std::endl;
    std::cout << (training_data->low_memory? "Low Memory Implementation" : "Inefficient Memory Implementation") << std::endl;

    // Train a regressoin forest
    std::cout << "\nAttempting training" << std::endl;
    try
    {
        std::unique_ptr<Forest<PixelSubtractionResponse, DiffEntropyAggregator> > forest =
            Regressor<PixelSubtractionResponse>::TrainPar(*training_data, progParams.Tpr);

        forest->Serialize(filename);
        std::cout << "Training complete, forest saved in :" << filename << std::endl;
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << "Training Failed" << std::endl;
        std::cerr << e.what() << std::endl;
    }

    return 0;
}

int regressOnline(std::string dir_path)
{
    if (!IPUtils::dirExists(dir_path))
        return 0;

    if(dir_path.back() != '/')
        dir_path += "/";

    std::cout << "Looking in:\t" << dir_path << std::endl << "Filename?\t";
    std::string filename;
    std::cin >> filename;
    std::string forest_path = dir_path + filename;
    std::cout << "Attempting to deserialize forest from " << forest_path << std::endl;

    std::string img_path = dir_path + "img";
    std::string pathstring;
    cv::Mat test_image = cv::Mat(cv::Size(640, 480), CV_8UC1);
    std::vector<uint16_t> reg_result;
    cv::Mat reg_mat;
    cv::Mat result_thresh(480, 640, CV_16UC1);

    // load forest
    std::unique_ptr<Forest<PixelSubtractionResponse, DiffEntropyAggregator> > forest =
        Forest<PixelSubtractionResponse, DiffEntropyAggregator>::Deserialize(forest_path);
    // Create ForestShared from loaded forest
    std::unique_ptr<ForestShared<PixelSubtractionResponse, DiffEntropyAggregator> > forest_shared =
        ForestShared<PixelSubtractionResponse, DiffEntropyAggregator>::ForestSharedFromForest(*forest);
    // Delete original forest. May roll these steps into one later if we don't need a regular forest application.
    forest->~Forest();
    forest.release();

    for (int i = 0; i < 106; i++)
    {
        int64 start_time = cv::getTickCount();
        pathstring = img_path + std::to_string(i) + "ir.png";
        test_image = cv::imread(pathstring, -1);
        if (!test_image.data)
            continue;

        std::unique_ptr<DataPointCollection> test_data1 = DataPointCollection::LoadMat(test_image, cv::Size(640, 480));
        reg_result = Regressor<PixelSubtractionResponse>::ApplyMat(*forest_shared, *test_data1);

        reg_mat = cv::Mat(480, 640, CV_16UC1, (uint16_t*)reg_result.data());
        IPUtils::threshold16(reg_mat, result_thresh, 1201, 65535, 4);
        result_thresh.convertTo(result_thresh, CV_16U, 54);
        cv::imshow("output", result_thresh);
        int64 process_time = (((cv::getTickCount() - start_time) / cv::getTickFrequency()) * 1000);
        std::cout << "Process time: " << std::to_string(process_time) << std::endl;
        int wait_time = std::max(2, (int)(LOOP_DELAY - process_time));
        if(cv::waitKey(wait_time) >= 0)
            break;
    }
    cv::startWindowThread();
    cv::destroyAllWindows();
    return 0;
}

int classifyOnline(std::string dir_path)
{
    if (!IPUtils::dirExists(dir_path))
        return 0;

    if(dir_path.back() != '/')
        dir_path += "/";

    std::cout << "Looking in:\t" << dir_path << std::endl << "Filename?\t";
    std::string filename;
    std::cin >> filename;
    std::string forest_path = dir_path + filename;
    std::cout << "Attempting to deserialize forest from " << forest_path << std::endl;

    std::string img_path = dir_path + "img";
    std::string pathstring;
    cv::Mat test_image = cv::Mat(cv::Size(640, 480), CV_8UC1);
    cv::Mat bins_mat;
    cv::Mat result_norm1;

    // load forest
    std::unique_ptr<Forest<PixelSubtractionResponse, HistogramAggregator> > forest =
        Forest<PixelSubtractionResponse, HistogramAggregator>::Deserialize(forest_path);
    // Create ForestShared from loaded forest
    std::unique_ptr<ForestShared<PixelSubtractionResponse, HistogramAggregator> > forest_shared =
        ForestShared<PixelSubtractionResponse, HistogramAggregator>::ForestSharedFromForest(*forest);
    // Delete original forest. May roll these steps into one later if we don't need a regular forest application.
    forest->~Forest();
    forest.release();

    for (int i = 0; i < 106; i++)
    {
        int64 start_time = cv::getTickCount();
        pathstring = img_path + std::to_string(i) + "ir.png";
        test_image = cv::imread(pathstring, -1);
        if (!test_image.data)
            continue;

        std::unique_ptr<DataPointCollection> test_data1 = DataPointCollection::LoadMat(test_image, cv::Size(640, 480));
        bins_mat = Classifier<PixelSubtractionResponse>::ApplyMat(*forest_shared, *test_data1);
        std::vector<uchar> bins_vec = IPUtils::vectorFromBins(bins_mat, cv::Size(640, 480));
        cv::Mat result_mat1 = cv::Mat(480, 640, CV_8UC1, (int8_t*)bins_vec.data());
        result_mat1.convertTo(result_norm1, CV_8U, 63);
        cv::imshow("output", result_norm1);
        int64 process_time = (((cv::getTickCount() - start_time) / cv::getTickFrequency()) * 1000);
        std::cout << "Process time: " << std::to_string(process_time) << std::endl;
        int wait_time = std::max(2, (int)(LOOP_DELAY - process_time));
        if(cv::waitKey(wait_time) >= 0)
            break;
    }
    cv::startWindowThread();
    cv::destroyAllWindows();
    
    return 0;
}

int applyMultiLevel(std::string forest_path)
{
    if(forest_path.back() != '/')
        forest_path += "/";

    std::cout << "Looking in:\t" << forest_path << std::endl << "File prefix?\t";
    std::string filename;
    std::cin >> filename;

    std::string class_path = forest_path + filename + "_classifier.frst";
    std::string e_path[] = {forest_path + filename + "_expert1.frst",
                            forest_path + filename + "_expert2.frst",
                            forest_path + filename + "_expert3.frst",
                            forest_path + filename + "_expert4.frst"};
    std::vector<std::string> expert_path (e_path, e_path+4);
    std::vector<std::unique_ptr<ForestShared<PixelSubtractionResponse, DiffEntropyAggregator> > > experts;
    std::unique_ptr<ForestShared<PixelSubtractionResponse, HistogramAggregator> > classifier;
    int bins = 5;

    // Load the classifier and expert regressors.
    try{
        std::cout << "Loading classifier" << std::endl;
        // load classifier
        std::unique_ptr<Forest<PixelSubtractionResponse, HistogramAggregator> > c_forest =
            Forest<PixelSubtractionResponse, HistogramAggregator>::Deserialize(class_path);
        // Create ForestShared from loaded forest
        std::unique_ptr<ForestShared<PixelSubtractionResponse, HistogramAggregator> > c_forest_shared =
            ForestShared<PixelSubtractionResponse, HistogramAggregator>::ForestSharedFromForest(*c_forest);
        // Delete original forest. May roll these steps into one later if we don't need a regular forest application.
        c_forest->~Forest();
        c_forest.release();
        classifier = move(c_forest_shared);
        std::cout << "Classifier loaded with " << std::to_string(classifier->TreeCount()) << " trees" << std::endl;
        for(int i=0;i<bins;i++)
        {
            std::cout << "Loading expert " << std::to_string(i+1) << std::endl;
            std::unique_ptr<Forest<PixelSubtractionResponse, DiffEntropyAggregator> > e_forest =
                Forest<PixelSubtractionResponse, DiffEntropyAggregator>::Deserialize(expert_path[i]);
            // Create ForestShared from loaded forest
            std::unique_ptr<ForestShared<PixelSubtractionResponse, DiffEntropyAggregator> > e_forest_shared =
                ForestShared<PixelSubtractionResponse, DiffEntropyAggregator>::ForestSharedFromForest(*e_forest);
            // Delete original forest. May roll these steps into one later if we don't need a regular forest application.
            e_forest->~Forest();
            e_forest.release();
            std::cout << "Expert loaded with " << std::to_string(e_forest_shared->TreeCount()) << " trees" << std::endl;
            experts.push_back(std::move(e_forest_shared));
        }
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Forest loading Failed" << std::endl;
        std::cerr << e.what() << std::endl;
    }


    // Load up an image, apply it to classifier to get weights
    // apply it to all experts, then ans = weighted sum
    std::string img_path = "/media/james/data_wd/training_realsense/img";
    std::string pathstring;
    std::string pathstringd;
    cv::Mat test_image;
    cv::Mat depth_image;
    cv::Mat result_thresh(480, 640, CV_16UC1);
    cv::Mat depth_thresh(480, 640, CV_16UC1);
    cv::Mat bins_mat;
    cv::Mat reg_mat;
    std::vector<float> weights_vec(bins-1);
    
    cv::namedWindow("output");
    cv::namedWindow("depth");

    for (int i = 0; i < 106; i++)
    {
        int64 start_time = cv::getTickCount();
        pathstring = img_path + std::to_string(i) + "ir.png";
        test_image = cv::imread(pathstring, -1);
        pathstringd = img_path + std::to_string(i) + "depth.png";
        depth_image = cv::imread(pathstringd, -1);
        if (!test_image.data)
            continue;

        
        std::unique_ptr<DataPointCollection> test_data1 = DataPointCollection::LoadMat(test_image, cv::Size(640, 480));
        bins_mat = Classifier<PixelSubtractionResponse>::ApplyMat(*classifier, *test_data1);
        // Get the weights for weighted sum from  classifiaction results. 
        // Essentially represents the probability for any pixel in the image to be in 
        // a certain bin.
        std::vector<uchar> bins_vec = IPUtils::vectorFromBins(bins_mat, cv::Size(640, 480));
        weights_vec = IPUtils::weightsFromBins(bins_mat, cv::Size(640,480), false);
        std::vector<uint16_t> sum_weighted_output((640*480), 0);

        for(int j=0;j<bins-1;j++)
        {
            std::vector<uint16_t> expert_output = Regressor<PixelSubtractionResponse>::ApplyMat(*experts[j], *test_data1);
            for(int k=0;k<expert_output.size();k++)
            {
                if(bins_vec[k] == 0)
                    continue;
                else
                    sum_weighted_output[k] = sum_weighted_output[k] + uint16_t(expert_output[k] * weights_vec[j]);
            }
        }
        reg_mat = cv::Mat(480, 640, CV_16UC1, (uint16_t*)sum_weighted_output.data());
        IPUtils::threshold16(reg_mat, result_thresh, 1201, 65535, 4);
        result_thresh.convertTo(result_thresh, CV_16U, 54);
        IPUtils::threshold16(depth_image, depth_thresh, 1201, 65535, 4);
        depth_thresh.convertTo(depth_thresh, CV_16U, 54);
        cv::imshow("output", result_thresh);
        cv::imshow("depth", depth_thresh);
        
        int64 process_time = (((cv::getTickCount() - start_time) / cv::getTickFrequency()) * 1000);
        std::cout << "Process time: " << std::to_string(process_time) << std::endl;
        int wait_time = std::max(2, (int)(LOOP_DELAY - process_time));
        if(cv::waitKey(wait_time) >= 0)
            break;
    }
    cv::startWindowThread();
    cv::destroyAllWindows();
    
    return 0;
    
}

int testForestAlternate(std::string forest_path,
    std::string forest_prefix,
    std::string test_image_path,
    std::string test_image_prefix,
    int num_images)
{
    if(!IPUtils::dirExists(forest_path))
        throw std::runtime_error("Failed to find forest directory:" + forest_path);

    if(!IPUtils::dirExists(test_image_path))
        throw std::runtime_error("Failed to find test image directory:" + test_image_path);        

    // check path ends in '/'
    if(forest_path.back() != '/')
        forest_path += "/";
    if(test_image_path.back() != '/')
        test_image_path += "/";

    // Initialize classification forest path string and vector of expert forest class strings
    std::string class_path = forest_path + forest_prefix + "_classifier.frst";
    std::string e_path[] = {forest_path + forest_prefix + "_expert0.frst",
                            forest_path + forest_prefix + "_expert1.frst",
                            forest_path + forest_prefix + "_expert2.frst",
                            forest_path + forest_prefix + "_expert3.frst",
                            forest_path + forest_prefix + "_expert4.frst"};
    std::vector<std::string> expert_path (e_path, e_path+5);

     // Init a vector of pointers to forests... essentially a vector of expert regressors
    std::vector<std::unique_ptr<ForestShared<PixelSubtractionResponse, DiffEntropyAggregator> > > experts;
    // Init a pointer to a classifier
    std::unique_ptr<ForestShared<PixelSubtractionResponse, HistogramAggregator> > classifier;
    int bins = 5;

    // Load the classifier and expert regressors.
    try
    {
        std::cout << "Loading classifier" << std::endl;
        // load classifier
        std::unique_ptr<Forest<PixelSubtractionResponse, HistogramAggregator> > c_forest =
            Forest<PixelSubtractionResponse, HistogramAggregator>::Deserialize(class_path);
        // Create ForestShared from loaded forest
        std::unique_ptr<ForestShared<PixelSubtractionResponse, HistogramAggregator> > c_forest_shared =
            ForestShared<PixelSubtractionResponse, HistogramAggregator>::ForestSharedFromForest(*c_forest);
        // Delete original forest. May roll these steps into one later if we don't need a regular forest application.
        c_forest->~Forest();
        c_forest.release();
        classifier = move(c_forest_shared);
        std::cout << "Classifier loaded with " << std::to_string(classifier->TreeCount()) << " trees" << std::endl;
        for(int i=0;i<bins;i++)
        {
            std::cout << "Loading expert " << std::to_string(i) << std::endl;
            std::unique_ptr<Forest<PixelSubtractionResponse, DiffEntropyAggregator> > e_forest =
                Forest<PixelSubtractionResponse, DiffEntropyAggregator>::Deserialize(expert_path[i]);
            // Create ForestShared from loaded forest
            std::unique_ptr<ForestShared<PixelSubtractionResponse, DiffEntropyAggregator> > e_forest_shared =
                ForestShared<PixelSubtractionResponse, DiffEntropyAggregator>::ForestSharedFromForest(*e_forest);
            // Delete original forest. May roll these steps into one later if we don't need a regular forest application.
            e_forest->~Forest();
            e_forest.release();
            std::cout << "Expert loaded with " << std::to_string(e_forest_shared->TreeCount()) << " trees" << std::endl;
            experts.push_back(std::move(e_forest_shared));
        }
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Forest loading Failed" << std::endl;
        std::cerr << e.what() << std::endl;
    }
    // TODO: change this prefix from img to test
    std::string img_path = test_image_path+test_image_prefix;
    std::string img_full_path;
    std::string depth_path = test_image_path+test_image_prefix;
    std::string depth_full_path;
    cv::Mat sse_t = cv::Mat::zeros(480, 640, CV_32SC1);
    cv::Mat err_t = cv::Mat::zeros(480, 640, CV_32SC1);
    cv::Mat msse_t(480, 640, CV_32SC1);
    cv::Mat sse_nt = cv::Mat::zeros(480, 640, CV_32SC1);
    cv::Mat err_nt = cv::Mat::zeros(480, 640, CV_32SC1);
    cv::Mat msse_nt(480, 640, CV_32SC1);
    cv::Mat depth_image(480, 640, CV_16UC1);
    cv::Mat test_image(480, 640, CV_8UC1);
    cv::Mat reg_mat = cv::Mat::zeros(480, 640, CV_16UC1);
    cv::Mat bins_mat;
    cv::Mat result_thresh(480, 640, CV_16UC1);
    cv::Mat depth_thresh(480, 640, CV_16UC1);
    cv::Mat temp_mat(480, 640, CV_16UC1);
    std::vector<float> weights_vec(bins-1);
    int images_processed = 0;

    for(int i=0;i<num_images;i++)
    {
        //TODO change this so it's right 
        int image_index = i * 5;
        img_full_path = img_path + std::to_string(image_index) + "ir.png";
        depth_full_path = depth_path + std::to_string(image_index) + "depth.png";
        
        test_image = cv::imread(img_full_path, -1);
        depth_image = cv::imread(depth_full_path, -1);

        if((!depth_image.data)||(!test_image.data))
        {
            std::cerr << "Error loading images:\n\t" << img_full_path << "\n\t" << depth_full_path << std::endl;
            continue;
        }

        // Pre process the test image (need to do this for the alternate training where we don't use zero inputs)
        // TODO get some parameters into this
        test_image = IPUtils::preProcess(test_image);

        std::unique_ptr<DataPointCollection> test_data1 = DataPointCollection::LoadMat(test_image, cv::Size(640, 480));
        bins_mat = Classifier<PixelSubtractionResponse>::ApplyMat(*classifier, *test_data1);
        // Get the weights for weighted sum from  classifiaction results. 
        // Essentially represents the probability for any pixel in the image to be in 
        // a certain bin.
        std::vector<uchar> bins_vec = IPUtils::vectorFromBins(bins_mat, cv::Size(640, 480));
        weights_vec = IPUtils::weightsFromBins(bins_mat, cv::Size(640,480), true);
        std::vector<uint16_t> sum_weighted_output((640*480), 0);

        for(int j=0;j<bins;j++)
        {
            std::vector<uint16_t> expert_output = Regressor<PixelSubtractionResponse>::ApplyMat(*experts[j], *test_data1);
            for(int k=0;k<expert_output.size();k++)
            {
                if(bins_vec[k] == 0)
                    continue;
                else
                    sum_weighted_output[k] = sum_weighted_output[k] + uint16_t(expert_output[k] * weights_vec[j]);
            }
        }

        int output_index = 0;
        for(int r=0;r<640;r++)
        {
            uchar* test_image_pix = test_image.ptr<uchar>(r);
            uint16_t* reg_mat_pix = reg_mat.ptr<uint16_t>(r);
            for(int c=0;c<480;c++)
            {
                if(test_image_pix[c] == 0)
                {
                    continue;
                }
                else
                {
                    // this might fail if we go out of bounds somehow.
                    reg_mat_pix[c] = sum_weighted_output[output_index];
                    output_index++;
                }
            }
        }

        IPUtils::threshold16(reg_mat, result_thresh, 1200, 65535, 4);
        IPUtils::threshold16(depth_image, depth_thresh, 1200, 65535, 4);
        err_nt = IPUtils::getError(depth_image, reg_mat);
        err_t = IPUtils::getError(depth_thresh, result_thresh); 
        // result_thresh.convertTo(result_thresh, CV_16U, 54);
        // depth_thresh.convertTo(depth_thresh, CV_16U, 54);
        // cv::imshow("error_nthresh", err_nt);
        // cv::imshow("error_thresh", err_t);
        // cv::imshow("depth", depth_thresh);
        // cv::imshow("result", result_thresh);
        // cv::waitKey(30);
        sse_t = sse_t + err_t;
        sse_nt = sse_nt + err_nt;
        images_processed++;
    }

    float alpha = 1.0 / images_processed;
    msse_t = sse_t * alpha;
    msse_nt = sse_nt * alpha;
    cv::Scalar mean_nt, mean_t;
    cv::Scalar std_dev_nt, std_dev_t;
    cv::meanStdDev(msse_t, mean_t, std_dev_t);
    cv::meanStdDev(msse_nt, mean_nt, std_dev_nt);
    std::cout << "Mean (not thresholded): " << std::to_string(mean_nt[0]) << std::endl;
    std::cout << "Std dev (not thresholded): " << std::to_string(std_dev_nt[0]) << std::endl;
    std::cout << "Mean (thresholded): " << std::to_string(mean_t[0]) << std::endl;
    std::cout << "Std dev (thresholded): " << std::to_string(std_dev_t[0]) << std::endl;
}

int testForest(std::string forest_path,
    std::string forest_prefix,
    std::string test_image_path,
    std::string test_image_prefix,
    int num_images)
{
    if(!IPUtils::dirExists(forest_path))
        throw std::runtime_error("Failed to find forest directory:" + forest_path);

    if(!IPUtils::dirExists(test_image_path))
        throw std::runtime_error("Failed to find test image directory:" + test_image_path);        

    // check path ends in '/'
    if(forest_path.back() != '/')
        forest_path += "/";
    if(test_image_path.back() != '/')
        test_image_path += "/";

    // Initialize classification forest path string and vector of expert forest class strings
    std::string class_path = forest_path + forest_prefix + "_classifier.frst";
    std::string e_path[] = {forest_path + forest_prefix + "_expert1.frst",
                            forest_path + forest_prefix + "_expert2.frst",
                            forest_path + forest_prefix + "_expert3.frst",
                            forest_path + forest_prefix + "_expert4.frst"};
    std::vector<std::string> expert_path (e_path, e_path+4);

    // Init a vector of pointers to forests... essentially a vector of expert regressors
    std::vector<std::unique_ptr<ForestShared<PixelSubtractionResponse, DiffEntropyAggregator> > > experts;
    // Init a pointer to a classifier
    std::unique_ptr<ForestShared<PixelSubtractionResponse, HistogramAggregator> > classifier;
    int bins = 5;

    // Load the classifier and expert regressors.
    try
    {
        std::cout << "Loading classifier" << std::endl;
        // load classifier
        std::unique_ptr<Forest<PixelSubtractionResponse, HistogramAggregator> > c_forest =
            Forest<PixelSubtractionResponse, HistogramAggregator>::Deserialize(class_path);
        // Create ForestShared from loaded forest
        std::unique_ptr<ForestShared<PixelSubtractionResponse, HistogramAggregator> > c_forest_shared =
            ForestShared<PixelSubtractionResponse, HistogramAggregator>::ForestSharedFromForest(*c_forest);
        // Delete original forest. May roll these steps into one later if we don't need a regular forest application.
        c_forest->~Forest();
        c_forest.release();
        classifier = move(c_forest_shared);
        std::cout << "Classifier loaded with " << std::to_string(classifier->TreeCount()) << " trees" << std::endl;
        for(int i=0;i<bins-1;i++)
        {
            std::cout << "Loading expert " << std::to_string(i+1) << std::endl;
            std::unique_ptr<Forest<PixelSubtractionResponse, DiffEntropyAggregator> > e_forest =
                Forest<PixelSubtractionResponse, DiffEntropyAggregator>::Deserialize(expert_path[i]);
            // Create ForestShared from loaded forest
            std::unique_ptr<ForestShared<PixelSubtractionResponse, DiffEntropyAggregator> > e_forest_shared =
                ForestShared<PixelSubtractionResponse, DiffEntropyAggregator>::ForestSharedFromForest(*e_forest);
            // Delete original forest. May roll these steps into one later if we don't need a regular forest application.
            e_forest->~Forest();
            e_forest.release();
            std::cout << "Expert loaded with " << std::to_string(e_forest_shared->TreeCount()) << " trees" << std::endl;
            experts.push_back(std::move(e_forest_shared));
        }
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Forest loading Failed" << std::endl;
        std::cerr << e.what() << std::endl;
    }
    // TODO: change this prefix from img to test
    std::string img_path = test_image_path+test_image_prefix;
    std::string img_full_path;
    std::string depth_path = test_image_path+test_image_prefix;
    std::string depth_full_path;
    cv::Mat sse_t = cv::Mat::zeros(480, 640, CV_32SC1);
    cv::Mat err_t = cv::Mat::zeros(480, 640, CV_32SC1);
    cv::Mat msse_t(480, 640, CV_32SC1);
    cv::Mat sse_nt = cv::Mat::zeros(480, 640, CV_32SC1);
    cv::Mat err_nt = cv::Mat::zeros(480, 640, CV_32SC1);
    cv::Mat msse_nt(480, 640, CV_32SC1);
    cv::Mat depth_image(480, 640, CV_16UC1);
    cv::Mat test_image(480, 640, CV_8UC1);
    cv::Mat reg_mat;
    cv::Mat bins_mat;
    cv::Mat result_thresh(480, 640, CV_16UC1);
    cv::Mat depth_thresh(480, 640, CV_16UC1);
    std::vector<float> weights_vec(bins-1);
    int images_processed = 0;

    for(int i=0;i<num_images;i++)
    {
        int image_index = i * 5;
        img_full_path = img_path + std::to_string(image_index) + "ir.png";
        depth_full_path = depth_path + std::to_string(image_index) + "depth.png";
        
        test_image = cv::imread(img_full_path, -1);
        depth_image = cv::imread(depth_full_path, -1);

        if((!depth_image.data)||(!test_image.data))
        {
            std::cerr << "Error loading images:\n\t" << img_full_path << "\n\t" << depth_full_path << std::endl;
            continue;
        }

        std::unique_ptr<DataPointCollection> test_data1 = DataPointCollection::LoadMat(test_image, cv::Size(640, 480));
        bins_mat = Classifier<PixelSubtractionResponse>::ApplyMat(*classifier, *test_data1);
        // Get the weights for weighted sum from  classifiaction results. 
        // Essentially represents the probability for any pixel in the image to be in 
        // a certain bin.
        std::vector<uchar> bins_vec = IPUtils::vectorFromBins(bins_mat, cv::Size(640, 480));
        weights_vec = IPUtils::weightsFromBins(bins_mat, cv::Size(640,480), false);
        std::vector<uint16_t> sum_weighted_output((640*480), 0);

        for(int j=0;j<bins-1;j++)
        {
            std::vector<uint16_t> expert_output = Regressor<PixelSubtractionResponse>::ApplyMat(*experts[j], *test_data1);
            for(int k=0;k<expert_output.size();k++)
            {
                if(bins_vec[k] == 0)
                    continue;
                else
                    sum_weighted_output[k] = sum_weighted_output[k] + uint16_t(expert_output[k] * weights_vec[j]);
            }
        }
        reg_mat = cv::Mat(480, 640, CV_16UC1, (uint16_t*)sum_weighted_output.data());
        IPUtils::threshold16(reg_mat, result_thresh, 1200, 65535, 4);
        IPUtils::threshold16(depth_image, depth_thresh, 1200, 65535, 4);
        err_nt = IPUtils::getError(depth_image, reg_mat);
        err_t = IPUtils::getError(depth_thresh, result_thresh); 
        // result_thresh.convertTo(result_thresh, CV_16U, 54);
        // depth_thresh.convertTo(depth_thresh, CV_16U, 54);
        // cv::imshow("error_nthresh", err_nt);
        // cv::imshow("error_thresh", err_t);
        // cv::imshow("depth", depth_thresh);
        // cv::imshow("result", result_thresh);
        // cv::waitKey(30);
        sse_t = sse_t + err_t;
        sse_nt = sse_nt + err_nt;
        images_processed++;
    }

    float alpha = 1.0 / images_processed;
    msse_t = sse_t * alpha;
    msse_nt = sse_nt * alpha;
    cv::Scalar mean_nt, mean_t;
    cv::Scalar std_dev_nt, std_dev_t;
    cv::meanStdDev(msse_t, mean_t, std_dev_t);
    cv::meanStdDev(msse_nt, mean_nt, std_dev_nt);
    std::cout << "Mean (not thresholded): " << std::to_string(mean_nt[0]) << std::endl;
    std::cout << "Std dev (not thresholded): " << std::to_string(std_dev_nt[0]) << std::endl;
    std::cout << "Mean (thresholded): " << std::to_string(mean_t[0]) << std::endl;
    std::cout << "Std dev (thresholded): " << std::to_string(std_dev_t[0]) << std::endl;
}

void testFunction()
{
}

int growSomeForests(ProgramParameters& progParams)
{
    
    if(progParams.ForestType == ForestDescriptor::Regression)
    {
        try
        {
            std::cout << "\nAttempting to grow regressor" << std::endl;
            trainRegressionPar(progParams, -1);
        }
        catch (const std::runtime_error& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    if(progParams.ForestType == ForestDescriptor::ExpertRegressor)
    {
        try
        {
            std::cout << "\nAttempting to grow expert regressor " << std::to_string(progParams.ExpertClassNo) << std::endl;
            trainRegressionPar(progParams, progParams.ExpertClassNo);
        }
        catch (const std::runtime_error& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    if(progParams.ForestType == ForestDescriptor::All)
    {
        int start = progParams.TrainOnZeroIR? 1 : 0;
        for (int i=start;i<progParams.Bins;i++)
        {
            try
            {
                std::cout << "\nAttempting to grow expert regressor " << std::to_string(i) << std::endl;
                trainRegressionPar(progParams, i);
            }
            catch (const std::runtime_error& e)
            {
                std::cerr << e.what() << std::endl;
            }
        }
    }   
    
    if(progParams.ForestType == ForestDescriptor::Classification || progParams.ForestType == ForestDescriptor::All)
    {
        try
        {
            std::cout << "\nAttempting to grow classifier" << std::endl;
            trainClassificationPar(progParams);
            
        }
        catch (const std::runtime_error& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }
    return -1;
}

void printMenu()
{
    std::cout << "*************************Forest training and testing*************************";
    std::cout << std::endl;
    std::cout << "Enter 1 to train Classification Forest in parallel" << std::endl;
    std::cout << "Enter 2 to train Regression Forest in parallel" << std::endl;
    std::cout << "Enter 6 to do applyMultiLevel" << std::endl;
    std::cout << "Enter 7 to test Regression Forest" << std::endl;
    std::cout << "Enter 8 to test Classification Forest" << std::endl;
    std::cout << "Enter q to quit" << std::endl;
    std::cout << "\n" << std::endl;
}

void interactiveMode()
{
    bool cont = true;
    std::string in;
    std::string forest_path = "/home/james/workspace/forest_files/";
    //std::string forest_path = "/media/james/data_wd/training_realsense/";
    printMenu();

    // Poll for user input to chose program mode
    while (cont)
    {
        in == "";
        std::cin >> in;

        if (in.compare("6") == 0)
        {
            applyMultiLevel(forest_path);
            printMenu();
        }
        else if (in.compare("7") == 0)
        {
            regressOnline(forest_path);
            printMenu();
        }
        else if (in.compare("8") == 0)
        {
            classifyOnline(forest_path);
            printMenu();
        }
        else if (in.compare("1") == 0)
        {
            //trainClassificationPar(FILE_PATH, forest_path, 100);
            printMenu();
        }
        else if (in.compare("2") == 0)
        {
            //trainRegressionPar(FILE_PATH, forest_path, 100);
            printMenu();
        }
        else if (in.compare("q") == 0)
            testFunction();
            cont = false;

        // Refresh cin buffer
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

void printUsage(bool incorrect=false)
{
    if(incorrect)
        std::cout << "Incorrect input." << std::endl;

    std::cout << "Command Line Useage:\n" << std::endl;
    std::cout << "To grow [a/some] forest[s]: \n\t./FTT -g /path/to/params.params" << std::endl;
    std::cout << "To run a test on multi-level forests: \n\t ./FTT -t";
    std::cout << " /path/to/forest/ forest_prefix";
    std::cout << " /path/to/test/images test_image_prefix";
    std::cout << " num_test_images" << std::endl;
    std::cout << "Note, when passing prefixes, things like _classifier.frst" << std::endl;
    std::cout << "and _expert0.frst and _testir.png and _testdepth.png will" << std::endl;
    std::cout << "appended automatically" << std::endl;
    std::cout << std::endl << "To use interactively, pass no cla's" << std::endl;
}

ProgramParameters getParamsFromFile(std::string& params_path)
{
    ProgramParameters return_params;
    std::string line;
    // If you change this, MAKE SURE you change to value of num_categories!!!
    std::string categories [] = {"TRAINING_IMAGE_PATH",
                                    "TRAINING_IMAGES",
                                    "IMAGES_START",
                                    "DEPTH_BINS",
                                    "PATCH_SIZE",
                                    "DEPTH_RAW",
                                    "TYPE",
                                    "TREES",
                                    "CLASS_LEVELS",
                                    "REG_LEVELS",
                                    "CANDIDATE_FEATURES",
                                    "THRESHOLDS_PER_FEATURE",
                                    "VERBOSE",
                                    "EXPERT",
                                    "MAX_THREADS",
                                    "SPLIT_FUNCTION",
                                    "FOREST_OUTPUT",
                                    "INPUT_PREFIX",
                                    "IMG_WIDTH",
                                    "IMG_HEIGHT",
                                    "TRAIN_ON_ZERO_IR"};

    int num_categories = 21;
    try
    {
        ifstream params_file(params_path);
        if(params_file.is_open())
        {
            for(int i=0;i<num_categories;i++)
            {
                // line by line in the file
                // find category string in line
                // if found, send next line to return_params.setParam(categories[i], line)
                // clear and reset file
                while(std::getline(params_file, line))
                {
                    if(line.find("#") == 0)
                        continue;
                    else if(line.find(categories[i]) != std::string::npos)
                    {
                        std::getline(params_file, line);
                        return_params.setParam(categories[i], line);
                        break;
                    }
                }
                params_file.clear();
                params_file.seekg(0,ios::beg);
            }
            params_file.close();
        }
        else
            throw std::runtime_error("File could not be opened: " + params_path);
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << "Error parsing params file" << std::endl;
        std::cerr << e.what() << std::endl;
    }

    return return_params;
}

ProgramParameters getParamsFromFile(std::string& params_path, std::string& images_path)
{
    ProgramParameters return_params = getParamsFromFile(params_path);
    return_params.setParam("TRAINING_IMAGE_PATH", images_path);

    return return_params;
}

int main(int argc, char *argv[])
{
    ProgramParameters progParams;
    if( argc < 2 )
    {
        printUsage();
        std::cout << "Interactive mode" << std::endl;
        interactiveMode();    
    }
    else if(argc == 3)
    {
        std::string frst_arg = argv[1];
        if(frst_arg.compare("-g") == 0)
        {
            std::string params_path = argv[2];
            progParams = getParamsFromFile(params_path);
            progParams.prettyPrint();
            growSomeForests(progParams);
        }
        else
        {
            printUsage(true); 
        }

    }
    else if(argc == 7)
    {
        std::string frst_arg = argv[1];
        if(frst_arg.compare("-t") == 0)
        {
            std::string forest_path = argv[2];
            std::string forest_prefix = argv[3];
            std::string test_image_path = argv[4];
            std::string test_image_prefix = argv[5];
            int num_test_images = std::stoi(std::string(argv[6]));
            std::cout << "Forest path: " << forest_path << std::endl;
            std::cout << "Forest prefix: " << forest_prefix << std::endl;
            std::cout << "Test image path: " << test_image_path << std::endl;
            std::cout << "Test image prefix: " << test_image_prefix << std::endl;
            std::cout << "Images to use in testing: " << std::to_string(num_test_images) << std::endl;
            testForest(forest_path, 
                forest_prefix, 
                test_image_path, 
                test_image_prefix, 
                num_test_images);
        }
        else
        {
            printUsage(true); 
        }
    }
    else
    {
        printUsage(true); 
    }   

}
