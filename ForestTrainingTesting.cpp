/*
This file contains main() and other top-level functions, like testing and 
training.

    As a general precursor to all the code in this project: the closer it got to
    the deadline of the project, the worse the code got. 
    There's a few functions in this file which do very similar things, which 
    could easily be rolled into one given a little more program control.
    
    James Andrew
    jamesrobertandrew1993@gmail.com
*/

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
#define THRESHOLD_PARAM 1200
// Hard-coded output paths for some results and of accuracy testing 
#define OUT_PATH "/home/james/workspace/forest_output_files/temp/"
#define IMAGE_OUT "/media/james/data_wd/output_images/"

// Some default paths for tinkering
#ifdef _WIN32
const std::string FILE_PATH = "D:\\";
#endif
#ifdef __linux__
const std::string FILE_PATH = "/media/james/data_wd/";
#endif

///<summary> Trains a classification forest using the multi-threaded training 
/// algorithm. Saves a classification forest to the forest output path as 
/// specified in function input object. </summary>
///<param name="progParams"> A reference to a program parameters object 
/// which contains information about the input images, forest parameters
/// and output path and naming </param>
int trainClassificationPar(ProgramParameters& progParams)
{
    // Ensure the path to the training images ends in a /
    if(progParams.TrainingImagesPath.back() != '/')
        progParams.TrainingImagesPath += "/";

    std::string filename = progParams.TrainingImagesPath + progParams.OutputFilename + "_classifier.frst";
     
    // load up the training data
    std::cout << "Searching for some IR and depth images in " << progParams.TrainingImagesPath << std::endl;
    std::unique_ptr<DataPointCollection> training_data = DataPointCollection::LoadImages(progParams, true);
    
    // Check and display some data for reference
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

///<summary> Trains a regression forest using the multi-threaded training 
/// algorithm. Saves a regression forest to the forest output path as 
/// specified in function input object. </summary>
///<param name="progParams"> A reference to a program parameters object 
/// which contains information about the input images, forest parameters
/// and output path and naming </param>
///<param name="class_expert_no">If we're training an expert regressor, this
/// is the integer class label for the expert. If -1, it's a general regressor
/// </param>
int trainRegressionPar(ProgramParameters& progParams, int class_expert_no = -1)
{
    std::string file_suffix;
    
    // adapt output file name depending on inputs
    if(class_expert_no != -1)
        file_suffix = "_expert" + std::to_string(class_expert_no) + ".frst";
    else
        file_suffix = "_regressor.frst";

    // check path delimiter
    if(progParams.TrainingImagesPath.back() != '/')
        progParams.TrainingImagesPath += "/";

    std::string filename = progParams.TrainingImagesPath + progParams.OutputFilename + file_suffix;
    std::cout << "Searching for some IR and depth images in " << progParams.TrainingImagesPath << std::endl;
    
    // create a DataPointCollection in the regression format
    std::unique_ptr<DataPointCollection> training_data = DataPointCollection::LoadImages(progParams, false, class_expert_no); 

    // some checking and simple data output
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

///<summary> Trains a classification forest using the multi-threaded training 
/// algorithm using the RandomHyperplane split-function. 
/// Saves a classification forest to the forest output path as specified in 
/// function input object. </summary>
///<param name="progParams"> A reference to a program parameters object 
/// which contains information about the input images, forest parameters
/// and output path and naming </param>
int trainClassificationRH(ProgramParameters& progParams)
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
        std::unique_ptr<Forest<RandomHyperplaneFeatureResponse, HistogramAggregator> > forest = 
            Classifier<RandomHyperplaneFeatureResponse>::TrainPar(*training_data, progParams.Tpc);

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

///<summary> Trains a regression forest using the multi-threaded training 
/// algorithm using the RandomHyperplane split-function. 
/// Saves a regression forest to the forest output path as specified in 
/// function input object. </summary>
///<param name="progParams"> A reference to a program parameters object 
/// which contains information about the input images, forest parameters
/// and output path and naming </param>
///<param name="class_expert_no">If we're training an expert regressor, this
/// is the integer class label for the expert. If -1, it's a general regressor
/// </param>
int trainRegressionRH(ProgramParameters& progParams, int class_expert_no = -1)
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
        std::unique_ptr<Forest<RandomHyperplaneFeatureResponse, DiffEntropyAggregator> > forest =
            Regressor<RandomHyperplaneFeatureResponse>::TrainPar(*training_data, progParams.Tpr);

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

// Get and display output from regression forest. Used only for testing.
// This is old, probably doesn't work anymore due to things being hardcoded
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

        // Create a DataPointCollection from a single input image
        std::unique_ptr<DataPointCollection> test_data1 = DataPointCollection::LoadMat(test_image, cv::Size(640, 480));
        reg_result = Regressor<PixelSubtractionResponse>::ApplyMat(*forest_shared, *test_data1);

        // reform vector of results into cv::Mat of results
        reg_mat = cv::Mat(480, 640, CV_16UC1, (uint16_t*)reg_result.data());
        // Threshold out outputs >= 1201 mm (max range)
        IPUtils::threshold16(reg_mat, result_thresh, 1201, 65535, 4);
        // Multiply all by 54 to scale 1200 to ~65535
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

///<summary>Loads a classification forest, then applies a number of images to 
/// the classification forest for evaluation. Test image pixels are discretised 
/// into depth classes using the classification forest, and the resulting images
/// are saved to a hardcoded output path (OUT_PATH) 
/// NOTE, for ease at the time, the test functions pick test images differently
/// based on the test_image_prefix parameter </summary>
///<param name="forest_path">Path to directory containing forest file</param>
///<param name="forest_prefix">eg for test_forest_classifier.frst, 
///  prefix = test_forest </param>
///<param name="test_image_path">path to directory of test images</param>
///<param name="test_image_prefix">eg for images names as img125ir.png and 
/// img125depth.png, prefix=img</param>
///<param name="num_images">integer number of test images to ecaluate</param>
int classifyOnline(std::string forest_path,
    std::string forest_prefix,
    std::string test_image_path,
    std::string test_image_prefix,
    int num_images = 0)
{
    // Dir checks
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
    std::cout << "Attempting to deserialize forest from " << class_path << std::endl;
    // Init a pointer to a classifier
    std::unique_ptr<ForestShared<PixelSubtractionResponse, HistogramAggregator> > classifier;

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
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Forest loading Failed" << std::endl;
        std::cerr << e.what() << std::endl;
    }

    cv::Mat test_image = cv::Mat(cv::Size(640, 480), CV_8UC1);
    cv::Mat bins_mat;
    cv::Mat result_norm1;

    std::string img_path = test_image_path+test_image_prefix;
    std::string img_full_path;
    bool realsense = false;

    int threshold_value = 38;
    size_t t_pos = forest_prefix.find("T");
    // Hacky way to pull threshold value out of forest prefix
    // if there's a 'T' in it, the threshold value follows it
    if(t_pos != std::string::npos)
    {
        threshold_value = std::stoi(forest_prefix.substr(t_pos+1, t_pos+2));

    }
    std::cout << "Threshold value used " << std::to_string(threshold_value) << std::endl;


    Random random;
    int use_images = 0;
    std::vector<int> rand_ints;
    if (num_images != 0)
    {
        use_images = num_images;
        rand_ints.resize(num_images);
        if(test_image_prefix.compare("img")==0)
        {
            std::cout << "testing on training_realsense training set" << std::endl; 
            realsense = true;
            rand_ints = random.RandomVector(0,1200,num_images,false);
        }
        else if(test_image_prefix.compare("test")==0)
        {
            std::cout << "testing on training_images_2 test set" << std::endl; 
            realsense = true;
            rand_ints = random.RandomVector(10000,11000,num_images,false);
        }
        else
        {
            std::cout << "testing on training_realsense_2 test set" << std::endl; 
            realsense = true;
            rand_ints = random.RandomVector(1200,1500,num_images,false);   
        }    
    }
    else
    {
        if(test_image_prefix.compare("img")==0)
        {
            use_images = 1200;
            rand_ints.resize(1200);
            std::cout << "testing on training_realsense training set" << std::endl; 
            realsense = true;
            std::iota(rand_ints.begin(), rand_ints.end(), 0);
        }
        else if(test_image_prefix.compare("test")==0)
        {
            use_images = 1000;
            rand_ints.resize(1000);
            std::cout << "testing on training_images_2 test set" << std::endl; 
            realsense = true;
            std::iota(rand_ints.begin(), rand_ints.end(), 10000);
        }
        else
        {
            use_images = 300;
            rand_ints.resize(300);
            std::cout << "testing on training_realsense_2 test set" << std::endl; 
            realsense = true;
            std::iota(rand_ints.begin(), rand_ints.end(), 1200);
        }
    }
    
    std::string ir_image_suffix = "ir.png";
    if(forest_prefix.find("cam") != std::string::npos)
    {
        ir_image_suffix = "cam.png";
        threshold_value = 79;
    }

    std::string savepath = IMAGE_OUT + forest_prefix + test_image_prefix + "Binned";

    for (int i = 0; i < use_images; i++)
    {
        int image_index;
        if(realsense)
            image_index = rand_ints[i];
        else
            image_index = 10000+i;

        img_full_path = img_path + std::to_string(image_index) + ir_image_suffix;
        test_image = cv::imread(img_full_path, -1);

        std::cout << img_full_path << std::endl;

         if(!test_image.data)
        {
            std::cerr << "Error loading image:\n\t" << img_full_path << std::endl;
            continue;
        }

        // Pre process the test image (need to do this for the alternate training where we don't use zero inputs)
        // TODO get some parameters into this
        test_image = IPUtils::preProcess(test_image, threshold_value);

        std::unique_ptr<DataPointCollection> test_data1 = DataPointCollection::LoadMat(test_image, cv::Size(640, 480), false, false);
        bins_mat = Classifier<PixelSubtractionResponse>::ApplyMat(*classifier, *test_data1);
        std::vector<uchar> bins_vec = IPUtils::vectorFromBins(bins_mat, cv::Size(640, 480));

        int output_index = 0;
        cv::Mat result_mat = cv::Mat::zeros(480, 640, CV_8UC1);
        // iterate through image, and slot in ouputs for all non-zero inputs
        for(int r=0;r<480;r++)
        {
            uchar* test_image_pix = test_image.ptr<uchar>(r);
            uchar* res_mat_pix = result_mat.ptr<uchar>(r);
            for(int c=0;c<640;c++)
            {
                if(test_image_pix[c] == 0)
                {
                    continue;
                }
                else
                {
                    // this might fail if we go out of bounds somehow.
                    res_mat_pix[c] = bins_vec[output_index];
                    output_index++;
                }
            }
        }

        result_mat.convertTo(result_norm1, CV_8U, 63);
        cv::Mat colourized;
        cv::applyColorMap(result_norm1, colourized, cv::COLORMAP_JET);
        cv::imshow("output", colourized);

        int blah = cv::waitKey(100);
        cv::imwrite(savepath+"gs"+std::to_string(rand_ints[i])+".png", result_norm1);
        cv::imwrite(savepath+"Colour"+std::to_string(rand_ints[i])+".png", colourized);
        if(blah == 113)
        {
            break;
        }
    }
    cv::startWindowThread();
    cv::destroyAllWindows();
    
    return 0;
}

// This is a pretty old function, used in early testing. Probably doesn't work 
// anymore. Uses hard-coded paths.
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

///<summary>Loads a set of multi-layer forests, then applies a number of images
/// to the forests for evaluation. Image pixels are classified into depth bins,
/// which are used to form weightings, then used in a weighted sum of all expert
/// regression forest outputs. THe "Alternate" part refers to the fact that this
/// application does not use zero input IR values in estimation of depth.
/// This function prints mean and standard deviation of per-pixel depth error
/// and processing time information to cout.
/// Also creates a file containing average depth vs depth error
/// NOTE, for ease at the time, the test functions pick test images differently
/// based on the test_image_prefix parameter </summary>
///<param name="forest_path">Path to directory containing forest file</param>
///<param name="forest_prefix">eg for test_forest_classifier.frst, 
///  prefix = test_forest </param>
///<param name="test_image_path">path to directory of test images</param>
///<param name="test_image_prefix">eg for images names as img125ir.png and 
/// img125depth.png, prefix=img</param>
///<param name="num_images">integer number of test images to ecaluate</param>
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
    // steady state error, error, mean sse - thresholded
    cv::Mat sse_t = cv::Mat::zeros(480, 640, CV_32SC1);
    cv::Mat err_t = cv::Mat::zeros(480, 640, CV_32SC1);
    cv::Mat msse_t(480, 640, CV_32SC1);
    // steady state error, error, mean sse - not thresholded
    cv::Mat sse_nt = cv::Mat::zeros(480, 640, CV_32SC1);
    cv::Mat err_nt = cv::Mat::zeros(480, 640, CV_32SC1);
    std::vector<int32_t> gt_error(THRESHOLD_PARAM, 0);
    std::vector<int32_t> gt_inc(THRESHOLD_PARAM, 0);
    cv::Mat msse_nt(480, 640, CV_32SC1);
    cv::Mat depth_image(480, 640, CV_16UC1);
    cv::Mat depth_norm;
    cv::Mat err_norm;
    cv::Mat test_image(480, 640, CV_8UC1);
    cv::Mat bins_mat;
    cv::Mat result_thresh(480, 640, CV_16UC1);
    cv::Mat depth_thresh(480, 640, CV_16UC1);
    cv::Mat temp_mat(480, 640, CV_16UC1);
    std::vector<float> weights_vec(bins);
    int images_processed = 0;
    bool realsense = false;

    // hacky way to extract desired threshold value from forest prefix
    int threshold_value = 36;
    size_t t_pos = forest_prefix.find("T");
    if(t_pos != std::string::npos)
    {
        threshold_value = std::stoi(forest_prefix.substr(t_pos+1, t_pos+2));

    }
    
    std::string ir_image_suffix = "ir.png";

    if(forest_prefix.find("cam") != std::string::npos)
    {
        ir_image_suffix = "cam.png";
        threshold_value = 79;
    }

    std::cout << "Threshold value used " << std::to_string(threshold_value) << std::endl;

    Random random;
    std::vector<int> rand_ints(num_images);
    if(test_image_prefix.compare("img")==0)
    {
        std::cout << "testing on training_realsense training set" << std::endl; 
        realsense = true;
        rand_ints = random.RandomVector(0,1200,num_images,false);
    }
    else if(test_image_prefix.compare("test")==0)
    {
        std::cout << "testing on training_images_2 test set" << std::endl; 
        realsense = true;
        rand_ints = random.RandomVector(10000,11000,num_images,false);
    }
    else
    {
        std::cout << "testing on training_realsense_2 test set" << std::endl; 
        realsense = true;
        rand_ints = random.RandomVector(1200,1500,num_images,false);   
    }

    std::string savepath = IMAGE_OUT + forest_prefix + test_image_prefix + "Result";

    int64 start_time;
    int64 process_time = 0;
    for(int i=0;i<num_images;i++)
    {
        int image_index;
        if(realsense)
            image_index = rand_ints[i];
        else
            image_index = 10000+i;

        
        cv::Mat reg_mat = cv::Mat::zeros(480, 640, CV_16UC1);
        img_full_path = img_path + std::to_string(image_index) + ir_image_suffix;
        depth_full_path = depth_path + std::to_string(image_index) + "depth.png";
        
        start_time = cv::getTickCount();

        test_image = cv::imread(img_full_path, -1);
        depth_image = cv::imread(depth_full_path, -1);

        if((!depth_image.data)||(!test_image.data))
        {
            std::cerr << "Error loading images:\n\t" << img_full_path << "\n\t" << depth_full_path << std::endl;
            continue;
        }

        // Pre process the test image (need to do this for the alternate training where we don't use zero inputs)
        // TODO get some parameters into this
        test_image = IPUtils::preProcess(test_image, threshold_value);

        std::unique_ptr<DataPointCollection> test_data1 = DataPointCollection::LoadMat(test_image, cv::Size(640, 480), false, false);
        bins_mat = Classifier<PixelSubtractionResponse>::ApplyMat(*classifier, *test_data1);
        // Get the weights for weighted sum from  classifiaction results. 
        // Essentially represents the probability for any pixel in the image to be in 
        // a certain bin.
        
        weights_vec = IPUtils::weightsFromBins(bins_mat, cv::Size(640,480), true);
        std::vector<uint16_t> sum_weighted_output(test_data1->Count(), 0);
        for(int j=0;j<bins;j++)
        {
            std::vector<uint16_t> expert_output = Regressor<PixelSubtractionResponse>::ApplyMat(*experts[j], *test_data1);
            for(int k=0;k<expert_output.size();k++)
            {
                sum_weighted_output[k] = sum_weighted_output[k] + uint16_t(expert_output[k] * weights_vec[j]);
            }
        }
        
        int output_index = 0;
        for(int r=0;r<480;r++)
        {
            uchar* test_image_pix = test_image.ptr<uchar>(r);
            uint16_t* reg_mat_pix = reg_mat.ptr<uint16_t>(r);
            for(int c=0;c<640;c++)
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
        process_time += (((cv::getTickCount() - start_time) / cv::getTickFrequency()) * 1000);

        IPUtils::threshold16(reg_mat, result_thresh, THRESHOLD_PARAM, 65535, 4);
        IPUtils::threshold16(depth_image, depth_thresh, THRESHOLD_PARAM, 65535, 4);
        err_nt = IPUtils::getError(depth_image, reg_mat);        
        err_t = IPUtils::getError(depth_thresh, result_thresh); 
        
        double min_err, max_err;
        cv::minMaxLoc(err_t, &min_err, &max_err);
        std::cout << "Min " << std::to_string(min_err) << "Max " << std::to_string(max_err) << std::endl;
        cv::normalize(err_t, err_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        //err_t.convertTo(err_norm, CV_8U, 0.2125);
        result_thresh.convertTo(depth_norm, CV_16U, 54);
        // cv::imshow("error_nthresh", err_nt);
        // cv::imshow("error_thresh", err_t);
        cv::Mat err_colourized(480, 640, CV_8UC3);
        cv::applyColorMap(err_norm, err_colourized, cv::COLORMAP_JET);
        cv::Mat img_with_key = IPUtils::AddKey(int(min_err), int(max_err), err_colourized);
        //cv::imshow("error", err_colourized);
        //cv::imshow("depth", depth_norm);
        //cv::imshow("with key", img_with_key);
        
        if(max_err <= 800)
        {
            cv::imwrite(savepath+"depth"+std::to_string(rand_ints[i])+".png", depth_norm);
            cv::imwrite(savepath+"error"+std::to_string(rand_ints[i])+".png", img_with_key);
        }

        int rows = depth_thresh.size().height;
        int cols = depth_thresh.size().width;

        for(int r=0;r<rows;r++)
        {
            uint16_t* depth_pix = depth_thresh.ptr<uint16_t>(r);
            int32_t* pixel_error = err_t.ptr<int32_t>(r);
            for(int c=0;c<cols;c++)
            {
                gt_error[depth_pix[c]] += pixel_error[c];
                gt_inc[depth_pix[c]]++;
            }
        }
        sse_t = sse_t + err_t;
        sse_nt = sse_nt + err_nt;
        images_processed++;
        //cv::waitKey(30);
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

    float avg_process_time = alpha * process_time;
    std::cout << "\nAverage per-frame process time: " << std::to_string(avg_process_time) << " ms" << std::endl;
    std::cout << "\nAverage framerate: " << std::to_string(1000/avg_process_time) << " Hz" << std::endl;

    // Create file of depth vs depth error
    ofstream out_file;
    out_file.open(OUT_PATH + forest_prefix + test_image_prefix);
    int temp;
    if(out_file.is_open())
    {
        for(int i=0;i<gt_error.size();i++)
        {
            if(gt_inc[i] != 0)
                temp = gt_inc[i];
            else
                temp = 1;

            out_file << to_string(round((float(gt_error[i])/temp))) << ";";
        }
        out_file.close();
    }
    else
    {
        std::cout << "Unable to open file" << std::endl;
    }

}

///<summary>Loads a set of multi-layer forests, then applies a number of images
/// to the forests for evaluation. Image pixels are classified into depth bins,
/// which are used to form weightings, then used in a weighted sum of all expert
/// regression forest outputs. THe "Alternate" part refers to the fact that this
/// application does not use zero input IR values in estimation of depth.
/// This function prints mean and standard deviation of per-pixel depth error
/// and processing time information to cout.
/// Also creates a file containing average depth vs depth error
/// NOTE, for ease at the time, the test functions pick test images differently
/// based on the test_image_prefix parameter 
/// *** AS ABOVE BUT WITH RandomHyperplane split function *** </summary>
///<param name="forest_path">Path to directory containing forest file</param>
///<param name="forest_prefix">eg for test_forest_classifier.frst, 
///  prefix = test_forest </param>
///<param name="test_image_path">path to directory of test images</param>
///<param name="test_image_prefix">eg for images names as img125ir.png and 
/// img125depth.png, prefix=img</param>
///<param name="num_images">integer number of test images to ecaluate</param>
int testForestAlternateRH(std::string forest_path,
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
    std::vector<std::unique_ptr<ForestShared<RandomHyperplaneFeatureResponse, DiffEntropyAggregator> > > experts;
    // Init a pointer to a classifier
    std::unique_ptr<ForestShared<RandomHyperplaneFeatureResponse, HistogramAggregator> > classifier;
    int bins = 5;

    // Load the classifier and expert regressors.
    try
    {
        std::cout << "Loading classifier" << std::endl;
        // load classifier
        std::unique_ptr<Forest<RandomHyperplaneFeatureResponse, HistogramAggregator> > c_forest =
            Forest<RandomHyperplaneFeatureResponse, HistogramAggregator>::Deserialize(class_path);
        // Create ForestShared from loaded forest
        std::unique_ptr<ForestShared<RandomHyperplaneFeatureResponse, HistogramAggregator> > c_forest_shared =
            ForestShared<RandomHyperplaneFeatureResponse, HistogramAggregator>::ForestSharedFromForest(*c_forest);
        // Delete original forest. May roll these steps into one later if we don't need a regular forest application.
        c_forest->~Forest();
        c_forest.release();
        classifier = move(c_forest_shared);
        std::cout << "Classifier loaded with " << std::to_string(classifier->TreeCount()) << " trees" << std::endl;
        for(int i=0;i<bins;i++)
        {
            std::cout << "Loading expert " << std::to_string(i) << std::endl;
            std::unique_ptr<Forest<RandomHyperplaneFeatureResponse, DiffEntropyAggregator> > e_forest =
                Forest<RandomHyperplaneFeatureResponse, DiffEntropyAggregator>::Deserialize(expert_path[i]);
            // Create ForestShared from loaded forest
            std::unique_ptr<ForestShared<RandomHyperplaneFeatureResponse, DiffEntropyAggregator> > e_forest_shared =
                ForestShared<RandomHyperplaneFeatureResponse, DiffEntropyAggregator>::ForestSharedFromForest(*e_forest);
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
    std::vector<int32_t> gt_error(THRESHOLD_PARAM, 0);
    std::vector<int32_t> gt_inc(THRESHOLD_PARAM, 0);
    cv::Mat msse_nt(480, 640, CV_32SC1);
    cv::Mat depth_image(480, 640, CV_16UC1);
    cv::Mat test_image(480, 640, CV_8UC1);
    cv::Mat bins_mat;
    cv::Mat result_thresh(480, 640, CV_16UC1);
    cv::Mat depth_thresh(480, 640, CV_16UC1);
    cv::Mat temp_mat(480, 640, CV_16UC1);
    std::vector<float> weights_vec(bins);
    int images_processed = 0;
    bool realsense = false;

    int threshold_value = 38;
    size_t t_pos = forest_prefix.find("T");
    if(t_pos != std::string::npos)
    {
        threshold_value = std::stoi(forest_prefix.substr(t_pos+1, t_pos+2));

    }
    
    Random random;
    std::vector<int> rand_ints(num_images);
    if(test_image_prefix.compare("img")==0)
    {
        std::cout << "testing on training_realsense training set" << std::endl; 
        realsense = true;
        rand_ints = random.RandomVector(0,1200,num_images,false);
    }
    else if(test_image_prefix.compare("test")==0)
    {
        std::cout << "testing on training_images_2 test set" << std::endl; 
        realsense = true;
        rand_ints = random.RandomVector(10000,11000,num_images,false);
    }
    else
    {
        std::cout << "testing on training_realsense_2 test set" << std::endl; 
        realsense = true;
        rand_ints = random.RandomVector(1200,1500,num_images,false);   
    }
    
    std::string ir_image_suffix = "ir.png";
    if(forest_prefix.find("cam") != std::string::npos)
    {
        ir_image_suffix = "cam.png";
        threshold_value = 79;
    }

    std::cout << "Threshold value used " << std::to_string(threshold_value) << std::endl;

    int64 start_time;
    int64 process_time = 0;

    for(int i=0;i<num_images;i++)
    {
        int image_index;
        if(realsense)
            image_index = rand_ints[i];
        else
            image_index = 10000+i;

        cv::Mat reg_mat = cv::Mat::zeros(480, 640, CV_16UC1);
        img_full_path = img_path + std::to_string(image_index) + ir_image_suffix;
        depth_full_path = depth_path + std::to_string(image_index) + "depth.png";
        
        start_time = cv::getTickCount();

        test_image = cv::imread(img_full_path, -1);
        depth_image = cv::imread(depth_full_path, -1);

        if((!depth_image.data)||(!test_image.data))
        {
            std::cerr << "Error loading images:\n\t" << img_full_path << "\n\t" << depth_full_path << std::endl;
            continue;
        }

        // Pre process the test image (need to do this for the alternate training where we don't use zero inputs)
        // TODO get some parameters into this
        test_image = IPUtils::preProcess(test_image, threshold_value);

        std::unique_ptr<DataPointCollection> test_data1 = DataPointCollection::LoadMat(test_image, cv::Size(640, 480), false, false);
        bins_mat = Classifier<RandomHyperplaneFeatureResponse>::ApplyMat(*classifier, *test_data1);
        // Get the weights for weighted sum from  classifiaction results. 
        // Essentially represents the probability for any pixel in the image to be in 
        // a certain bin.
        
        weights_vec = IPUtils::weightsFromBins(bins_mat, cv::Size(640,480), true);
        std::vector<uint16_t> sum_weighted_output(test_data1->Count(), 0);
        for(int j=0;j<bins;j++)
        {
            std::vector<uint16_t> expert_output = Regressor<RandomHyperplaneFeatureResponse>::ApplyMat(*experts[j], *test_data1);
            for(int k=0;k<expert_output.size();k++)
            {
                sum_weighted_output[k] = sum_weighted_output[k] + uint16_t(expert_output[k] * weights_vec[j]);
            }
        }
        
        int output_index = 0;
        for(int r=0;r<480;r++)
        {
            uchar* test_image_pix = test_image.ptr<uchar>(r);
            uint16_t* reg_mat_pix = reg_mat.ptr<uint16_t>(r);
            for(int c=0;c<640;c++)
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

        process_time += (((cv::getTickCount() - start_time) / cv::getTickFrequency()) * 1000);

        IPUtils::threshold16(reg_mat, result_thresh, THRESHOLD_PARAM, 65535, 4);
        IPUtils::threshold16(depth_image, depth_thresh, THRESHOLD_PARAM, 65535, 4);
        err_nt = IPUtils::getError(depth_image, reg_mat);        
        err_t = IPUtils::getError(depth_thresh, result_thresh); 
        // result_thresh.convertTo(result_thresh, CV_16U, 54);
        // depth_thresh.convertTo(depth_thresh, CV_16U, 54);
        // cv::imshow("error_nthresh", err_nt);
        // cv::imshow("error_thresh", err_t);
        // cv::imshow("depth", depth_thresh);
        // cv::imshow("result", result_thresh);
        int rows = depth_thresh.size().height;
        int cols = depth_thresh.size().width;

        for(int r=0;r<rows;r++)
        {
            uint16_t* depth_pix = depth_thresh.ptr<uint16_t>(r);
            int32_t* pixel_error = err_t.ptr<int32_t>(r);
            for(int c=0;c<cols;c++)
            {
                gt_error[depth_pix[c]] += pixel_error[c];
                gt_inc[depth_pix[c]]++;
            }
        }
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

    float avg_process_time = alpha * process_time;
    std::cout << "\nAverage per-frame process time: " << std::to_string(avg_process_time) << " ms" << std::endl;
    std::cout << "\nAverage framerate: " << std::to_string(1000/avg_process_time) << " Hz" << std::endl;

    ofstream out_file;
    out_file.open(OUT_PATH + forest_prefix + test_image_prefix);
    int temp;
    if(out_file.is_open())
    {
        for(int i=0;i<gt_error.size();i++)
        {
            if(gt_inc[i] != 0)
                temp = gt_inc[i];
            else
                temp = 1;

            out_file << to_string(round((float(gt_error[i])/temp))) << ";";
        }
        out_file.close();
    }
    else
    {
        std::cout << "Unable to open file" << std::endl;
    }
}

///<summary>Loads a set of multi-layer forests, then applies a number of images
/// to the forests for evaluation. Image pixels are classified into depth bins,
/// which are used to form weightings, then used in a weighted sum of all expert
/// regression forest outputs.
/// This function prints mean and standard deviation of per-pixel depth error
/// and processing time information to cout.
/// Also creates a file containing average depth vs depth error
/// NOTE, for ease at the time, the test functions pick test images differently
/// based on the test_image_prefix parameter </summary>
///<param name="forest_path">Path to directory containing forest file</param>
///<param name="forest_prefix">eg for test_forest_classifier.frst, 
///  prefix = test_forest </param>
///<param name="test_image_path">path to directory of test images</param>
///<param name="test_image_prefix">eg for images names as img125ir.png and 
/// img125depth.png, prefix=img</param>
///<param name="num_images">integer number of test images to ecaluate</param>
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
    std::vector<int32_t> gt_error(THRESHOLD_PARAM, 0);
    std::vector<int32_t> gt_inc(THRESHOLD_PARAM, 0);
    cv::Mat msse_nt(480, 640, CV_32SC1);
    cv::Mat depth_image(480, 640, CV_16UC1);
    cv::Mat test_image(480, 640, CV_8UC1);
    cv::Mat reg_mat;
    cv::Mat bins_mat;
    cv::Mat result_thresh(480, 640, CV_16UC1);
    cv::Mat depth_thresh(480, 640, CV_16UC1);
    std::vector<float> weights_vec(bins-1);
    int images_processed = 0;
    bool realsense = false;

    Random random;
    std::vector<int> rand_ints(num_images);
    if(test_image_prefix.compare("img")==0)
    {
        std::cout << "testing on training_realsense training set" << std::endl; 
        realsense = true;
        rand_ints = random.RandomVector(0,1200,num_images,false);
    }
    else if(test_image_prefix.compare("test")==0)
    {
        std::cout << "testing on training_images_2 test set" << std::endl; 
        realsense = true;
        rand_ints = random.RandomVector(10000,11000,num_images,false);
    }
    else
    {
        std::cout << "testing on training_realsense_2 test set" << std::endl; 
        realsense = true;
        rand_ints = random.RandomVector(1200,1500,num_images,false);   
    }

    
    std::string ir_image_suffix = "ir.png";
    if(forest_prefix.find("cam") != std::string::npos)
        ir_image_suffix = "cam.png";

    int64 start_time;
    int64 process_time = 0;

    for(int i=0;i<num_images;i++)
    {
        int image_index;
        if(realsense)
            image_index = rand_ints[i];
        else
            image_index = 10000+i;

        img_full_path = img_path + std::to_string(image_index) + ir_image_suffix;
        depth_full_path = depth_path + std::to_string(image_index) + "depth.png";
        
        start_time = cv::getTickCount();
        
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

        process_time += (((cv::getTickCount() - start_time) / cv::getTickFrequency()) * 1000);

        IPUtils::threshold16(reg_mat, result_thresh, THRESHOLD_PARAM, 65535, 4);
        IPUtils::threshold16(depth_image, depth_thresh, THRESHOLD_PARAM, 65535, 4);
        err_nt = IPUtils::getError(depth_image, reg_mat);
        err_t = IPUtils::getError(depth_thresh, result_thresh); 
        // result_thresh.convertTo(result_thresh, CV_16U, 54);
        // depth_thresh.convertTo(depth_thresh, CV_16U, 54);
        // cv::imshow("error_nthresh", err_nt);
        // cv::imshow("error_thresh", err_t);
        // cv::imshow("depth", depth_thresh);
        // cv::imshow("result", result_thresh);
        // cv::waitKey(30);
        int rows = depth_thresh.size().height;
        int cols = depth_thresh.size().width;

        for(int r=0;r<rows;r++)
        {
            uint16_t* depth_pix = depth_thresh.ptr<uint16_t>(r);
            int32_t* pixel_error = err_t.ptr<int32_t>(r);
            for(int c=0;c<cols;c++)
            {
                gt_error[depth_pix[c]] += pixel_error[c];
                gt_inc[depth_pix[c]]++;
            }
        }

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

    float avg_process_time = alpha * process_time;
    std::cout << "\nAverage per-frame process time: " << std::to_string(avg_process_time) << " ms" << std::endl;
    std::cout << "\nAverage framerate: " << std::to_string(1000/avg_process_time) << " Hz" << std::endl;

    // output file of mean depth error (mm) against depth (mm)
    ofstream out_file;
    out_file.open(OUT_PATH + forest_prefix + test_image_prefix);
    int temp;
    if(out_file.is_open())
    {
        for(int i=0;i<gt_error.size();i++)
        {
            if(gt_inc[i] != 0)
                temp = gt_inc[i];
            else
                temp = 1;

            out_file << to_string(round((float(gt_error[i])/temp))) << ";";
        }
        out_file.close();
    }
    else
    {
        std::cout << "Unable to open file" << std::endl;
    }
    
}

// Output a file for a dataset with a histogram of "beth threshold values"
// Useful for chosing or identifying an ideal threshold value
// Hardcoded paths
void testFunction()
{
    Random random;
    int size = 1000;
    std::vector<int> threshold_values(size);
    std::vector<int> error_values(size);
    for(int i=0;i<size;i++)
    {
        int random_int = random.Next(10000,11000);
        std::string file_name = "/media/james/data_wd/training_images_2/test" + to_string(random_int);
        cv::Mat ir_image = cv::imread(file_name + "ir.png", -1);
        cv::Mat depth_image = cv::imread(file_name + "depth.png", -1);

        if(ir_image.data && depth_image.data)
            std::cout << "Testing threshold with image " << file_name << std::endl;

        threshold_values[i] = IPUtils::getBestThreshold(ir_image, depth_image, 1000, error_values[i]);

        std::cout << "Best threshold: " << std::to_string(threshold_values[i])  << " " << std::to_string(error_values[i]) << std::endl;
    }
    sort(threshold_values.begin(), threshold_values.end());
    int median;
    int sum_t = 0;
    uint32_t sum_e = 0;
    int mean_e;
    int mean;
    std::vector<int> histogram(255);

    for(int i=0;i<size;i++)
    {
        sum_t += threshold_values[i];
        sum_e += uint32_t(error_values[i]);
        histogram[threshold_values[i]]++;
    }

    mean = round(float(sum_t) / size);
    mean_e = round(float(sum_e) / size);

    if(size%2==0)
    {
        median = (threshold_values[size/2] + threshold_values[(size/2) - 1])/2;
    }
    else
    {
        median = threshold_values[size/2];
    }

    ofstream histogram_file;
    histogram_file.open("/home/james/workspace/big_realsense_histogram_file.csv");
    if(histogram_file.is_open())
    {
        for(int i=0;i<255;i++)
        {
            histogram_file << to_string(histogram[i]) << ",";
        }
        histogram_file.close();
    }
    else
    {
        std::cout << "Unable to open file" << std::endl;
    }

    std::cout << "Mean: " << std::to_string(mean) << std::endl;
    std::cout << "Median: " << std::to_string(median) << std::endl;
    std::cout << "Mean Error: " << std::to_string(mean_e) << std::endl;

}

// Redirect to correct forest training functions based on program parameters
int growSomeForests(ProgramParameters& progParams)
{
    bool rh = (progParams.SplitFunctionType == SplitFunctionDescriptor::RandomHyperplane);
    
    if(progParams.ForestType == ForestDescriptor::Regression)
    {
        try
        {
            std::cout << "\nAttempting to grow regressor" << std::endl;
            if(!rh)
                trainRegressionPar(progParams, -1);
            else
                trainRegressionRH(progParams, -1);
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
            if(!rh)
                trainRegressionPar(progParams, progParams.ExpertClassNo);
            else
                trainRegressionRH(progParams, progParams.ExpertClassNo);
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
                if(!rh)
                    trainRegressionPar(progParams, i);
                else
                    trainRegressionRH(progParams, i);
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
            if(!rh)
                trainClassificationPar(progParams);
            else
                trainClassificationRH(progParams);
            
        }
        catch (const std::runtime_error& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }
    return -1;
}

// Outputs a file of ir intensity value vs ground-truth depth for a set of
// images. Choses image indices differently based on test image prefix
int testIrLimit(std::string test_image_path,
    std::string test_image_prefix,
    int num_images)
{
    if(!IPUtils::dirExists(test_image_path))
        throw std::runtime_error("Failed to find test image directory:" + test_image_path);  

    if(test_image_path.back() != '/')
        test_image_path += "/";

    std::vector<uint64_t> ir_acc(THRESHOLD_PARAM, 0);
    std::vector<int> depth_count(THRESHOLD_PARAM, 0);

    bool realsense = false;
    Random random;
    std::vector<int> rand_ints(num_images);
    std::string ir_image_suffix = "ir.png";
    std::string temp_string = test_image_prefix;
    int input_threshold = 38;
    if(test_image_prefix.compare("img")==0)
    {
        std::cout << "testing on training_realsense training set" << std::endl; 
        realsense = true;
        rand_ints = random.RandomVector(0,1200,num_images,false);
    }
    else if(test_image_prefix.compare("test")==0)
    {
        std::cout << "testing on training_images_2 ir test set" << std::endl; 
        realsense = true;
        rand_ints = random.RandomVector(10000,11000,num_images,false);
    }
    else if(test_image_prefix.compare("test_cam")==0)
    {
        std::cout << "testing on training_images_2 cam test set" << std::endl; 
        realsense = true;
        rand_ints = random.RandomVector(10000,11000,num_images,false);
        temp_string = "test";
        ir_image_suffix = "cam.png";
        input_threshold = 79;
    }
    else
    {
        std::cout << "testing on training_realsense_2 test set" << std::endl; 
        realsense = true;
        rand_ints = random.RandomVector(1200,1500,num_images,false);   
    }

    std::cout << "Threshold value used " << std::to_string(input_threshold) << std::endl;

    std::string img_path = test_image_path+temp_string;
    std::string depth_path = test_image_path+temp_string;
    std::string img_full_path;
    std::string depth_full_path;
    cv::Mat depth_image;
    cv::Mat test_image;
    cv::Mat test_image_pp;
    cv::Mat depth_th(480, 640, CV_16UC1);

    for(int i=0;i<num_images;i++)
    {
        int image_index;
        if(realsense)
            image_index = rand_ints[i];
        else
            image_index = 10000+i;

        img_full_path = img_path + std::to_string(image_index) + ir_image_suffix;
        depth_full_path = depth_path + std::to_string(image_index) + "depth.png";
        
        test_image = cv::imread(img_full_path, -1);
        depth_image = cv::imread(depth_full_path, -1);

        if((!depth_image.data)||(!test_image.data))
        {
            std::cerr << "Error loading images:\n\t" << img_full_path << "\n\t" << depth_full_path << std::endl;
            continue;
        }

        IPUtils::threshold16(depth_image, depth_th, THRESHOLD_PARAM, 65535, 4);
        test_image_pp = IPUtils::preProcess(test_image, input_threshold);

        for(int r=0;r<480;r++)
        {
            uint8_t* ir_pix = test_image_pp.ptr<uint8_t>(r);
            uint16_t* depth_pix = depth_th.ptr<uint16_t>(r);

            for(int c=0;c<640;c++)
            {
                ir_acc[depth_pix[c]] += ir_pix[c];
                depth_count[depth_pix[c]]++;
            }
        }

    }

    std::cout << "Finished, Saving data to " << OUT_PATH + test_image_prefix << std::endl;

    ofstream out_file;
    out_file.open(OUT_PATH + test_image_prefix);
    int temp;
    if(out_file.is_open())
    {
        for(int i=0;i<ir_acc.size();i++)
        {
            if(depth_count[i] != 0)
                temp = depth_count[i];
            else
                temp = 1;

            out_file << to_string(round((float(ir_acc[i])/temp))) << ";";
        }
        out_file.close();
    }
    else
    {
        std::cout << "Unable to open file" << std::endl;
    }

    return 0;
}

// Print "interactive mode" menu
void printMenu()
{
    std::cout << "*************************Forest training and testing*************************";
    std::cout << std::endl;
    std::cout << "Enter 1 to train Classification Forest in parallel" << std::endl;
    std::cout << "Enter 2 to train Regression Forest in parallel" << std::endl;
    std::cout << "Enter 6 to do applyMultiLevel" << std::endl;
    std::cout << "Enter 7 to test Regression Forest" << std::endl;
    std::cout << "Enter 8 to test Classification Forest" << std::endl;
    std::cout << "Enter 3 to test Colourisation" << std::endl;
    std::cout << "Enter 4 to test Thresholding" << std::endl;
    std::cout << "Enter q to quit" << std::endl;
    std::cout << "\n" << std::endl;
}

// access to some small bits of functionality, entirely for testing purposes
// hence hardcoded paths
void interactiveMode()
{
    bool cont = true;
    std::string in;
    std::string forest_path = "/home/james/workspace/forest_files/";
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
            //classifyOnline(forest_path);
            std::cout << "Nothing Here" << std::endl << std::endl;
            printMenu();
        }
        else if (in.compare("1") == 0)
        {
            //trainClassificationPar(FILE_PATH, forest_path, 100);
            std::cout << "Nothing Here" << std::endl << std::endl;
            printMenu();
        }
        else if (in.compare("2") == 0)
        {
            //trainRegressionPar(FILE_PATH, forest_path, 100);
            std::cout << "Nothing Here" << std::endl << std::endl;
            printMenu();
        }
        else if (in.compare("3") == 0)
        {
            std::cout << "Nothing Here" << std::endl << std::endl;
            printMenu();
        }
        else if (in.compare("4") == 0)
        {
            testFunction();
            printMenu();
        }
        else if (in.compare("q") == 0)
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

// Parses a program parameters file.
// INPUT: params_path - path to the .params file
// OUTPUT: ProgramParameters object containing all relevent processing info
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
                                    "TRAIN_ON_ZERO_IR",
                                    "MAX_RANGE",
                                    "TH_VALUE",
                                    "WEBCAM",
                                    "IGNORE_CLOSE"};

    int num_categories = 25;
    try
    {
        // Open the .params file
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

int main(int argc, char *argv[])
{
    // Parse input parameters
    
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
    else if(argc == 5)
    {
        std::string frst_arg = argv[1];
        if(frst_arg.compare("-ir") == 0)
        {
            std::string test_image_path = argv[2];
            std::string test_image_prefix = argv[3];
            int num_test_images = std::stoi(std::string(argv[4]));
            std::cout << "Test image path: " << test_image_path << std::endl;
            std::cout << "Test image prefix: " << test_image_prefix << std::endl;
            std::cout << "Images to use in testing: " << std::to_string(num_test_images) << std::endl;
            testIrLimit(test_image_path, 
                test_image_prefix, 
                num_test_images);
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
        else if(frst_arg.compare("-ta")==0)
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
            testForestAlternate(forest_path, 
                forest_prefix, 
                test_image_path, 
                test_image_prefix, 
                num_test_images);    
        }
        else if(frst_arg.compare("-tr")==0)
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
            testForestAlternateRH(forest_path, 
                forest_prefix, 
                test_image_path, 
                test_image_prefix, 
                num_test_images);    
        }
        else if(frst_arg.compare("-c")==0)
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
            classifyOnline(forest_path, 
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
