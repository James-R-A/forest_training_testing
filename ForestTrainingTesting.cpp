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
#define DEF_TREES 3
#define DEF_CAND_FEAT 10
#define DEF_THRESH 10
#define DEF_REG_LEVELS 20
#define DEF_CLASS_LEVELS 22
#define DEF_BINS 5
#define RHFR_FLAG false
#define PSR_FLAG true

#ifdef _WIN32
const std::string FILE_PATH = "D:\\";
#endif
#ifdef __linux__
const std::string FILE_PATH = "/media/james/data_wd/";
#endif

int trainClassification(std::string path,
    std::string save_path,
    int number_of_trees = DEF_TREES,
    int candidate_features = DEF_CAND_FEAT,
    int thresholds_per_feature = DEF_THRESH,
    int max_decision_levels = DEF_CLASS_LEVELS)
{
    
    std::cout << "Output forest filename?\t";
    std::string filename;
    std::cin >> filename;
    filename = save_path + filename;

    // Setup the program training parameters
    TrainingParameters training_parameters;
    training_parameters.NumberOfTrees = number_of_trees;
    training_parameters.MaxDecisionLevels = max_decision_levels;
    training_parameters.NumberOfCandidateFeatures = candidate_features;
    training_parameters.NumberOfCandidateThresholdsPerFeature = thresholds_per_feature;
    training_parameters.Verbose = false;
    training_parameters.max_threads = omp_get_max_threads();

    std::string file_path = path;
    std::cout << "Searching for some IR and depth images in " << file_path << std::endl;
    std::unique_ptr<DataPointCollection> training_data = DataPointCollection::LoadImagesClass(file_path,
        cv::Size(640, 480),
        false, 1, 0, 5);

    std::cout << "Data loaded here's how many samples: " << training_data->Count() << std::endl;
    std::cout << " each with dimensionality: " << training_data->Dimensions() << std::endl;

    if (RHFR_FLAG)
    {
        std::cout << "\nAttempting training" << std::endl;
        try
        {
            std::unique_ptr<Forest<RandomHyperplaneFeatureResponse, HistogramAggregator> > forest =
                Classifier<RandomHyperplaneFeatureResponse>::Train(*training_data, training_parameters);

            forest->Serialize(filename);
        }
        catch (const std::runtime_error& e)
        {
            std::cout << "Training Failed" << std::endl;
            std::cerr << e.what() << std::endl;;
        }
    }
    else if (PSR_FLAG)
    {
        std::cout << "\nAttempting training" << std::endl;
        try
        {
            std::unique_ptr<Forest<PixelSubtractionResponse, HistogramAggregator> > forest =
                Classifier<PixelSubtractionResponse>::Train(*training_data, training_parameters);

            forest->Serialize(filename);
        }
        catch (const std::runtime_error& e)
        {
            std::cout << "Training Failed" << std::endl;
            std::cerr << e.what() << std::endl;
        }
    }
    else
    {
        std::cout << "Invalid Feature response flags. Exiting" << std::endl;
        return -1;
    }

    std::cout << "Training complete, forest saved in :" << filename << std::endl;

    return 0;
}

int trainRegression(std::string path,
    std::string save_path,
    int number_of_trees = DEF_TREES,
    int candidate_features = DEF_CAND_FEAT,
    int thresholds_per_feature = DEF_THRESH,
    int max_decision_levels = DEF_REG_LEVELS)
{
    std::cout << "Output forest filename?\t";
    std::string filename;
    std::cin >> filename;
    filename = save_path + filename;

    // Setup the program training parameters
    TrainingParameters training_parameters;
    training_parameters.NumberOfTrees = number_of_trees;
    training_parameters.MaxDecisionLevels = max_decision_levels;
    training_parameters.NumberOfCandidateFeatures = candidate_features;
    training_parameters.NumberOfCandidateThresholdsPerFeature = thresholds_per_feature;
    training_parameters.Verbose = false;
    training_parameters.max_threads = omp_get_max_threads();

    // init the file path
    std::string file_path = path;
    std::cout << "Searching for some IR and depth images in " << file_path << std::endl;
    // create a DataPointCollection in the regression format
    std::unique_ptr<DataPointCollection> training_data = DataPointCollection::LoadImagesRegression(file_path,
        cv::Size(640, 480),
        false, 10);

    std::cout << "Data loaded here's how many samples: " << training_data->Count() << std::endl;
    std::cout << " each with dimensionality: " << training_data->Dimensions() << std::endl;

    // Train a regressoin forest
    std::cout << "\nAttempting training" << std::endl;
    try
    {
        std::unique_ptr<Forest<PixelSubtractionResponse, DiffEntropyAggregator> > forest = 
            Regressor<PixelSubtractionResponse>::Train(*training_data,training_parameters);

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

int trainClassificationPar(std::string path,
    std::string save_path,
    int training_images,
    int depth_bins = DEF_BINS,
    bool verbose_ = false,
    int number_of_trees = DEF_TREES,
    int candidate_features = DEF_CAND_FEAT,
    int thresholds_per_feature = DEF_THRESH,
    int max_decision_levels = DEF_CLASS_LEVELS)
{

    std::cout << "Output forest filename?\t";
    std::string filename;
    std::cin >> filename;
    filename = save_path + filename;

    // Setup the program training parameters
    TrainingParameters training_parameters;
    training_parameters.NumberOfTrees = number_of_trees;
    training_parameters.MaxDecisionLevels = max_decision_levels;
    training_parameters.NumberOfCandidateFeatures = candidate_features;
    training_parameters.NumberOfCandidateThresholdsPerFeature = thresholds_per_feature;
    training_parameters.Verbose = verbose_;
    training_parameters.max_threads = omp_get_max_threads();

    std::string file_path = path;
    std::cout << "Searching for some IR and depth images in " << file_path << std::endl;
    std::unique_ptr<DataPointCollection> training_data = DataPointCollection::LoadImagesClass(file_path,
        cv::Size(640, 480),
        false, training_images, 0, depth_bins);

    std::cout << "Data loaded here's how many samples: " << training_data->Count() << std::endl;
    std::cout << " each with dimensionality: " << training_data->Dimensions() << std::endl;

    if (RHFR_FLAG)
    {
        std::cout << "\nAttempting training" << std::endl;
        try
        {
            std::unique_ptr<Forest<RandomHyperplaneFeatureResponse, HistogramAggregator> > forest =
                Classifier<RandomHyperplaneFeatureResponse>::TrainPar(*training_data, training_parameters);

            forest->Serialize(filename);
        }
        catch (const std::runtime_error& e)
        {
            std::cout << "Training Failed" << std::endl;
            std::cerr << e.what() << std::endl;
        }
    }
    else if (PSR_FLAG)
    {
        std::cout << "\nAttempting training" << std::endl;
        try
        {
            std::unique_ptr<Forest<PixelSubtractionResponse, HistogramAggregator> > forest = 
                Classifier<PixelSubtractionResponse>::TrainPar(*training_data, training_parameters);

            forest->Serialize(filename);
        }
        catch (const std::runtime_error& e)
        {
            std::cout << "Training Failed" << std::endl;
            std::cerr << e.what() << std::endl;
        }
    }
    else
    {
        std::cout << "Invalid Feature response flags. Exiting" << std::endl;
        return -1;
    }

    std::cout << "Training complete, forest saved in :" << filename << std::endl;

    return 0;
}

int trainRegressionPar(std::string path,
    std::string save_path,
    int training_images,
    int depth_bins = DEF_BINS,
    int expert_class_no = -1,
    bool verbose_ = false,
    int number_of_trees = DEF_TREES,
    int candidate_features = DEF_CAND_FEAT,
    int thresholds_per_feature = DEF_THRESH,
    int max_decision_levels = DEF_REG_LEVELS)
{
    std::cout << "Output forest filename?\t";
    std::string filename;
    std::cin >> filename;
    filename = save_path + filename;

    // Setup the program training parameters
    TrainingParameters training_parameters;
    training_parameters.NumberOfTrees = number_of_trees;
    training_parameters.MaxDecisionLevels = max_decision_levels;
    training_parameters.NumberOfCandidateFeatures = candidate_features;
    training_parameters.NumberOfCandidateThresholdsPerFeature = thresholds_per_feature;
    training_parameters.Verbose = verbose_;
    training_parameters.max_threads = omp_get_max_threads();

    // init the file path
    std::string file_path = path;
    std::cout << "Searching for some IR and depth images in " << file_path << std::endl;
    // create a DataPointCollection in the regression format
    std::unique_ptr<DataPointCollection> training_data = DataPointCollection::LoadImagesRegression(file_path,
        cv::Size(640, 480),
        false, training_images, 0, depth_bins, true, expert_class_no);

    std::cout << "Data loaded here's how many samples: " << training_data->Count() << std::endl;
    std::cout << " each with dimensionality: " << training_data->Dimensions() << std::endl;

    // Train a regressoin forest
    std::cout << "\nAttempting training" << std::endl;
    try
    {
        std::unique_ptr<Forest<PixelSubtractionResponse, DiffEntropyAggregator> > forest =
            Regressor<PixelSubtractionResponse>::TrainPar(*training_data, training_parameters);

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

int testMethod(std::string dir_path)
{
    if (!IPUtils::dirExists(dir_path))
    {
        std::cout << dir_path << " not found. Ensure all drives are mounted" << std::endl;
        return 0;
    }

    std::cout << "Looking in:\t" << dir_path << std::endl << "Filename?\t";
    std::string filename;
    std::cin >> filename;
    std::string full_path = dir_path + filename;
    std::cout << "Attempting to deserialize forest from " << full_path << std::endl;


    std::unique_ptr<Forest<PixelSubtractionResponse, HistogramAggregator> > forest =
        Forest<PixelSubtractionResponse, HistogramAggregator>::Deserialize(full_path);

    //std::unique_ptr<Forest<RandomHyperplaneFeatureResponse, HistogramAggregator> > forest =
    //  Forest<RandomHyperplaneFeatureResponse, HistogramAggregator>::Deserialize(full_path);

    std::cout << "Forest loaded:" << std::endl;
    std::cout << "Trees:\t" << std::to_string(forest->TreeCount()) << std::endl;

    int64 start_time = cv::getTickCount(); //////////////////
    std::unique_ptr<DataPointCollection> test_data = DataPointCollection::LoadImagesClass(FILE_PATH,
        cv::Size(640, 480),
        false, 1, 11);
    int64 process_time = (((cv::getTickCount() - start_time) / cv::getTickFrequency()) * 1000);///////////////
    std::cout << "Process time data load:" << std::to_string(process_time) << std::endl;


    cv::Mat testMat;
    testMat = cv::imread(FILE_PATH+"test11ir.png", -1);
    if (!testMat.data)
        throw;

    start_time = cv::getTickCount(); //////////////////
    std::unique_ptr<DataPointCollection> test_data1 = DataPointCollection::LoadMat(testMat, cv::Size(640, 480));
    process_time = (((cv::getTickCount() - start_time) / cv::getTickFrequency()) * 1000);///////////////
    std::cout << "Process time data load:" << std::to_string(process_time) << std::endl;

    start_time = cv::getTickCount();
    cv::Mat bins_mat = Classifier<PixelSubtractionResponse>::ApplyMat(*forest, *test_data1);


    std::vector<uchar> bins_vec = IPUtils::vectorFromBins(bins_mat, cv::Size(640, 480));
    cv::Mat result_mat1 = cv::Mat(480, 640, CV_8UC1, (uint8_t*)bins_vec.data());
    cv::Mat result_norm1;
    cv::normalize(result_mat1, result_norm1, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::Mat depth_mat = cv::Mat(480, 640, CV_8UC1, (uint8_t*)test_data->labels_.data());
    cv::Mat depth_norm;

    cv::normalize(depth_mat, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);


    cv::imshow("test", depth_norm);
    cv::imshow("result1", result_norm1);
    process_time = (((cv::getTickCount() - start_time) / cv::getTickFrequency()) * 1000);
    std::cout << "Process time:" << std::to_string(process_time) << std::endl;

    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;

}

int regressOnline(std::string dir_path)
{
    if (!IPUtils::dirExists(dir_path))
    {
        std::cout << dir_path << " not found. Ensure all drives are mounted" << std::endl;
        return 0;
    }

    std::cout << "Looking in:\t" << dir_path << std::endl << "Filename?\t";
    std::string filename;
    std::cin >> filename;
    std::string forest_path = dir_path + filename;
    std::cout << "Attempting to deserialize forest from " << forest_path << std::endl;

    std::string img_path = dir_path + "test";
    std::string pathstring;
    cv::Mat test_image = cv::Mat(cv::Size(640, 480), CV_8UC1);
    std::vector<int16_t> reg_result;
    cv::Mat reg_mat;
    cv::Mat result_norm1;

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
        cv::normalize(reg_mat, result_norm1, 0, 65535, cv::NORM_MINMAX, CV_16UC1);
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

int classifyOnline(std::string dir_path)
{
    if (!IPUtils::dirExists(dir_path))
    {
        std::cout << dir_path << " not found. Ensure all drives are mounted" << std::endl;
        return 0;
    }

    std::cout << "Looking in:\t" << dir_path << std::endl << "Filename?\t";
    std::string filename;
    std::cin >> filename;
    std::string forest_path = dir_path + filename;
    std::cout << "Attempting to deserialize forest from " << forest_path << std::endl;

    std::string img_path = dir_path + "test";
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
        cv::Mat result_mat1 = cv::Mat(480, 640, CV_8UC1, (uint8_t*)bins_vec.data());
        cv::normalize(result_mat1, result_norm1, 0, 255, cv::NORM_MINMAX, CV_8UC1);
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

void printMenu()
{
    std::cout << "*************************Forest training and testing*************************";
    std::cout << std::endl;
    std::cout << "Enter 6 to compare a depth image and classified image" << std::endl;
    std::cout << "Enter 7 to test Regression Forest" << std::endl;
    std::cout << "Enter 8 to test Classification Forest" << std::endl;
    std::cout << "Enter r to train Regression Forest" << std::endl;
    std::cout << "Enter c to train Classification Forest" << std::endl;
    std::cout << "Enter 1 to train Classification Forest in parallel" << std::endl;
    std::cout << "Enter 2 to train Regression Forest in parallel" << std::endl;
    std::cout << "Enter q to quit" << std::endl;
    std::cout << "\n" << std::endl;
}

int main(int argc, char *argv[])
{
    bool cont = true;
    std::string in;
    std::string forest_path = FILE_PATH;
    printMenu();

    // Poll for user input to chose program mode
    while (cont)
    {
        in == "";
        std::cin >> in;

        if (in.compare("6") == 0)
        {
            testMethod(forest_path);
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
        else if (in.compare("c") == 0)
        {
            trainClassification(FILE_PATH, forest_path);
            printMenu();
        }
        else if (in.compare("r") == 0)
        {
            trainRegression(FILE_PATH, forest_path);
            printMenu();
        }
        else if (in.compare("1") == 0)
        {
            trainClassificationPar(FILE_PATH, forest_path, 100);
            printMenu();
        }
        else if (in.compare("2") == 0)
        {
            trainRegressionPar(FILE_PATH, forest_path, 100);
            printMenu();
        }
        else if (in.compare("q") == 0)
            cont = false;

        // Refresh cin buffer
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }


}