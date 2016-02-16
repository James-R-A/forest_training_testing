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

int trainClassification(std::string path,
    std::string save_path,
    int max_decision_levels = DEF_CLASS_LEVELS)
{
    
    std::cout << "Output forest filename?\t";
    std::string filename;
    std::cin >> filename;
    filename = save_path + filename;

    // Setup the program training parameters
    TrainingParameters training_parameters;
    training_parameters.MaxDecisionLevels = max_decision_levels;

    std::string file_path = path;
    std::cout << "Searching for some IR and depth images in " << file_path << std::endl;
    std::unique_ptr<DataPointCollection> training_data = DataPointCollection::LoadImagesClass(file_path,
        cv::Size(640, 480),
        false, 1, 0, 5);

    std::cout << "Data loaded here's how many samples: " << training_data->Count() << std::endl;
    std::cout << " each with dimensionality: " << training_data->Dimensions() << std::endl;

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
     

    std::cout << "Training complete, forest saved in :" << filename << std::endl;

    return 0;
}

int trainRegression(std::string path,
    std::string save_path,
    int max_decision_levels = DEF_REG_LEVELS)
{
    std::cout << "Output forest filename?\t";
    std::string filename;
    std::cin >> filename;
    filename = save_path + filename;

    // Setup the program training parameters
    TrainingParameters training_parameters;
    training_parameters.MaxDecisionLevels = max_decision_levels;

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
    int max_decision_levels = DEF_CLASS_LEVELS)
{

    if (!IPUtils::dirExists(path))
        return 0;

    if (!IPUtils::dirExists(save_path))
        return 0;

    std::cout << "Output forest filename?\t";
    std::string filename;
    std::cin >> filename;
    filename = save_path + filename;

    // Setup the program training parameters
    TrainingParameters training_parameters;
    training_parameters.MaxDecisionLevels = max_decision_levels;
    
    std::string file_path = path;
    std::cout << "Searching for some IR and depth images in " << file_path << std::endl;
    std::unique_ptr<DataPointCollection> training_data = DataPointCollection::LoadImagesClass(file_path,
        cv::Size(640, 480),
        false, training_images, 0);

    std::cout << "Data loaded here's how many samples: " << training_data->Count() << std::endl;
    std::cout << " each with dimensionality: " << training_data->Dimensions() << std::endl;
 
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

    std::cout << "Training complete, forest saved in :" << filename << std::endl;

    return 0;
}

int trainRegressionPar(std::string path,
    std::string save_path,
    int training_images,
    int max_decision_levels = DEF_REG_LEVELS)
{

    if (!IPUtils::dirExists(path))
        return 0;

    if (!IPUtils::dirExists(save_path))
        return 0;

    std::cout << "Output forest filename?\t";
    std::string filename;
    std::cin >> filename;
    filename = save_path + filename;

    // Setup the program training parameters
    TrainingParameters training_parameters;
    training_parameters.MaxDecisionLevels = max_decision_levels;
    
    // init the file path
    std::string file_path = path;
    std::cout << "Searching for some IR and depth images in " << file_path << std::endl;
    // create a DataPointCollection in the regression format
    std::unique_ptr<DataPointCollection> training_data = DataPointCollection::LoadImagesRegression(file_path,
        cv::Size(640, 480),
        false, training_images, 0); // TODO change

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
        return 0;
    

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
        return 0;

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
        return 0;

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

int trainExperts(std::string path, std::string save_path, std::string save_prefix)
{
    for(int i=0;i<5;i++)
    {
        std::string filename = save_prefix + to_string(i);

    }
}

void printMenu()
{
    std::cout << "*************************Forest training and testing*************************";
    std::cout << std::endl;
    std::cout << "Enter 6 to compare a depth image and classified image" << std::endl;
    std::cout << "Enter 7 to test Regression Forest" << std::endl;
    std::cout << "Enter 8 to test Classification Forest" << std::endl;
    // std::cout << "Enter r to train Regression Forest" << std::endl;
    // std::cout << "Enter c to train Classification Forest" << std::endl;
    std::cout << "Enter 1 to train Classification Forest in parallel" << std::endl;
    std::cout << "Enter 2 to train Regression Forest in parallel" << std::endl;
    std::cout << "Enter 3 to train Expert Regressors" << std::endl;
    std::cout << "Enter q to quit" << std::endl;
    std::cout << "\n" << std::endl;
}

void interactiveMode()
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

void printUseage(bool incorrect=false)
{
    if(incorrect)
        std::cout << "Incorrect input." << std::endl;

    std::cout << "Command Line Useage:\n\t./FTT /path/to/params.params /path/to/trainingImages" << std::endl;
    std::cout << "Or, if your params file contains the path to training images, pass only the params file" << std::endl;
    std::cout << "To use interactively, pass no cla's" << std::endl;
}

ProgramParameters getParamsFromFile(std::string& params_path)
{
    ProgramParameters return_params;
    std::string line;
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
                                    "MAX_THREADS"};

    try
    {
        ifstream params_file(params_path);
        if(params_file.is_open())
        {
            for(int i=0;i<15;i++)
            {
                // line by line in the file
                // find category string in line
                // if found, send next line to return_params.setParam(categories[i], line)
                // clear and reset file
                while(std::getline(params_file, line))
                {
                    if(line.find(categories[i]) != std::string::npos)
                    {
                        std::getline(params_file, line);
                        std::cout << line;
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
    return_params.setParam("TRAINING_IMAGES_PATH", images_path);

    return return_params;
}

int main(int argc, char *argv[])
{
    if( argc < 2 )
    {
        printUseage();
        std::cout << "Interactive mode" << std::endl;
        interactiveMode();    
    }
    else if(argc == 2)
    {
        std::string params_path = argv[1];
        ProgramParameters progParams = getParamsFromFile(params_path);
    }
    else if(argc == 3)
    {
        std::string params_path = argv[1];
        std::string images_path = argv[2];
        ProgramParameters progParams = getParamsFromFile(params_path, images_path);
    }
    else
    {
        printUseage(true);
    }

}