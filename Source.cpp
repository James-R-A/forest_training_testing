#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <algorithm>
#include <Windows.h>

#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <pxccapture.h>

#include "IPUtils.h"

#include "Sherwood.h"
#include "DataPointCollection.h"
#include "Classification.h"
#include "Regression.h"

using namespace std;
using namespace MicrosoftResearch::Cambridge::Sherwood;

#define LOOP_DELAY 30
#define FILE_PATH "D:\\"
#define SAVE_PATH "D:\\test"
#define RHFR_FLAG false
#define PSR_FLAG true

bool dirExists(const std::string& dirName_in)
{
	DWORD ftyp = GetFileAttributesA(dirName_in.c_str());
	if (ftyp == INVALID_FILE_ATTRIBUTES)
		return false;  //something is wrong with your path!

	if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
		return true;   // this is a directory!

	return false;    // this is not a directory!
}

int trainClassification(string path,
	string save_path,
	int number_of_trees = 1,
	int candidate_features = 10,
	int thresholds_per_feature = 10,
	int max_decision_levels = 5)
{
	/*
	Training data path
	Forest output path
	Forest input path
	Test data path
	Validation data path

	set training parameters
	Max decision levels
	Number of candidate features
	Number of candidate thresholds per feature
	Number of trees]

	load training data, or load Some training data, train and plot test set.
	*/

	std::cout << "Output forest filename?\t";
	string filename;
	std::cin >> filename;
	filename = save_path + filename;

	// Setup the program training parameters
	TrainingParameters training_parameters;
	training_parameters.NumberOfTrees = number_of_trees;
	training_parameters.MaxDecisionLevels = max_decision_levels;
	training_parameters.NumberOfCandidateFeatures = candidate_features;
	training_parameters.NumberOfCandidateThresholdsPerFeature = thresholds_per_feature;
	training_parameters.Verbose = false;

	string file_path = path;
	std::cout << "Searching for some IR and depth images in " << file_path << endl;
	std::auto_ptr<DataPointCollection> training_data = DataPointCollection::LoadImages(file_path,
		DataDescriptor::Classes,
		cv::Size(640, 480),
		false, 1, 0, 5);

	std::cout << "Data loaded here's how many samples: " << training_data->Count() << std::endl;
	std::cout << " each with dimensionality: " << training_data->Dimensions() << std::endl;

	if (RHFR_FLAG)
	{
		std::cout << "\nAttempting training" << endl;
		std::auto_ptr<Forest<RandomHyperplaneFeatureResponse, HistogramAggregator>> forest = Classifier<RandomHyperplaneFeatureResponse>::Train(
			*training_data,
			training_parameters);

		forest->Serialize(filename);
	}
	else if (PSR_FLAG)
	{
		std::cout << "\nAttempting training" << endl;
		std::auto_ptr<Forest<PixelSubtractionResponse, HistogramAggregator>> forest = Classifier<PixelSubtractionResponse>::Train(
			*training_data,
			training_parameters);

		forest->Serialize(filename);
	}
	else
	{
		std::cout << "Invalid Feature response flags. Exiting" << endl;
		return -1;
	}

	std::cout << "Training complete, forest saved in :" << filename << endl;

	return 0;
}

int trainRegression(string path,
	string save_path,
	int number_of_trees = 3,
	int candidate_features = 10,
	int thresholds_per_feature = 10,
	int max_decision_levels = 20)
{
	std::cout << "Output forest filename?\t";
	string filename;
	std::cin >> filename;
	filename = save_path + filename;

	// Setup the program training parameters
	TrainingParameters training_parameters;
	training_parameters.NumberOfTrees = number_of_trees;
	training_parameters.MaxDecisionLevels = max_decision_levels;
	training_parameters.NumberOfCandidateFeatures = candidate_features;
	training_parameters.NumberOfCandidateThresholdsPerFeature = thresholds_per_feature;
	training_parameters.Verbose = false;

	string file_path = path;
	std::cout << "Searching for some IR and depth images in " << file_path << endl;
	std::auto_ptr<DataPointCollection> training_data = DataPointCollection::LoadImages(file_path,
		DataDescriptor::TargetValues,
		cv::Size(640, 480),
		false, 100);

	std::cout << "Data loaded here's how many samples: " << training_data->Count() << std::endl;
	std::cout << " each with dimensionality: " << training_data->Dimensions() << std::endl;

	std::cout << "\nAttempting training" << endl;
	std::auto_ptr<Forest<PixelSubtractionResponse, DiffEntropyAggregator>> forest = Regressor<PixelSubtractionResponse>::Train(
		*training_data,
		training_parameters);

	forest->Serialize(filename);
	std::cout << "Training complete, forest saved in :" << filename << endl;

	return 0;

}

int testMethod(string dir_path)
{
	if (!dirExists(dir_path))
		return 0;

	std::cout << "Looking in:\t" << dir_path << endl << "Filename?\t";
	string filename;
	std::cin >> filename;
	string full_path = dir_path + filename;
	std::cout << "Attempting to deserialize forest from " << full_path << endl;


	std::auto_ptr<Forest<PixelSubtractionResponse, HistogramAggregator>> forest =
		Forest<PixelSubtractionResponse, HistogramAggregator>::Deserialize(full_path);

	//std::auto_ptr<Forest<RandomHyperplaneFeatureResponse, HistogramAggregator>> forest =
	//	Forest<RandomHyperplaneFeatureResponse, HistogramAggregator>::Deserialize(full_path);

	std::cout << "Forest loaded:" << endl;
	std::cout << "Trees:\t" << to_string(forest->TreeCount()) << endl;

	int64 start_time = cv::getTickCount(); //////////////////
	std::auto_ptr<DataPointCollection> test_data = DataPointCollection::LoadImages(FILE_PATH,
		DataDescriptor::Classes,
		cv::Size(640, 480),
		false, 1, 11);
	int64 process_time = (((cv::getTickCount() - start_time) / cv::getTickFrequency()) * 1000);///////////////
	std::cout << "Process time data load:" << to_string(process_time) << endl;


	cv::Mat testMat;
	testMat = cv::imread("D:\\test11ir.png", -1);
	if (!testMat.data)
		throw;

	start_time = cv::getTickCount(); //////////////////
	std::auto_ptr<DataPointCollection> test_data1 = DataPointCollection::LoadMat(testMat, cv::Size(640, 480));
	process_time = (((cv::getTickCount() - start_time) / cv::getTickFrequency()) * 1000);///////////////
	std::cout << "Process time data load:" << to_string(process_time) << endl;

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
	std::cout << "Process time:" << to_string(process_time) << endl;

	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;

}

int regressOnline()
{
	string dir_path = "D:\\test";
	string pathstring;
	cv::Mat test_image = cv::Mat(cv::Size(640, 480), CV_8UC1);
	std::vector<int16_t> reg_result;
	cv::Mat reg_mat;
	cv::Mat result_norm1;

	// load forest
	std::auto_ptr<Forest<PixelSubtractionResponse, DiffEntropyAggregator>> forest =
		Forest<PixelSubtractionResponse, DiffEntropyAggregator>::Deserialize("D:\\reg3t20l100i");
	// Create ForestShared from loaded forest
	std::unique_ptr<ForestShared<PixelSubtractionResponse, DiffEntropyAggregator>> forest_shared =
		ForestShared<PixelSubtractionResponse, DiffEntropyAggregator>::ForestSharedFromForest(*forest);
	// Delete original forest. May roll these steps into one later if we don't need a regular forest application.
	forest->~Forest();

	for (int i = 0; i < 106; i++)
	{
		int64 start_time = cv::getTickCount();
		pathstring = dir_path + to_string(i) + "ir.png";
		test_image = cv::imread(pathstring, -1);
		if (!test_image.data)
			continue;

		std::auto_ptr<DataPointCollection> test_data1 = DataPointCollection::LoadMat(test_image, cv::Size(640, 480));
		reg_result = Regressor<PixelSubtractionResponse>::ApplyMat(*forest_shared, *test_data1);

		reg_mat = cv::Mat(480, 640, CV_16UC1, (uint16_t*)reg_result.data());
		cv::normalize(reg_mat, result_norm1, 0, 65535, cv::NORM_MINMAX, CV_16UC1);
		cv::imshow("output", result_norm1);
		int64 process_time = (((cv::getTickCount() - start_time) / cv::getTickFrequency()) * 1000);
		std::cout << "Process time: " << to_string(process_time) << endl;
		int wait_time = std::max(2, (int)(LOOP_DELAY - process_time));
		cv::waitKey(wait_time);
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}

int classifyOnline()
{
	string dir_path = "D:\\test";
	string pathstring;
	cv::Mat test_image = cv::Mat(cv::Size(640, 480), CV_8UC1);
	cv::Mat bins_mat;
	cv::Mat result_norm1;

	// load forest
	std::auto_ptr<Forest<PixelSubtractionResponse, HistogramAggregator>> forest =
		Forest<PixelSubtractionResponse, HistogramAggregator>::Deserialize("D:\\t3l22s100");
	// Create ForestShared from loaded forest
	std::unique_ptr<ForestShared<PixelSubtractionResponse, HistogramAggregator>> forest_shared =
		ForestShared<PixelSubtractionResponse, HistogramAggregator>::ForestSharedFromForest(*forest);
	// Delete original forest. May roll these steps into one later if we don't need a regular forest application.
	forest->~Forest();

	for (int i = 0; i < 106; i++)
	{
		int64 start_time = cv::getTickCount();
		pathstring = dir_path + to_string(i) + "ir.png";
		test_image = cv::imread(pathstring, -1);
		if (!test_image.data)
			continue;

		std::auto_ptr<DataPointCollection> test_data1 = DataPointCollection::LoadMat(test_image, cv::Size(640, 480));
		bins_mat = Classifier<PixelSubtractionResponse>::ApplyMat(*forest_shared, *test_data1);
		std::vector<uchar> bins_vec = IPUtils::vectorFromBins(bins_mat, cv::Size(640, 480));
		cv::Mat result_mat1 = cv::Mat(480, 640, CV_8UC1, (uint8_t*)bins_vec.data());
		cv::normalize(result_mat1, result_norm1, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::imshow("output", result_norm1);
		int64 process_time = (((cv::getTickCount() - start_time) / cv::getTickFrequency()) * 1000);
		std::cout << "Process time: " << to_string(process_time) << endl;
		int wait_time = std::max(2, (int)(LOOP_DELAY - process_time));
		cv::waitKey(wait_time);
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}

void printMenu()
{
	std::cout << "*************************Forest training and testing*************************";
	std::cout << endl;
	std::cout << "Enter 6 to compare a depth image and classified image" << endl;
	std::cout << "Enter 7 to test Regression Forest" << endl;
	std::cout << "Enter 8 to test Classification Forest" << endl;
	std::cout << "Enter r to train Regression Forest" << endl;
	std::cout << "Enter c to train Classification Forest" << endl;
	std::cout << "Enter q to quit" << endl;
	std::cout << "\n" << endl;
}

int main()
{
	bool cont = true;
	string in;
	string forest_path = "D:\\";
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
			regressOnline();
			printMenu();
		}
		else if (in.compare("8") == 0)
		{
			classifyOnline();
			printMenu;
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
		else if (in.compare("q") == 0)
			cont = false;

		// Refresh cin buffer
		std::cin.clear();
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}


}