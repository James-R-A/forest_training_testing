#include "DataPointCollection.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
	bool dirExists(const std::string& dirName_in)
	{
		DWORD ftyp = GetFileAttributesA(dirName_in.c_str());
		if (ftyp == INVALID_FILE_ATTRIBUTES)
			return false;  //something is wrong with your path!

		if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
			return true;   // this is a directory!

		return false;    // this is not a directory!
	}

	// Iterares through a depth image (16 bit uint) and classifies each pixel into correct bin as 
	// specified in the pix_to_label look-up table
	cv::Mat createLabelMatrix(cv::Mat depth_image, std::vector<int> pix_to_label)
	{
		cv::Size mat_size = depth_image.size();
		cv::Mat label_mat(mat_size, CV_8UC1);
		int max = pix_to_label.size() - 1;
		// iterate through pixels in depth image, bin them and assign the depth label
		for (int r = 0; r < mat_size.height; r++)
		{
			uint16_t* d_pixel = depth_image.ptr<uint16_t>(r);
			uchar* label_pixel = label_mat.ptr<uchar>(r);
			for (int c = 0; c < mat_size.width; c++)
			{
				if (d_pixel[c] <= max)
					label_pixel[c] = pix_to_label[d_pixel[c]];
				else
					label_pixel[c] = 0;
			}
		}

		return label_mat;
	}

	std::auto_ptr<DataPointCollection> DataPointCollection::LoadImagesClass(
		std::string path, 
		cv::Size img_size,
		bool depth_raw,
		int number, 
		int start,
		int num_classes,
		bool zero_class_label,
		int patch_size)
	{
		if (!dirExists(path))
			throw std::runtime_error("Failed to find directory:\t" + path);

		if (patch_size % 2 == 0)
			throw std::runtime_error("Patch size must be odd");
		
		int first = start;
		int last = first + number -1;

		// Set up DataPointCollection object
		std::auto_ptr<DataPointCollection> result = std::auto_ptr<DataPointCollection>(new DataPointCollection());
		result->dimension_ = patch_size * patch_size;
		result->depth_raw = depth_raw;
		result->image_size = img_size;
		result->data_vec_size = number * img_size.height * img_size.width;

		// Data allocated using data_.resize(result->data_vec_size) 
		// then result->data_[n] = value; because it's faster than push_back
		// speed change ~= x2
		result->images_.resize(number);
		result->data_.resize(result->data_vec_size);
		int img_no = 0;
		int datum_no = 0;
		//int target_no = 0;
		int label_no = 0;
		//if (bHasTargetValues)
		// 	result->targets_.resize(result->data_vec_size);

		result->labels_.resize(result->data_vec_size);
		
		// Variables affecting class formation
		bool zero_class = zero_class_label;
		int total_classes = num_classes;
		// This max parameter is important. Don't forget this is aimed at 16 bit unsigned ints, 
		// so the viable range is 0-65535. 
		// If not using RAW depth data format, it's measured in mm, so be sensible (i.e. 1000 - 1500 mm?)
		int max = depth_raw ? 65000 : 1200;
		result->pixelLabels_ = IPUtils::generateDepthBinMap(zero_class, total_classes, max);
		
		cv::Mat ir_image, ir_preprocessed, depth_image, depth_labels;
		std::string ir_path;
		std::string depth_path;
		cv::Size ir_size, depth_size;

		for (int i = first; i <= last;i++)
		{
			// generate individual image paths
			ir_path = path + "test" + to_string(i) + "ir.png";
			depth_path = path + "test" + to_string(i) + "depth.png";

			std::cout << to_string(i) << endl;
			// read depth and ir images
			ir_image = cv::imread(ir_path, -1);
			depth_image = cv::imread(depth_path, -1);

			// if program fails to open image
			if(!ir_image.data)
				throw std::runtime_error("Failed to open image:\n\t" + ir_path);

			if (!depth_image.data)
				throw std::runtime_error("Failed to open image:\n\t" + depth_path);

			// If the datatypes in the images are incorrect
			if (IPUtils::getTypeString(ir_image.type()) != "8UC1")
				throw std::runtime_error("Encountered image with unexpected content type:\n\t" + ir_path);

			if (IPUtils::getTypeString(depth_image.type()) != "16UC1")
				throw std::runtime_error("Encountered image with unexpected content type:\n\t" + depth_path);

			ir_size = ir_image.size();
			depth_size = depth_image.size();
			if (ir_size != depth_size)
				throw std::runtime_error("Depth and IR images not the same size:\n\t" + ir_path + depth_path);

			// Send the ir image for preprocessing, default values used for now
			ir_preprocessed = IPUtils::preProcess(ir_image);
			result->images_[img_no] = ir_preprocessed;

			// Create matrix of depth labels (ie depth bins)
			depth_labels = createLabelMatrix(depth_image, result->pixelLabels_);
			// iterate through depth_labels matrix and add each element
			// to results.
			// Also set up data vector
			for (int r = 0; r < depth_size.height; r++)
			{
				uchar* label_pixel = depth_labels.ptr<uchar>(r);
				for (int c = 0; c < depth_size.width; c++)
				{
					result->labels_[label_no] = label_pixel[c];
					label_no++;
					result->data_[datum_no] = tuple<cv::Mat*, cv::Point>(&(result->images_[img_no]), cv::Point(c, r));
					datum_no++;
				}
			}
			img_no++;
		}
		
		return result;
	}

	std::auto_ptr<DataPointCollection> DataPointCollection::LoadImagesRegression(
		std::string path,
		cv::Size img_size,
		bool depth_raw,
		int number,
		int start,
		int num_classes,
		bool zero_class_label,
		int class_number,
		int patch_size)
	{
		if (!dirExists(path))
			throw std::runtime_error("Failed to find directory:\t" + path);

		if (patch_size % 2 == 0)
			throw std::runtime_error("Patch size must be odd");

		int first = start;
		int last = first + number - 1;

		// Set up DataPointCollection object
		std::auto_ptr<DataPointCollection> result = std::auto_ptr<DataPointCollection>(new DataPointCollection());
		result->dimension_ = patch_size * patch_size;
		result->depth_raw = depth_raw;
		result->image_size = img_size;
		result->data_vec_size = number * img_size.height * img_size.width;

		// Data allocated using data_.resize(result->data_vec_size) 
		// then result->data_[n] = value; because it's faster than push_back
		// speed change ~= x2
		result->images_.resize(number);
		result->data_.resize(result->data_vec_size);
		int img_no = 0;
		int datum_no = 0;
		int target_no = 0;
		
		result->targets_.resize(result->data_vec_size);

		// Variables affecting class formation
		bool zero_class = zero_class_label;
		int total_classes = num_classes;
		// This max parameter is important. Don't forget this is aimed at 16 bit unsigned ints, 
		// so the viable range is 0-65535. 
		// If not using RAW depth data format, it's measured in mm, so be sensible (i.e. 1000 - 1500 mm?)
		int max = depth_raw ? 65000 : 1200;
		result->pixelLabels_ = IPUtils::generateDepthBinMap(zero_class, total_classes, max);

		cv::Mat ir_image, ir_preprocessed, depth_image, depth_labels;
		std::string ir_path;
		std::string depth_path;
		cv::Size ir_size, depth_size;

		for (int i = first; i <= last; i++)
		{
			// generate individual image paths
			ir_path = path + "test" + to_string(i) + "ir.png";
			depth_path = path + "test" + to_string(i) + "depth.png";

			std::cout << to_string(i) << endl;
			// read depth and ir images
			ir_image = cv::imread(ir_path, -1);
			depth_image = cv::imread(depth_path, -1);

			// if program fails to open image
			if (!ir_image.data)
				throw std::runtime_error("Failed to open image:\n\t" + ir_path);

			if (!depth_image.data)
				throw std::runtime_error("Failed to open image:\n\t" + depth_path);

			// If the datatypes in the images are incorrect
			if (IPUtils::getTypeString(ir_image.type()) != "8UC1")
				throw std::runtime_error("Encountered image with unexpected content type:\n\t" + ir_path);

			if (IPUtils::getTypeString(depth_image.type()) != "16UC1")
				throw std::runtime_error("Encountered image with unexpected content type:\n\t" + depth_path);

			ir_size = ir_image.size();
			depth_size = depth_image.size();
			if (ir_size != depth_size)
				throw std::runtime_error("Depth and IR images not the same size:\n\t" + ir_path + depth_path);

			// Send the ir image for preprocessing, default values used for now
			ir_preprocessed = IPUtils::preProcess(ir_image);
			result->images_[img_no] = ir_preprocessed;
			
			// Create matrix of depth labels (ie depth bins)
			depth_labels = createLabelMatrix(depth_image, result->pixelLabels_);

			// Iterate through the depth image and add each element
			// to results.
			// Also set up data vector
			for (int r = 0; r < depth_size.height; r++)
			{
				uint16_t* depth_pixel = depth_image.ptr<uint16_t>(r);
				uchar* label_pixel = depth_labels.ptr<uchar>(r);
				for (int c = 0; c < depth_size.width; c++)
				{
					if (class_number == label_pixel[c] || class_number == -1)
					{
						result->targets_[target_no] = depth_pixel[c];
						target_no++;
						result->data_[datum_no] = tuple<cv::Mat*, cv::Point>(&(result->images_[img_no]), cv::Point(c, r));
						datum_no++;
					}
				}
			}

			img_no++;
		}
		// Resize data and targets vector to however full they are.
		result->data_.resize(datum_no);
		result->data_.shrink_to_fit();
		// Shrink to fit new size to free up excess memory.
		result->targets_.resize(target_no);
		result->targets_.shrink_to_fit();

		return result;
	}

	std::auto_ptr<DataPointCollection> DataPointCollection::LoadMat(cv::Mat mat_in, cv::Size img_size)
	{
		// If the datatypes in the images are incorrect
		if (IPUtils::getTypeString(mat_in.type()) != "8UC1")
			throw std::runtime_error("Incorrect image type, expecting CV_8UC1");

		// Set up DataPointCollection object
		std::auto_ptr<DataPointCollection> result = std::auto_ptr<DataPointCollection>(new DataPointCollection());
		result->dimension_ = 1;
		result->image_size = img_size;
		result->images_.resize(1);
		// Send the ir image for preprocessing, default values used for now
		cv::Mat ir_preprocessed = IPUtils::preProcess(mat_in);
		result->images_[0] = ir_preprocessed;
		result->data_vec_size = img_size.height * img_size.width;
		result->data_.resize(result->data_vec_size);

		int datum_no = 0;
		for (unsigned int r = 0; r < img_size.height; r++)
		{

			for (unsigned int c = 0; c < img_size.width; c++)
			{
				result->data_[datum_no] = tuple<cv::Mat*, cv::Point>(&(result->images_[0]), cv::Point(c, r));
				datum_no++;
			}
		}

		return result;
	}

}	}	}