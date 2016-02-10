#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>
#include <tuple>
#include <windows.h>
#include <algorithm>
#include <fstream>
#include <sstream>

#include "IPUtils.h"

#include "Interfaces.h"


namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
	/// <summary>
	/// Used to describe the expected use of the loaded images (used
	/// in DataPointCollection::Load()).
	/// </summary>

	class DataDescriptor
	{
	public:
		enum e
		{
			Both = 0x0,
			Classes = 0x1,
			TargetValues = 0x2,
			None = 0x4
		};
	};

	/// <summary>
	/// A collection of data points, each represented by an int and (optionally)
	/// associated with a string class label and/or a float??? target value.
	/// </summary>
	class DataPointCollection: public IDataPointCollection
	{
		std::vector<std::tuple<cv::Mat*, cv::Point>> data_;
		std::vector<cv::Mat> images_;
		cv::Size image_size;
		int dimension_;
		long data_vec_size;

		bool depth_raw;
		
		// vector of pixel-to-label mapping
		std::vector<int> pixelLabels_;
		

	public:
		// for classified data
		std::vector<uint8_t> labels_;
		// For regression, equiv of exact depth data
		std::vector<uint16_t> targets_;

		static const int UnknownClassLabel = -1;

		/// <summary>
		/// Loads a data set from a directory of IR and depth images
		/// Loads in a classification problem format
		/// </summary>
		/// <param name="path">Path of directory containing images to be read.</param>
		/// <param name="img_size">The expected image size of the input data</param>
		/// <param name="depth_raw"> depth data type, raw, or in mm </param>
		/// <param name="number">Number of images to be loaded</param>
		/// <param name="start">Image number to start at (default 0)</param>
		/// <param name="num_classes"> Number of classes to classify depth data </param>
		/// <param name="zero_class_label"> Is there a seperate class for zero output? </param>
		/// <param name="patch_size">Size of pixel patch to load (default 1, i.e. single pixel)</param>
		static  std::auto_ptr<DataPointCollection> LoadImagesClass(
			std::string path, 
			cv::Size img_size,
			bool depth_raw, 
			int number, 
			int start=0,
			int num_classes = 5,
			bool zero_class_label = true,
			int patch_size = 25);

		/// <summary>
		/// Loads a data set from a directory of IR and depth images
		/// Loads in a regression problem format
		/// </summary>
		/// <param name="path">Path of directory containing images to be read.</param>
		/// <param name="img_size">The expected image size of the input data</param>
		/// <param name="depth_raw"> depth data type, raw, or in mm </param>
		/// <param name="number">Number of images to be loaded</param>
		/// <param name="start">Image number to start at (default 0)</param>
		/// <param name="num_classes"> Number of classes to classify depth data </param>
		/// <param name="zero_class_label"> Is there a seperate class for zero output? </param>
		/// <param name="class_number"> which class data subset to load, -1 for all </param>
		/// <param name="patch_size">Size of pixel patch to load (default 1, i.e. single pixel)</param>
		static  std::auto_ptr<DataPointCollection> LoadImagesRegression(
			std::string path,
			cv::Size img_size,
			bool depth_raw,
			int number,
			int start = 0,
			int num_classes = 5,
			bool zero_class_label = true,
			int class_number = -1,
			int patch_size = 25);

		static std::auto_ptr<DataPointCollection> LoadMat(cv::Mat, cv::Size img_size);

		/// <summary>
		/// Do these data have class labels?
		/// </summary>
		bool HasLabels() const
		{
			return labels_.size() != 0;
		}

		/// <summary>
		/// How many unique class labels are there?
		/// </summary>
		/// <returns>The number of unique class labels</returns>
		int CountClasses() const
		{
			if (!HasLabels())
				throw std::runtime_error("Unlabelled data.");
			return (*std::max_element(pixelLabels_.begin(), pixelLabels_.end())) + 1;
		}

		/// <summary>
		/// Do these data have target values (e.g. for regression)?
		/// </summary>
		bool HasTargetValues() const
		{
			return targets_.size() != 0;
		}

		/// <summary>
		/// Count the data points in this collection.
		/// </summary>
		/// <returns>The number of data points</returns>
		unsigned int Count() const
		{
			return data_.size();
		}

		/// <summary>
		/// Basically the patch area.
		/// </summary>
		int Dimensions() const
		{
			return dimension_;
		}

		/// <summary>
		/// Get the specified data point.
		/// </summary>
		/// <param name="i">Zero-based data point index.</param>
		/// <returns>Pointer to the first element of the data point.</returns>
		const tuple<cv::Mat*,cv::Point>* GetDataPoint(int i) const
		{
			return &data_[i];
		}

		/// <summary>
		/// Get the class label for the specified data point (or raise an
		/// exception if these data points do not have associated labels).
		/// </summary>
		/// <param name="i">Zero-based data point index</param>
		/// <returns>A zero-based integer class label.</returns>
		int GetIntegerLabel(int i) const
		{
			if (!HasLabels())
				throw std::runtime_error("Data have no associated class labels.");

			return (int)labels_[i]; // may throw an exception if index is out of range
		}

		/// <summary>
		/// Get the target value for the specified data point (or raise an
		/// exception if these data points do not have associated target values).
		/// </summary>
		/// <param name="i">Zero-based data point index.</param>
		/// <returns>The target value.</returns>
		float GetTarget(int i) const
		{
			if (!HasTargetValues())
				throw std::runtime_error("Data have no associated target values.");

			return float(targets_[i]); // may throw an exception if index is out of range
		}

		bool DepthRaw() const
		{
			return depth_raw;
		}
	};

	// Here would be a good place to add parsing functionality and 
	// things like string to float conversion
	// A couple of file parsing utilities, exposed here for testing only.

}	}	}