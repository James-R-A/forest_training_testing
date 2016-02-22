#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>
#include <tuple>
#include <algorithm>
#include <fstream>
#include <sstream>

#include "IPUtils.h"
#include "TrainingParameters.h"
#include "Interfaces.h"


namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
    /// <summary>
    /// Used to describe the expected use of the loaded images (used
    /// in DataPointCollection::Load()).
    /// </summary>

    //class DataDescriptor
    //{
    //public:
    //  enum e
    //  {
    //      Both = 0x0,
    //      Classes = 0x1,
    //      TargetValues = 0x2,
    //      None = 0x4
    //  };
    //};

    /// <summary>
    /// A collection of data points, each represented by an int and (optionally)
    /// associated with a string class label and/or a float??? target value.
    /// </summary>
    class DataPointCollection: public IDataPointCollection
    {
        std::vector< std::tuple<cv::Mat*, cv::Point> > data_;
        std::vector<cv::Mat> images_;
        cv::Size image_size;
        int dimension_;
        int64_t data_vec_size;

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
        /// <param name="progParams">A program parameters object which defines 
        ///  how the program will run, its inputs, and its outputs </param>
        static  std::unique_ptr<DataPointCollection> LoadImagesClass(ProgramParameters& progParams);

        /// <summary>
        /// Loads a data set from a directory of IR and depth images
        /// Loads in a regression problem format
        /// </summary>
        /// <param name="progParams">A program parameters object which defines 
        ///  how the program will run, its inputs, and its outputs </param>
        static  std::unique_ptr<DataPointCollection> LoadImagesRegression(ProgramParameters& progParams, int class_number=-1);

        static std::unique_ptr<DataPointCollection> LoadMat(cv::Mat, cv::Size img_size);

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

        int CountImages() const
        {
            return images_.size();
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
        const std::tuple<cv::Mat*,cv::Point>* GetDataPoint(int i) const
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

    class LMDataPointCollection: public IDataPointCollection
    {
        // If we're doing expert regression, we need all the point locations
        // otherwise, this isn't used
        //std::vector<int64_t> data_points_;

        // Vector of input infrared images
        std::vector<cv::Mat> images_;
        // vector of depth labels
        std::vector<uint8_t> labels_;
        // because this weird return type is what we need for GetDataPoint()
        std::tuple<cv::Mat*, cv::Point> data_point;
        // Basically number of pixels in an image
        int step;
        cv::Size image_size;
        // Related to patch size, gets passed on to other stuff
        int dimension_;
        // Number of images in the DataPointCollection
        int image_vec_size;
        // Raw depth flag. used in classification
        bool depth_raw;
        // vector of pixel-to-label mapping
        std::vector<int> pixelLabels_;
        // Expected number of data points.
         int64_t n_data_points;
        
    public:

        static std::unique_ptr<LMDataPointCollection> LoadImagesClass(
            ProgramParameters& progParams);

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

        int CountImages() const
        {
            return image_vec_size;
        }

        /// <summary>
        /// Do these data have target values (e.g. for regression)?
        /// </summary>
        bool HasTargetValues() const
        {
            return false;
        }

        /// <summary>
        /// Count the data points in this collection.
        /// </summary>
        /// <returns>The number of data points</returns>
        unsigned int Count() const
        {
            return n_data_points;
        }

        /// <summary>
        /// Basically the patch area.
        /// </summary>
        int Dimensions() const
        {
            return dimension_;
        }

        bool DepthRaw() const
        {
            return depth_raw;
        }

        /// <summary>
        /// Get the specified data point.
        /// </summary>
        /// <param name="i">Zero-based data point index.</param>
        /// <returns>Pointer to the first element of the data point.</returns>
        const std::tuple<cv::Mat*,cv::Point>* GetDataPoint(int i)
        {
            // assuming compiler is clever enough to get quotient and remainder 
            // in singe operation
            int image_index = i / step;
            int position_rem = i % step;
            int row = position_rem / image_size.width;
            int column = position_rem % image_size.width;
            std::get<0>(data_point) = &images_[image_index];
            std::get<1>(data_point) = cv::Point(column, row);
            
            return &data_point;
        }

        /// <summary>
        /// Get the class label for the specified data point (or raise an
        /// exception if these data points do not have associated labels).
        /// </summary>
        /// <param name="i">Zero-based data point index</param>
        /// <returns>A zero-based integer class label.</returns>
        int GetIntegerLabel(int i) const
        {
            //TODO
            if (!HasLabels())
                throw std::runtime_error("Data have no associated class labels.");

            return (int)labels_[i]; // may throw an exception if index is out of range
        }
        
    };
}   }   }