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
    /// A collection of data points, each represented by an int and (optionally)
    /// associated with a string class label or a float target value.
    /// Two types of DataPointCollection are supported. A Low memory DataPointCollection 
    /// has low_memory flag as true and contains every pixel in every image.
    /// By default, full regressors and all classifiers are low memory. Only experts filter by depth bin.
    /// </summary>
    class DataPointCollection: public IDataPointCollection
    {
        // Data vector is actually the index of the actual data point ie. the index of 
        // the central pixel.
        std::vector< uint32_t > data_;
        std::vector<cv::Mat> images_;
        cv::Size image_size;
        int dimension_;
        uint32_t data_vec_size;
        bool depth_raw;
        // Basically number of pixels in an image
        int step;
        // vector of pixel-to-label mapping
        std::vector<int> pixelLabels_;
        

    public:
        // for classified data
        std::vector<uint8_t> labels_;
        // For regression, equiv of exact depth data
        std::vector<uint16_t> targets_;
        // flag to show if the DPC is a low memory implementation.
        bool low_memory;
        // flag to show if we bother training on zero intensity pixels
        bool train_on_zero;

        static const int UnknownClassLabel = -1;

        /// <summary>
        /// Loads a data set from a directory of IR and depth images
        /// </summary>
        /// <param name="progParams">A program parameters object which defines 
        ///  how the program will run, its inputs, and its outputs </param>
        /// <param name="classification"> set true if for classification, 
        ///  false if for regression </param>
        /// <param name="class_number">Expert class number, parameter used in
        ///  training regressors or experts </param>
        static  std::unique_ptr<DataPointCollection> LoadImages(
            ProgramParameters& progParams,
            bool classification=true,
            int class_number=-1);

        /// <summary>
        /// Loads a single cv::Mat into a DataPointCollection object for 
        /// evaluation.
        /// Returns a std::unique_ptr to the DataPointCollection
        /// </summary>
        /// <param name="mat_in"> The cv::Mat to load up </param>
        /// <param name="img_size"> cv::Size of input image </param>
        /// <param name="mat_in"> boolean. If true, zero input IR are included 
        ///                       in the DataPointCollection </param>
        /// <param name="pre_process"> if true, image is pre-processed </param>
        /// <param name="pp_value"> Threshold value for use in pre-processing 
        ///                  (everything below this value is set to 0) </param>
        static std::unique_ptr<DataPointCollection> LoadMat(cv::Mat mat_in, cv::Size img_size, bool inc_zero = true, bool pre_process = true, int pp_value = 36);

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
            if(!low_memory)
                return data_.size();
            else
                return data_vec_size;
        }

        /// <summary>
        /// Basically the patch area.
        /// </summary>
        int Dimensions() const
        {
            return dimension_;
        }

        /// <summary>
        /// Get the specified data point from a low memory DataPointCollection
        /// </summary>
        /// <param name="i">Zero-based data point index.</param>
        /// <returns>tuple containing cv::Mat pointer to image and cv::point.</returns>
        std::tuple<const cv::Mat*,cv::Point> GetDataPointLM(uint32_t i) const
        {
            // assuming compiler is clever enough to get quotient and remainder 
            // in singe operation
            uint32_t image_index = i / step;
            uint32_t position_rem = i % step;
            uint32_t row = position_rem / image_size.width;
            uint32_t column = position_rem % image_size.width;
                      
            return std::tuple<const cv::Mat*, cv::Point>(&images_[image_index], cv::Point(column, row));
        }

        /// <summary>
        /// Get the specified data point from a regular DataPointCollection
        /// </summary>
        /// <param name="i">Zero-based data point index.</param>
        /// <returns>tuple containing cv::Mat pointer to image and cv::point.</returns>
        std::tuple<const cv::Mat*, cv::Point> GetDataPointRegular(uint32_t i) const
        {
            uint32_t image_index = data_[i] / step;
            uint32_t position_rem = data_[i] % step;
            uint32_t row = position_rem / image_size.width;
            uint32_t column = position_rem % image_size.width;

            return std::tuple<const cv::Mat*, cv::Point>(&images_[image_index], cv::Point(column, row));
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

        int GetStep()
        {
            return step;
        }
    };

}   }   }