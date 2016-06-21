#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#ifdef _WIN32
#include <Windows.h>
#endif
#ifdef __linux__
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv/highgui.h>

// This class is basically just a number of unrelated functions which dealt with either 
// image processing, or wrangling the outputs of the Random Decision Forests

class IPUtils
{
public:
    /// <summary>
    /// Returns the exponential transform of the input image using the input arguments
    ///</summary> 
    static cv::Mat getExponential(cv::Mat image_in, int expConst = 10, int expMult = 1);

    /// <summary>
    /// Returns the logarithmic transform of the input image using the input arguments
    ///</summary> 
    static cv::Mat getLogarithmic(cv::Mat image_in, int logConst = 10, int logMult = 1);

    /// <summary>
    /// Wrapper for cv::threshold which returns the thresholded image
    ///</summary> 
    static cv::Mat getThresholded(cv::Mat image_in, int threshold_value = 0, int threshold_type = 0);

    /// <summary>
    /// Wrapper for opencv bilateral filter which returns the filtered image
    ///</summary> 
    static cv::Mat getBilateralFiltered(cv::Mat image_in, int value=30);

    /// <summary>
    /// Returns a string which defines the type of an opencv cv::Mat object
    /// <param name="type"> Integer value for type (mat.type()) </param>
    ///</summary> 
    static std::string getTypeString(int type);

    /// <summary>
    /// Pre-process routine for infrared intensity images input to forest training
    /// or evaluation using forest. Ensure the same variables are used for each.
    /// Applies bilateral filter then a static threshold.
    /// </summary>
    /// <param name="image_in">The cv::Mat image for pre-processing.</param>
    /// <param name="bilat_param">The bilateral filter parameter (default 30),
    ///                           higher values give more "cartoonish" apprearence </param>
    /// <param name="threshold_value">Value for static threshold, pixels with intensity 
    ///                               below this value are set to 0 </param>
    /// <param name="threshold_type">Defaults to 3. don't change it for forest training</param>
    static cv::Mat preProcess(cv::Mat image_in, int threshold_value=36, int bilat_param=30, int threshold_type=3);

    /// <summary>
    /// Generates a pixel intensity to bin number look-up-table
    /// Bins are of even size between zero and max, with the option to include a separate zero bin
    /// </summary>
    /// <param name="zero_bin"> flag to indicate if a zero class is required </param>
    /// <param name="total_bins"> Total number of bins, including zero bin if required </param>
    /// <param name="max"> Maximum value to be included in the bins </param> 
    /// <returns> a vector look-up table representing the depth bins</returns>
    static std::vector<int> generateDepthBinMap(bool zero_bin, int total_bins, int max);

    /// <summary>
    /// captures a patch centered on a point and returns it as a cv::Mat
    /// </summary>
    static cv::Mat getPatch(cv::Mat image, cv::Point center, int patch_size);

    /// <summary>
    /// finds the tallest bin in each row of bin_mat.
    /// Returns vector containing tallest depth bin indexes for each row.
    ///</summary> 
    static std::vector<uchar> vectorFromBins(cv::Mat bin_mat, cv::Size expected_size);

    /// <summary>
    /// Calculates a simplistic weighting for each depth bin 
    /// The weighting for bin x is as follows:
    /// w(x) = p(x|V) where V is the vector containing the dallest depth bin 
    /// for each row (ie V = the output of vectorFromBins ).
    ///</summary> 
    static std::vector<float> weightsFromBins(cv::Mat bin_mat, cv::Size image_size, bool include_zero);

    /// <summary>
    /// Returns true is the directory input exists
    ///</summary> 
    static bool dirExists(const std::string& dirName_in);

    /// <summary>
    /// Thresholds a 16 bit image. The usage is the same as cv::threshold()
    ///</summary> 
    static double threshold16(cv::Mat& input_image, cv::Mat& output_image, int thresh, int maxval, int type);

    /// <summary>
    /// Generates a cv::Mat where each pixel is the squared error
    /// return pixel(x,y) = ( mat_a(x,y) - mat_b(x,y) )^2;
    /// accepted input matrix types are CV_16UC1
    /// inputs must be same size and type
    /// </summary>
    /// <param name="mat_a"> first input matrix </param>
    /// <param name="mat_a"> second input matrix </param>
    static cv::Mat getError(cv::Mat mat_a, cv::Mat mat_b);
    
    /// <summary>
    /// sweeps threshold value to find best binary threshold to maximise overlap between two input images
    ///</summary> 
    static int getBestThreshold(cv::Mat ir_image, cv::Mat depth_image, int depth_max, int& best_error_out);

    /// <summary>
    /// Given cv::Mat of bins, outputs the tallest bin across whole image
    ///</summary> 
    static int getTallestBin(cv::Mat& binned_image, int num_bins = 5, bool ignore_zero = true);

    /// <summary>
    /// Used for making a key, returns a vector of numbers between min and max
    /// normalised to between 0 and 255
    ///</summary> 
    static std::vector<uint8_t> generateGradientValues(int min, int max, int min_h = 120, int max_h = 0, bool inv = true);

    /// <summary>
    /// Colourise an image
    ///</summary> 
    static int Colourize(cv::Mat& in, cv::Mat& out, bool zero_black = true);

    /// <summary>
    /// colourise an image
    ///</summary> 
    static int Colourize(cv::Mat& in, cv::Mat& out, int min_h, int max_h, bool inv, bool zero_black = true);

    /// <summary>
    /// Add a key to an already colourized image
    ///</summary> 
    static int AddKey(cv::Mat& original, cv::Mat& colour, int min_h=120, int max_h=0, bool inv=true);

    /// <summary>
    /// Output an image with an added key
    ///</summary> 
    static cv::Mat AddKey(int min, int max, cv::Mat& mat_in);

    IPUtils();
    ~IPUtils();
};

