#include "IPUtils.h"

// Gets the exponential transform of an image. Input should be grayscale.
// Non-grayscale input will be converted and used anyway.
// The exponential transform essentialy increases the contrast of brighter regions.
cv::Mat IPUtils::getExponential(cv::Mat image_in, int expConst, int expMult)
{
    cv::Mat input_image;
    input_image = image_in;
    cv::Mat grayscale_image;
    cv::Size im_size = image_in.size();

    // Variables for operation.
    int constant = expConst;
    int alpha_mult = expMult;
    double alpha = 0.001;

    // Check image in is grayscale, if not, change it.
    if (input_image.channels() > 1)
        cv::cvtColor(input_image, grayscale_image, CV_BGR2GRAY);
    else
        grayscale_image = input_image;

    cv::Mat temp;
    grayscale_image.convertTo(temp, CV_64F);
    cv::Mat temp_1(im_size, CV_64F);
    // Do the exponential transformation
    // output[i,j] = constant * (b^input[i,j] - 1)
    // where b = 1 + (alpha * alpha_mult) 
    double b = 1.0 + (alpha * alpha_mult);
    for (int row = 0; row < temp.rows; row++)
    {
        double* pixel = temp.ptr<double>(row);
        for (int col = 0; col < temp.cols; col++)
        {
            temp_1.at<double>(row, col) = (pow(b, *pixel) - 1.0);
            pixel++;
        }
    }
    temp_1 = constant * temp_1;

    // Convert back to an 8 bit image and normalise across range.
    convertScaleAbs(temp_1, temp_1);
    cv::Mat output_image;
    normalize(temp_1, output_image, 0, 255, cv::NORM_MINMAX);

    return output_image;
}

// Gets the logarithmic transform of an image. Input should be grayscale.
// Non-grayscale will be converted and used anyway.
// The logarithmic transform essentially reduces the contrast of brighter regions.
cv::Mat IPUtils::getLogarithmic(cv::Mat image_in, int logConst, int logMult)
{
    cv::Mat input_image;
    input_image = image_in;
    cv::Mat grayscale_image;


    // variables for operation.
    int constant = logConst;
    int omega_mult = logMult;
    double omega = 0.01;

    // Check image in is grayscale, if not, change it.
    if (input_image.channels() > 1)
        cvtColor(input_image, grayscale_image, CV_BGR2GRAY);
    else
        grayscale_image = input_image;

    cv::Mat temp;
    grayscale_image.convertTo(temp, CV_32F);
    // Do the logarithmic transformation
    // output[i,j] = constant * log(1 + b|input[i,j]|)
    // where b = exp(omega * omega_mult) - 1
    
    double b = exp(omega*omega_mult) - 1.0;
    temp = 1 + (b * temp);
    cv::log(temp, temp);
    temp = constant * temp;

    // Convert back to an 8 bit image and normalise across range.
    convertScaleAbs(temp, temp);
    cv::Mat output_image;
    normalize(temp, output_image, 0, 255, cv::NORM_MINMAX);

    return output_image;
}

cv::Mat IPUtils::getThresholded(cv::Mat image_in, int threshold_value, int threshold_type)
{
    cv::Mat grayscale_image;

    // Check image in is grayscale, if not, change it.
    if (image_in.channels() > 1)
        cvtColor(image_in, grayscale_image, CV_BGR2GRAY);
    else
        grayscale_image = image_in;

    cv::Mat output_image = grayscale_image;
    cv::threshold(grayscale_image, output_image, threshold_value, 255, threshold_type);

    return output_image;
}

cv::Mat IPUtils::getBilateralFiltered(cv::Mat image_in, int value)
{
    cv::Mat grayscale_image;

    // Check image in is grayscale, if not, change it.
    if (image_in.channels() > 1)
        cvtColor(image_in, grayscale_image, CV_BGR2GRAY);
    else
        grayscale_image = image_in;

    cv::Mat output_image;
    cv::bilateralFilter(grayscale_image, output_image, 5, value, value);

    return output_image;
}

std::string IPUtils::getTypeString(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

cv::Mat IPUtils::preProcess(cv::Mat image_in, int threshold_value, int bilat_param, int threshold_type)
{
    cv::Mat temp;

    temp = IPUtils::getBilateralFiltered(image_in, bilat_param);
    return IPUtils::getThresholded(temp, threshold_value, threshold_type);
}

std::vector<int> IPUtils::generateDepthBinMap(bool zero_bin, int total_bins, int max)
{
    std::vector<int> binMap(max+1);
    int ranged_classes = zero_bin ? total_bins - 1 : total_bins;
    int division = (int)ceil((float)max / ranged_classes);
    int class_no = zero_bin ? 1 : 0;
    int d = 1;
    binMap[0] = 0;
    for (int i = 1; i <= max; i++)
    {
        binMap[i] = class_no;
        if (d++ == division)
        {
            class_no++;
            d = 1;
        }
    }
    return binMap;
}

cv::Mat IPUtils::getPatch(cv::Mat image, cv::Point center, int patch_size)
{
    int max_offset = (patch_size - 1) / 2;
    cv::Point patch_lb((center.x - max_offset), (center.y - max_offset));
    cv::Point patch_ub((center.x + max_offset), (center.y + max_offset));
    cv::Rect boundary(cv::Point(0, 0), image.size());
    cv::Point point_check;
    cv::Mat ret_patch = cv::Mat::zeros(patch_size, patch_size, CV_8UC1);

    int pr = 0, pc = 0;
    for (int r = patch_lb.y; r <= patch_ub.y; r++)
    {
        pc = 0;
        point_check = cv::Point(0, r);
        if (point_check.inside(boundary))
        {
            uchar* im_pix = image.ptr<uchar>(r);
            uchar* patch_pix = ret_patch.ptr<uchar>(pr);

            for (int c = patch_lb.x; c <= patch_ub.x; c++)
            {
                point_check = cv::Point(c, r);
                if (point_check.inside(boundary))
                {
                    patch_pix[pc] = im_pix[c];
                    pc++;
                }
                else
                {
                    pc++;
                    continue;
                }
            }

            pr++;
        }
        else
        {
            pr++;
            continue;
        }
    }

    return ret_patch;
}

std::vector<uchar> IPUtils::vectorFromBins(cv::Mat bin_mat, cv::Size expected_size)
{
    int samples = bin_mat.size().height;
    int bins = bin_mat.size().width;
    
    std::vector<uchar> out_vec(samples);
    
    int row_max;
    int row_max_index;
    // TODO change from at to ptr.
    for (int i = 0; i < samples; i++)
    {
        row_max = bin_mat.at<int>(i,0);
        row_max_index = 0;
        for (int j = 1; j < bins; j++)
        {
            if (row_max < bin_mat.at<int>(i, j))
            {
                row_max = bin_mat.at<int>(i, j);
                row_max_index = j;
            }
        }
        out_vec[i] = uchar(row_max_index);
    }
    
    return out_vec;
}

std::vector<float> IPUtils::weightsFromBins(cv::Mat bin_mat, cv::Size image_size, bool include_zero)
{
    int samples = bin_mat.size().height;
    int bins = bin_mat.size().width;
    //TODO pass this in by reference instead of calculating
    std::vector<uchar> bin_vector = vectorFromBins(bin_mat, image_size);
    std::vector<int> bin_totals(bins, 0);
    std::vector<float> weights(1, 0);
    int start = include_zero? 0:1;
    
    for(int i=0;i<samples;i++)
    {
        bin_totals[bin_vector[i]]++;
    }
    

    if(include_zero)
    {
        weights.resize(bins, 0);
        for(int i=0;i<bins;i++)
        {
            weights[i] = float(bin_totals[i]) / samples;
        }
    }
    else
    {
        weights.resize(bins-1, 0);
        int total = 0;
        for(int i=1;i<bins;i++)
        {
            total += bin_totals[i];
        }
        for(int i=0;i<bins-1;i++)
        {
            weights[i] = float(bin_totals[i+1])/total;
        }
    }
    
    return weights;
}

double IPUtils::threshold16(cv::Mat& input_image, cv::Mat& output_image, int thresh, int maxval, int type)
{
    int input_type = input_image.type();
    uchar input_depth = input_type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (input_type >> CV_CN_SHIFT);

    if((chans != 1) || (input_depth != CV_16U))
        throw std::runtime_error("invalid input format");

    if(input_type != output_image.type())
        throw std::runtime_error("image types not equal");

    int rows = input_image.size().height;
    int cols = input_image.size().width;

    if(type == 0)
    {
        for(int r=0;r<rows;r++)
        {
            uint16_t* src_pix = input_image.ptr<uint16_t>(r);
            uint16_t* dst_pix = output_image.ptr<uint16_t>(r);
            for(int c=0;c<cols;c++)
            {
                if(src_pix[c] > thresh)
                    dst_pix[c] = maxval;
                else
                    dst_pix[c] = 0;
            }
        }
    }
    else if(type == 1)
    {
        for(int r=0;r<rows;r++)
        {
            uint16_t* src_pix = input_image.ptr<uint16_t>(r);
            uint16_t* dst_pix = output_image.ptr<uint16_t>(r);
            for(int c=0;c<cols;c++)
            {
                if(src_pix[c] > thresh)
                    dst_pix[c] = 0;
                else
                    dst_pix[c] = maxval;
            }
        }
    }
    else if(type == 2)
    {
        for(int r=0;r<rows;r++)
        {
            uint16_t* src_pix = input_image.ptr<uint16_t>(r);
            uint16_t* dst_pix = output_image.ptr<uint16_t>(r);
            for(int c=0;c<cols;c++)
            {
                if(src_pix[c] > thresh)
                    dst_pix[c] = thresh;
                else
                    dst_pix[c] = src_pix[c];
            }
        }
    }
    else if(type == 3 )
    {
        for(int r=0;r<rows;r++)
        {
            uint16_t* src_pix = input_image.ptr<uint16_t>(r);
            uint16_t* dst_pix = output_image.ptr<uint16_t>(r);
            for(int c=0;c<cols;c++)
            {
                if(src_pix[c] > thresh)
                    dst_pix[c] = src_pix[c];
                else
                    dst_pix[c] = 0;
            }
        }
    }
    else if(type == 4)
    {
        for(int r=0;r<rows;r++)
        {
            uint16_t* src_pix = input_image.ptr<uint16_t>(r);
            uint16_t* dst_pix = output_image.ptr<uint16_t>(r);
            for(int c=0;c<cols;c++)
            {
                if(src_pix[c] > thresh)
                    dst_pix[c] = 0;
                else
                    dst_pix[c] = src_pix[c];
            }
        }
    }
    else
    {
        throw std::runtime_error("Invalid threshold type");
    }

    return 0.0;
}

cv::Mat IPUtils::getError(cv::Mat mat_a, cv::Mat mat_b)
{

    if(mat_a.size() != mat_b.size())
        throw std::runtime_error("Matrix sizes are not equal");

    if(mat_a.type() != mat_b.type())
        throw std::runtime_error("Matrix types are not equal");

    int input_type = mat_a.type();
    uchar input_depth = input_type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (input_type >> CV_CN_SHIFT);
    if((chans != 1) || (input_depth != CV_16U))
        throw std::runtime_error("invalid input format");

    //cv::Mat ret_mat(mat_a.size(), CV_32SC1);
    int rows = mat_a.size().height;
    int cols = mat_a.size().width;

    cv::Mat ret_mat(rows, cols, CV_32SC1);
    uint16_t temp=0;

    for(int r=0;r<rows;r++)
    {
        int32_t* ret_pix = ret_mat.ptr<int32_t>(r);
        uint16_t* a_pix = mat_a.ptr<uint16_t>(r);
        uint16_t* b_pix = mat_b.ptr<uint16_t>(r);
        for(int c=0;c<cols;c++)
        {
            temp = abs(a_pix[c] - b_pix[c]);
            ret_pix[c] = temp;
        }
    }
    return ret_mat;
}

int IPUtils::getBestThreshold(cv::Mat ir_image, cv::Mat depth_image, int depth_max, int& best_error_out)
{
    if(ir_image.size() != depth_image.size())
        throw std::runtime_error("Matrix sizes are not equal");

    int ir_type = ir_image.type();
    uchar input_depth = ir_type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (ir_type >> CV_CN_SHIFT);
    if((chans != 1) || (input_depth != CV_8U))
        throw std::runtime_error("invalid ir format");

    int depth_type = depth_image.type();
    input_depth = depth_type & CV_MAT_DEPTH_MASK;
    chans = 1 + (depth_type >> CV_CN_SHIFT);
    if((chans != 1) || (input_depth != CV_16U))
        throw std::runtime_error("invalid depth format");


    int rows = ir_image.size().height;
    int cols = ir_image.size().width;

    cv::Mat depth_thresh_tmp(rows, cols, CV_16UC1);
    cv::Mat depth_thresh(rows, cols, CV_16UC1);
    // take everything above depth_max to zero
    threshold16(depth_image, depth_thresh_tmp, depth_max, 65535, 4);
    // all non-zero to 255 - binary threshold
    threshold16(depth_thresh_tmp, depth_thresh, 0, 255, 0);
    depth_thresh.convertTo(depth_thresh, CV_8U);
    
    int best_error = rows*cols;
    int best_threshold_value =  0;
    // binary search for best threshold value to match the two binary images
    for(int i=0;i<=255;i++)
    {        
       cv::Mat ir_thresh;
       cv::threshold(ir_image, ir_thresh, i, 255, 0);
       int error = 0;
       for(int r=0;r<rows;r++)
       {
            uint8_t* d_pix = depth_thresh.ptr<uint8_t>(r);
            uint8_t* ir_pix = ir_thresh.ptr<uint8_t>(r);
            for(int c=0;c<cols;c++)
            {
                if(ir_pix[c] != d_pix[c])
                {
                    error++;
                }
            }
       }
       if(error < best_error)
       {
            best_threshold_value = i;
            best_error = error;
       }
    }
    best_error_out = best_error;
    return best_threshold_value;

}

int IPUtils::getTallestBin(cv::Mat& binned_mat, int num_bins, bool ignore_zero)
{
    int rows = binned_mat.size().height;
    int cols = binned_mat.size().width;
    std::vector<int> bins_accumulator(num_bins);

    for(int r = 0 ;r<rows;r++)
    {
        uint8_t* bin_pix = binned_mat.ptr<uint8_t>(r);
        for(int c=0;c<cols;c++)
        {
            bins_accumulator[bin_pix[c]]++;
        }
    }

    int start = ignore_zero? 1 : 0;
    int max = 0;
    int tallest_bin = 0;
    for(int i=start;i<num_bins;i++)
    {
        if(bins_accumulator[i]>=max)
        {
            max = bins_accumulator[i];
            tallest_bin = i;
        }
    } 
    return tallest_bin;

}

std::vector<uint8_t> IPUtils::generateGradientValues(int min, int max, int min_h, int max_h, bool inv)
{
    int vec_size = (max-min)+1;
    std::vector<uint8_t> h_ints(vec_size);
    std::vector<float> h_values(vec_size);
    float increment;

    if(inv)
    {
        if(min_h <= max_h)
        {
            increment = float(180 - (max_h-min_h))/255;
        }
        else
        {
            increment = float(min_h-max_h)/255;
        }
        
        h_values[0] = min_h;
        for(int i=1;i<vec_size;i++)
        {
            h_values[i] = h_values[i-1] - increment;
            if(h_values[i] >= 179.5)
                h_values[i] -= 180;
            if(h_values[i] < -0.5)
                h_values[i] += 180;
        }

        for(int i=0;i<vec_size;i++)
        {
            h_ints[i] = uint8_t(round(h_values[i]));
        }
    }
    else
    {
        if(min_h >= max_h)
        {
            increment = float(180 - (min_h-max_h))/255;
        }
        else
        {
            increment = float(max_h-min_h)/255;
        }
        
        h_values[0] = min_h;
        for(int i=1;i<vec_size;i++)
        {
            h_values[i] = h_values[i-1] + increment;
            if(h_values[i] >= 179.5)
                h_values[i] -= 180;
            if(h_values[i] < -0.5)
                h_values[i] += 180;
        }

        for(int i=0;i<vec_size;i++)
        {
            h_ints[i] = uint8_t(round(h_values[i]));
        }
    }

    return h_ints;

}

int IPUtils::Colourize(cv::Mat& in, cv::Mat& out, bool zero_black)
{

    std::vector<uint8_t> h_ints = generateGradientValues(0,255);

    int rows = in.size().height;
    int cols = in.size().width;
    cv::Mat in_gs(rows,cols,CV_8UC1);
    cv::Mat out_hsv(rows, cols, CV_8UC3, cv::Scalar(0,255,255));
    
    in.convertTo(in_gs, CV_8U);

    for(int r=0;r<rows;r++)
    {
        uint8_t* hue_pixel = out_hsv.ptr<uint8_t>(r);
        uint8_t* gs_pixel = in_gs.ptr<uint8_t>(r);
        for(int c=0;c<cols;c++)
        {
            if(gs_pixel[c] != 0 || !zero_black)
            {
                hue_pixel[c*3] = h_ints[gs_pixel[c]];
            }
            else
            {
                hue_pixel[(c*3) + 2] = 0;
            }
        }
    }
    cv::cvtColor(out_hsv, out, CV_HSV2BGR);
    return 0;
}

int IPUtils::Colourize(cv::Mat& in, cv::Mat& out, int min_h, int max_h, bool inv, bool zero_black)
{

    std::vector<uint8_t> h_ints = generateGradientValues(0,255, min_h, max_h, inv);

    int rows = in.size().height;
    int cols = in.size().width;
    cv::Mat in_gs(rows,cols,CV_8UC1);
    cv::Mat out_hsv(rows, cols, CV_8UC3, cv::Scalar(0,255,255));
    
    in.convertTo(in_gs, CV_8U);

    for(int r=0;r<rows;r++)
    {
        uint8_t* hue_pixel = out_hsv.ptr<uint8_t>(r);
        uint8_t* gs_pixel = in_gs.ptr<uint8_t>(r);
        for(int c=0;c<cols;c++)
        {
            if(gs_pixel[c] != 0 || !zero_black)
            {
                hue_pixel[c*3] = h_ints[gs_pixel[c]];
            }
            else
            {
                hue_pixel[(c*3) + 2] = 0;
            }
        }
    }
    cv::cvtColor(out_hsv, out, CV_HSV2BGR);
    return 0;
}

int IPUtils::AddKey(cv::Mat& original, cv::Mat& colour, int min_h, int max_h, bool inv)
{
    int min, max;
    double min_d, max_d;

    cv::minMaxLoc(original, &min_d, &max_d);
    min = int(min_d);
    max = int(max_d);

    std::vector<uint8_t> h_ints = generateGradientValues(0,255,min_h, max_h, inv);

    cv::Mat key(256, 50,CV_8UC3, cv::Scalar(0,255,255));
    cv::Mat key_bgr;
    for(int i=0;i<256;i++)
    {
        uint8_t* key_pixel = key.ptr<uint8_t>(i);
        for(int c=0;c<50;c++)
        {
            key_pixel[c*3] = h_ints[255 - i];
        }
    }
    cv::cvtColor(key, key_bgr, CV_HSV2BGR);
    cv::rectangle(key_bgr, cv::Point(0,0), cv::Point(49,255), cv::Scalar(0,0,0), 2);
    cv::Mat region_of_interest(colour, cv::Rect(25,50,50,256));
    key_bgr.copyTo(region_of_interest);
    cv::putText(colour, std::to_string(max), cv::Point(25,46), 3, 1, cv::Scalar(0,0,0), 1, 8);
    cv::putText(colour, std::to_string(min), cv::Point(25,330), 3, 1, cv::Scalar(0,0,0), 1, 8);
    
    return 0;
}

cv::Mat IPUtils::AddKey(int min, int max, cv::Mat& mat_in)
{
    cv::Mat img_matches(480, 850, CV_8UC3, cv::Scalar(255,255,255));

    float interval;
    interval = float(255) / 460;
    //std::cout << std::to_string(interval) << std::endl;
    float current_value = 255;
    float temp = current_value;
    
    cv::Mat key(460, 50,CV_8UC1);
    cv::Mat key_bgr;
    for(int i=0;i<460;i++)
    {
        temp = current_value;
        //std::cout << std::to_string(current_value) << std::endl;
        uint8_t* key_pixel = key.ptr<uint8_t>(i);
        for(int c=0;c<50;c++)
        {
            key_pixel[c] = uint8_t(round(temp));
        }
        current_value = current_value - interval;
    }
    cv::applyColorMap(key, key_bgr, cv::COLORMAP_JET);
    cv::rectangle(key_bgr, cv::Point(0,0), cv::Point(49,459), cv::Scalar(0,0,0), 3);

    cv::Mat left(img_matches, cv::Rect(0, 0, 640, 480)); // Copy constructor
    mat_in.copyTo(left);
    cv::Mat right(img_matches, cv::Rect(645, 10, 50, 460)); // Copy constructor
    key_bgr.copyTo(right);

    cv::putText(img_matches, std::to_string(max)+"mm", cv::Point(695,30), 2, 1, cv::Scalar(0,0,0), 1, 8);
    cv::putText(img_matches, std::to_string(min)+"mm", cv::Point(695,469), 2, 1, cv::Scalar(0,0,0), 1, 8);
    
    return img_matches;
}

#ifdef __WIN32
bool IPUtils::dirExists(const std::string& dirName_in)
{
    unsigned long ftyp = GetFileAttributesA(dirName_in.c_str());
    if (ftyp == INVALID_FILE_ATTRIBUTES)
    {
        std::cout << "Directory not found: " << dirName_in << std::endl;
        return false;  //something is wrong with your path!
    }

    if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
        return true;   // this is a directory!

    std::cout << "Directory not found: " << dirName_in << std::endl;
    return false;    // this is not a directory!
}
#endif
#ifdef __linux__
bool IPUtils::dirExists(const std::string& dirName_in)
{
    const char *dirName_carr = dirName_in.c_str();
    bool is_dir = false;
    struct stat st;
    if(stat(dirName_carr, &st)==0)
        is_dir = S_ISDIR(st.st_mode);

    if(!is_dir)
        std::cout << "Directory not found: " << dirName_in << std::endl;
    
    return is_dir;
}
#endif

IPUtils::IPUtils()
{
}

IPUtils::~IPUtils()
{
}
