#include "DataPointCollection.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
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

    std::unique_ptr<DataPointCollection> DataPointCollection::LoadImagesClass(ProgramParameters& progParams)
    {
        std::string prefix = progParams.InputPrefix;
        cv::Size img_size = cv::Size(progParams.ImgWidth, progParams.ImgHeight);

        // for shorthand
        std::string path = progParams.TrainingImagesPath;
        if (!IPUtils::dirExists(path))
            throw std::runtime_error("Failed to find directory:\t" + path);

        if (progParams.PatchSize % 2 == 0)
            throw std::runtime_error("Patch size must be odd");
        
        int number = progParams.NumberTrainingImages;
        int first = progParams.TrainingImagesStart;
        int last = first + number -1;

        // Set up DataPointCollection object
        std::unique_ptr<DataPointCollection> result = std::unique_ptr<DataPointCollection>(new DataPointCollection());
        result->dimension_ = progParams.PatchSize * progParams.PatchSize;
        result->depth_raw = progParams.DepthRaw;
        result->image_size = img_size;
        result->data_vec_size = number * img_size.height * img_size.width;

        // Data allocated using data_.resize(result->data_vec_size) 
        // then result->data_[n] = value; because it's faster than push_back
        // speed change ~= x2
        result->images_.resize(number);
        result->data_.resize(result->data_vec_size);
        int img_no = 0;
        int datum_no = 0;
        int label_no = 0;
        
        result->labels_.resize(result->data_vec_size);
        
        // Variables affecting class formation
        bool zero_class = true;
        int total_classes = progParams.Bins;
        // This max parameter is important. Don't forget this is aimed at 16 bit unsigned ints, 
        // so the viable range is 0-65535. 
        // If not using RAW depth data format, it's measured in mm, so be sensible (i.e. 1000 - 1500 mm?)
        int max = progParams.DepthRaw ? 65000 : 1200;
        result->pixelLabels_ = IPUtils::generateDepthBinMap(true, total_classes, max);
        
        cv::Mat ir_image, ir_preprocessed, depth_image, depth_labels;
        std::string ir_path;
        std::string depth_path;
        cv::Size ir_size, depth_size;

        for (int i = first; i <= last;i++)
        {
            // generate individual image paths
            ir_path = path + "/" + prefix + std::to_string(i) + "ir.png";
            depth_path = path + "/" + prefix + std::to_string(i) + "depth.png";

            //std::cout << std::to_string(i) << std::endl;
            // read depth and ir images
            ir_image = cv::imread(ir_path, -1);
            depth_image = cv::imread(depth_path, -1);

            // if program fails to open image
            if(!ir_image.data)
            {
                std::cerr << "Failed to open image:\n\t" + ir_path << std::endl;
                continue;
            }
            if (!depth_image.data)
            {
                std::cerr << "Failed to open image:\n\t" + depth_path << std::endl;
                continue;
            }
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
                    result->data_[datum_no] = std::tuple<cv::Mat*, cv::Point>(&(result->images_[img_no]), cv::Point(c, r));
                    datum_no++;
                }
            }
            img_no++;
        }
        
        return result;
    }

    std::unique_ptr<DataPointCollection> DataPointCollection::LoadImagesRegression(ProgramParameters& progParams, int class_number)
    {
        std::string prefix = progParams.InputPrefix;
        cv::Size img_size = cv::Size(progParams.ImgWidth, progParams.ImgHeight);

        std::string path = progParams.TrainingImagesPath;
        if (!IPUtils::dirExists(path))
            throw std::runtime_error("Failed to find directory:\t" + path);

        if (progParams.PatchSize % 2 == 0)
            throw std::runtime_error("Patch size must be odd");

        if ((class_number >= progParams.Bins)||(class_number<-1))
            throw std::runtime_error("class number outside class range.");

        int number = progParams.NumberTrainingImages;
        int first = progParams.TrainingImagesStart;
        int last = first + number - 1;

        // Set up DataPointCollection object
        std::unique_ptr<DataPointCollection> result = std::unique_ptr<DataPointCollection>(new DataPointCollection());
        result->dimension_ = progParams.PatchSize * progParams.PatchSize;
        result->depth_raw = progParams.DepthRaw;
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
        bool zero_class = true;
        int total_classes = progParams.Bins;
        // This max parameter is important. Don't forget this is aimed at 16 bit unsigned ints, 
        // so the viable range is 0-65535. 
        // If not using RAW depth data format, it's measured in mm, so be sensible (i.e. 1000 - 1500 mm?)
        int max = progParams.DepthRaw ? 65000 : 1200;
        result->pixelLabels_ = IPUtils::generateDepthBinMap(zero_class, total_classes, max);

        cv::Mat ir_image, ir_preprocessed, depth_image, depth_labels;
        std::string ir_path;
        std::string depth_path;
        cv::Size ir_size, depth_size;

        for (int i = first; i <= last; i++)
        {
            // generate individual image paths
            ir_path = path + "/" + prefix + std::to_string(i) + "ir.png";
            depth_path = path + "/" + prefix + std::to_string(i) + "depth.png";

            //std::cout << std::to_string(i) << std::endl;
            // read depth and ir images
            ir_image = cv::imread(ir_path, -1);
            depth_image = cv::imread(depth_path, -1);

            // if program fails to open image
            if (!ir_image.data)
            {
                std::cerr <<"Failed to open image:\n\t" + ir_path << std::endl;
                continue;
            }
            if (!depth_image.data)
            {
                std::cerr << "Failed to open image:\n\t" + depth_path << std::endl;
                continue;
            }

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
                        result->data_[datum_no] = std::tuple<cv::Mat*, cv::Point>(&(result->images_[img_no]), cv::Point(c, r));
                        datum_no++;
                    }
                }
            }

            img_no++;
        }
        // TODO need to deal with empty data point collections. Otherwise errors caused in training
        // Resize data and targets vector to however full they are.
        result->data_.resize(datum_no);
        result->data_.shrink_to_fit();
        // Shrink to fit new size to free up excess memory.
        result->targets_.resize(target_no);
        result->targets_.shrink_to_fit();

        return result;
    }

    std::unique_ptr<DataPointCollection> DataPointCollection::LoadMat(cv::Mat mat_in, cv::Size img_size)
    {
        // If the datatypes in the images are incorrect
        if (IPUtils::getTypeString(mat_in.type()) != "8UC1")
            throw std::runtime_error("Incorrect image type, expecting CV_8UC1");

        // Set up DataPointCollection object
        std::unique_ptr<DataPointCollection> result = std::unique_ptr<DataPointCollection>(new DataPointCollection());
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
                result->data_[datum_no] = std::tuple<cv::Mat*, cv::Point>(&(result->images_[0]), cv::Point(c, r));
                datum_no++;
            }
        }

        return result;
    }

}   }   }
