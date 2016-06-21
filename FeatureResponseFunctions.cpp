#include "FeatureResponseFunctions.h"

namespace MicrosoftResearch {
    namespace Cambridge {
        namespace Sherwood
        {
            
            RandomHyperplaneFeatureResponse RandomHyperplaneFeatureResponse::CreateRandom(Random& random, unsigned int dimensions)
            {
                return RandomHyperplaneFeatureResponse(random, dimensions);
            }

            float RandomHyperplaneFeatureResponse::GetResponse(const IDataPointCollection& data, unsigned int index) const
            {
                const DataPointCollection& concreteData = (const DataPointCollection&)(data);
                // Get a tuple containing a pointer to an image and the coordinate of 
                // the pixel of interest
                std::tuple<const cv::Mat*, cv::Point> datum;
                if(concreteData.low_memory)
                    datum = concreteData.GetDataPointLM(index);
                else
                    datum = concreteData.GetDataPointRegular(index);

                // extract from tuple
                const cv::Mat* datum_matp = std::get<0>(datum);
                cv::Point datum_point = std::get<1>(datum);
                cv::Size datum_mat_size = datum_matp->size();
                // define image boundaries
                cv::Rect boundry = cv::Rect(0, 0, datum_mat_size.width, datum_mat_size.height);

                std::vector<cv::Point> probe_point(dimensions);
                std::vector<float> pixel_value(dimensions);
                
                // Sum a number of pixel values together if they are within
                // the image boundary
                float response = 0;
                for (unsigned int c = 0; c < dimensions; c++)
                {
                    probe_point[c] = datum_point + offset[c];
                    if (probe_point[c].inside(boundry))
                        pixel_value[c] = datum_matp->at<uchar>(probe_point[c]);
                    else
                        pixel_value[c] = 0;

                    response += float(pixel_value[c]);
                }
                
                return response;
            }

            PixelSubtractionResponse PixelSubtractionResponse::CreateRandom(Random& random, unsigned int dimensions)
            {
                return PixelSubtractionResponse(random, dimensions);
            }
            
            float PixelSubtractionResponse::GetResponse(const IDataPointCollection& data, unsigned int index) const
            {
                const DataPointCollection& concreteData = (const DataPointCollection&)(data);
                // Get a tuple containing a pointer to an image and the coordinate of 
                // the pixel of interest
                std::tuple<const cv::Mat*, cv::Point> datum;
                if(concreteData.low_memory)
                {
                    datum = concreteData.GetDataPointLM(index);
                }
                else
                {
                    datum = concreteData.GetDataPointRegular(index);
                }

                // extract from tuple
                const cv::Mat* datum_matp = std::get<0>(datum);
                cv::Point datum_point = std::get<1>(datum);
                cv::Size datum_mat_size = datum_matp->size();
                // define image boundaries
                cv::Rect boundry = cv::Rect(0, 0, datum_mat_size.width, datum_mat_size.height);
                cv::Point probe_point_0;
                cv::Point probe_point_1;
                float pixel_value_0;
                float pixel_value_1;

                // Get pixel value 0 from probe point 0
                probe_point_0 = datum_point + offset_0;
                if (probe_point_0.inside(boundry))
                    pixel_value_0 = (float)(datum_matp->at<uchar>(probe_point_0));
                else
                    pixel_value_0 = 0;

                // Get pixel value 1 from probe point 1
                probe_point_1 = datum_point + offset_1;
                if (probe_point_1.inside(boundry))
                    pixel_value_1 = (float)(datum_matp->at<uchar>(probe_point_1));
                else
                    pixel_value_1 = 0;
                
                float response = pixel_value_0 - pixel_value_1;
                return response;
            }
            
        }
    }
}

