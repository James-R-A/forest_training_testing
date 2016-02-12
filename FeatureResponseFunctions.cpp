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
				const std::tuple<cv::Mat*, cv::Point>* datum = concreteData.GetDataPoint(index);

				cv::Mat* datum_matp = std::get<0>(*datum);
				cv::Point datum_point = std::get<1>(*datum);
				cv::Size datum_mat_size = datum_matp->size();
				cv::Rect boundry = cv::Rect(0, 0, datum_mat_size.width, datum_mat_size.height);

				std::vector<cv::Point> probe_point;
				probe_point.resize(dimensions);
				std::vector<float> pixel_value;
				pixel_value.resize(dimensions);
				float response = 0;
				for (unsigned int c = 0; c < dimensions; c++)
				{
					probe_point[c] = datum_point + offset[c];
					if (probe_point[c].inside(boundry))
						pixel_value[c] = datum_matp->at<uchar>(probe_point[c]);
					else
						pixel_value[c] = 0;

					response += n[c] * pixel_value[c];
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

				const std::tuple<cv::Mat*, cv::Point>* datum = concreteData.GetDataPoint(index);
				
				cv::Mat* datum_matp = std::get<0>(*datum);
				cv::Point datum_point = std::get<1>(*datum);
				cv::Size datum_mat_size = datum_matp->size();
				cv::Rect boundry = cv::Rect(0, 0, datum_mat_size.width, datum_mat_size.height);

				std::vector<cv::Point> probe_point(2);
				//probe_point.resize(2);
				std::vector<float> pixel_value(2);
				//pixel_value.resize(2);
				for (int i = 0; i < 2; i++)
				{
					probe_point[i] = datum_point + offset[i];
					if (probe_point[i].inside(boundry))
						pixel_value[i] = (float)(datum_matp->at<uchar>(probe_point[i]));
					else
						pixel_value[i] = 0;
				}
				
				float response = pixel_value[0];
				
				for (int i = 1; i < 2; i++)
				{
					response = response - pixel_value[i];
				}
				
				return response;
			}
		}
	}
}