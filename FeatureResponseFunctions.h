#pragma once

// This file defines some IFeatureResponse implementations used by the example code in
// Classification.h, DensityEstimation.h, etc. Note we represent IFeatureResponse
// instances using simple structs so that all tree data can be stored
// contiguously in a linear array.

#include <string>
#include <math.h>
#include <vector>
#include <cmath>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "DataPointCollection.h"
#include "Interfaces.h"
#include "Random.h"


namespace MicrosoftResearch {
    namespace Cambridge {
        namespace Sherwood
        {
            class Random;
            
            // Generate a normal distributed number given a uniform random number generator
            // Box-muller transform.
            static float randn(Random& random)
            {
                float u = (2 * random.NextDouble()) - 1;
                float v = (2 * random.NextDouble()) - 1;
                float w = u * u + v * v;

                if (w == 0 || w > 1) 
                    return randn(random);

                float x = sqrt(-2 * log(w) / w);
                return u * x;
            }

            /// <summary>   f(x,n) = Sum(n*p(x)) where n is a randomly generated integer,
            ///             and p(x) is a pixel in the patch surrounding pixel x. patch size^2 = dimensions</summary>
            class RandomHyperplaneFeatureResponse
            {
            public:
                unsigned dimensions;
                std::vector<float> n;
                std::vector<cv::Point> offset;

                RandomHyperplaneFeatureResponse() {
                    dimensions = 0;
                }

                RandomHyperplaneFeatureResponse(Random& random,
                    unsigned int dimensions)
                    : dimensions(dimensions)
                {

                    n.resize(dimensions);
                    offset.resize(dimensions);

                    int ub = (int)((sqrt(dimensions) -1) / 2);
                    int lb = 0 - ub;
                    // Normal distributed numbers to gives an unbiased random unit vector.
                    for (unsigned int c = 0; c < dimensions; c++) {
                        n[c] = randn(random);
                    }

                    int i = 0;
                    for (int r = lb; r <= ub; r++)
                    {
                        for (int c = lb; c <= ub; c++)
                        {
                            offset[i] = cv::Point(c, r);
                            i++;
                        }
                    }

                }

                static RandomHyperplaneFeatureResponse CreateRandom(Random& random, unsigned int dimensions);

                // IFeatureResponse implementation
                float GetResponse(const IDataPointCollection& data, unsigned int index) const;

            };

            /// <summary>   f(x,u,v) = I(x+u) - I(x+v) where x is the evaluated pixel in image I
            ///             and u and v are random 2-d pixel offsets within (sqrt(dimension)-1)/2 
            ///             of the pixel being evaluated. (equiv to (patch_size-1)/2)  </summary>
            class PixelSubtractionResponse
            {
            public:
                unsigned dimensions;
                cv::Point offset_0;
                cv::Point offset_1;

                PixelSubtractionResponse() {
                    dimensions = 0;
                    offset_0 = cv::Point(0,0);
                    offset_1 = cv::Point(0,0);
                }

                PixelSubtractionResponse(Random& random,
                    unsigned int dimensions)
                    : dimensions(dimensions)
                {
                    
                    int ub = (int)ceil(sqrt(dimensions) / 2);
                    int lb = 0 - ub;

                    offset_0 = cv::Point(random.Next(lb,ub), random.Next(lb,ub));
                    offset_1 = cv::Point(random.Next(lb,ub), random.Next(lb,ub));
                }

                static PixelSubtractionResponse CreateRandom(Random& random, unsigned int dimensions);
                
                // IFeatureResponse implementation
                float GetResponse(const IDataPointCollection& data, unsigned int index) const;

            };
        }
    }
}
