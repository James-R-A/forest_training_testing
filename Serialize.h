#pragma once
#include "FeatureResponseFunctions.h"
#include "StatisticsAggregators.h"

// Default serialization functions used for serializing all types.
// Works fine for basic types, but if implementations are not simple 
// value types then use explicit template specialization to override.
// Some examples are shown in this file.
namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{

    template<typename T>
    void  binary_write(std::ostream& o, const T& value)
    {
        o.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template<typename T>
    void binary_read(std::istream& i, T& value)
    {
        i.read(reinterpret_cast<char*>(&value), sizeof(T));
    }

    template<typename T>
    static void Serialize_(std::ostream& o, const T& t)
    {
        binary_write(o, t);
    }

    template<typename T>
    static void Deserialize_(std::istream& o, T& t)
    {
        binary_read(o, t);
    }

    // If it's microsoft visual c++, it insists on these explicit
    // template instances being specified static
    #ifdef _MSC_VER
        template<>
        static void Serialize_<PixelSubtractionResponse>(std::ostream& o, const PixelSubtractionResponse& b)
        {
            binary_write(o, b.dimensions);
            binary_write(o, b.offset_0.x);
            binary_write(o, b.offset_0.y);
            binary_write(o, b.offset_1.x);
            binary_write(o, b.offset_1.y);
        }

        template<>
        static void Deserialize_<PixelSubtractionResponse>(std::istream& o, PixelSubtractionResponse& b)
        {
            binary_read(o, b.dimensions);
            binary_read(o, b.offset_0.x);
            binary_read(o, b.offset_0.y);
            binary_read(o, b.offset_1.x);
            binary_read(o, b.offset_1.y);
        }

        template<>
        static void Serialize_<RandomHyperplaneFeatureResponse>(std::ostream& o, const RandomHyperplaneFeatureResponse& b)
        {
            unsigned int size = b.dimensions;
            binary_write(o, b.dimensions);
            for (unsigned int i = 0; i < size; i++)
            {
                binary_write(o, b.offset[i].x);
                binary_write(o, b.offset[i].y);
            }
            for (unsigned int i = 0; i < size; i++)
            {
                binary_write(o, b.n[i]);
            }
        }

        template<>
        static void Deserialize_<RandomHyperplaneFeatureResponse>(std::istream& o, RandomHyperplaneFeatureResponse& b)
        {
            unsigned int size;
            binary_read(o, size);
            b.dimensions = size;
            b.offset.resize(size);
            b.n.resize(size);
            for (unsigned int i = 0; i < size; i++)
            {
                binary_read(o, b.offset[i].x);
                binary_read(o, b.offset[i].y);
            }
            for (unsigned int i = 0; i < size; i++)
            {
                binary_read(o, b.n[i]);
            }
        }

        template<>
        static void Serialize_<HistogramAggregator>(std::ostream& o, const HistogramAggregator& b)
        {
            binary_write(o, b.binCount_);
            binary_write(o, b.sampleCount_);

            for (unsigned int i = 0; i < b.binCount_; i++)
            {
                binary_write(o, b.bins_[i]);
            }
        }

        template<>
        static void Deserialize_<HistogramAggregator>(std::istream& o, HistogramAggregator& b)
        {
            binary_read(o, b.binCount_);
            binary_read(o, b.sampleCount_);
            b.bins_.resize(b.binCount_);

            for (unsigned int i = 0; i < b.binCount_; i++)
            {
                binary_read(o, b.bins_[i]);
            }
        }
    // If it's not MS Visual C++, it doesn't like static in these 
    // explicit template instances. 
    #else
        template<>
        void Serialize_<PixelSubtractionResponse>(std::ostream& o, const PixelSubtractionResponse& b)
        {
            binary_write(o, b.dimensions);
            binary_write(o, b.offset_0.x);
            binary_write(o, b.offset_0.y);
            binary_write(o, b.offset_1.x);
            binary_write(o, b.offset_1.y);
        }

        template<>
        void Deserialize_<PixelSubtractionResponse>(std::istream& o, PixelSubtractionResponse& b)
        {
            binary_read(o, b.dimensions);
            binary_read(o, b.offset_0.x);
            binary_read(o, b.offset_0.y);
            binary_read(o, b.offset_1.x);
            binary_read(o, b.offset_1.y);
        }

        template<>
        void Serialize_<RandomHyperplaneFeatureResponse>(std::ostream& o, const RandomHyperplaneFeatureResponse& b)
        {
            unsigned int size = b.dimensions;
            binary_write(o, b.dimensions);
            for (unsigned int i = 0; i < size; i++)
            {
                binary_write(o, b.offset[i].x);
                binary_write(o, b.offset[i].y);
            }
            for (unsigned int i = 0; i < size; i++)
            {
                binary_write(o, b.n[i]);
            }
        }

        template<>
        void Deserialize_<RandomHyperplaneFeatureResponse>(std::istream& o, RandomHyperplaneFeatureResponse& b)
        {
            unsigned int size;
            binary_read(o, size);
            b.dimensions = size;
            b.offset.resize(size);
            b.n.resize(size);
            for (unsigned int i = 0; i < size; i++)
            {
                binary_read(o, b.offset[i].x);
                binary_read(o, b.offset[i].y);
            }
            for (unsigned int i = 0; i < size; i++)
            {
                binary_read(o, b.n[i]);
            }
        }

        template<>
        void Serialize_<HistogramAggregator>(std::ostream& o, const HistogramAggregator& b)
        {
            binary_write(o, b.binCount_);
            binary_write(o, b.sampleCount_);

            for (unsigned int i = 0; i < b.binCount_; i++)
            {
                binary_write(o, b.bins_[i]);
            }
        }

        template<>
        void Deserialize_<HistogramAggregator>(std::istream& o, HistogramAggregator& b)
        {
            binary_read(o, b.binCount_);
            binary_read(o, b.sampleCount_);
            b.bins_.resize(b.binCount_);

            for (unsigned int i = 0; i < b.binCount_; i++)
            {
                binary_read(o, b.bins_[i]);
            }
        }
    #endif


}}}