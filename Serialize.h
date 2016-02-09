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

	template<>
	static void Serialize_<PixelSubtractionResponse>(std::ostream& o, const PixelSubtractionResponse& b)
	{
		binary_write(o, b.dimensions);
		for (unsigned int i = 0; i < 2; i++)
		{
			binary_write(o, b.offset[i].x);
			binary_write(o, b.offset[i].y);
		}
		
	}

	template<>
	static void Deserialize_<PixelSubtractionResponse>(std::istream& o, PixelSubtractionResponse& b)
	{
		binary_read(o, b.dimensions);
		b.offset.resize(2);
		for (unsigned int i = 0; i < 2; i++)
		{
			binary_read(o, b.offset[i].x);
			binary_read(o, b.offset[i].y);
		}
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

}}}