#pragma once

// This file defines some IStatisticsAggregator implementations used by the
// example code in Classification.h, DensityEstimation.h, etc. Note we
// represent IStatisticsAggregator instances using simple structs so that all
// tree data can be stored contiguously in a linear array.

#include <math.h>

#include <limits>
#include <vector>
#include <iostream>

#include "Interfaces.h"
#include "DataPointCollection.h"

// Maximum number of categories allowed.
#define MAX_BINS 5

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood 
{
    // Histogram Aggregator is used for classification problems
    // Should be able to deal with multi-dimensional data (I think)
    struct HistogramAggregator
    {
    public:
        std::vector<unsigned int> bins_;
        unsigned int binCount_;

        unsigned int sampleCount_;

    public:
        /// <summary>
        /// Returns the Shannon entropy
        /// </summary>
        double Entropy() const;

        /// <summary>
        /// Creates a HistogramAggregator instance with number of bins set to 0
        /// </summary>
        HistogramAggregator();
        
        /// <summary>
        /// Creates a HistogramAggregator instance with number of bins set to nClasses
        /// </summary>
        /// <param name="nClasses"> Number of bins </param>
        HistogramAggregator(int nClasses);

        float GetProbability(int classIndex) const;

        unsigned int BinCount() const { return binCount_; }

        unsigned int SampleCount() const { return sampleCount_; }

        int FindTallestBinIndex() const;

        //////////// IStatisticsAggregator implementation ////////////////
        void Clear();

        void Aggregate(const IDataPointCollection& data, unsigned int index);

        void Aggregate(const HistogramAggregator& aggregator);

        HistogramAggregator DeepClone() const;
        //////////// END IStatisticsAggregator implementation ////////////////
    };

    // DiffEntropyAggregator is basically just a 1-d gaussian and stores 
    // the mean, variance, and sum squared error
    // The "entropy" value calculated is log(var_) = differential entropy
    // hence the name.
    struct DiffEntropyAggregator
    {
    public:
        float mean_;
        float sse_;
        float var_;
        unsigned int sample_count_;

        /// <summary>
        /// Returns the Shannon entropy
        /// </summary>
        double DifferentialEntropy() const;

        /// <summary>
        /// Creates a DiffEntropyAggregator instance with mean, sse, and var = 0
        /// </summary>
        DiffEntropyAggregator();

        unsigned int SampleCount() const { return sample_count_; }

        float GetMean() const { return mean_; }

        //////////// IStatisticsAggregator implementation ////////////////
        void Clear();

        void Aggregate(const IDataPointCollection& data, unsigned int index);

        void Aggregate(const DiffEntropyAggregator& aggregator);

        DiffEntropyAggregator DeepClone() const;
        //////////// END IStatisticsAggregator implementation ////////////////

        // For debugging
        void Aggregate(float datum);

    };

}   }   }
