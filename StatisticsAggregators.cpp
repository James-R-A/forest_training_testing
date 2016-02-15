#include "StatisticsAggregators.h"


namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
    double HistogramAggregator::Entropy() const
    {
        if (sampleCount_ == 0)
            return 0.0;

        double result = 0.0;
        for (unsigned int b = 0; b < BinCount(); b++)
        {
            double p = (double)bins_[b] / (double)sampleCount_;
            result -= p == 0.0 ? 0.0 : p * log(p) / log(2.0);
        }

        return result;
    }

    HistogramAggregator::HistogramAggregator()
    {
        binCount_ = 0;
        for (unsigned int b = 0; b < binCount_; b++)
            bins_[b] = 0;
        sampleCount_ = 0;
    }

    HistogramAggregator::HistogramAggregator(int nClasses)
    {
        std::string exceptionString = "HistogramAggregator supports a maximum of " + std::to_string(MAX_BINS) + " classes.";
        if (nClasses > MAX_BINS)
            throw std::runtime_error(exceptionString);
        binCount_ = nClasses;
        bins_.resize(nClasses);
        for (unsigned int b = 0; b < binCount_; b++)
            bins_[b] = 0;
        sampleCount_ = 0;
    }

    float HistogramAggregator::GetProbability(int classIndex) const
    {
        return (float)(bins_[classIndex]) / sampleCount_;
    }

    int HistogramAggregator::FindTallestBinIndex() const
    {
        unsigned int maxCount = bins_[0];
        int tallestBinIndex = 0;

        for (unsigned int i = 1; i < BinCount(); i++)
        {
            if (bins_[i] > maxCount)
            {
                maxCount = bins_[i];
                tallestBinIndex = i;
            }
        }

        return tallestBinIndex;
    }

    //////////// IStatisticsAggregator implementation ////////////////
    void HistogramAggregator::Clear()
    {
        for (unsigned int b = 0; b < BinCount(); b++)
            bins_[b] = 0;

        sampleCount_ = 0;
    }

    void HistogramAggregator::Aggregate(const IDataPointCollection& data, unsigned int index)
    {
        const DataPointCollection& concreteData = (const DataPointCollection&)(data);

        bins_[concreteData.GetIntegerLabel((int)index)]++;
        sampleCount_ += 1;
    }

    void HistogramAggregator::Aggregate(const HistogramAggregator& aggregator)
    {
        assert(aggregator.BinCount() == BinCount());

        for (unsigned int b = 0; b < BinCount(); b++)
            bins_[b] += aggregator.bins_[b];

        sampleCount_ += aggregator.sampleCount_;
    }

    HistogramAggregator HistogramAggregator::DeepClone() const
    {
        HistogramAggregator result(BinCount());

        for (unsigned int b = 0; b < BinCount(); b++)
            result.bins_[b] = bins_[b];

        result.sampleCount_ = sampleCount_;

        return result;
    }
    //////////// END IStatisticsAggregator implementation ////////////////

    DiffEntropyAggregator::DiffEntropyAggregator()
    {
        mean_ = 0;
        sample_count_ = 0;
        var_ = 0;
        sse_ = 0;
    }

    double DiffEntropyAggregator::DifferentialEntropy() const
    {
        double result;
        if (this->var_ == 0.0)
            result = 0.0;
        else
            result = log(sqrt(this->var_));

        return result;
    }

    //////////// IStatisticsAggregator implementation ////////////////

    void DiffEntropyAggregator::Aggregate(const IDataPointCollection& data,
        unsigned int index)
    {
        // http://stats.stackexchange.com/questions/72212/updating-variance-of-a-dataset

        const DataPointCollection& concreteData = (const DataPointCollection&)(data);
        this->sample_count_ = this->sample_count_ + 1;
        float err = concreteData.GetTarget(index) - this->mean_;
        this->mean_ = this->mean_ + (err / this->sample_count_);
        this->sse_ = this->sse_ + (err * (concreteData.GetTarget(index) - this->mean_));
        this->var_ = this->sse_ / (this->sample_count_ - 1);
    }

    void DiffEntropyAggregator::Aggregate(const DiffEntropyAggregator& aggregator)
    {
        // http://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-variance-of-two-groups-given-known-group-variances-mean
        int n = this->sample_count_;
        int m = aggregator.sample_count_;
        float var1 = this->var_;
        float var2 = aggregator.var_;
        float mean1 = this->mean_;
        float mean2 = aggregator.mean_;
        this->mean_ = ((n * mean1) + (m * mean2)) / (n + m);
        this->sample_count_ = n + m;
        float num = n * (var1 + (mean1 * mean1)) + m * (var2 + (mean2 * mean2));
        float den = this->sample_count_;
        this->var_ = (num / den) - (this->mean_ * this->mean_);
        this->sse_ = this->var_ * this->sample_count_;  

    }

    void DiffEntropyAggregator::Clear()
    {
        this->mean_ = 0;
        this->sample_count_ = 0;
        this->sse_ = 0;
        this->var_ = 0;
    }

    DiffEntropyAggregator DiffEntropyAggregator::DeepClone() const
    {
        DiffEntropyAggregator result;

        result.mean_ = this->mean_;
        result.sample_count_ = this->sample_count_;
        result.var_ = this->var_;
        result.sse_ = this->sse_;
        
        return result;
    }
    //////////// END IStatisticsAggregator implementation ////////////////

    // For debugging
    void DiffEntropyAggregator::Aggregate(float datum)
    {
        
        this->sample_count_ = this->sample_count_ + 1;
        float err = datum - this->mean_;
        this->mean_ = this->mean_ + (err / this->sample_count_);
        this->sse_ = this->sse_ + (err * (datum - this->mean_));
        this->var_ = this->sse_ / (this->sample_count_ - 1);
    }

}   }   }