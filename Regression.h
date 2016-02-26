#pragma once

// This file defines types used to illustrate the use of the decision forest
// library in simple 1D to 1D regression task.

#include "Sherwood.h"

#include "FeatureFactory.h"
#include "StatisticsAggregators.h"
#include "FeatureResponseFunctions.h"
#include "DataPointCollection.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{

    template<class F>
    class RegressionTrainingContext : public ITrainingContext<F, DiffEntropyAggregator> // where F:IFeatureResponse
    {
    private:
        int nClasses_;

        IFeatureResponseFactory<F>* featureFactory_;

    public:
        RegressionTrainingContext(IFeatureResponseFactory<F>* featureFactory)
        {
            featureFactory_ = featureFactory;
        }

    private:
        // Implementation of ITrainingContext
        F GetRandomFeature(Random& random)
        {
            return featureFactory_->CreateRandom(random);
        }

        DiffEntropyAggregator GetStatisticsAggregator()
        {
            return DiffEntropyAggregator();
        }

        double ComputeInformationGain(const DiffEntropyAggregator& allStatistics, const DiffEntropyAggregator& leftStatistics, const DiffEntropyAggregator& rightStatistics)
        {
            double entropyBefore = allStatistics.DifferentialEntropy();

            unsigned int nTotalSamples = leftStatistics.SampleCount() + rightStatistics.SampleCount();

            if (nTotalSamples <= 1)
                return 0.0;

            double entropyAfter = (leftStatistics.SampleCount() * leftStatistics.DifferentialEntropy() + rightStatistics.SampleCount() * rightStatistics.DifferentialEntropy()) / nTotalSamples;
        
            return entropyBefore - entropyAfter;
        }

        bool ShouldTerminate(const DiffEntropyAggregator& parent, const DiffEntropyAggregator& leftChild, const DiffEntropyAggregator& rightChild, double gain)
        {
            return gain < 0.01;
        }
    };

    /// <summary>
    /// A class for construction and application of regression based decision trees.
    /// </summary>
    template<class F>
    class Regressor
    {
    public:
        /// <summary>
        /// Create and train a classification forest (HistogramAggregator statistics) 
        /// If OpenMP is compiled, this function parallelises by evaluating node responses in parallel
        /// training one tree at a time.
        /// </summary>
        static std::unique_ptr<Forest<F, DiffEntropyAggregator> > TrainPar(
            const DataPointCollection& trainingData,
            const TrainingParameters& TrainingParameters) // where F : IFeatureResponse
        {

            if (trainingData.HasTargetValues() == false)
                throw std::runtime_error("Training data points must have target values.");

            // For random number generation.
            Random random;
            FeatureFactory<F> featureFactory(trainingData.Dimensions());
            RegressionTrainingContext<F> regressionContext(&featureFactory);
            ProgressStream progress_stream(std::cout, Interest);
            if (TrainingParameters.Verbose)
                progress_stream.makeVerbose();

            std::unique_ptr<Forest<F, DiffEntropyAggregator> > forest = ParallelForestTrainer<F, DiffEntropyAggregator>::TrainForest(
                random, TrainingParameters, regressionContext, trainingData, &progress_stream);

            return forest;
        }

                /// <summary>
        /// Create and train a classification forest (HistogramAggregator statistics) 
        /// If OpenMP is compiled, this function parallelises by evaluating node responses in parallel
        /// training one tree at a time.
        /// </summary>
        // static std::unique_ptr<Forest<F, DiffEntropyAggregator> > TrainPar(
        //     const LMDataPointCollection& trainingData,
        //     const TrainingParameters& TrainingParameters) // where F : IFeatureResponse
        // {
        //     std::cout << "foo" << std::endl;
        //     if (trainingData.HasTargetValues() == false)
        //         throw std::runtime_error("Training data points must have target values.");

        //     // For random number generation.
        //     Random random;

        //     FeatureFactory<F> featureFactory(trainingData.Dimensions());
        //     RegressionTrainingContext<F> regressionContext(&featureFactory);
        //     ProgressStream progress_stream(std::cout, Interest);
        //     if (TrainingParameters.Verbose)
        //         progress_stream.makeVerbose();

        //     std::unique_ptr<Forest<F, DiffEntropyAggregator> > forest = ParallelForestTrainer<F, DiffEntropyAggregator>::TrainForest(
        //         random, TrainingParameters, regressionContext, trainingData, &progress_stream);

        //     return forest;
        // }

        /// <summary>
        /// Sends an openCV Mat object down each tree of a forest (per-pixel) and 
        /// aggregates the results.
        /// returns a std::vector where each element corresponds to an input pixel's
        /// probability distribution's mean value. 
        /// Beware, due to the nature of Forest trees, use of this function 
        /// will put the trees out of scope. To avoid this, use a ForestShared
        /// object instead.
        /// </summary>
        static std::vector<uint16_t> ApplyMat(Forest<F, DiffEntropyAggregator>& forest, const DataPointCollection& regressData)
        {
            unsigned int samples = regressData.Count();
            std::vector<uint16_t> ret(samples);
            
            for (unsigned int t = 0; t < forest.TreeCount(); t++)
            {
                std::vector<int> leafNodeIndices;
                std::shared_ptr<Tree<F, DiffEntropyAggregator> > tree = forest.GetTreeShared(t);
                tree->Apply(regressData, leafNodeIndices);

                for (unsigned int i = 0; i < regressData.Count(); i++)
                {
                    // TODO potentially remove this declaration if it's taking too much time
                    DiffEntropyAggregator agg = tree->GetNode(leafNodeIndices[i]).TrainingDataStatistics;

                    ret[i] = uint16_t(round(agg.mean_));
                }
            }

            return ret;
        }

        /// <summary>
        /// Sends an openCV Mat object down each tree of a forest (per-pixel) and 
        /// aggregates the results.
        /// returns a std::vector where each element corresponds to an input pixel's
        /// probability distribution's mean value. 
        /// </summary>
        static std::vector<uint16_t> ApplyMat(ForestShared<F, DiffEntropyAggregator>& forest, const DataPointCollection& regressData)
        {
            unsigned int samples = regressData.Count();
            std::vector<uint16_t> ret(samples);
            
            for (unsigned int t = 0; t < forest.TreeCount(); t++)
            {
                std::vector<int> leafNodeIndices;
                Tree<F, DiffEntropyAggregator>& tree = forest.GetTree(t);
                tree.Apply(regressData, leafNodeIndices);

                for (unsigned int i = 0; i < regressData.Count(); i++)
                {
                    DiffEntropyAggregator agg = tree.GetNode(leafNodeIndices[i]).TrainingDataStatistics;

                    ret[i] = uint16_t(round(agg.mean_));
                }
            }

            return ret;
        }
    
    };

}   }   }