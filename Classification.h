#pragma once

// This file defines types used to illustrate the use of the decision forest
// library in simple multi-class classification task (2D data points).

#include <stdio.h>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

#include "Sherwood.h"

#include "FeatureFactory.h"
#include "StatisticsAggregators.h"
#include "FeatureResponseFunctions.h"
#include "DataPointCollection.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{

	template<class F>
	class ClassificationTrainingContext : public ITrainingContext<F, HistogramAggregator> // where F:IFeatureResponse
	{
	private:
		int nClasses_;

		IFeatureResponseFactory<F>* featureFactory_;

	public:
		ClassificationTrainingContext(int nClasses, IFeatureResponseFactory<F>* featureFactory)
		{
			nClasses_ = nClasses;
			featureFactory_ = featureFactory;
		}

	private:
		// Implementation of ITrainingContext
		F GetRandomFeature(Random& random)
		{
			return featureFactory_->CreateRandom(random);
		}

		HistogramAggregator GetStatisticsAggregator()
		{
			return HistogramAggregator(nClasses_);
		}

		double ComputeInformationGain(const HistogramAggregator& allStatistics, const HistogramAggregator& leftStatistics, const HistogramAggregator& rightStatistics)
		{
			double entropyBefore = allStatistics.Entropy();

			unsigned int nTotalSamples = leftStatistics.SampleCount() + rightStatistics.SampleCount();

			if (nTotalSamples <= 1)
				return 0.0;

			double entropyAfter = (leftStatistics.SampleCount() * leftStatistics.Entropy() + rightStatistics.SampleCount() * rightStatistics.Entropy()) / nTotalSamples;

			return entropyBefore - entropyAfter;
		}

		bool ShouldTerminate(const HistogramAggregator& parent, const HistogramAggregator& leftChild, const HistogramAggregator& rightChild, double gain)
		{
			return gain < 0.01;
		}
	};

	template<class F>
	class Classifier
	{

	public:
		static std::auto_ptr<Forest<F, HistogramAggregator> > Train(
			const DataPointCollection& trainingData,
			const TrainingParameters& TrainingParameters) // where F : IFeatureResponse
		{
							
			if (trainingData.HasLabels() == false)
				throw std::runtime_error("Training data points must be labelled.");

			Random random;

			FeatureFactory<F> featureFactory(trainingData.Dimensions());
			ClassificationTrainingContext<F> classificationContext(trainingData.CountClasses(), &featureFactory);
			ProgressStream progress_stream(std::cout, Interest);

			#if defined(_OPENMP)
				std::auto_ptr<Forest<F, HistogramAggregator> > forest
					= ForestTrainer<F, HistogramAggregator>::ParallelTrainForest(
						random, TrainingParameters, classificationContext,
						trainingData, omp_get_max_threads(), &progress_stream);
			#else
				std::auto_ptr<Forest<F, HistogramAggregator> > forest
					= ForestTrainer<F, HistogramAggregator>::TrainForest(
						random, TrainingParameters, classificationContext, trainingData, &progress_stream);
			#endif

			return forest;
		}

		static cv::Mat ApplyMat(Forest<F, HistogramAggregator>& forest, const DataPointCollection& classifyData)
		{
			unsigned int num_classes = forest.GetTreeShared(0)->GetNode(0).TrainingDataStatistics.BinCount();
			unsigned int samples = classifyData.Count();
			cv::Mat bin_mat = cv::Mat::zeros(samples, num_classes, CV_32S);

			for (unsigned int t = 0; t < forest.TreeCount(); t++)
			{
				std::vector<int> leafNodeIndices;
				std::shared_ptr<Tree<F, HistogramAggregator>> tree = forest.GetTreeShared(t);
				tree->Apply(classifyData, leafNodeIndices);
				
				for (unsigned int i = 0; i < classifyData.Count(); i++)
				{
					HistogramAggregator agg = tree->GetNode(leafNodeIndices[i]).TrainingDataStatistics;

					for (unsigned int c = 0; c < num_classes; c++)
					{
						bin_mat.at<int>(cv::Point(c, i)) += int(agg.bins_[c]);
					}
				}
			}
			
			return bin_mat;
		}

		static cv::Mat ApplyMat(ForestShared<F, HistogramAggregator>& forest, const DataPointCollection& classifyData)
		{
			unsigned int num_classes = forest.GetTree(0).GetNode(0).TrainingDataStatistics.BinCount();
			unsigned int samples = classifyData.Count();
			cv::Mat bin_mat = cv::Mat::zeros(samples, num_classes, CV_32S);

			for (unsigned int t = 0; t < forest.TreeCount(); t++)
			{
				std::vector<int> leafNodeIndices;
				Tree<F, HistogramAggregator>& tree = forest.GetTree(t);
				tree.Apply(classifyData, leafNodeIndices);

				for (unsigned int i = 0; i < classifyData.Count(); i++)
				{
					HistogramAggregator agg = tree.GetNode(leafNodeIndices[i]).TrainingDataStatistics;

					for (unsigned int c = 0; c < num_classes; c++)
					{
						bin_mat.at<int>(cv::Point(c, i)) += int(agg.bins_[c]);
					}
				}
			}

			return bin_mat;
		}

	};

}	}	}