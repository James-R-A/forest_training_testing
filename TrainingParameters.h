#pragma once

// This file defines the ParallelParallelTreeTraininer class, which is responsible for
// creating new Tree instances by learning from training data.

#include <assert.h>

#include <vector>
#include <string>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif


namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  /// <summary>
  /// Decision tree training parameters.
  /// </summary>
  struct TrainingParameters
  {
    TrainingParameters()
    {
      // Some sane defaults will need to be changed per application.
      NumberOfTrees = 1;
      NumberOfCandidateFeatures = 10;
      NumberOfCandidateThresholdsPerFeature = 10;
      MaxDecisionLevels = 5;
      Verbose = false;
      MaxThreads = omp_get_max_threads();     
    }

    int NumberOfTrees;
    int NumberOfCandidateFeatures;
    unsigned int NumberOfCandidateThresholdsPerFeature;
    int MaxDecisionLevels;
    bool Verbose;
    int MaxThreads;
  };

  class ForestDescriptor
  {
  public:
    enum e
    {
      Classification = 0x0,
      Regression = 0x1,
      ExpertRegressor = 0x2,
      All = 0x4
    };
  };

  ///<summary>
  /// Program params
  ///</summary>
  struct ProgramParameters
  {
    TrainingParameters Tpc;
    TrainingParameters Tpr;
    std::string OutputFilename;
    std::string TrainingImagesPath;
    int NumberTrainingImages;
    int TrainingImagesStart;
    ForestDescriptor::e ForestType;
    int ExpertClassNo;
    bool DepthRaw;
    int PatchSize;
    int Bins;

    ProgramParameters()
    {
      OutputFilename = "default";
      TrainingImagesPath = ".";
      ExpertClassNo = -1;
      ForestType = ForestDescriptor::Classification;
      DepthRaw = false;
      Bins = 5;
    }

    bool setParam(std::string parameter, std::string value)
    {
      if(parameter.compare("TRAINING_IMAGE_PATH")==0)
      {
        if(IPUtils::dirExists(value))
          TrainingImagesPath = value;
        else
          throw std::runtime_error("images path not found");
      }
      else if(parameter.compare("TRAINING_IMAGES") == 0)
      {
        int n = std::stoi(value);
        if( n >= 10 )
        {
          NumberTrainingImages = n;
        }
        else
          throw std::runtime_error("Number of training images must be at least 10");
      }
      else if(parameter.compare("IMAGES_START") == 0)
      {
        int n = std::stoi(value);
        TrainingImagesStart = n;
      }
      else if(parameter.compare("DEPTH_BINS") == 0)
      {
        int n= std::stoi(value);
        if((n>1)&&(n<6))
          Bins = n;
        else
          throw std::runtime_error("Accepted numbers of bins are 2-5");
      }
      else if(parameter.compare("PATCH_SIZE") == 0)
      {
        int n= std::stoi(value);
        if((n>=3)&&(n<256))
          PatchSize = n;
        else 
          throw std::runtime_error("Accepted patch sizes are 3-255");
      }
      else if(parameter.compare("DEPTH_RAW") == 0)
      {
        if(value.compare("NO")==0)
          DepthRaw = false;
        else if(value.compare("YES")==0)
          DepthRaw = true;
        else
          throw std::runtime_error("Invalid value for DEPTH_RAW");
      }
      else if(parameter.compare("TYPE") == 0)
      {
        if(value.compare("CLASS")==0)
          ForestType = ForestDescriptor::Classification;
        else if(value.compare("REG")==0)
          ForestType = ForestDescriptor::Regression;
        else if(value.compare("EXPREG")==0)
          ForestType = ForestDescriptor::ExpertRegressor;
        else if(value.compare("ALL")==0)
          ForestType = ForestDescriptor::All;
        else
          throw std::runtime_error("Invalid forest type. Accepted values are CLASS, REG, EXPREG, ALL");
      }
      else if(parameter.compare("TREES") == 0)
      {
        int n= std::stoi(value);
        Tpc.NumberOfTrees = n;
        Tpr.NumberOfTrees = n;
      }
      else if(parameter.compare("CLASS_LEVELS") == 0)
      {
        int n= std::stoi(value);
        Tpc.MaxDecisionLevels = n;
      }
      else if(parameter.compare("REG_LEVELS") == 0)
      {
        int n= std::stoi(value);
        Tpr.MaxDecisionLevels = n;
      }
      else if(parameter.compare("CANDIDATE_FEATURES") == 0)
      {
        int n= std::stoi(value);
        Tpc.NumberOfCandidateFeatures = n;
        Tpr.NumberOfCandidateFeatures = n;
      }
      else if(parameter.compare("THRESHOLDS_PER_FEATURE") == 0)
      {
        int n= std::stoi(value);
        Tpc.NumberOfCandidateThresholdsPerFeature = n;
        Tpr.NumberOfCandidateThresholdsPerFeature = n;
      }
      else if(parameter.compare("VERBOSE") == 0)
      {
        bool v;
        if(value.compare("NO")==0)
          v = false;
        else if(value.compare("YES")==0)
          v = true;
        else
          throw std::runtime_error("Invalid value for DEPTH_RAW");

        Tpc.Verbose = v;
        Tpr.Verbose = v;
      }
      else if(parameter.compare("EXPERT") == 0)
      {
        int n= std::stoi(value);
        if(n>=-1 && n<Bins)
          ExpertClassNo = n;
        else
          throw std::runtime_error("Expert outside resonable range");
      }
      else if(parameter.compare("MAX_THREADS") == 0)
      {
        int n= std::stoi(value);
        Tpc.MaxThreads = n;
        Tpr.MaxThreads = n;
      }
      else
        return false;

      return true;
    }

    void prettyPrint()
    {

      std::cout << "Program Parameters:" << std::endl;
      std::cout << std::endl;
      std::cout << "Training images path: \t" << TrainingImagesPath << std::endl;
      std::cout << "Forest output: \t" << OutputFilename << std::endl;
      std::cout << "NumberTrainingImages: \t" << to_string(NumberTrainingImages) << std::endl;
      std::cout << "Starting at image: \t" << to_string(TrainingImagesStart) << std::endl;
      std::cout << "Number of bins: \t" << to_string(Bins) << std::endl;
      std::cout << "Depth Raw" << std::endl;
      std::cout <<  << std::endl;
      std::cout <<  << std::endl;

    }
  };
} } }
