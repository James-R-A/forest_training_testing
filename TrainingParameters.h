#pragma once

// This file defines the Training and Program parameters objects which control the 
// operation of the program

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
      Classification = 0,
      Regression = 1,
      ExpertRegressor = 2,
      All = 3
    };
  };

  class SplitFunctionDescriptor
  {
  public:
    enum e
    {
      PixelDifference = 0,
      RandomHyperplane = 1
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
    std::string InputPrefix;
    int NumberTrainingImages;
    int TrainingImagesStart;
    ForestDescriptor::e ForestType;
    SplitFunctionDescriptor::e SplitFunctionType;
    int ExpertClassNo;
    bool DepthRaw;
    int PatchSize;
    int Bins;
    int ImgHeight;
    int ImgWidth;
    bool TrainOnZeroIR;
    int MR;

    ProgramParameters()
    {
      OutputFilename = "default";
      TrainingImagesPath = "/media/james/data_wd/training_realsense";
      InputPrefix = "img";
      NumberTrainingImages = 10;
      TrainingImagesStart = 0;
      ForestType = ForestDescriptor::Classification;
      SplitFunctionType = SplitFunctionDescriptor::PixelDifference;
      ExpertClassNo = -1;
      DepthRaw = false;
      PatchSize = 25;
      Bins = 5;
      ImgHeight = 480;
      ImgWidth = 640;
      TrainOnZeroIR = true;
      MR = 1200;
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
        if((n > omp_get_max_threads()) || (n == -1))
        {
          Tpc.MaxThreads = omp_get_max_threads();
          Tpr.MaxThreads = omp_get_max_threads();
        }
        else if(n < -1)
        {
          throw std::runtime_error("Invalid number of threads");
        }
        else
        {
          Tpc.MaxThreads = n;
          Tpr.MaxThreads = n;
        } 
      } 
      else if(parameter.compare("SPLIT_FUNCTION")==0)
      {
        if(value.compare("PIXEL_DIFFERENCE") == 0)
          SplitFunctionType = SplitFunctionDescriptor::PixelDifference;
        else if(value.compare("RANDOM_HYPERPLANE")==0)
          SplitFunctionType = SplitFunctionDescriptor::RandomHyperplane;
        else
          throw std::runtime_error("Invalid value for SPLIT_FUNCTION, accepted values are PIXEL_DIFFERENCE and RANDOM_HYPERPLANE");
      }
      else if(parameter.compare("FOREST_OUTPUT")==0)
      {
        OutputFilename = value;
      }
      else if(parameter.compare("INPUT_PREFIX")==0)
      {
        InputPrefix = value;
      }
      else if(parameter.compare("IMG_WIDTH") == 0)
      {
        int n= std::stoi(value);
        ImgWidth = n;
      }
      else if(parameter.compare("IMG_HEIGHT") == 0)
      {
        int n= std::stoi(value);
        ImgHeight = n;
      }
      else if(parameter.compare("TRAIN_ON_ZERO_IR") == 0)
      {
        if(value.compare("YES") == 0)
          TrainOnZeroIR = true;
        else if(value.compare("NO") == 0)
          TrainOnZeroIR = false;
        else
          throw std::runtime_error("Invalid value for TRAIN_ON_ZERO_IR, expected YES or NO");
      }
      else if(parameter.compare("MAX_RANGE")==0)
      {
        int n = std::stoi(value);
        if((n <= 1200)&&(n>=500))
          MR = n;
        else
          throw std::runtime_error("Max range must be between 500 and 1200");
      }
      else
        return false;

      return true;
    }

    void prettyPrint()
    {
      std::string forestTypes [] = {"Classification", "Regression", "ExpertRegressor", "All"};
      std::string splitTypes [] = {"Pixel Difference Response", "Random Hyperplane Response"};

      std::cout << "Program Parameters:" << std::endl;
      std::cout << std::endl;
      std::cout << "Forest Type: \t\t\t" << forestTypes[ForestType] << std::endl;
      std::cout << "Split Function Type: \t\t" << splitTypes[SplitFunctionType] << std::endl;
      std::cout << "Number of bins: \t\t" << std::to_string(Bins) << std::endl;
      std::cout << "Training images path: \t\t" << TrainingImagesPath << std::endl;
      std::cout << "Training images file prefix: \t" << InputPrefix << std::endl;
      std::cout << "NumberTrainingImages: \t\t" << std::to_string(NumberTrainingImages) << std::endl;
      std::cout << "Starting at image: \t\t" << std::to_string(TrainingImagesStart) << std::endl;
      std::string dr = DepthRaw? "True" : "False";
      std::cout << "Depth Raw: \t\t\t" << dr << std::endl;
      std::cout << "Patch Size: \t\t\t" << std::to_string(PatchSize) << std::endl;
      std::cout << "Max Depth Range: \t\t" << std::to_string(MR) << std::endl;
      std::cout << "Image Width:\t\t\t" << std::to_string(ImgWidth) << std::endl;
      std::cout << "Image Height:\t\t\t" << std::to_string(ImgHeight) << std::endl;
      std::cout << "Forest output prefix: \t\t" << OutputFilename << std::endl;
      std::cout << "Trees per Forest:\t\t" << std::to_string(Tpr.NumberOfTrees) <<std::endl;
      std::cout << "Regression Decision Levels: \t" << std::to_string(Tpr.MaxDecisionLevels) << std::endl;
      std::cout << "Classification Decision Levels:\t" << std::to_string(Tpc.MaxDecisionLevels) << std::endl;
      std::cout << "Candidate Features: \t\t" << std::to_string(Tpr.NumberOfCandidateFeatures) << std::endl;
      std::cout << "Thresholds per Features: \t" << std::to_string(Tpr.NumberOfCandidateThresholdsPerFeature) << std::endl;
      std::cout << "ExpertClassNo: \t\t\t" << std::to_string(ExpertClassNo) << std::endl;
      std::cout << "Verbose: \t\t\t" << (Tpc.Verbose? "Yes" : "no") << std::endl;
      std::cout << "Train on zero IR: \t\t" << (TrainOnZeroIR? "Yes" : "no") << std::endl;
      std::cout << "Max threads to use: \t\t" << std::to_string(Tpr.MaxThreads) << std::endl;
      

    }
  };
} } }
