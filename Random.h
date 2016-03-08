#pragma once
// Code taken from https://github.com/johannesu/sherwood-classify-matlab
// This implementation of Random.h aims to allow larger numbers in Windows by 
// not using RAND_MAX which is only 2^15 (not big enough). 
// Should work just the same though.
// This files defines the Random class, used throughout the forest training
// framework for random number generation.

#include <time.h>
#include <cstdlib>
#include <iostream> 

namespace MicrosoftResearch {
  namespace Cambridge {
    namespace Sherwood
    {
      // RAND_MAX on visual studio is just 2^15 which is way too small
      // This code also avoid the issue that srand is not thread safe.
      class Random
      {
      public:
        Random() : seed((unsigned int)(time(NULL))), a(214013), c(2531011), m(2147483648)
        {}

        Random(unsigned int s) : seed(s), a(214013), c(2531011), m(2147483648)
        {}

        int Next() {
          return(seed = (a * seed + c) % m);
        }

        double NextDouble()
        {
          return (double)(Next()) / m;
        }

        int Next(int minValue, int maxValue)
        {
          return minValue + Next() % (maxValue - minValue);
        }

        std::vector<int> RandomVector(int minValue, int maxValue, int length, bool replacement)
        {
          if((maxValue - minValue < length)&&!replacement)
            throw std::runtime_error("Not enough integers in that range");

          int rand_int;
          std::vector<int> random_vec;
          random_vec.resize(length);
          std::vector<int>::iterator it;
          bool continue_flag = false;
          bool continue_reset = replacement? true : false;
          for(int i=0;i<length;i++)
          {
            continue_flag = false;
            while(!continue_flag)
            {
              rand_int = Next(minValue,maxValue);
              it = find(random_vec.begin(), random_vec.end(), rand_int);
              if(it == random_vec.end())
              {
                continue_flag = true;
                random_vec[i] = rand_int;
              }
            }

          }
          return random_vec;
        }

      private:
        const int a, c;
        const unsigned int m;
        unsigned int seed;
      };

    }
  }
}

/*  OLD VERSION
#pragma once

// This files defines the Random class, used throughout the forest training
// framework for random number generation.

#include <time.h>
#include <cstdlib>

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  /// <summary>
  /// Encapsulates random number generation - so as to facilitate
  /// overriding of standard library behaviours.
  /// </summary>
  class Random
  {
  public:
    /// <summary>
    /// Creates a 'random number' generator using a seed derived from the system time.
    /// </summary>
    Random()
    {
      srand ( (unsigned int)(time(NULL)) );
    }

    /// <summary>
    /// Creates a deterministic 'random number' generator using the specified seed.
    /// May be useful for debugging.
    /// </summary>
    Random(unsigned int seed)
    {
      srand ( seed );
    }

    /// <summary>
    /// Generate a positive random number.
    int Next()
    {
      return rand();
    }

    /// <summary>
    /// Generate a random number in the range [0.0, 1.0).
    /// </summary>
    double NextDouble()
    {
      return (double)(rand())/RAND_MAX;
    }

    /// <summary>
    /// Generate a random integer within the sepcified range.
    /// </summary>
    /// <param name="minValue">Inclusive lower bound.</param>
    /// <param name="maxValue">Exclusive upper bound.</param>
    int Next(int minValue, int maxValue)
    {
      return minValue + rand()%(maxValue-minValue);
    }
  };
} } } */
