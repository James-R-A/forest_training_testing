#pragma once

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{ 

	template<class F>
	class IFeatureResponseFactory
	{
	public:
		virtual F CreateRandom(Random& random) = 0;
	};

	/// <summary> 
	/// Feature factory, used for random generation of a FeatureResponseFunction object
	/// </summary>
	template<typename F>
	class FeatureFactory : public IFeatureResponseFactory<F>
	{
	public:

		FeatureFactory(unsigned int dimensions)
			: dimensions(dimensions)
		{};

		F CreateRandom(Random& random)
		{
			return F::CreateRandom(random, dimensions);
		}
	private:
		unsigned int dimensions;

	};

}}}
