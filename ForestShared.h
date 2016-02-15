#pragma once

#include <memory>

#include "Forest.h"
#include "Tree.h"

// This class allows the repeated application of trees in a forest without the 
// individual trees going out of scope after a single use due to the use of auto_ptr's 
// Alternate solutions to this required construction of a shared_ptr every time a tree
// was called. This was wasteful and slow.

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{

    /// <summary>
    /// A decision forest, i.e. a collection of decision trees.
    /// </summary>
    template<class F, class S>
    class ForestShared // where F:IFeatureResponse where S:IStatisticsAggregator<S>
    {

    public:

        typedef typename std::vector< Tree<F, S>* >::size_type TreeIndex;

        std::vector<std::shared_ptr<Tree<F, S> > > trees_;

        ~ForestShared()
        {
            //for (TreeIndex t = 0; t<trees_.size(); t++)
            //  delete trees_[t].get();
        }

        /// <summary>
        /// Constructs a ForestShared where the vector of trees is a vector of std::shared_ptr's instead
        /// of depricated auto_ptr's. This allows the repeated calls to GetTree without the 
        /// individual trees going out of scope.
        /// </summary>
        /// <param name="forest">The regular forest as defined in the bog-standard Sherwood library</param>
        /// <Returns> A std::unique_ptr to the ForestShared.
        static std::unique_ptr<ForestShared<F, S> > ForestSharedFromForest(Forest<F, S>& forest)
        {
            std::unique_ptr<ForestShared<F, S> > forest_shared = std::unique_ptr<ForestShared<F, S> >(new ForestShared<F, S>);

            for (int i = 0; i < forest.TreeCount(); i++)
            {
                forest_shared->AddTree(forest.trees_[i]);
            }

            if (forest_shared->TreeCount() != forest.TreeCount())
                throw;

            return forest_shared;
        }

        /// <summary>
        /// Add another tree to the forest.
        /// </summary>
        /// <param name="path">The tree.</param>
        void AddTree(Tree<F, S>* tree)
        {
            tree->CheckValid();

            std::shared_ptr<Tree<F, S> > sp1(nullptr);
            sp1 = std::make_shared<Tree<F, S> >(*tree);

            trees_.push_back(sp1);
        }

        /// <summary>
        /// How many trees in the forest?
        /// </summary>
        int TreeCount() const
        {
            return trees_.size();
        }

        /// <summary>
        /// Access the specified tree.
        /// </summary>
        /// <param name="index">A zero-based integer index.</param>
        /// <returns>The tree.</returns>
        const Tree<F, S>& GetTree(int index) const
        {
            return *trees_[index];
        }

        /// <summary>
        /// Access the specified tree.
        /// </summary>
        /// <param name="index">A zero-based integer index.</param>
        /// <returns>The tree.</returns>
        Tree<F, S>& GetTree(int index)
        {
            return *trees_[index];
        }

    };

}}}