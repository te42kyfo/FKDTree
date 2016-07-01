#ifndef FKDTREE_FKDTREE_CPU_H_
#define FKDTREE_FKDTREE_CPU_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>
#include "FKDPoint.h"
#include "FKDTree.h"
#include "FQueue.h"
#define FLOOR_LOG2(X) ((unsigned int)(31 - __builtin_clz(X | 1)))
#define CEIL_LOG2(X) ((unsigned int)(X <= 1 ? 0 : 32 - __builtin_clz(X - 1)))

template <class TYPE, unsigned int numberOfDimensions>
class FKDTree_CPU : public FKDTree<TYPE, numberOfDimensions> {
 public:
  FKDTree_CPU(const std::vector<FKDPoint<TYPE, numberOfDimensions>>& points)
      : theDepth(FLOOR_LOG2(points.size())),
        theDimensions(numberOfDimensions, std::vector<TYPE>(points.size())),
        theIds(points.size()),
        thePoints(begin(points), end(points)) {}

  std::vector<unsigned int> search_in_the_box_branchless(
      const FKDPoint<TYPE, numberOfDimensions>& minPoint,
      const FKDPoint<TYPE, numberOfDimensions>& maxPoint) {
    FQueue<unsigned int> indecesToVisit(256);
    std::vector<unsigned int> foundPoints;
    foundPoints.reserve(16);
    indecesToVisit.push_back(0);
    unsigned int index;
    bool intersection;
    int dimension;
    unsigned int numberOfIndecesToVisitThisDepth;
    int maxNumberOfSonsToVisitNext;
    int numberOfSonsToVisitNext;
    unsigned int firstSonToVisitNext;
    for (unsigned int depth = 0; depth < theDepth + 1; ++depth) {
      dimension = depth % numberOfDimensions;
      numberOfIndecesToVisitThisDepth = indecesToVisit.size();
      for (unsigned int visitedIndecesThisDepth = 0;
           visitedIndecesThisDepth < numberOfIndecesToVisitThisDepth;
           visitedIndecesThisDepth++) {
        index = indecesToVisit[visitedIndecesThisDepth];
        intersection = intersects(index, minPoint, maxPoint, dimension);
        firstSonToVisitNext = leftSonIndex(index);
        maxNumberOfSonsToVisitNext =
            (firstSonToVisitNext < thePoints.size()) +
            ((firstSonToVisitNext + 1) < thePoints.size());

        if (intersection) {
          if (is_in_the_box(index, minPoint, maxPoint)) {
            foundPoints.emplace_back(theIds[index]);
          }
          numberOfSonsToVisitNext = maxNumberOfSonsToVisitNext;
        } else {
          numberOfSonsToVisitNext = std::min(maxNumberOfSonsToVisitNext, 1);
          firstSonToVisitNext +=
              (theDimensions[dimension][index] < minPoint[dimension]);
        }

        for (int whichSon = 0; whichSon < numberOfSonsToVisitNext; ++whichSon)
          indecesToVisit.push_back(firstSonToVisitNext + whichSon);
      }

      indecesToVisit.pop_front(numberOfIndecesToVisitThisDepth);
    }
    return foundPoints;
  }

  std::vector<unsigned int> search_in_the_box_BFS(
      const FKDPoint<TYPE, numberOfDimensions>& minPoint,
      const FKDPoint<TYPE, numberOfDimensions>& maxPoint) {
    // the queue could become a data member in case we don't want to run this in
    // parallel
    FQueue<unsigned int> indecesToVisit(256);
    std::vector<unsigned int> foundPoints;
    foundPoints.reserve(16);
    indecesToVisit.push_back(0);
    unsigned int index;
    bool intersection;
    int dimension;
    int depth;
    int maxNumberOfSonsToVisitNext;
    int numberOfSonsToVisitNext;
    unsigned int firstSonToVisitNext;

    while (!indecesToVisit.empty()) {
      index = indecesToVisit.pop_front();
      depth = FLOOR_LOG2(index + 1);

      dimension = depth % numberOfDimensions;

      intersection = intersects(index, minPoint, maxPoint, dimension);
      firstSonToVisitNext = leftSonIndex(index);
      maxNumberOfSonsToVisitNext =
          (firstSonToVisitNext < thePoints.size()) +
          ((firstSonToVisitNext + 1) < thePoints.size());

      if (intersection) {
        if (is_in_the_box(index, minPoint, maxPoint)) {
          foundPoints.emplace_back(theIds[index]);
        }
        numberOfSonsToVisitNext = maxNumberOfSonsToVisitNext;
      } else {
        bool willVisitOnlyRightSon =
            theDimensions[dimension][index] < minPoint[dimension];
        numberOfSonsToVisitNext =
            willVisitOnlyRightSon == maxNumberOfSonsToVisitNext
                ? 0
                : std::min(maxNumberOfSonsToVisitNext, 1);

        firstSonToVisitNext += willVisitOnlyRightSon;
      }

      for (int whichSon = 0; whichSon < numberOfSonsToVisitNext; ++whichSon)
        indecesToVisit.push_back(firstSonToVisitNext + whichSon);
    }
    return foundPoints;
  }

  std::vector<unsigned int> search_in_the_box(
      const FKDPoint<TYPE, numberOfDimensions>& minPoint,
      const FKDPoint<TYPE, numberOfDimensions>& maxPoint) const {
    FQueue<unsigned int> indecesToVisit(256);
    std::vector<unsigned int> result;
    result.reserve(16);
    indecesToVisit.push_back(0);
    int dimension;
    unsigned int numberOfIndecesToVisitThisDepth;
    unsigned int index;
    bool intersection;
    bool isLowerThanBoxMin;
    int startSon;
    int endSon;
    unsigned int indexToAdd;
    for (unsigned int depth = 0; depth < theDepth + 1; ++depth) {
      dimension = depth % numberOfDimensions;
      numberOfIndecesToVisitThisDepth = indecesToVisit.size();
      for (unsigned int visitedIndecesThisDepth = 0;
           visitedIndecesThisDepth < numberOfIndecesToVisitThisDepth;
           visitedIndecesThisDepth++) {
        index = indecesToVisit[visitedIndecesThisDepth];
        intersection = intersects(index, minPoint, maxPoint, dimension);

        if (intersection && is_in_the_box(index, minPoint, maxPoint))
          result.push_back(theIds[index]);

        isLowerThanBoxMin =
            theDimensions[dimension][index] < minPoint[dimension];

        startSon = isLowerThanBoxMin;  // left son = 0, right son =1

        endSon = isLowerThanBoxMin || intersection;

        for (int whichSon = startSon; whichSon < endSon + 1; ++whichSon) {
          indexToAdd = leftSonIndex(index) + whichSon;

          if (indexToAdd < thePoints.size()) {
            indecesToVisit.push_back(indexToAdd);
          }
        }
      }

      indecesToVisit.pop_front(numberOfIndecesToVisitThisDepth);
    }
    return result;
  }

  void search_in_the_box_recursive(
      const FKDPoint<TYPE, numberOfDimensions>& minPoint,
      const FKDPoint<TYPE, numberOfDimensions>& maxPoint,
      std::vector<unsigned int>& foundPoints, unsigned int index = 0,
      int dimension = 0) const {
    unsigned int firstSonToVisitNext = leftSonIndex(index);
    int maxNumberOfSonsToVisitNext =
        (firstSonToVisitNext < thePoints.size()) +
        ((firstSonToVisitNext + 1) < thePoints.size());
    bool intersection = intersects(index, minPoint, maxPoint, dimension);

    int numberOfSonsToVisitNext;
    if (intersection) {
      if (is_in_the_box(index, minPoint, maxPoint)) {
        foundPoints.emplace_back(theIds[index]);
      }
      numberOfSonsToVisitNext = maxNumberOfSonsToVisitNext;
    } else {
      bool isLowerThanBoxMin =
          theDimensions[dimension][index] < minPoint[dimension];
      numberOfSonsToVisitNext =
          isLowerThanBoxMin && (maxNumberOfSonsToVisitNext == 1)
              ? 0
              : std::min(maxNumberOfSonsToVisitNext, 1);
      firstSonToVisitNext += isLowerThanBoxMin;
    }

    if (numberOfSonsToVisitNext != 0)

    {
      auto nextDimension = (dimension + 1) % numberOfDimensions;
      for (int whichSon = 0; whichSon < numberOfSonsToVisitNext; ++whichSon)
        search_in_the_box_recursive(minPoint, maxPoint, foundPoints,
                                    firstSonToVisitNext + whichSon,
                                    nextDimension);
    }
  }

  std::vector<TYPE> const& getDimensionVector(const int dimension) const {
    if (dimension < numberOfDimensions) return theDimensions[dimension];
  }

  void build() {
    // gather kdtree building
    int dimension;
    std::vector<unsigned int> theIntervalLength(thePoints.size());
    std::vector<unsigned int> theIntervalMin(thePoints.size());
    theIntervalMin[0] = 0;
    theIntervalLength[0] = thePoints.size();

    for (unsigned int depth = 0; depth < theDepth; ++depth) {
      dimension = depth % numberOfDimensions;
      unsigned int firstIndexInDepth = (1 << depth) - 1;
      for (unsigned int indexInDepth = 0; indexInDepth < (1 << depth);
           ++indexInDepth) {
        unsigned int indexInArray = firstIndexInDepth + indexInDepth;
        unsigned int leftSonIndexInArray = 2 * indexInArray + 1;
        unsigned int rightSonIndexInArray = leftSonIndexInArray + 1;

        unsigned int whichElementInInterval =
            partition_complete_kdtree(theIntervalLength[indexInArray]);
        std::nth_element(
            thePoints.begin() + theIntervalMin[indexInArray],
            thePoints.begin() + theIntervalMin[indexInArray] +
                whichElementInInterval,
            thePoints.begin() + theIntervalMin[indexInArray] +
                theIntervalLength[indexInArray],
            [dimension](const FKDPoint<TYPE, numberOfDimensions>& a,
                        const FKDPoint<TYPE, numberOfDimensions>& b) -> bool {
              if (a[dimension] == b[dimension])
                return a.getId() < b.getId();
              else
                return a[dimension] < b[dimension];
            });
        add_at_position(
            thePoints[theIntervalMin[indexInArray] + whichElementInInterval],
            indexInArray);

        if (leftSonIndexInArray < thePoints.size()) {
          theIntervalMin[leftSonIndexInArray] = theIntervalMin[indexInArray];
          theIntervalLength[leftSonIndexInArray] = whichElementInInterval;
        }

        if (rightSonIndexInArray < thePoints.size()) {
          theIntervalMin[rightSonIndexInArray] =
              theIntervalMin[indexInArray] + whichElementInInterval + 1;
          theIntervalLength[rightSonIndexInArray] =
              (theIntervalLength[indexInArray] - 1 - whichElementInInterval);
        }
      }
    }

    dimension = theDepth % numberOfDimensions;
    unsigned int firstIndexInDepth = (1 << theDepth) - 1;
    for (unsigned int indexInArray = firstIndexInDepth;
         indexInArray < thePoints.size(); ++indexInArray) {
      add_at_position(thePoints[theIntervalMin[indexInArray]], indexInArray);
    }
  }

 private:
  void add_at_position(const FKDPoint<TYPE, numberOfDimensions>& point,
                       const unsigned int position) {
    for (unsigned int i = 0; i < numberOfDimensions; ++i)
      theDimensions[i][position] = point[i];
    theIds[position] = point.getId();
  }

  unsigned int partition_complete_kdtree(unsigned int length) {
    if (length == 1) return 0;
    unsigned int index = 1 << (FLOOR_LOG2(length));
    //		unsigned int index = 1 << ((int) log2(length));

    if ((index / 2) - 1 <= length - index)
      return index - 1;
    else
      return length - index / 2;
  }

  unsigned int leftSonIndex(unsigned int index) const { return 2 * index + 1; }

  unsigned int rightSonIndex(unsigned int index) const { return 2 * index + 2; }

  bool intersects(unsigned int index,
                  const FKDPoint<TYPE, numberOfDimensions>& minPoint,
                  const FKDPoint<TYPE, numberOfDimensions>& maxPoint,
                  int dimension) const {
    return (theDimensions[dimension][index] <= maxPoint[dimension] &&
            theDimensions[dimension][index] >= minPoint[dimension]);
  }

  bool is_in_the_box(unsigned int index,
                     const FKDPoint<TYPE, numberOfDimensions>& minPoint,
                     const FKDPoint<TYPE, numberOfDimensions>& maxPoint) const {
    for (unsigned int i = 0; i < numberOfDimensions; ++i) {
      if ((theDimensions[i][index] <= maxPoint[i] &&
           theDimensions[i][index] >= minPoint[i]) == false)
        return false;
    }

    return true;
    ;
  }

  unsigned int theDepth;
  std::vector<FKDPoint<TYPE, numberOfDimensions>> thePoints;
  std::vector<std::vector<TYPE>> theDimensions;
  std::vector<unsigned int> theIds;
};

#endif /* FKDTREE_FKDTREE_CPU_H_ */
