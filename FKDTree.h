#ifndef FKDTREE_FKDTREE_H_
#define FKDTREE_FKDTREE_H_

#include "FKDPoint.h"

template <class TYPE, unsigned int numberOfDimensions>
class FKDTree {
 public:
  FKDTree(const std::vector<FKDPoint<TYPE, numberOfDimensions>>& points) {}
  FKDTree() {}

  virtual std::vector<unsigned int> search_in_the_box(
      const FKDPoint<TYPE, numberOfDimensions>& minPoint,
      const FKDPoint<TYPE, numberOfDimensions>& maxPoint) const = 0;

  virtual std::vector<std::vector<unsigned int>> search_in_the_box_multiple(
      const std::vector<FKDPoint<TYPE, numberOfDimensions>>& minPoints,
      const std::vector<FKDPoint<TYPE, numberOfDimensions>>& maxPoints)
      const = 0;

  virtual void build() = 0;
  virtual std::vector<TYPE> const& getDimensionVector(const int dimension) = 0;
  virtual std::vector<unsigned int> const& getIdVector() = 0;

  bool test_correct_build(unsigned int index = 0, int dimension = 0)  {
    unsigned int theNumberOfPoints = getDimensionVector(0).size();
    unsigned int leftSonIndexInArray = 2 * index + 1;
    unsigned int rightSonIndexInArray = leftSonIndexInArray + 1;
    if (rightSonIndexInArray >= theNumberOfPoints &&
        leftSonIndexInArray >= theNumberOfPoints) {
      return true;
    } else {
      if (leftSonIndexInArray < theNumberOfPoints) {
        if (getDimensionVector(dimension)[index] >=
            getDimensionVector(dimension)[leftSonIndexInArray]) {
          test_correct_build(leftSonIndexInArray,
                             (dimension + 1) % numberOfDimensions);
        } else
          return false;
      }

      if (rightSonIndexInArray < theNumberOfPoints) {
        if (getDimensionVector(dimension)[index] <=
            getDimensionVector(dimension)[rightSonIndexInArray]) {
          test_correct_build(rightSonIndexInArray,
                             (dimension + 1) % numberOfDimensions);
        } else
          return false;
      }
    }
    return true;
  }
};

template <typename TYPE, int numberOfDimensions>
bool test_correct_search(
    const std::vector<FKDPoint<TYPE, numberOfDimensions>> points,
    const std::vector<unsigned int> foundPoints,
    const FKDPoint<TYPE, numberOfDimensions>& minPoint,
    const FKDPoint<TYPE, numberOfDimensions>& maxPoint) {
  bool testGood = true;
  for (unsigned int i = 0; i < points.size(); ++i) {
    bool shouldBeInTheBox = true;
    for (unsigned int dim = 0; dim < numberOfDimensions; ++dim) {
      shouldBeInTheBox &=
          (points[i][dim] <= maxPoint[dim] && points[i][dim] >= minPoint[dim]);
    }

    bool foundToBeInTheBox = std::find(foundPoints.begin(), foundPoints.end(),
                                       points[i].getId()) != foundPoints.end();

    if (foundToBeInTheBox == shouldBeInTheBox) {
      testGood &= true;
    } else {
      if (foundToBeInTheBox)
        std::cout << "Point " << points[i].getId()
                  << " was wrongly found to be in the box." << std::endl;
      else
        std::cout << "Point " << points[i].getId()
                  << " was wrongly found to be outside the box." << std::endl;

      testGood &= false;
    }
  }

  return testGood;
}

#endif /* FKDTREE_FKDTREE_H_ */
