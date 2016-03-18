/*
 * Queue.h
 *
 *  Created on: Mar 1, 2016
 *      Author: fpantale
 */

#ifndef FKDTREE_QUEUE_H_
#define FKDTREE_QUEUE_H_

#include <vector>
template<class T>

class FQueue
{
public:
	FQueue()
	{

		theSize = 0;
//		theBuffer(0);
		theFront = 0;
		theTail = 0;
		theCapacity = 0;

	}

	FQueue(unsigned int initialCapacity)
	{
//		theBuffer.resize(initialCapacity);
		theSize = 0;
		theFront = 0;
		theTail = 0;

		theCapacity = initialCapacity;
	}

	FQueue(const FQueue<T> & v)
	{
		theSize = v.theSize;
		theBuffer = v.theBuffer;
		theFront = v.theFront;
		theTail = v.theTail;
		theCapacity = v.theCapacity;
	}

	FQueue(FQueue<T> && other) :
			theSize(0), theFront(0), theTail(0)
	{
//		theBuffer.clear();
		theCapacity = other.theCapacity;
		theSize = other.theSize;
		theFront = other.theFront;
		theTail = other.theTail;
		theBuffer = other.theBuffer;
		other.theSize = 0;
		other.theFront = 0;
		other.theTail = 0;
	}

	FQueue<T>& operator=(FQueue<T> && other)
	{

		if (this != &other)
		{
//			theBuffer.clear();
			theSize = other.theSize;
			theFront = other.theFront;
			theTail = other.theTail;
			theBuffer = other.theBuffer;
			other.theSize = 0;
			other.theFront = 0;
			other.theTail = 0;
		}
		return *this;

	}

	FQueue<T> & operator=(const FQueue<T>& v)
	{
		if (this != &v)
		{
//			theBuffer.clear();
			theSize = v.theSize;
			theBuffer = v.theBuffer;
			theFront = v.theFront;
			theTail = v.theTail;
		}
		return *this;

	}
	~FQueue()
	{
	}

	unsigned int size() const
	{
		return theSize;
	}
	bool empty() const
	{
		return theSize == 0;
	}
	T & front()
	{

		return theBuffer[theFront];

	}
	T & tail()
	{
		return theBuffer[theTail];
	}

	void push_back(const T & value)
	{

//		if (theSize >= theCapacity)
//		{
//			theBuffer.reserve(theCapacity + theTail);
//			if (theFront != 0)
//			{
//				for (unsigned int i = 0; i < theTail; ++i)
//				{
//					theBuffer.push_back(theBuffer[i]);
//				}
//				theCapacity += theTail;
//
//				theTail = 0;
//			}
//			else
//			{
//
//				theBuffer.resize(theCapacity + 16);
//				theTail += theCapacity;
//				theCapacity += 16;
//
//			}
//
//		}

		theBuffer[theTail] = value;
		theTail = (theTail + 1) % theCapacity;

		theSize++;


	}

	void print()
	{
		std::cout << "printing the content of the queue:" << std::endl;
		for(unsigned int i = theFront;  i  != theTail; i = ( i+ 1) % theCapacity)
			std::cout << theBuffer[i] << " at position " << i << std::endl;

	}
	T pop_front()
	{

		if (theSize > 0)
		{
			T element = theBuffer[theFront];
			theFront = (theFront + 1) % theCapacity;
			theSize--;


			return element;
		}
	}

	void pop_front(const unsigned int numberOfElementsToPop)
	{
		unsigned int elementsToErase =
				theSize > numberOfElementsToPop ?
						numberOfElementsToPop : theSize;
		theSize -= elementsToErase;
		theFront = (theFront + elementsToErase) % theCapacity;
	}

	void reserve(unsigned int capacity)
	{
//		theBuffer.reserve(capacity);
	}
	void resize(unsigned int capacity)
	{
//		theBuffer.resize(capacity);

	}

	T & operator[](unsigned int index)
	{
		return theBuffer[(theFront + index) % theCapacity];
	}

	void clear()
	{
//		theBuffer.clear();
		theSize = 0;
		theFront = 0;
		theTail = 0;
	}
private:
	unsigned int theSize;
	unsigned int theFront;
	unsigned int theTail;
//	std::vector<T> theBuffer;
	std::array<T, 1000> theBuffer;
	unsigned int theCapacity;

};

#endif
