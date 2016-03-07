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
	FQueue();
	FQueue(unsigned int capacity);
	FQueue(const FQueue<T> & v);
	FQueue(FQueue<T> && other) :
			theSize(0), theFront(0), theTail(0)
	{
		theBuffer.clear();
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
			theBuffer.clear();
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
	~FQueue();

	unsigned int capacity() const;
	unsigned int size() const;
	bool empty() const;
	T & front();
	T & tail();
	void push_back(const T & value);
	void pop_front();
	void pop_front(const unsigned int numberOfElementsToPop);

	void reserve(unsigned int capacity);
	void resize(unsigned int capacity);

	T & operator[](unsigned int index);
	FQueue<T> & operator=(const FQueue<T> &);
	void clear();
private:
	unsigned int theSize;
	unsigned int theFront;
	unsigned int theTail;
	std::vector<T> theBuffer;

};

// Your code goes here ...
template<class T>
FQueue<T>::FQueue()
{
	theSize = 0;
	theBuffer(0);
	theFront = 0;
	theTail = 0;

}

template<class T>
FQueue<T>::FQueue(const FQueue<T> & v)
{
	theSize = v.theSize;
	theBuffer = v.theBuffer;
	theFront = v.theFront;
	theTail = v.theTail;
}

template<class T>
FQueue<T>::FQueue(unsigned int capacity)
{
	theBuffer.resize(capacity);
	theSize = 0;
	theFront = 0;
	theTail = 0;
}

template<class T>
FQueue<T> & FQueue<T>::operator =(const FQueue<T> & v)
{
	if (this != &v)
	{
		theBuffer.clear();
		theSize = v.theSize;
		theBuffer = v.theBuffer;
		theFront = v.theFront;
		theTail = v.theTail;
	}
	return *this;

}

template<class T>
T& FQueue<T>::front()
{
	return theBuffer[theFront];
}

template<class T>
T& FQueue<T>::tail()
{
	return theBuffer[theTail];
}

template<class T>
void FQueue<T>::push_back(const T & v)
{
//	std::cout << "head tail and size before pushing " << theFront << " " << theTail << " " << theSize << std::endl;
//	std::cout << "content before pushing" << std::endl;
//	for(int i =0; i< theSize; i++)
//		std::cout << theBuffer.at((theFront+i)%theBuffer.capacity()) << std::endl;
	if (theSize >= theBuffer.size())
	{
		auto oldCapacity = theBuffer.size();
		auto oldTail = theTail;
		theBuffer.reserve(oldCapacity + theTail);

		if (theFront != 0)
		{
//			std::copy(theBuffer.begin(), theBuffer.begin() + theTail, theBuffer.begin() + oldCapacity);
			for (int i = 0; i < theTail; ++i)
			{
				theBuffer.push_back(theBuffer[i]);
			}
			theTail = 0;

		}
		else
		{
			theBuffer.resize(oldCapacity + 16);
			theTail += oldCapacity;
		}
//		theTail += oldCapacity;

//		std::cout << "resized" << std::endl;
	}

	theBuffer[theTail] = v;
	theTail = (theTail + 1) % theBuffer.size();
	theSize++;
//	std::cout << "head and tail after pushing " << theFront << " " << theTail << " " << theSize << std::endl;
//
//	std::cout << "content after pushing" << std::endl;
//	for(int i =0; i< theSize; i++)
//		std::cout << theBuffer.at((theFront+i)%theBuffer.capacity()) << std::endl;
//	std::cout << "\n\n" << std::endl;

}

template<class T>
void FQueue<T>::pop_front()
{
	if (theSize > 0)
	{
		theFront = (theFront + 1) % theBuffer.size();
		theSize--;
	}
}

template<class T>
void FQueue<T>::reserve(unsigned int capacity)
{
	theBuffer.reserve(capacity);
}

template<class T>
unsigned int FQueue<T>::size() const //
{
	return theSize;
}

template<class T>
void FQueue<T>::resize(unsigned int capacity)
{
	theBuffer.resize(capacity);

}

template<class T>
T& FQueue<T>::operator[](unsigned int index)
{
	return theBuffer[(theFront + index) % theBuffer.size()];
}

template<class T>
unsigned int FQueue<T>::capacity() const
{
	return theBuffer.capacity();
}

template<class T>
FQueue<T>::~FQueue()
{

}

template<class T>
void FQueue<T>::clear()
{
	theBuffer.clear();
	theSize = 0;
	theFront = 0;
	theTail = 0;
}

template<class T>
void FQueue<T>::pop_front(const unsigned int numberOfElementsToPop)
{
	unsigned int elementsToErase =
			theSize > numberOfElementsToPop ? numberOfElementsToPop : theSize;
	theSize -= elementsToErase;
	theFront = (theFront + elementsToErase) % theBuffer.size();
}

#endif
