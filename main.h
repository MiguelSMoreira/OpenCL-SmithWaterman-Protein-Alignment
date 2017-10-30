#ifndef _MAIN_H
#define _MAIN_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <vector>
#include <map>

#define WHOLE_AMOUNT_OF(x,y) ((x+y-1)/y)
#define PADDING_SEQ_NAME "PADDING | PADDING | PADDING"

const int NUM_AMINO_ACIDS = 24;
const char* NUCLEOTIDES = "ACGTURYKMSWBDHVNX";
const char AMINO_ACIDS[NUM_AMINO_ACIDS+1] = "ABCDEFGHIKLMNPQRSTVWYZX";
const char* UNSUPPORTED_LETTERS = "UO*"; // http://faculty.virginia.edu/wrpearson/fasta/fasta36/
const char* UNSUPPORTED_LETTERS_REPLACEMENTS = "CKX";

typedef unsigned long long timestamp_t;

struct Options
{
	char* sequenceFile;
	char* dbFile;
	char* matrix;
	char* topSequenceFile;
	int gapWeight;
	int extendWeight;
	unsigned int listSize;
	bool dna;
}; /**< Program options, globally accessible */

extern struct Options options;

struct Result
{
        int index;
        int score;
};

//-------------------------------------------------------------------------------------



//typedef unsigned short scoreType;
typedef cl_ushort scoreType;

typedef cl_char2 seqType2;

typedef cl_char4 seqType4;
typedef cl_char8 seqType8;
typedef cl_char4 queryType;
typedef cl_ushort2 scoreType2;
typedef cl_ushort4 scoreType4;

struct TempData
{
	scoreType F;
	scoreType Ix;
}__attribute__ ((aligned (4)));

struct  TempData2
{
	TempData a;
	TempData b;
}__attribute__ ((aligned (8)));

struct TempData4
{
	TempData a;
	TempData b;
	TempData c;
	TempData d;
}__attribute__ ((aligned (16)));

//-------------------------------------------------------------------------------------

// GPUDB variables

	static const size_t BLOCK_SIZE = 16; /**< How many sequences make up a block. Should be device half-warp size. Should be power of 2. */
	static const size_t LOG2_BLOCK_SIZE = 4; /**< To be able to use a shift instead of division in the device code */
	static const size_t ALIGNMENT = 64;
	static const size_t SUBBLOCK_SIZE = 8; /**< The number of symbols of each sequence to interleave at a time; to be able to read multiple symbols in one access */
	static const char SEQUENCE_GROUP_TERMINATOR = ' ';
	static const char SUB_SEQUENCE_TERMINATOR = '#';
	

	//typedef unsigned int blockOffsetType;
	typedef cl_uint blockOffsetType;
	//typedef unsigned int seqSizeType;
	typedef cl_uint seqSizeType;
	//typedef unsigned int seqNumType;
	typedef cl_uint seqNumType;
	//typedef char seqType;
	typedef cl_char seqType;
	char* blob; /**< Whole database loaded into memory */
	size_t blobSize; /**< Size in bytes */
	
	/** Pointers to host and device side sequence lenghts and sequence letters in the blob */
	blockOffsetType* blockOffsets;
	seqNumType* seqNums; 
	seqType* sequences;

	char* descriptionBuffer; /**< All descriptions */
	std::vector<char*> descriptions; /**< Pointers into description buffer */
	static struct
	{
		size_t numSequences;
		size_t numSymbols;
		size_t numBlocks;
		size_t alignmentPadding1;
		size_t alignmentPadding2;		
	} metadata; /**< General info on the database */

	//size_t* sequenceOffsets; /**< Pointers to sequences in blob */
	cl_ulong* sequenceOffsets; /**< Pointers to sequences in blob */
	
	//size_t* test;
	cl_ulong* test;

	std::ofstream outFile;

//-------------------------------------------------------------------------------------
// FastaFile variables

	char* buffer; /**< Buffer that is whole file loaded into memory */
	struct FastaRecord
	{
		char* sequence;
		char* description;
		size_t length;
	};
	std::vector<FastaRecord> records; /**< Records index into buffer */
	unsigned int numSymbols;

//-------------------------------------------------------------------------------------
// FastaFile variables for Database

char* buffer2; /**< Buffer that is whole file loaded into memory */
std::vector<FastaRecord> drecords; /**< Records index into buffer */
unsigned int numSymbols2;


//-------------------------------------------------------------------------------------
// Substitution Matrix variables

	static const int MATRIX_SIZE = NUM_AMINO_ACIDS*NUM_AMINO_ACIDS;
	typedef cl_char substType;
	typedef cl_char4 queryType;

	substType matrix[MATRIX_SIZE];
	substType* d_matrix;
	substType* queryProfile;
	cl_ulong queryProfileLength;
	std::map< char,std::map<char,int> > mapMatrix; /**< Map for convenient host-side storage of matrix */
	
//-------------------------------------------------------------------------------------
// Functions

	size_t getSequenceLength(size_t sequenceNum);
	char* getSequence(size_t sequenceNum);
	size_t getNumSequences();
	size_t getDBSizeInBytes();
	size_t getNumSymbols();
	size_t getNumBlocks();
	const char* getDescription(unsigned int index);
	bool writeToFile(size_t sequenceNum);
	static timestamp_t get_timestamp();
	const char *get_error_string(cl_int error);
	static bool resultComparisonFunc(Result r1, Result r2);

    // Functions used to implement the HeapSort algorithm used to efficiently sort the database sequences by sequence length
    void heap_sort(int*, int*, int );
    void  build_max_heap(int*, int*, int );
    void  max_heapify(int*, int*, int, int );


#endif
