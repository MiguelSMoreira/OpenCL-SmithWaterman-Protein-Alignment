#include "kernel_cl.h"

/*****************************************************
Kernel Function
 ****************************************************/
__kernel void mul_kernel( global char *Database, global unsigned short *Scores, int numbSeq, int gapPenalty, int gapExtendPenalty, int queryLength, global int *DBSeqLength, global int *mapMatrix, global char *Query){

    // Kernel is ran one time per database sequence (each kernel is a different db sequence)
    int id = get_global_id(0);

    // Calculating offset necessary to access correct Database values for this db sequence
    int addressDatabase=0, maxScore=0;
    for(int f=1; f<=id; f++ ) addressDatabase+=DBSeqLength[f-1];
    
    if (id < numbSeq){
        Scores[id] = 0;
        // Initializing of auxiliary structures for Smith-Waterman algorithm
        int FColumn[5750], IxColumn[5750];
        for( uint j=1; j<queryLength+1; j++ ){      FColumn[j]=0;
                                                    IxColumn[j]=0;      }
        
        // Scoping through the entire database sequence, "i" at a time
        for(uint i=1;i<DBSeqLength[id]+1;i++){
            int top=0, left=0, topLeft=0, topIy=0;
            
            //"Scoping" through the entire query sequence, "j" at a time
            for( uint j=1; j<queryLength+1; j++ ){
                // Implementation of the Smith-Waterman algorithm
                int score;
                left = FColumn[j];
                int subs = mapMatrix[ Query[j-1] * NUM_AMINO_ACIDS + Database[ addressDatabase + i - 1] ];
                int align = topLeft + subs;
                score = max(align,0);
                // Used for supporting affine gap penalties in the algorithm
                int newIx= max(left+gapPenalty+gapExtendPenalty,IxColumn[j]+gapExtendPenalty);
                score = max(score,newIx);
            
                int newIy = max(top+gapPenalty+gapExtendPenalty,topIy+gapExtendPenalty);
                score = max(score,newIy);
                maxScore = max( score, maxScore );
                // Refreshing score column w/values to left of currently processing column
                FColumn[j] = score;
                IxColumn[j] = newIx;
                top = score;
                topLeft = left;
                topIy = newIy;
            }
        }
    }
    // Writing final calculated value to output score vector
    Scores[id]=maxScore;
}
