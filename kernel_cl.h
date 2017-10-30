#ifndef _kernel_cl_h
#define _kernel_cl_h

#define max(X,Y) (X>Y ? X:Y)
__constant int NUM_AMINO_ACIDS = 24;
__constant char AMINO_ACIDS[25] = "ABCDEFGHIKLMNPQRSTVWYZX";

typedef char seqType;
typedef char substType;
typedef ushort scoreType;

#endif
