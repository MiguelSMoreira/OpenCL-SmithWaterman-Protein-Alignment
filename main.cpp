#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "stdafx.h"
#include "main.h"
#include <wchar.h>
#include <string.h>
#include <algorithm>
#include <limits>
#include <string.h>
#include <fstream>
#include <cstddef>
#include <vector>
#include <assert.h>
#include <malloc.h>
#include <sys/time.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define MAX(X,Y) (X>Y ? X:Y)

Options options;

int main(int argc, char* argv[]){

    options.gapWeight = -10;
    options.extendWeight = -2;
    options.listSize = 20;
    options.dna = 0;
    
    //Show help info if not enough command line options provided
    if(argc != 4){
        puts("Usage: gpu <blosum> <sequence> <database>");
        puts("Sequence: FASTA format file; Database: DBCONV .fasta file.");
        return EXIT_SUCCESS;
    }

    options.matrix = argv[1];
    options.sequenceFile = argv[2];
    options.dbFile = argv[3];
    
//--------------------------------------------------------- 
    //-----------------------------------------------------
    // STEP 1: Discover and initialize the platforms
    //-----------------------------------------------------
    timestamp_t clinit = get_timestamp();
    
    cl_int status;
    cl_uint numPlatforms = 0;
    cl_platform_id *platforms = NULL;
    
    // Use clGetPlatformIDs() to retrieve the number of platforms
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    
    // Allocate enough space for each platform
    platforms = (cl_platform_id*) malloc( numPlatforms * sizeof(cl_platform_id) );
    
    // Fill in platforms with clGetPlatformIDs()
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if(status != CL_SUCCESS){
        printf("error in step 1\n");
        exit(-1);
    }
    
//--------------------------------------------------------- 
    //-----------------------------------------------------
    // STEP 2: Discover and initialize the devices
    //-----------------------------------------------------
    cl_uint numDevices = 0, numSpecDev = 0;
    cl_device_id *devices = NULL;
    
    // Use clGetDeviceIDs() to retrieve the number of devices present
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
    // Allocate enough space for each device
    devices = (cl_device_id*) malloc( numDevices * sizeof(cl_device_id));
    
    // Fill in devices with clGetDeviceIDs()
    status |= clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, &numSpecDev);
    
    // Select the device which will be used
    int device_id = 0;
    if((status != CL_SUCCESS) || ((unsigned int) device_id >= numDevices)){
        printf("error in step 2\n");
        exit(-1);
    }
    
    /*cl_uint multiProcessorCount;
    char buf[512];
    // Print the selected compute device and respective number of available SMs
    clGetDeviceInfo(devices[device_id], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(multiProcessorCount), &multiProcessorCount, NULL);
    clGetDeviceInfo(devices[device_id], CL_DEVICE_NAME, sizeof(buf), buf, NULL);
    printf("Using %s accelerator, with %u available streaming multiprocessors.\n", buf, (unsigned int)multiProcessorCount);
    fflush(stdout);*/

//---------------------------------------------------------
    //-----------------------------------------------------
    // STEP 3: Create a context
    //-----------------------------------------------------
    cl_context context = NULL;
    
    // Create a context using clCreateContext() and associate it with the devices
    context = clCreateContext(NULL, 1, &devices[device_id], NULL, NULL, &status);
    if(status != CL_SUCCESS){
        printf("error in step 3\n");
        exit(-1);
    }
    
//---------------------------------------------------------   
    //-----------------------------------------------------
    // STEP 4: Create a command queue
    //-----------------------------------------------------
    cl_command_queue cmdQueue;
    
    // Create a command queue using clCreateCommandQueue(),and associate it with the
    //device you want to execute on
    cmdQueue = clCreateCommandQueue( context, devices[device_id], 0, &status);
    if(status != CL_SUCCESS){
        printf("error in step 4\n");
        exit(-1);
    }

    timestamp_t clend = get_timestamp();
    timestamp_t clinittime = clend - clinit;
    //printf("CLInit: %llu\n",clinittime);
    
//---------------------------------------------------------
    //-----------------------------------------------------
    // STEP 5: Load query, database and substitution matrix
    //-----------------------------------------------------
    timestamp_t dbinit = get_timestamp();

    //-----------------------------------------------------
    // STEP 5.1: Load query into records[i] data structure where size of
    //  records is records.size and we have data records[i].description and
    //  records[i].sequence
    //-----------------------------------------------------
    const char *fileNamequ = options.sequenceFile;
    bool dna = options.dna;
    const char* ALPHABET;
    if(dna)
        ALPHABET = NUCLEOTIDES;
    else
        ALPHABET = AMINO_ACIDS;

    std::ifstream filequ;
    filequ.open(fileNamequ,std::ios::binary);
    if(!filequ.is_open()){
        printf("\nCannot open sequence file: %s\nUsage: gpu <blosum> <sequence> <database>\nSequence: FASTA format file; Database: DBCONV .fasta file.\n", fileNamequ);
        return EXIT_FAILURE;
    }
    
    //Get size
    unsigned int fsize;
    filequ.seekg(0, std::ios::end);
    fsize = filequ.tellg();
    if(fsize < 1){ //Empty file
        filequ.close();
        printf("Sequence file empty\n");
        return EXIT_FAILURE;
    }

    //Read file into memory
    buffer = new char[fsize+1];
    filequ.seekg(0);
    filequ.read(buffer,fsize);
    buffer[fsize]= '\0';                            // Buffer now has our sequence file info
    filequ.close();                                 // which is query to compare to database
    if(filequ.bad()){
        printf("reading sequence file error\n");
        return EXIT_FAILURE;
    }

    //Process records
    FastaRecord r;                              //Loading sequence records into Fasta Record
    r.length = 0;                               //type variable r
    //char* context;
    char* tokStr=buffer+1;//Skip initial '>'
    
    // Goes through entire sequence file, loads sequences description in r.description and the sequence data itself in r.sequence
    while(1){
        r.description=strtok(tokStr,"\n");      //Breaks down tokStr string into the several
        r.sequence=strtok('\0',">");            //smaller string separated with \n in tokStr
        if(!r.description || !r.sequence)
            break;
        records.push_back(r);
        tokStr='\0';
    }   

    //Strip unwanted characters
    for(size_t i=0;i<records.size();i++){
        char* badChar;
        //Strip newlines from description
        while((badChar=strpbrk(records[i].description,"\r\n"))) // Searched for str1 in str2
            *badChar='\0';
        int copyAmt = 0;

        //For each bad character, increase the shift amount so we don't have to perform any superfluous copies
        size_t recLen = strlen(records[i].sequence)+1;
        //+1 so NULL gets copied
        for(char* c=records[i].sequence;c<records[i].sequence+recLen;c++){
            *c=toupper(*c);
            const char* badResidue;
            if(!dna){
                //Replace unsupported symbols by replacements
                if(*c!='\0'&&(badResidue=strchr(UNSUPPORTED_LETTERS,*c))!='\0'){
                    *c=UNSUPPORTED_LETTERS_REPLACEMENTS[badResidue-UNSUPPORTED_LETTERS];
                }
            }

            const char* residueIndex;
            //Invalid character, skip it
            if((residueIndex=strchr(ALPHABET,*c))=='\0'){
                copyAmt--;
                //Usually the unsupported characters should only be whitespace and newlines
                if(*c!='\n' && *c!='\r' && *c != ' '){
                    printf("Deleted unknown character %c\n",*c);
                }
            }

            else{
                if(*c!='\0'){
                    records[i].length++;
                    numSymbols++;
                    *c=residueIndex-ALPHABET; //Replace symbol with its alphabetic index
                }
                if(copyAmt!=0)
                    *(c+copyAmt)=*c;
            }
        }
    }
    printf("MATRIX: %s\n", options.matrix);
    printf("SEQUENCE: %s, input sequence length is %zd.\n",options.sequenceFile,getSequenceLength(0));

    
//---------------------------------------------------------------
    //-----------------------------------------------------------
    // STEP 5.2: Load substitution matrix (blosum 62 FILE) into mapMatrix[row][columnName]
    //-----------------------------------------------------------
    const char *fileNamebm = options.matrix;
    if(!fileNamebm){
        puts("No blosum File\n");
        return EXIT_FAILURE;
    }
    std::ifstream filebm;
    filebm.open(fileNamebm);
    if(!filebm.is_open()){
        printf("\nCannot open blosum File: %s\nUsage: gpu <blosum> <sequence> <database>\nSequence: FASTA format file; Database: DBCONV .fasta file.\n", fileNamebm);
        return EXIT_FAILURE;
    }           
    bool readHeader = false;
    std::string header;
    char row=' ';
    size_t column=0;
    while(1){
        char c;
        filebm >> c;
        if( filebm.eof() )
            break;

        //Skip comments
        if(c=='#')    {
            filebm.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
            continue;
        }
        
        if(!readHeader){ //Read row of amino acid letters
            if(header.find(c)!=std::string::npos) //Duplicate character: have gone past header into rows
                readHeader=true;
            else                
                header.append(1,c);
        }
        
        if(readHeader){ //Interpret rows of values
            if(header.find(c)!=std::string::npos){ //Amino
                row=c;
                column=0;
            }
            else{ //Value
                if(header.length()<column+1){
                    puts("Error in sub matrix\n");
                    return EXIT_FAILURE;
                }
                char columnName = header.at(column);
                column++;
                filebm.putback(c);
                int val;
                filebm>>val;
                mapMatrix[row][columnName]=val;
            }
        }
    }
    
    filebm.close();
    
    
//-----------------------------------------------------------------
    //-------------------------------------------------------------
    // STEP 5.3: Create queryprofile into queryProfile[j*queryProfileLength+i], which
    //  is an improved implementation of a substitution matrix
    //-------------------------------------------------------------
/*
    if(dna)
        ALPHABET = NUCLEOTIDES;
    else
        ALPHABET = AMINO_ACIDS;

    size_t seqLength = getSequenceLength(0);
    
    queryProfileLength = WHOLE_AMOUNT_OF(getSequenceLength(0),sizeof(queryType))*sizeof(queryType);
    //printf("queryProfileLength : %d \n", queryProfileLength);
    char* seq = getSequence(0);
    size_t qpSize = queryProfileLength*NUM_AMINO_ACIDS;
    queryProfile = new substType[qpSize];
    if(!queryProfile){
        puts("queryProfile error\n");
        return EXIT_FAILURE;
    }
    for(size_t j=0;j<NUM_AMINO_ACIDS;j++)
    {
        char d = ALPHABET[j];
        for(size_t i=0;i<queryProfileLength;i++)
        {
            substType s;
            if(i>=seqLength) //If query sequence is too short, pad profile with zeroes
                s=0;
            else
            {
                // CUIDADO USAR (INT) PODE CAUSAR PROBLEMAS NO FUNCIONAMENTO
                char q = ALPHABET[(int)seq[i]];
                s = mapMatrix[q][d];
            }
            queryProfile[j*queryProfileLength+i] = s;

        }
    }
*/
    
//------------------------------------------------------------
    //--------------------------------------------------------
    // STEP 5.4: Load database (swiss prot) into drecords[i] data structure,
    //   the same type used for the query sequence file
    //--------------------------------------------------------
    const char *fileNamequeu = options.dbFile;
    if(dna)
        ALPHABET = NUCLEOTIDES;
    else
        ALPHABET = AMINO_ACIDS;
    
    std::ifstream filequeu;
    filequeu.open(fileNamequeu,std::ios::binary);
    if(!filequeu.is_open()){
        printf("\nCannot open database sequence file: %s\nUsage: gpu <blosum> <sequence> <database>\nSequence: FASTA format file; Database: DBCONV .fasta file.\n", fileNamequeu);
        return EXIT_FAILURE;
    }
    
    //Get size
    unsigned int fsize2;
    filequeu.seekg(0, std::ios::end);
    fsize2 = filequeu.tellg();
    if(fsize2 < 1){ //Empty file
        filequeu.close();
        printf("Database sequence file empty\n");
        return EXIT_FAILURE;
    }
    
    //Read file into memory
    buffer2 = new char[fsize2+1];
    filequeu.seekg(0);
    filequeu.read(buffer2,fsize2);
    buffer2[fsize2]= '\0';                            // Buffer now has our sequence file info
    filequeu.close();                                 // which is query to compare to database
    if(filequeu.bad()){
        printf("reading database sequence file error\n");
        return EXIT_FAILURE;
    }
    
    //Process database records
    FastaRecord d;                              //Loading sequence records into Fasta Record
    d.length = 0;                               //type variable r
    //char* context;
    char *tokStr2 = buffer2+1;//Skip initial '>'
    
    // Goes through entire database sequence file, loads sequences description in r.description and the sequence data itself in r.sequence
    while(1){
        d.description=strtok(tokStr2,"\n");      //Breaks down tokStr string into the several
        d.sequence=strtok('\0',">");            //smaller string separated with \n in tokStr
        if(!d.description || !d.sequence)
            break;
        drecords.push_back(d);
        tokStr2='\0';
    }
    
    //Strip unwanted characters
    for(size_t t=0;t<drecords.size();t++ ){
        char* badChar2;
        //Strip newlines from description
        while((badChar2=strpbrk(drecords[t].description,"\r\n"))) // Searched for str1 in str2
            *badChar2='\0';
        int copyAmt2 = 0;
        
        //For each bad character, increase the shift amount so we don't have to perform any superfluous copies
        size_t recLen2 = strlen(drecords[t].sequence)+1;
        
        for(char* g=drecords[t].sequence;g<drecords[t].sequence+recLen2;g++){
            *g=toupper(*g);
            const char* badResidue2;
            if(!dna){
                //Replace unsupported symbols by replacements
                if(*g!='\0'&&(badResidue2=strchr(UNSUPPORTED_LETTERS,*g))!='\0'){
                    *g=UNSUPPORTED_LETTERS_REPLACEMENTS[badResidue2-UNSUPPORTED_LETTERS];
                }
            }
            
            const char* residueIndex2;
            //Invalid character, skip it
            if((residueIndex2=strchr(ALPHABET,*g))=='\0'){
                copyAmt2--;
                //Usually the unsupported characters should only be whitespace and newlines
                if(*g!='\n' && *g!='\r' && *g != ' '){
                    printf("Deleted unknown character %c\n",*g);
                }
            }
            
            else{
                if(*g!='\0'){
                    drecords[t].length++;
                    numSymbols2++;
                    *g=residueIndex2-ALPHABET; //Replace symbol with its alphabetic index
                }
                if(copyAmt2!=0)
                    *(g+copyAmt2)=*g;
            }
        }
    }
    
    printf("DATABASE: %s, %d symbols in %lu sequence(s).\n",options.dbFile, numSymbols2, getNumSequences());
    
    timestamp_t dbstop = get_timestamp();
    timestamp_t dbinittime = (dbstop-dbinit);
    
//--------------------------------------------------------------------
    //----------------------------------------------------------------
    // STEP 6: Copy to GPU
    //----------------------------------------------------------------
    timestamp_t seconds;
    timestamp_t cpstart = get_timestamp();
    
    //----------------------------------------------------------------
    // STEP 6.1: Copy query sequence
    //----------------------------------------------------------------
    cl_mem bufferQuery;
    bufferQuery = clCreateBuffer(
                                        context,
                                        CL_MEM_READ_ONLY,
                                        records[0].length*sizeof(cl_char),
                                        NULL,
                                        &status);
    
    status = clEnqueueWriteBuffer (
                                   cmdQueue,
                                   bufferQuery,
                                   CL_FALSE,
                                   0,
                                   records[0].length*sizeof(cl_char),
                                   records[0].sequence,
                                   0,
                                   NULL,
                                   NULL);
    
    if(status != CL_SUCCESS){
        printf("Error in step 6.1, creating buffer for Query Profile\n");
        exit(-1);
    }
    
    
    //----------------------------------------------------------------
    // STEP 6.2: Create a version of the mapMatrix addressable with the Query and
    //   Database values (integers) instead of the ALPHABET to port it to OpenCL
    //   specification. Subsequent copy to GPU global memory
    //----------------------------------------------------------------
    int* mapMatrix4cpy = (int*) malloc( NUM_AMINO_ACIDS*NUM_AMINO_ACIDS*sizeof(int) );

    for(size_t j=0;j<NUM_AMINO_ACIDS;j++){
        char d = ALPHABET[j];
        for(size_t i=0;i<NUM_AMINO_ACIDS;i++){
            char q = ALPHABET[i];
            mapMatrix4cpy[j*NUM_AMINO_ACIDS+i] = mapMatrix[q][d];
        }
    }
    
    cl_mem buffermapMatrix;
    buffermapMatrix = clCreateBuffer(
                                        context,
                                        CL_MEM_READ_ONLY,
                                        NUM_AMINO_ACIDS*NUM_AMINO_ACIDS*sizeof(cl_int),
                                        NULL,
                                        &status);
    
    status = clEnqueueWriteBuffer (
                                   cmdQueue,
                                   buffermapMatrix,
                                   CL_FALSE,
                                   0,
                                   NUM_AMINO_ACIDS*NUM_AMINO_ACIDS*sizeof(cl_int),
                                   mapMatrix4cpy,
                                   0,
                                   NULL,
                                   NULL);
    
    if(status != CL_SUCCESS){
        printf("Error in step 6.1.1, writing mapMatrix data to buffer.\n");
        exit(-1);
    }
    
    
//--------------------------------------------------------------------
    //----------------------------------------------------------------
    // STEP 6.3: Prepare Auxiliary Structure with Length of Database Sequences used
    //   inside of the kernel to have each thread (which represents a diferent database
    //   sequence) correctly access the Database structure. Copy it to GPU global memory
    //----------------------------------------------------------------
    int totalDBsize=0;
    cl_int *DBSeqLength = (int*) malloc( sizeof(cl_int) * drecords.size() );
    
    for( int i=0; i< (int)drecords.size(); i++ ){
        DBSeqLength[i] = drecords[i].length;
        totalDBsize += drecords[i].length;
    }
    
    // Using an implementation of the HeapSort algorithm to create a "lookup vector"
    // which maps the entries in the drecords[] host database to their correct position
    // (sorted by sequence length). This will be used to copy the database to GPU memory
    // while sorting it (therefore not incorring in overhead for sorting)
    int* lookup = (int*) malloc( sizeof(int)*drecords.size() );
    for( int i=0; i< (int)drecords.size(); i++ ) lookup[i]=i;
    heap_sort(DBSeqLength, lookup, drecords.size()  );
    
    // The auxiliary vector of database sequence lengths if copied (already being sorted)
    cl_mem bufferDBSeqLength;
    bufferDBSeqLength = clCreateBuffer(
                             context,
                             CL_MEM_READ_WRITE,
                             sizeof(cl_int)*drecords.size(),
                             NULL,
                             &status);
    
    status = clEnqueueWriteBuffer (
                                   cmdQueue,
                                   bufferDBSeqLength,
                                   CL_FALSE,
                                   0,
                                   sizeof(cl_int)*drecords.size(),
                                   DBSeqLength,
                                   0,
                                   NULL,
                                   NULL);
    if(status != CL_SUCCESS){
        printf("Error in step 6.2, writing Converted Database data to buffer.\n");
        exit(-1);
    }

//--------------------------------------------------------------------
    //----------------------------------------------------------------
    // STEP 6.4: Copy converted database to GPU global memory
    //----------------------------------------------------------------
    cl_mem bufferDatabase;
    
    char *auxDB4cpy = (char*) malloc( totalDBsize * sizeof(cl_char) );
    char *auxPointerDB;
    
    // Creation of the most simple database structure possible (in terms of memory) to be
    // copied to the device. This consists of a vector with all database sequences
    // sequentially placed one after the other. No termination caracters are necessary
    // as the DBSeqLength vector will be used inside the kernel to correctly address this
    // struture. By creating it as simple as possible, we hope to minimize memory transfer
    // overheads from the Host to Device. This struture is created already sorted by sequence
    // length by using the "lookup vector" that maps database sequence positions in the
    // drecords[] structure with their respective sorted position
    auxPointerDB = auxDB4cpy;
    for( int e=0; e < (int)drecords.size(); e++ ){
        for( int n=0;n< (int)drecords[ lookup[e] ].length ;n++ ){
            *auxPointerDB = drecords[ lookup[e] ].sequence[n];
            auxPointerDB+=sizeof(drecords[ lookup[e] ].sequence[n]);
        }
    }

    // Copy of database data structure to GPU global memory
    bufferDatabase = clCreateBuffer(
                                    context,
                                    CL_MEM_READ_ONLY,
                                    totalDBsize*sizeof(cl_char),
                                    NULL,
                                    &status);
    
    status = clEnqueueWriteBuffer (
                                   cmdQueue,
                                   bufferDatabase,
                                   CL_FALSE,
                                   0,
                                   totalDBsize*sizeof(cl_char),
                                   auxDB4cpy,
                                   0,
                                   NULL,
                                   NULL);
    
    if(status != CL_SUCCESS){
        printf("Error in step 6.2, writing Converted Database data to buffer.\n");
        exit(-1);
    }
    
    
//--------------------------------------------------------------------
    //----------------------------------------------------------------
    // STEP 6.4.1: Prepare Auxiliary Structure with Size of Database Sequences
    //----------------------------------------------------------------
    // Allocating auxiliary Matrix big enough to allow Fmatrix and IyMatrix por each of the db.getNumSequences kernels (each with a size of max(maxDBSeqLength               +1,query.getSequenceLength(0)+1) )
    //int matSize = MAX(maxDBSeqLength+1, queryLength + 1);
    
 /*   printf("The size of the private buffers for the big sequence and database files is: %lu\n", queryProfileLength );
    
    cl_int auxFill = 0;
    cl_mem FMatrix;
    FMatrix = clCreateBuffer(context,CL_MEM_READ_WRITE,queryProfileLength*drecords.size()*sizeof(cl_int),NULL,&status);
    
    if(status != CL_SUCCESS){
        printf("Error in step 6.3.1, allocating auxiliary buffer FMatrix \n");
        exit(-1);
    }
    
    status = clEnqueueFillBuffer (cmdQueue,FMatrix,&auxFill,sizeof(cl_int),0,queryProfileLength*drecords.size()*sizeof(cl_int),0, NULL,NULL);
  
    if(status != CL_SUCCESS){
        printf("Error in step 6.3.1, creating auxiliary buffer FMatrix \n");
        exit(-1);
    }
    
    cl_mem IxMatrix;
    IxMatrix = clCreateBuffer(context,CL_MEM_READ_WRITE,queryProfileLength*drecords.size()*sizeof(cl_int), NULL,&status);
    
    status = clEnqueueFillBuffer (cmdQueue,IxMatrix,&auxFill,sizeof(cl_int), 0, queryProfileLength*drecords.size()*sizeof(cl_int),0,NULL, NULL);
    
    if(status == CL_OUT_OF_RESOURCES) printf("Error creating auxiliary buffer IxMatrix for failure to allocate resources on the device.\n");
    if(status == CL_OUT_OF_HOST_MEMORY) printf("Error creating auxiliary buffer IxMatrix for failure to allocate resources on the host.\n");
    
    if(status != CL_SUCCESS){
        printf("Error in step 6.3.1, creating auxiliary buffer IxMatrix.\n");
        exit(-1);
    }*/
    
//--------------------------------------------------------------------  
    //----------------------------------------------------------------
    // STEP 6.5: Prepare Output array for device to host
    //----------------------------------------------------------------
    scoreType* scores;
    unsigned int  scoreArraySize = sizeof(scoreType)*drecords.size();
    scores = (scoreType*)malloc(scoreArraySize);

    cl_mem bufferScores;
    bufferScores = clCreateBuffer(   context,
                                     CL_MEM_WRITE_ONLY,
                                     scoreArraySize,
                                     NULL,
                                     &status);
    
    if(status != CL_SUCCESS){
        printf("error in step 6.3, creating buffer for bufferC\n");
        exit(-1);
    }

    timestamp_t cpend = get_timestamp();
    seconds = cpend - cpstart;
    

//---------------------------------------------------------------------
    //-----------------------------------------------------------------
    // STEP 7: Create and compile the program
    //-----------------------------------------------------------------
    timestamp_t argsstart = get_timestamp();

    // Reading the kernel file into buffer for compilation
    char const *mulFileName;
    char *mulBuffer;
    mulFileName = "kernel.cl";
    FILE *mulFile;
    mulFile = fopen(mulFileName, "r");
    if(mulFile == NULL){
        printf("\nCannot open .cl file\n");
        printf("Current path: %s\n", mulFileName);
        exit(-1);
    }
    fseek(mulFile, 0, SEEK_END);
    size_t mulSize = ftell(mulFile);
    rewind(mulFile);
    
    // Read kernel source into buffer
    mulBuffer = (char*) malloc(mulSize + 1);
    mulBuffer[mulSize] = '\0';
    fread(mulBuffer, sizeof(char), mulSize, mulFile);
    fclose(mulFile);
    
    // Creating the Program
    cl_program program = clCreateProgramWithSource(
                                                   context,
                                                   1,
                                                   (const char**) &mulBuffer,
                                                   &mulSize,
                                                   &status);

    free(mulBuffer);
    
    if(status != CL_SUCCESS){
        printf("Error in step 7 creating the program.\n");
        exit(-1);
    }
    
    // Build (compile) the program for the devices with clBuildProgram()
    const char cloptions[] = "-cl-std=CL1.2";
    status |= clBuildProgram(
                             program,
                             1,
                             &devices[device_id],
                             cloptions,
                             NULL,
                             NULL);

    
    if(status != CL_SUCCESS){
        printf("\nError in step 7 compiling the program.\n");
        //Debug Program Building Info
        /*size_t len = 0;
        cl_uint ret = clGetProgramBuildInfo(program, devices[device_id], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        char buff [len];
        ret = clGetProgramBuildInfo(program, devices[device_id], CL_PROGRAM_BUILD_LOG, len, buff, NULL);
        printf("%s", buff); fflush(stdout);*/
        exit(-1);
    }


//--------------------------------------------------------------------- 
    //-----------------------------------------------------------------
    // STEP 8: Create Kernel
    //-----------------------------------------------------------------
    cl_kernel mulKernel = NULL;
    
    // Use clCreateKernel() to create a kernel from the
    mulKernel = clCreateKernel(program, "mul_kernel", &status);
    if(status != CL_SUCCESS){
        printf("Error in step 8 creating the Kernel.\n");
        exit(-1);
    }
    

//---------------------------------------------------------------------
    //-----------------------------------------------------------------
    // STEP 9: Set Kernel Arguments
    //-----------------------------------------------------------------
    // Associate the input and output buffers with the kernel
    
    status |= clSetKernelArg(
                             mulKernel,
                             0,
                             sizeof(cl_mem),
                             &bufferDatabase);

    status |= clSetKernelArg(
                             mulKernel,
                             1,
                             sizeof(cl_mem),
                             &bufferScores);
    
    //Passing number of sequences in database
    int auxNumSeq = drecords.size();
    status |= clSetKernelArg(
                             mulKernel,
                             2,
                             sizeof(cl_uint),
                             &auxNumSeq);
    
    int gapWeight = options.gapWeight;
    status |= clSetKernelArg(
                             mulKernel,
                             3,
                             sizeof(int),
                             &gapWeight);
    
    int extendWeight = options.extendWeight;
    status |= clSetKernelArg(
                             mulKernel,
                             4,
                             sizeof(int),
                             &extendWeight);
    
    cl_int auxqueryLength = (int) getSequenceLength(0);
    status |= clSetKernelArg(
                             mulKernel,
                             5,
                             sizeof(int),
                             &auxqueryLength);
    
    status |= clSetKernelArg(
                             mulKernel,
                             6,
                             sizeof(cl_mem),
                             &bufferDBSeqLength);
    
    status |= clSetKernelArg(
                             mulKernel,
                             7,
                             sizeof(cl_mem),
                             &buffermapMatrix);

  /*  status |= clSetKernelArg(mulKernel,8,sizeof(cl_mem),&FMatrix);
    status |= clSetKernelArg(mulKernel,9,sizeof(cl_mem),&IxMatrix);*/
    
    status |= clSetKernelArg(
                             mulKernel,
                             8,
                             sizeof(cl_mem),
                             &bufferQuery);
    
    if(status != CL_SUCCESS){
        printf("Error in step 9 setting number of sequences.\n");
        exit(-1);
    }
    
//-----------------------------------------------------------------------
    //-------------------------------------------------------------------
    // EXTRA 10: Configure the Work-item structure
    //-------------------------------------------------------------------
    // The localworkgroup sized was set to 128 because this card supports a maximum of 2048
    // resident threads and 16 resident blocks per multiprocessor, and so 128 is the smallest
    // workgroupsize that utilizes all. Being the smallest allows us to manage workload
    // distribution discrepancies by minimazing the amount of threads that are prevented from
    // running when a thread in a executing block takes longer to complete
    size_t localWorkSize[1];
    localWorkSize[0] = 128;
    
    // One instance of the kernel was run for every alignment between a database sequence and
    // query sequence, so our globalworksize was defined as the number of database sequences.
    // Because we are forced to make the localworksize a multiple of the globalworksize,
    // we chose the lowest multiple of 128 that is larger than the db sequence number and
    // implemented the kernel to prevent incorrect memory accesses from the extra threads
    // created
    size_t globalWorkSize[1];
    globalWorkSize[0] = drecords.size();
    while( globalWorkSize[0]%128 != 0 ) globalWorkSize[0]++;
    
    // Printing of profiling information
    timestamp_t argsend = get_timestamp();
    timestamp_t argstime = argsend - argsstart;
    printf("\nOpenCL initTime: %llu(us)\t",clinittime);
    printf("Matrix, query and database host initTime: %llu(us)\n",dbinittime);
    printf("Host-Dev Memcpy Time: %llu(us)\t", seconds);
    printf("Program creation and kernel arg setting: %llu(us)\n", argstime);
    
//-----------------------------------------------------------------------
    //-------------------------------------------------------------------
    // STEP 11: Start the kernel
    //-------------------------------------------------------------------
    cl_event mulDone;
    
    status |= clEnqueueNDRangeKernel(
                                     cmdQueue,
                                     mulKernel,
                                     1,
                                     NULL,
                                     globalWorkSize,
                                     localWorkSize,
                                     0,
                                     NULL,
                                     &mulDone);
    
    // Make sure all requested threads ran, so to get a correct value for the kernel runtime
    status |= clFinish(cmdQueue);
    
    if(status != CL_SUCCESS){
        // Debug
        //printf("%s", get_error_string(status) ); fflush(stdout);
        clWaitForEvents (1,&mulDone);
        printf("Error in Starting or Completing the Kernel\n");
        exit(-1);
    }
  
    //Kernel Profiling
    timestamp_t comp = get_timestamp();
    timestamp_t comptime = comp - argsend;
    printf("Kernel Compute Time: %llu(us)\t", comptime );

    
//------------------------------------------------------------------------
    //--------------------------------------------------------------------
    // STEP 12: Copy results back to host
    //--------------------------------------------------------------------
    clEnqueueReadBuffer(
                        cmdQueue,
                        bufferScores,
                        CL_TRUE,
                        0,
                        scoreArraySize,
                        scores,
                        1,
                        &mulDone,
                        NULL);
    
    if(status != CL_SUCCESS){
        printf("Error in reading back resulting scores.\n");
        exit(-1);
    }

    timestamp_t results = get_timestamp();
    timestamp_t restime = results - comp;
    printf("Result write back Time: %llu(us)\n\n", restime );
    
    
//-------------------------------------------------------------------------
    //---------------------------------------------------------------------
    // Sort scores and print results
    //---------------------------------------------------------------------
    //Scores after reconverting to host indexes
    scoreType* correctScores = (scoreType*) malloc(scoreArraySize);
    for( int e=0; e < (int)drecords.size(); e++ ) correctScores[ lookup[e] ] = scores[ e ];
    
    //puts("\nSorting results...");
    // Creating vector for the sorted results
    std::vector<Result> sortScores;
    sortScores.resize(getNumSequences());
    for(size_t i=0;i<sortScores.size();i++){
        sortScores[i].index = i;
        sortScores[i].score = correctScores[i];
    }
    free(scores);
    std::sort(sortScores.begin(),sortScores.end(),&resultComparisonFunc);
 
    
    //Printing relevant metrics
    //printf("\nUsing %zu as globalWorkSize and %zu as localWorkSize for a database with %lu sequences\n",globalWorkSize[0], localWorkSize[0], drecords.size());
    int queryLength = (int) getSequenceLength(0);
    double numCells = queryLength*(double)numSymbols2;
    
    float GCUPS = numCells/comptime/1000;
    printf("Number of Cells: %0.f\tCompute time: %llu (usec)\tGCUPS: %f\n\n", numCells,comptime, GCUPS);
    
    //Display results
    puts("Results:");
    for(size_t i=0;i < (size_t) std::min(20, (int)getNumSequences());i++){
        printf("%3ld. %-50.50s\t SCORE: %d\n",i,drecords[sortScores[i].index].description,sortScores[i].score);
    }

//---------------------------------------------------------------------------
    //-----------------------------------------------------------------------
    // STEP 12: Free OpenCL and C buffers
    //-----------------------------------------------------------------------
    // Free OpenCL resources
    clReleaseKernel(mulKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseContext(context);
    
    clReleaseMemObject(bufferQuery);
    clReleaseMemObject(buffermapMatrix);
    clReleaseMemObject(bufferDatabase);
    clReleaseMemObject(bufferDBSeqLength);
    clReleaseMemObject(bufferScores);
    
    
    // Free up memory
    free(lookup);
    free(devices);
    free(platforms);
    free(auxDB4cpy);
    free(DBSeqLength);
    free(correctScores);
    free(mapMatrix4cpy);
    delete[] buffer;
    delete[] buffer2;
    delete[] queryProfile;
    delete[] sequenceOffsets;
    delete[] descriptionBuffer;

    return EXIT_SUCCESS;
}


// printf("%s:%d\n",__FILE__, __LINE__); fflush(stdout);

//-----------------------------------------------------
// EXTRA
//-----------------------------------------------------
// Implements the Heapsort algorithm to eficiently sort the database sequences by length.
// Given the size of the database a algorithm of complexity (in worst case scenario) of
// O(nlog(n)) and best case scenario O(n) was required to make sorting the db feasable
void  max_heapify(int a[], int lookup[], int i, int heapsize)
{
    int tmp, largest;
    int l = (2 * i) + 1;
    int r = (2 * i) + 2;
    if ((l <= heapsize) && (a[l] > a[i]))
        largest = l;
    else
        largest = i;
    if ((r <= heapsize) && (a[r] > a[largest]))
        largest = r ;
    if (largest != i)
    {
        tmp = a[i];
        a[i] = a[largest];
        a[largest] = tmp;
        tmp = lookup[i];
        lookup[i] = lookup[largest];
        lookup[largest] = tmp;
        max_heapify(a, lookup ,largest, heapsize);
    }
    
}



void  build_max_heap(int a[], int lookup[], int heapsize)
{
    int i;
    for (i = heapsize/2; i >= 0; i--)
    {
        max_heapify(a, lookup, i, heapsize);
    }
    
}



void heap_sort(int a[], int lookup[], int heapsize)
{
    int i, tmp;
    build_max_heap(a, lookup, heapsize);
    for (i = heapsize; i > 0; i--)
    {
        tmp = a[i];
        a[i] = a[0];
        a[0] = tmp;
        tmp = lookup[i];
        lookup[i] = lookup[0];
        lookup[0] = tmp;
        heapsize--;
        max_heapify(a, lookup, 0, heapsize);
    }
}



static bool resultComparisonFunc(Result r1, Result r2){
    return (r1.score>r2.score);
}


static timestamp_t get_timestamp(){
    struct timeval now;
    gettimeofday (&now, NULL);
    return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}


bool writeToFile(size_t sequenceNum){
    if(strcmp(descriptions[sequenceNum],PADDING_SEQ_NAME) == 0)
        return true;
    outFile << '>' << descriptions[sequenceNum] << std::endl;

    size_t offset = sequenceOffsets[sequenceNum];
    
    seqType* ptr = sequences+offset;
            
    while(*ptr!=SUB_SEQUENCE_TERMINATOR&&*ptr!=SEQUENCE_GROUP_TERMINATOR)
    {
        for(size_t i=0;i<SUBBLOCK_SIZE;i++)
        {
            seqType val = ptr[i];
            if(val== ' ') //Subgroup padding
                break;
            outFile << AMINO_ACIDS[val];            
        }
        ptr+=BLOCK_SIZE*SUBBLOCK_SIZE;
    }
    outFile << '\n';
    return true;
}

// Returns the length of the sequences of number "sequenceNum" FROM THE DATABASE
size_t getSequenceLength(size_t sequenceNum){
    if(sequenceNum >= records.size())
        return 0;

    return records[sequenceNum].length;
}

char* getSequence(size_t sequenceNum){
    if(sequenceNum >= records.size())
        return NULL;
    
    return records[sequenceNum].sequence;

}

size_t getNumSequences(){
    return drecords.size();
}

size_t getDBSizeInBytes(){
    return blobSize;
}

size_t getNumSymbols(){
    return metadata.numSymbols;
}

size_t getNumBlocks(){
    return metadata.numBlocks;
}

const char* getDescription(unsigned int index){
    if(index>=descriptions.size())
        return NULL;
    return descriptions[index];
}

const char *get_error_string(cl_int error){
    switch (error) {
        // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

/*
switch(status){
     case CL_SUCCESS:                            printf("Success!");break;
     case CL_DEVICE_NOT_FOUND:                   printf("Device not found.");break;
     case CL_DEVICE_NOT_AVAILABLE:               printf("Device not available");break;
     case CL_COMPILER_NOT_AVAILABLE:             printf("Compiler not available");break;
     case CL_MEM_OBJECT_ALLOCATION_FAILURE:      printf("Memory object allocation failure");break;
     case CL_OUT_OF_RESOURCES:                   printf("Out of resources");break;
     case CL_OUT_OF_HOST_MEMORY:                 printf("Out of host memory");break;
     case CL_PROFILING_INFO_NOT_AVAILABLE:       printf("Profiling information not available");break;
     case CL_MEM_COPY_OVERLAP:                   printf("Memory copy overlap");break;
     case CL_IMAGE_FORMAT_MISMATCH:              printf("Image format mismatch");break;
     case CL_IMAGE_FORMAT_NOT_SUPPORTED:         printf("Image format not supported");break;
     case CL_BUILD_PROGRAM_FAILURE:              printf("Program build failure");break;
     case CL_MAP_FAILURE:                        printf("Map failure");break;
     case CL_INVALID_VALUE:                     printf("Invalid value");break;
     case CL_INVALID_DEVICE_TYPE:                printf("Invalid device type");break;
     case CL_INVALID_PLATFORM:                   printf("Invalid platform");break;
     case CL_INVALID_DEVICE:                     printf("Invalid device");break;
     case CL_INVALID_CONTEXT:                    printf("Invalid context");break;
     case CL_INVALID_QUEUE_PROPERTIES:           printf("Invalid queue properties");break;
     case CL_INVALID_COMMAND_QUEUE:              printf("Invalid command queue");break;
     case CL_INVALID_HOST_PTR:                   printf("Invalid host pointer");break;
     case CL_INVALID_MEM_OBJECT:                 printf("Invalid memory object");break;
     case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    printf("Invalid image format descriptor");break;
     case CL_INVALID_IMAGE_SIZE:                 printf("Invalid image size");break;
     case CL_INVALID_SAMPLER:                    printf("Invalid sampler");break;
     case CL_INVALID_BINARY:                     printf("Invalid binary");break;
     case CL_INVALID_BUILD_OPTIONS:              printf("Invalid build options");break;
     case CL_INVALID_PROGRAM:                    printf("Invalid program");break;
     case CL_INVALID_PROGRAM_EXECUTABLE:         printf("Invalid program executable");break;
     case CL_INVALID_KERNEL_NAME:                printf("Invalid kernel name");break;
     case CL_INVALID_KERNEL_DEFINITION:          printf("Invalid kernel definition");break;
     case CL_INVALID_KERNEL:                     printf("Invalid kernel");break;
     case CL_INVALID_ARG_INDEX:                  printf("Invalid argument index");break;
     case CL_INVALID_ARG_VALUE:                  printf("Invalid argument value");break;
     case CL_INVALID_ARG_SIZE:                   printf("Invalid argument size");break;
     case CL_INVALID_KERNEL_ARGS:                printf("Invalid kernel arguments");break;
     case CL_INVALID_WORK_DIMENSION:             printf("Invalid work dimension");break;
     case CL_INVALID_WORK_GROUP_SIZE:            printf("Invalid work group size");break;
     case CL_INVALID_WORK_ITEM_SIZE:             printf("Invalid work item size");break;
     case CL_INVALID_GLOBAL_OFFSET:              printf("Invalid global offset");break;
     case CL_INVALID_EVENT_WAIT_LIST:            printf("Invalid event wait list");break;
     case CL_INVALID_EVENT:                      printf("Invalid event");break;
     case CL_INVALID_OPERATION:                  printf("Invalid operation");break;
     case CL_INVALID_GL_OBJECT:                  printf("Invalid OpenGL object");break;
     case CL_INVALID_BUFFER_SIZE:                printf("Invalid buffer size");break;
     case CL_INVALID_MIP_LEVEL:                  printf("Invalid mip-map level");break;
 }
*/
