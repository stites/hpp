## Machine Problem Questions
##### 1 .How many floating operations are being performed in your vector add kernel? explain.
`inputLength` + 1 floating point operations. `inputLength` for each operation to add vectors, plus one to get the byte length of the vectors in:
int byteLen = inputLength * sizeof(float);

##### 2 .How many global memory reads are being performed by your kernel? explain.
3 reads are performed via the three `cudaMemcpy` calls.

##### 3 .How many global memory writes are being performed by your kernel? explain.
4 writes occur on the invocations of `malloc` and `cudaMalloc`s.

##### 4 .Describe what possible optimizations can be implemented to your kernel to achieve a performance speedup.
Given vectors of the appropriately small size, we can perform vector addition in Shared memory - as opposed to Global memory.

##### 5 .Name three applications of vector addition.
Analysis of SQL databases. Analytics on BAM files (from bioinformatics). Work with fluid dynamics. Sections of Logistic Regression.
