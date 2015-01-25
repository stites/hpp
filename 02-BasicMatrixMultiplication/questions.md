## Machine Problem Questions
##### 1 .How many floating operations are being performed in your matrix multiply kernel? explain.
the equivalent number as (numARows * numAColumns) + (numBRows * numBColumns) +
(numCRows * numCColumns) + 3 for size of float calculations.

##### 2 .How many global memory reads are being performed by your kernel?  explain.
(numARows * numAColumns) + (numBRows * numBColumns) - one for each declaration of space.
Then all of the Memcpys - so plus 3.

##### 3 .How many global memory writes are being performed by your kernel?  explain.
(numCRows * numCColumns) to move everything from device to space on disk, plus the writes needed
for overhead in declaring memory for all host variables.

##### 4 .Describe what possible optimizations can be implemented to your kernel to achieve a performance speedup.
Tiling.

##### 5 .Name three applications of matrix multiplication.
Working with adjacency matrices, generating Laplacian matricies, determining pagerank using power iteration.