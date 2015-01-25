### Machine Problem Questions
##### 1 .What is the compute capability of the NVIDIA Fermi architecture?
1536 threads

##### 2 .What are the maximum block dimensions for GPUs with 3.0 compute capability?
1024 x 1024 x 64

##### 3 .Suppose you are launching a one dimensional grid and block. If the hardware's maximum grid dimension is 65535 and the maximum block dimension is 512, what is the maximum number threads can be launched on the GPU?
dimgrid = 65535; dimblock= 512; one dimension;
dimgrid/dimblock == 127.998046875;

so less than the max grid size. if we break the block into warps, we can squeeze some more juice out of this:
(512*127) + (15/16)*512 == 65504

65504 threads? A little stuck on this one due to the overly-simple example (compared to what we've seen so far).
Otherwise 65024 threads without the extra warps. Actually that I'll stick with 65024 since we can't "squeeze out" a partial, physical block. Reasoning my way out of this, it's terrible.

##### 4 .Under what conditions might a programmer choose not want to launch the maximum number of threads?
I can think of two conditions: if they need more threads in a block than will fit into the grid evenly, as in Question 3. Or if the resources delegated per-thread are depleted before the program can be fully utilized - in which case it may be smarter to have less threads with higher resource delegation.

##### 5 .What can limit a program from launching the maximum number of threads on a GPU?
The answers to Question 4, primarily if the resources delegated per-thread are depleted before the program fully delegates the maximum number of threads.

##### 6 .What is shared memory?
An on-chip, read-write memory which can be accessed at high speeds and with high parallelism. It's allocated by thread block making it an efficient way for threads to share input/intermediate/final data on a per-block basis.

##### 7 .What is global memory?
Synonymous to the memory referenced in the von-Neumann model. This is off-chip, meaning that it has long access latency and has low-access in comparison to shared memory or registers, and is accessible on a per-grid level.

##### 8 .What is constant memory?
Constant memory is memory which is read-only per-grid, only usable by device functions — a function that can only be called by a kernel or another device function. We don't know too much about constant memory just yet, however if it's read-only for device functions, I'd take a stab that it is read-write for kernel functions.

##### 9 .What does warp size signify on a GPU?
A warp is a scheduling unit for a streaming multiprocessor. The size of a warp (32-threads) signifies a group of threads that will be run in a SIMD-executed style (SIMD: single instruction, multiple data).

##### 10 .Is double precision supported on GPUs with 1.3 compute capability?
Yes. According to nVidia's FAQ ( https://developer.nvidia.com/cuda-faq ) — not sure if it was covered in lecture.
