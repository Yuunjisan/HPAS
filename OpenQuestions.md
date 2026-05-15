## Question 2:
t-SNE is a non-convex optimization problem,  each run will look different from the last. Starting
with random coordinates means it takes longer to converge and is more prone to crowding or
creating embeddings where clusters that are globally similar in high dimensions end up far apart
in 2D.
Whereas PCA is deterministic, the points will always start in the same positions. It places points
based on the largest variances in the data. This helps t-SNE maintain the relationship between
distant clusters better than random starts. Starting with a reasonably organized layout allows the
gradient descent to reach a stable state in fewer iterations.
These impacts can be observed by toggling the initialization in the framework.

Some Alternate initializations include:
Spectral Initialization: The  Laplacian of the similarity graph is used to place points. It is
particularly useful when the underlying structure needs to be captured.
Multi-dimensional scaling: It is similar to PCA, but it is particularly used when pair-wise
distances are mentioned and need to be preserved.

Another method would be to add a random noise to PCA, which prevents the optimization from
being done only in local minimas. Although it has the same issue that t-SNE does.



## Question 3:


PCA can be used for dimensionality reduction in various scenarios.
Noise reduction: High-dimensional datasets like MNIST contain noise. This can complicate and
confuse distance calculations, and in these cases, PCA acts as a noise filter.

t-SNE requires the p-matrix, which is the similarity matrix. Calculating Euclidean distances in
high dimensions is significantly slower than calculating distances in lower dimensions.

In extremely high dimensions, the distance between the nearest and farthest neighbors
becomes very similar, making it harder for t-SNE to define clear neighborhoods.  PCA helps
alleviate this by identifying the direction in which the data varies the most.
By reducing the data to its most significant dimensions, PCA makes the high-dimensional
similarities more robust and computationally efficient to calculate, which is essential for a
high-performance system.




# Open Questions

## 5. FFT Benchmark Results

The benchmark compares the recursive and iterative FFT implementations across
increasing input sizes. Lower values are better.

|     n | Recursive FFT ms | Iterative FFT ms |
| ----: | ---------------: | ---------------: |
|    16 |         0.016915 |         0.001625 |
|    32 |         0.033780 |         0.003485 |
|    64 |         0.071175 |         0.008170 |
|   128 |         0.154730 |         0.018120 |
|   256 |         0.300195 |         0.040170 |
|   512 |         0.628615 |         0.090575 |
|  1024 |         1.323940 |         0.199580 |
|  2048 |         2.276840 |         0.284830 |
|  4096 |         3.151090 |         0.629885 |
|  8192 |         6.366990 |         1.294360 |
| 16384 |        13.192600 |         2.784160 |
| 32768 |        25.923900 |         5.892080 |
| 65536 |        52.150000 |        12.701500 |

### Observations

To empirically determine the asymptotic runtime of the recursive and iterative
FFT implementations, we benchmarked both algorithms on one-dimensional
real-valued input arrays of increasing power-of-two sizes and we observe how the implementations behave as the input becomes very large. The tested sizes
ranged from `n = 16` to `n = 65536`. For each size, a random input vector was
generated and both FFT implementations were run multiple times. 

The reported values are the mean runtime per FFT call over 20 repetitions, measured using the x64-Release configuration.

The results show that both implementations scale consistently with the expected
`O(n log n)` complexity. When the input size doubles,
the runtime increases by slightly more than a factor of two for larger inputs,
which matches the additional logarithmic factor in `n log n`.

The iterative FFT was consistently faster than the recursive FFT. For example,
for `n = 1024`, the recursive FFT took approximately `1.32 ms`, while the
iterative FFT took approximately `0.20 ms`. For `n = 65536`, the recursive FFT
took approximately `52.15 ms`, while the iterative FFT took approximately
`12.70 ms`. Thus, at the largest tested input size, the iterative implementation
was about `4.1` times faster.

This difference is expected. The recursive implementation repeatedly splits the
input into even and odd arrays, creating many temporary arrays and pilling up recursive calls overhead. The iterative implementation first applies
bit-reversal ordering and then performs butterfly operations over blocks of
increasing size. This avoids much of the recursion and allocation overhead and
therefore has a smaller practical constant factor. Both algorithms are theoretically `O(n log n)`, but the iterative implementation has a smaller 
constant factor, making it more efficient for large inputs.
