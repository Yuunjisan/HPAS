

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
