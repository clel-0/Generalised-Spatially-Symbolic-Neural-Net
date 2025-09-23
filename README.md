We model a continuous transformation field; a parameterized operator attached to every point in a vector space (linear map + activation). This operator field acts on input point-sets, enabling the network’s structure to morph with the input.

Summarised Description:

The model can be described through interactions between the following class of objects. While object-oriented in conceptual design, the implementation uses purely tensors and JAX functions in order to avoid the increase in memory and processing time produced by the overhead of objects and classes, and furthermore to allow for parallel computation through GPU threads:

  1) Input Points: holds a 'mass vector' (a given input vector), position vector, and velocity vector. Its mass vector is transformed via the transformation field each frame (see function ApplyT), and furthermore is used within the transformation of its own velocity vector (as well as other velocity vectors) through the function 'ApplyG'. Each element of the position vector relative to the position of 'parameter points' (see below) scales its associated matrix (dimension-wise) within the rank-3 tensor of each parameter, before the sum of scaled matrices, is applied to the mass vector, through the function 'ApplyT'. The velocity vector is used to update the position vector of the input point, each frame, through the function 'Move' (in terms of a neural net, this essentially takes the mass vector to its next layer, ready to be transformed via 'ApplyT').

  2) Parameter Points: holds a rank-3 transformation tensor that is used in the transformation of the mass vecs of each input point within the space via 'ApplyT', and holds a position vector. Parameter points do not move (i.e. their position vector remains constant). This promotes the emergence of a non-volatile transformation field that over-parameterises (an already existent weakness of this symbolic neural net that I am to reduce through refinement).

  3) Immoveable Points: holds a mass and position vector, aimed to guide points through the transformation field via the 'ApplyG' function. K-values are used to further scale the change in a given input's velocity vector, allowing each immoveable point to apply 'gravity' (change in velocity) of differing strengths to each input point, allowing for emergent positional encoding. As the name suggests, the position vector of an immoveable point remains constant.

  4) Output Locations: holds a set of output locations specific to each input point, where after each frame, each scaled input mass is added to its respective output vector, where the scale to the input mass is determined by the inverse of the squared norm of the distance between the input point and output location. A variance factor is also applied to the scaling, with variance being unique to each output location. This allows for each input point to take on a range of emergent tasks, from a one-off, discrete output to a cumulative production of outputs. (Done through the function 'CheckOutput'). 

  5) Boundaries: smooth boundaries for each dimension localises the environment, causing input points to actively interact with each other, using tanh to produce moderate, differentiable changes in the position of the input points, through the function 'ApplyBoundary'.


This current model is only a small sub-component of the overall model aimed to be built. The key idea is to create a modular system of composed, generalised neural networks that will act as a buil
