We model a continuous transformation field; a parameterized operator attached to every point in a vector space (linear map + activation). This operator field acts on input point-sets, enabling the network’s structure to morph with the input.

Specifications:

The model can be described through interactions between the following class of objects. While object-oriented in conceptual design, the implementation uses purely tensors and JAX functions in order to avoid the increase in memory and processing time produced by the overhead of objects and classes, and furthermore to allow for parallel computation through GPU threads:

  1) Input Points: holds a 'mass vector' (a given input vector), position vector, and velocity vector. Its mass vector is transformed via the transformation field each frame (see function ApplyT), and furthermore is used within the transformation of its own velocity vector (as well as other velocity vectors) through the function 'ApplyG'. Each element of the position vector relative to the position of 'parameter points' (see below) scales its associated matrix (dimension-wise) within the rank-3 tensor of each parameter, before the sum of scaled matrices, is applied to the mass vector. Additionally, the difference in positions between the input and parameter point is parsed through 

This current model is only a small sub-component of the overall model aimed to be built. The key idea is to create a modular system of composed, generalised neural networks that will act as a buil
