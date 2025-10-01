import json
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as random
import jax.tree_util as tree
from polynomial import *





#key = jax.random.PRNGKey(seed)
def initStructure(nInp,nImm,nParam,D,X,key): #note: X is the mass vec size
    """Initialise a structure state with randomised tensors for points and parameters.

    Args:
        nInp: Number of input points in the structure.
        nImm: Number of immoveable points exerting forces on inputs.
        nParam: Number of parameter points defining transformation tensors.
        D: Dimensionality of the spatial domain.
        X: Size of each mass vector associated with a point.
        key: PRNG key used to generate reproducible random parameters.

    Returns:
        dict: Mapping of structure components (positions, velocities, tensors, etc.).
    """
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
    #everything needs to be wrapped as a jnp.array()
    return{
        'inputPositions': jax.random.uniform(k1, (nInp, D))*0.1, #(nInp,D)
        'inputVelocities': jax.random.normal(k2, (nInp, D)) * 0.01, #(nInp,D)
        #'inputMasses': jax.random.normal(k3, (nInp, X)), #(nInp, X)
        'immoveablePositions': jax.random.uniform(k4, (nImm, D)), #(nImm,D)
        'immoveableMasses': jax.random.normal(k5, (nImm, X)), #(nImm,X)
        'T': jax.random.normal(k6, (nParam, D, X, X))*0.1, #(nParam, D, X, X)
        'b': jax.random.normal(k6, (nParam, X))*0.1, #(nParam, D, X) #bias term for T application
        'parameterPos': jax.random.uniform(k7, (nParam, D)), #(nParam, D)
        'kValues': jax.random.normal(key, (nInp, nImm + nInp, X)) * 0.1, #(nInp, nImm + nInp, X)
        #'outputList': jnp.zeros((nInp, X)),  # start from zero 
        'outputLocations': jax.random.normal(k3, (nInp, D)), #(nInp,D)
        'outputVars': jnp.ones((nInp,))*10, #1D array of len nInp
        'boundaries': jnp.ones((D,)) * 1.0,
        'frequency': jnp.array(100.0),
        #'iterations': 50.0,
        'boundarySharpness': jnp.ones((D,))*0.5, #(D)
        'maxV' : jnp.array(4.0)


    }




def apply_boundary(state):
    """Apply differentiable boundary reflections to input velocities.

    Args:
        state: Structure state dictionary containing positions, velocities, boundaries,
            and sharpness parameters.

    Returns:
        dict: Updated state with boundary-adjusted `inputVelocities`.
    """
    inpPos = state['inputPositions'] #(nInp,D)
    inpV = state['inputVelocities'] #(nInp,D)
    boundaries = state['boundaries'] #(D,)
    sharpness = state['boundarySharpness'] #(scalar)
    def perInput(pos,vel):
        def perDim(d):
            wall_force = (1+jax.nn.tanh((pos[d] - boundaries[d]) * sharpness[d]))/2#tanh wall instead of non-diff wall
            flip_factor = 1.0 - 2.0 * wall_force
            return vel[d] * flip_factor
        
        return(jax.vmap(perDim)(jnp.arange(boundaries.shape[0])))

    newV = jax.vmap(perInput)(inpPos,inpV)
    
    return state | {'inputVelocities' : newV} #(nInp,D)

def move(state):
    """Advance input positions using stored velocities and global frequency.

    Args:
        state: Structure state dictionary containing `inputPositions`, `inputVelocities`,
            and `frequency`.

    Returns:
        dict: Updated state with advanced `inputPositions`.
    """
    inpPos = state['inputPositions'] #(nInp,D)
    inpV = state['inputVelocities'] #(nInp,D)
    newPos = inpPos + (1 / (state['frequency']**2 + 1e-2)) * inpV
    return state | {'inputPositions': newPos} #(nInp,D)
            
#Not used
def stateCreator(inputPointList,immoveableList,parameterPointList,kValues,outputLocations,outputVars,boundaries,frequency,sharpness,maxV):
    """Convert object-based inputs into the tensor-backed structure state format.

    Args:
        inputPointList: Iterable of input point objects exposing `position` and `velocity`.
        immoveableList: Iterable of immoveable point objects with `position` and `mass`.
        parameterPointList: Iterable providing `T_tensor` and `position` per parameter point.
        kValues: Precomputed interaction coefficients with shape `(nInp, nInp + nImm, X)`.
        outputLocations: Array of target output positions per input point.
        outputVars: Variance scalars controlling output falloff.
        boundaries: Soft boundary extents for each dimension.
        frequency: Scalar simulation frequency.
        sharpness: Boundary sharpness controls for each dimension.
        maxV: Maximum velocity scale used elsewhere in simulation.

    Returns:
        dict: Structure state dictionary mirroring `initStructure` output.
    """
    return {
        #shape is listed on the side
        #in the future, shapes may be mutable using T's that arent square 
        'inputPositions' : jnp.stack([input.position for input in inputPointList]), #(nInp,D)
        'inputVelocities': jnp.stack([input.velocity for input in inputPointList]), #(n,D)
        #'inputMasses' : jnp.stack([input.mass for input in inputPointList]), #(n,X)
        'immoveablePositions' : jnp.stack([immov.position for immov in immoveableList]), #(m,D)
        'immoveableMasses' : jnp.stack([immov.mass for immov in immoveableList]), #(m,X)
        'T': jnp.stack(param.T_tensor for param in parameterPointList), #(l, D, X, X) #note: T are dim-wise matrices applied to the mass of each input point
        'parameterPos' : jnp.stack(param.position for param in parameterPointList), #(l,D)
        'kValues' : kValues, # (n,n+m,X) kValues are now a rank-3 tensor, where for each input-mass pair, there is a list of coefficients, used in the triple dot product between inpM,immM or inpM and kVec
        #'outputList' : outputList, # (n,D) (Note: this is a list of the final masses)
        'outputLocations' : outputLocations, #(n,D)
        'outputVars' : outputVars, #(n)
        'boundaries' : boundaries, #(D)
        'frequency' : frequency, #scalar
        #'iterations' : iterations, #scalar
        'boundarySharpness' : sharpness, #scalar
        'maxV' : maxV #scalar
    }
       

def applyT(state,inputMasses):# (nInput, X)
    """Apply the parametric transformation field to each input mass vector.

    Args:
        state: Structure state providing tensors `T`, biases `b`, `inputPositions`, and
            `parameterPos`.
        inputMasses: Array of shape `(nInp, X)` containing current input mass vectors.

    Returns:
        jnp.ndarray: Transformed masses with the same shape as `inputMasses`.
    """
    T = state["T"]                      # (nParam, D, X, X)
    inputPositions = state["inputPositions"]  # (nInput, D)
    paramPos = state["parameterPos"]          # (nParam, D)
    b = state["b"]                      # (nParam, X)
       
    # Addressing the shape of T: (nParam, D, X, X), from left to right (where T is applied to each input) (and where (X,X) is the shape of the matrix applied)
    def perInput(i_pos, i_mass):
        def perParam(p_idx):
            p_pos = paramPos[p_idx]           # (D,)
            def perDim(d):
                dist = (1/(1+(p_pos[d] - i_pos[d])**2))  # scalar
                return dist * (T[p_idx, d] @ i_mass) # (X,)
                # T[p_idx, d] has shape (X, X), i_mass has shape (X,)
                # The matrix multiplication results in shape (X,)
                # The sigmoid is applied element-wise, resulting in shape (X,)

            transformed = jax.vmap(perDim)(jnp.arange(T.shape[1]))  # (D, X)
            return jnp.sum(transformed, axis=0) + b[p_idx]  # (X,)

        all_params = jax.vmap(perParam)(jnp.arange(T.shape[0]))  # (nParam, X)
        return jnp.sum(all_params, axis=0)  # (X,)

    updated_masses = jax.vmap(perInput)(inputPositions, inputMasses)  # (nInput, X)

    return updated_masses #NOW RETURNS UPDATED MASSES

    

    

def applyG(state, inputMasses):
    """Update input velocities via learned pairwise interactions.

    Args:
        state: Structure state containing positions, velocities, immoveable masses, and
            interaction coefficients `kValues`.
        inputMasses: Array of shape `(nInp, X)` representing current input masses.

    Returns:
        dict: Updated state with new `inputVelocities` reflecting interaction forces.
    """
    
    kValues = state['kValues'] #(nInp, nInp+nImm, X)
    inpPos = state['inputPositions'] #(nInp,D)
    immovPos = state['immoveablePositions'] #(nImm,D)
    immovM = state['immoveableMasses'] #(nImm,X)
    inpM = inputMasses #(nImp,X)
    inpV = state['inputVelocities'] #(nInp,D)
    allPos = jnp.concatenate([immovPos, inpPos], axis=0)   # shape (M + N, D) (Note that for allPos and allM, its immovs, THEN inputs)
    allM = jnp.concatenate([immovM, inpM], axis=0)      # shape (M + N, X)

    def perPair(inpInd,targetInd): #per input-mass pair
        distance = jnp.sum((inpPos[inpInd] - allPos[targetInd]) ** 2)
        direction = inpPos[inpInd] - allPos[targetInd] # shape (D,)
        strength = (1/(1+(distance**2))) * jnp.sum(kValues[inpInd,targetInd] * inpM[inpInd] * allM[targetInd]) #Triple dot product has valid dimensions; Shape: (X)*(X)*(X)
        
        is_self = (targetInd == inpInd + immovPos.shape[0])
        return jnp.where(is_self, 0.0, strength) * direction #returns 0 strength for itself (Shape: D)
        

    def perInput(inpInd):
        dvs = jax.vmap(lambda targetInd: perPair(inpInd, targetInd))(jnp.arange(allM.shape[0])) #arange in this instance gives list from 0 to len(allM-1). This allows a vmap of perPair applied to each mass point in the structure.
        totalDV = jnp.sum(dvs) #sum all elements in vmap
        return inpV[inpInd] + totalDV #Shape: D
    
    updated_inpV = jax.vmap(perInput)(jnp.arange(inpV.shape[0])) #return vmap with elements being the new velocities

    
    return state | {'inputVelocities' : updated_inpV} #Shape: (nInp,D)

    

def checkOutput(state,inputMasses,outputList): 
    """Aggregate each input's mass into its output accumulator based on distance falloff.

    Args:
        state: Structure state providing positions, output locations, and variance scalars.
        inputMasses: Array of transformed masses `(nInp, X)`.
        outputList: Running array of accumulated outputs `(nInp, X)`.

    Returns:
        jnp.ndarray: Updated output accumulator aligned with `outputList`.
    """
    
    inpPos = state["inputPositions"] #(nInp,D)
    inpM = inputMasses #(nInp,X)
    outLoc = state["outputLocations"] #(nInp,D)
    outVar = state["outputVars"] #(nInp)
    outList = outputList #(nInp,X)
    def perInput(inpInd):
        distance = (jnp.sum((inpPos[inpInd]-outLoc[inpInd])**2)) #distance-squared
        return outList[inpInd] + inpM[inpInd] * (1 / (1 + (outVar[inpInd] * distance**2))) #Shape: (X). Allows cumulative (large variance) or approx. one-time (low variance) output
    updated_outList = jax.vmap(perInput)(jnp.arange(inpM.shape[0])) #Shape: (nInp, X)
    return updated_outList #NOW RETURNS UPDATED OUTPUTLIST ONLY


        

def StructureFrame(state,inputMasses,outputList):
    """Perform a single simulation frame of interactions, motion, and output updates.

    Args:
        state: Structure state dictionary in the format produced by `initStructure`.
        inputMasses: Mass tensor `(nInp, X)` to transform during the frame.
        outputList: Accumulator `(nInp, X)` for output mass collections.

    Returns:
        tuple: `(state, inputMasses, outputList)` after one frame of updates.
    """
    #Frequency will be specified to how long one run through of runStructure actually takes
    state = applyG(state,inputMasses)
    state = move(state)
    inputMasses = applyT(state,inputMasses)
    state = apply_boundary(state)
    outputList = checkOutput(state,inputMasses,outputList)
    return state,inputMasses,outputList
    #ensure these are assigned properly

def runStructure(state,inputMasses,outputList):
    """Advance the structure for a fixed number of frames using a JAX loop primitive.

    Args:
        state: Initial structure state to evolve.
        inputMasses: Initial mass tensor `(nInp, X)` for the simulated inputs.
        outputList: Initial output accumulator `(nInp, X)`.

    Returns:
        tuple: Final `(state, inputMasses, outputList)` after the configured number of frames.
    """
    ITERATIONS = 100.0 #number of frames
    def body_fn(i, carry):
        s, iM, oL = carry
        return StructureFrame(s, iM, oL)
    
    init_carry = (state, inputMasses, outputList)
    return lax.fori_loop(0, 100, body_fn, init_carry) #acts as a jnp-compatible for loop through the frames. Args: (lowerbound,upperbound,function,functionArg) #Note: lax forLoop can only have one function arg
