
import json
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as random
import jax.tree_util as tree
from polynomial import *



#not used
class InputPoints:
    def __init__(self,mass,position,velocity):
        self.mass = mass
        self.position = position
        self.velocity = velocity

class Immoveables:
    def __init__(self, mass, position):
        self.mass = mass
        self.position = position

class ParameterPoints:
    def __init__(self, tList, position):
        self.tList = tList #list of matrices that are scaled by each dimension upon application
        self.position = position
        self.T_tensor = jnp.stack(tList)





def apply_boundary(state):
    inpPos = state['inputPositions'] #(nInp,D)
    inpV = state['inputVelocities'] #(nInp,D)
    boundaries = state['boundaries'] #(D,)
    sharpness = state['boundarySharpness'] #(scalar)
    def perInput(pos,vel):
        def perDim(d):
            wall_force = (1+jax.nn.tanh((pos[d] - boundaries[d]) * sharpness[d]))/2#sigmoid wall instead of non-diff wall
            flip_factor = 1.0 - 2.0 * wall_force
            return vel[d] * flip_factor
        
        return(jax.vmap(perDim)(jnp.arange(boundaries.shape[0])))

    newV = jax.vmap(perInput)(inpPos,inpV)
    
    return state | {'inputVelocities' : newV} #(nInp,D)

def move(state):
    inpPos = state['inputPositions'] #(nInp,D)
    inpV = state['inputVelocities'] #(nInp,D)
    newPos = inpPos + (1 / (state['frequency']**2 + 1e-2)) * inpV
    return state | {'inputPositions': newPos} #(nInp,D)
            
#Not used
def stateCreator(inputPointList,immoveableList,parameterPointList,kValues,outputLocations,outputVars,boundaries,frequency,sharpness,maxV):
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
    T = state["T"]                      # (nParam, D, X, X)
    inputPositions = state["inputPositions"]  # (nInput, D)
    paramPos = state["parameterPos"]          # (nParam, D)
       
    # Addressing the shape of T: (nParam, D, X, X), from left to right (where T is applied to each input) (and where (X,X) is the shape of the matrix applied)
    def perInput(i_pos, i_mass):
        def perParam(p_idx):
            p_pos = paramPos[p_idx]           # (D,)
            def perDim(d):
                dist = (1/(1+(p_pos[d] - i_pos[d])**2))  # scalar
                return dist * (T[p_idx, d] @ i_mass)           # (X,)
            transformed = jax.vmap(perDim)(jnp.arange(T.shape[1]))  # (D, X)
            return jnp.sum(transformed, axis=0)  # (X,)

        all_params = jax.vmap(perParam)(jnp.arange(T.shape[0]))  # (nParam, X)
        return jnp.sum(all_params, axis=0)  # (X,)

    updated_masses = jax.vmap(perInput)(inputPositions, inputMasses)  # (nInput, X)

    return updated_masses #NOW RETURNS UPDATED MASSES

    

    

def applyG(state, inputMasses):
    
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
    #Gaussian during training (variance is slowly decreased as location parameters converge to the right set of values).
    #For a given simulation, a preset positions are already chosen
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
    #Frequency will be specified to how long one run through of runStructure actually takes
    state = applyG(state,inputMasses)
    state = move(state)
    inputMasses = applyT(state,inputMasses)
    state = apply_boundary(state)
    outputList = checkOutput(state,inputMasses,outputList)
    return state,inputMasses,outputList
    #ensure these are assigned properly

def runStructure(state,inputMasses,outputList):
    ITERATIONS = 100.0 #number of frames
    def body_fn(i, carry):
        s, iM, oL = carry
        return StructureFrame(s, iM, oL)
    
    init_carry = (state, inputMasses, outputList)
    return lax.fori_loop(0, 100, body_fn, init_carry) #acts as a jnp-compatible for loop through the frames. Args: (lowerbound,upperbound,function,functionArg) #Note: lax forLoop can only have one function arg

def lossFunction(outputList,trueOutputs):

    


    def perOutput(n):
        return(jnp.sum(outputList[n]-trueOutputs[n])**2) 
    
    return jnp.sum(jax.vmap(perOutput)(jnp.arange(outputList.shape[0]))) #sums the square of loss for each input-output pair
    
@jax.jit
def run_and_loss(state, inputMasses, outputList, true_outputs):
#    softState = state | { #makes positions and velocity, as well as sharpness, bounded.
#    'inputPositions': jax.nn.sigmoid(state['inputPositions'])*state['boundaries'], #(nInp,D)
#    'parameterPos': jax.nn.sigmoid(state['parameterPos'])*state['boundaries'],
#    'immoveablePositions': jax.nn.sigmoid(state['immoveablePositions'])*state['boundaries'],
#    'boundarySharpness': jax.nn.sigmoid(state['boundarySharpness']),
#    'outputLocations': jax.nn.sigmoid(state['outputLocations'])*state['boundaries'],
#    'inputVelocities': jax.nn.sigmoid(state['inputVelocities'])*state['maxV'],
#    }

    from jax.tree_util import tree_flatten_with_path
    from jax.tree_util import DictKey


    #This applies the abs() to each leaf. This pushes values to 0, as d/dx(abs(x)) = sign(x)
    def l1_regularization(tree, λ, allowed_keys=('T', 'kValues', 'immoveableMasses')):
        flat_info, _ = tree_flatten_with_path(tree)
    
        total_reg = 0.0
        for path, leaf in flat_info:
            # Check if this leaf is keyed by one of the allowed dictionary keys
            for node in path:
                if isinstance(node, DictKey) and node.key in allowed_keys:
                    total_reg += jnp.sum(jnp.abs(leaf))
                    break  # Avoid double-counting if multiple matches in path
        return λ * total_reg



    final_state, final_inputMasses, final_outputList = runStructure(state, inputMasses, outputList)
    return lossFunction(final_outputList, true_outputs) + l1_regularization(state, λ=1e-2)
   

def normalize_grads(grads, max_norm=1.0): #only on condition that grads has gradient larger than 1
        flat, tree_def = jax.tree_util.tree_flatten(grads)
        total_norm = jnp.sqrt(sum([jnp.sum(x**2) for x in flat]))
        scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-8))
        flat_scaled = [g * scale for g in flat]
        return jax.tree_util.tree_unflatten(tree_def, flat_scaled)

def gradDescentStep_andRefinementCheck(state, inputMasses, outputList, true_outputs, lr, subkey, noise_scale, currentVelocity, momentum, grad_history, step, grad_dir_buffer, check_every, refineDotThresh, refineNormThresh, lr_decay,refinement_started,refinementThresh):
    grads = jax.grad(run_and_loss, argnums=0)(state, inputMasses, outputList, true_outputs)

    

    #used to take certain values to 0
    def filter_tree(tree, include_keys):
        flat, treedef = jax.tree_util.tree_flatten(tree)
        flat_filtered = [v if k in include_keys else jnp.zeros_like(v) for k, v in zip(tree.keys(), flat)]
        return jax.tree_util.tree_unflatten(treedef, flat_filtered)



    
    
    
    
    if step == 0:
        grads_flat, _ = tree.tree_flatten(normalize_grads(grads))
        for i, g in enumerate(grads_flat):
            jax.debug.print("GRAD {} → max: {}, min: {}, has_nan: {}", i, jnp.max(g), jnp.min(g), jnp.isnan(g).any())


    

    #used to update each parameter
    def updateStateWithNoiseAndMomentum(param,grad,key,newVel): #key in this case refers to a seed that gives a specifc, random generation
        newparam = param + newVel
        #print("param:", type(param), param.shape)
        #print("grad:", type(grad), grad.shape)
        
        #print("newparam:", newparam.shape)
        
        return newparam 


    def updateVelWithNoiseAndMomentum(param,grad,key,currentVelocity):
        noise = noise_scale * random.normal(key, shape=grad.shape) #shape is dim of normally chosen array
        newVel = currentVelocity*jnp.tanh(step/20)*momentum - lr * (grad + noise) #using jnp.tanh() allows momentum to start initially low, which prohibits system from overshooting with momentum, at the initial steps where variance in descents may be high due to complexity of structure
        #print("currentVelocity:", type(currentVelocity), currentVelocity.shape)
        #print("newVel:", newVel.shape)
        return newVel


    def checkRefinement(refinement_started, grad_history, lr, noise_scale):
        # Track flattened grad direction
        flat_grads, _ = tree.tree_flatten(grads) #flattens tree into leaves
        grad_vec = jnp.concatenate([g.ravel() for g in flat_grads]) #This takes each array g in flat_grads, flattens it into a 1D vector using .ravel(), then concatenates all of them into one big 1D vector.
        grad_norm = jnp.linalg.norm(grad_vec) #finds the gradient vector size, of the whole PyTree: grads
        grad_history.append(grad_vec / (grad_norm + 1e-4)) #appends the unit vec to grad_history

        

        if not refinement_started and grad_norm < refineNormThresh:
        # Decide on refinement 
            if not refinement_started and step > grad_dir_buffer and step % check_every == 0:
                avg_dot = jnp.mean(jnp.array([jnp.dot(grad_vec, g) for g in grad_history]))
                if avg_dot > refineDotThresh and grad_norm < refineNormThresh:
                    refinement_started = True
                    lr *= lr_decay
                    noise_scale = 0.0
                    
                    print(f"🔍 Refinement triggered at step {step} (avg_dot={avg_dot:.4f}, grad_norm={grad_norm:.6f})")

        return noise_scale, grad_history, refinement_started, lr


    #These lines produce a copy of the state PyTree, and produce RNG keys for each array in the tree  
    flat_keys = random.split(subkey, len(jax.tree_util.tree_leaves(grads))) #flattens it into all the arrays, assigns keys to each array
    keys_tree = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(grads), flat_keys) #unflattens the PyTree

    new_velocity = jax.tree_util.tree_map(updateVelWithNoiseAndMomentum, state, grads, keys_tree,currentVelocity) #creates a tree for the velocity, with gradient descent applied
    new_state = jax.tree_util.tree_map(updateStateWithNoiseAndMomentum, state, grads, keys_tree,new_velocity) #creates a tree for the state, with gradient descent applied



##CHECKING STRUCTURE AND NUMBER OF LEAVES
    #from jax.tree_util import tree_flatten_with_path

    #print("\n🔎 Flattened tree structure:")
    #flat_info, _ = tree_flatten_with_path(updated_tree)

    #tuple_count = 0
    #for path, leaf in flat_info:
    #    print(f"{path}: {type(leaf)} shape={getattr(leaf, 'shape', 'scalar')}")
    #    if isinstance(leaf, tuple):
    #        print("⚠️ Tuple leaf found:", leaf)
    #        tuple_count += 1

    #print(f"\n✅ Total tuple leaves: {tuple_count}")
    #print(f"📦 Total leaves: {len(flat_info)}")


 ##CHECKING NUMBER OF PARAMVEL PAIRS FROM CUSTOM CLASS
    #flat_info, _ = jax.tree_util.tree_flatten_with_path(new_velocity)

    #tuple_count = 0
    #for path, leaf in flat_info:
    #    print(f"{path}: {type(leaf)}")
        


                                                                         
    noise_scale, grad_history, refinement_started, lr = checkRefinement(refinement_started, grad_history, lr, noise_scale)

    return new_state, new_velocity, noise_scale, grad_history, refinement_started, lr, momentum


#note that * in the zip, turns the output from ((param1,vel1),...,(paramN,velN)) to (param1,...,paramN),(vel1,...velN)


#key = jax.random.PRNGKey(seed)
def initStructure(nInp,nImm,nParam,D,X,key): #note: X is the mass vec size
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
    #everything needs to be wrapped as a jnp.array()
    return{
        'inputPositions': jax.random.uniform(k1, (nInp, D))*0.1, #(nInp,D)
        'inputVelocities': jax.random.normal(k2, (nInp, D)) * 0.01, #(nInp,D)
        #'inputMasses': jax.random.normal(k3, (nInp, X)), #(nInp, X)
        'immoveablePositions': jax.random.uniform(k4, (nImm, D)), #(nImm,D)
        'immoveableMasses': jax.random.normal(k5, (nImm, X)), #(nImm,X)
        'T': jax.random.normal(k6, (nParam, D, X, X))*0.1, #(nParam, D, X, X)
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












from collections import deque

def train_loop(state, rng_key, #input_Output_data has shape (max_steps, nInp), where each element is a mass input paired with the target output
               lr=1e-10, noise_scale=1e-5, momentum=0.9,
               lr_decay=1e-2, check_every=10, grad_dir_buffer=20,
               refineDotThresh=0.9, refineNormThresh=1e-2,
               refinementThresh = 500,
               loss_threshold=1e-4, consecutive=10, max_steps=5000, 
               save_path="saved_structure.pkl"):
    
    currentVelocity = tree.tree_map(jnp.zeros_like, state)
    loss_history = []
    refinement_started = False
    grad_history = deque(maxlen=grad_dir_buffer) #note: Deque is a rolling list that has the cool property of (once max size is reached), popping the earliest added element, when a new element is added. This allows us to only look at the most recent gradients to determine whether refinement is required or not.
    stable_steps = 0
    refineSteps = 0
    

    try:

        for step in range(max_steps):
            
            inputs, targets = generate_polynomial_dataset(10, 20, (-10, 10), step)
            
            inputMasses = inputs
            true_outputs = targets

            rng_key, subkey = random.split(rng_key) 

            outputList = jnp.zeros_like(inputMasses)

            from jax.tree_util import tree_flatten_with_path

            if step == 0:
                testState = runStructure(state, inputMasses, outputList)

                print("🔎 STRUCTURE AFTER RUNNING")
                flat_info, _ = tree_flatten_with_path(testState)

                for path, leaf in flat_info:
                    print(f"{path}: max={jnp.max(leaf):.5f}, min={jnp.min(leaf):.5f}, has_nan={jnp.isnan(leaf).any()}")



            # Apply update, and check for refinement
            new_state, new_Velocity, noise_scale, grad_history, refinement_started, lr, momentum = gradDescentStep_andRefinementCheck(
                state, inputMasses, outputList, true_outputs, lr, subkey, noise_scale, currentVelocity, momentum, grad_history, step, grad_dir_buffer, check_every, refineDotThresh, refineNormThresh, lr_decay, refinement_started, refinementThresh
            )
            state = normalize_grads(new_state)
            currentVelocity = new_Velocity
            
            # Track loss
            loss = run_and_loss(state, inputMasses, outputList, true_outputs).block_until_ready()
            loss_history.append(loss)

            if loss <= refinementThresh and refinement_started == False:
                refineSteps += 1
            else:
                refineSteps = 0

            #if refineSteps>=20 and refinement_started == False:
            #    lr *= lr_decay
            #    noise_scale = 0.0
            #    momentum = 0.5
            #    refinement_started = True


            if loss <= loss_threshold and refinement_started == True:
                stable_steps += 1
            else:
                stable_steps = 0

            if step % 50 == 0:
                print(f"[Step {step}] Loss = {loss:.6f} | Refinement = {refinement_started} | Stable = {stable_steps}/{consecutive}")

            if stable_steps >= consecutive:
                print(f"✅ Converged for {consecutive} steps. Saving model...")
                print(f"💾 Model saved to {save_path}")
                save_state(state, save_path)
                break
        
    
    except KeyboardInterrupt:

        print("🛑 Training interrupted — saving model...")
        print(f"💾 Model saved to {save_path}")
        save_state(state, save_path)
        # metadata = {
        #     'step': step,
        #     'final_loss': float(loss),
        #     'refinement_started': refinement_started,
        #     'loss_history': [float(l) for l in loss_history],
        # }
        # with open("train_log.json", "w") as f:
        #     json.dump(metadata, f)
    

    
    return state, loss_history









import pickle
import os

def save_state(state, path="saved_structure.pkl"):
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(state), f)

def load_state(path="saved_structure.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)



if __name__ == "__main__":
    print("Hello")
    
