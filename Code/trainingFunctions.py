import json
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as random
import jax.tree_util as tree
from structureFunctions import *
from polynomial import *




def lossFunction(outputList,trueOutputs):
    """Compute mean-squared error between predicted and true output masses.

    Args:
        outputList: Array `(nInp, X)` with accumulated outputs from the structure.
        trueOutputs: Target output array with the same shape as `outputList`.

    Returns:
        jnp.ndarray: Scalar loss measuring squared differences per input summed globally.
    """

    #Mean Squared Error between outputList and trueOutputs
    #Both have shape (nInp,X)
    #We sum the square of the differences for each input-output pair, then sum these values
    #This gives a single scalar value for the loss
    #This is then used in the training loop to guide gradient descent


    def perOutput(n):
        return(jnp.sum(outputList[n]-trueOutputs[n])**2) 
    
    return jnp.sum(jax.vmap(perOutput)(jnp.arange(outputList.shape[0]))) #sums the square of loss for each input-output pair
    




@jax.jit
def run_and_loss(state, inputMasses, outputList, true_outputs):
    """Run the structure forward pass and evaluate regularised loss.

    Args:
        state: Structure state PyTree supplying parameters and simulation buffers.
        inputMasses: Input mass tensor `(nInp, X)` for the current batch.
        outputList: Output accumulator initialised for the run.
        true_outputs: Target outputs to compare against the simulated results.

    Returns:
        jnp.ndarray: Scalar loss combining task error and L1 regularisation.
    """

    #This runs the structure, and then computes the loss
    #It also applies L1 regularization to certain parameters in the structure, to push them to 0 if not needed


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
    """Rescale a gradient PyTree to enforce an ℓ2 norm ceiling.

    Args:
        grads: Gradient PyTree matching the structure state.
        max_norm: Maximum allowed ℓ2 norm for the flattened gradient vector.

    Returns:
        PyTree: Gradients scaled proportionally when the norm exceeds `max_norm`.
    """
    flat, tree_def = jax.tree_util.tree_flatten(grads)
    total_norm = jnp.sqrt(sum([jnp.sum(x**2) for x in flat]))
    scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-8))
    flat_scaled = [g * scale for g in flat]
    return jax.tree_util.tree_unflatten(tree_def, flat_scaled)






def gradDescentStep_andRefinementCheck(state, inputMasses, outputList, true_outputs, lr, subkey, noise_scale, currentVelocity, momentum, grad_history, step, grad_dir_buffer, check_every, refineDotThresh, refineNormThresh, lr_decay,refinement_started,refinementThresh):
    """Apply one noisy momentum update and detect when to enter refinement phase.

    Args:
        state: Current structure state PyTree.
        inputMasses: Mass tensor `(nInp, X)` for the current batch.
        outputList: Output accumulator aligned with `inputMasses`.
        true_outputs: Target outputs to drive training.
        lr: Learning rate applied to gradient updates.
        subkey: PRNG key used for noise injection.
        noise_scale: Standard deviation for gradient noise.
        currentVelocity: Momentum buffer PyTree matching `state`.
        momentum: Scalar momentum coefficient.
        grad_history: Deque tracking recent gradient directions for refinement checks.
        step: Current optimisation step index.
        grad_dir_buffer: Number of recent gradients maintained in the history.
        check_every: Interval of steps between refinement evaluations.
        refineDotThresh: Cosine similarity threshold triggering refinement.
        refineNormThresh: Gradient norm threshold triggering refinement.
        lr_decay: Factor for shrinking the learning rate when refining.
        refinement_started: Boolean flag indicating refinement mode.
        refinementThresh: Loss threshold used to infer refinement readiness.

    Returns:
        tuple: Updated `(state, velocity, noise_scale, grad_history, refinement_started,
            lr, momentum)` reflecting the optimisation step.
    """
    grads = jax.grad(run_and_loss, argnums=0)(state, inputMasses, outputList, true_outputs)
    #This function applies one step of gradient descent to the structure parameters, using the gradients computed from run_and_loss
    #It also checks if refinement should be started, based on the gradient history and norms
    #If refinement is started, it reduces the learning rate and noise scale to allow finer adjustments
    
    
    
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
