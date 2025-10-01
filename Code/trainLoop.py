
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as random
import jax.tree_util as tree
from polynomial import *
from structureFunctions import *
from trainingFunctions import *




from collections import deque

def train_loop(x,state, rng_key, #input_Output_data has shape (max_steps, nInp), where each element is a mass input paired with the target output
               lr=1e-10, noise_scale=1e-5, momentum=0.9,
               lr_decay=1e-2, check_every=10, grad_dir_buffer=20,
               refineDotThresh=0.9, refineNormThresh=1e-2,
               refinementThresh = 500,
               loss_threshold=1e-4, consecutive=10, max_steps=5000, 
               save_path="saved_structure.pkl"):
    """Train the structure parameters with noisy momentum SGD and optional refinement.

    Args:
        state: Initial structure state PyTree to optimise.
        rng_key: Base PRNG key for stochastic components.
        lr: Initial learning rate applied before refinement.
        noise_scale: Magnitude of Gaussian noise added to gradients.
        momentum: Momentum coefficient for the velocity buffer.
        lr_decay: Multiplier applied to `lr` when refinement begins.
        check_every: Step interval for assessing refinement criteria.
        grad_dir_buffer: Length of gradient history stored for direction checks.
        refineDotThresh: Cosine similarity threshold that signals refinement.
        refineNormThresh: Gradient norm threshold that signals refinement.
        refinementThresh: Loss threshold used to gauge readiness for refinement.
        loss_threshold: Target loss for determining convergence.
        consecutive: Number of successive low-loss steps required to stop training.
        max_steps: Maximum number of optimisation steps to execute.
        save_path: Filesystem path used when persisting the trained state.

    Returns:
        tuple: `(state, loss_history)` containing the final state and list of loss values.
    """
    
    currentVelocity = tree.tree_map(jnp.zeros_like, state)
    loss_history = []
    refinement_started = False
    grad_history = deque(maxlen=grad_dir_buffer) #note: Deque is a rolling list that has the cool property of (once max size is reached), popping the earliest added element, when a new element is added. This allows us to only look at the most recent gradients to determine whether refinement is required or not.
    stable_steps = 0
    refineSteps = 0
    

    try:

        for step in range(max_steps):
            
            inputs, targets = generate_polynomial_dataset(state['inputPositions'].shape[0], x-1, (-10, 10), step)
            
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
    """Serialise the provided structure state to disk using pickle."""
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(state), f)

def load_state(path="saved_structure.pkl"):
    """Load a previously serialised structure state from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)



if __name__ == "__main__":
    print("Hello")
    
