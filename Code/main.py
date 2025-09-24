import json
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as random
import jax.tree_util as tree
from polynomial import *
from collections import deque
from trainLoop import *
import matplotlib.pyplot as plt



if __name__ == '__main__':
   key = jax.random.PRNGKey(42)
   
   nInp = 10
   nImm = 2
   nParam = 2
   d = 3
   x = 21
   

   state = initStructure(nInp,nImm,nParam,d,x,key)
   s_flat, _ = tree.tree_flatten(state)
   print("INITIAL STRUCTURE")
   for g in s_flat:
      print("state max:", jnp.max(g), "min:", jnp.min(g), "has nan:", jnp.isnan(g).any())

   #For this one; the inputs and true_outputs are created within the training loop 

   state,loss_history = train_loop(state,key)

   # Plot the values
   plt.plot(jnp.log(jnp.array(loss_history)))
   plt.title("List of loss Values")
   plt.xlabel("Step")
   plt.ylabel("log of Value")
   plt.grid(True)
   plt.show()

   

