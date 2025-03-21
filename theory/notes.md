1. How to Escape Saddle Points Efficiently

- **Learnings**
  - For a fuction that is l-gradient lipschitz, it will well known that gradient descent finds an e-first-order stationary point within l(f(x_0) - f\*)/e^2 iterations.
  - You can still find local minimums by adding random perturbations or randomly initializing gradient descent, but the number of steps needed to reach any local minimum is not bounded.
- **Key Terms**

  - **saddle points** : stationary point where the gradient is zero but the hessian has both positive and negative eigenvalues.
    - If the hessian is positive semidefinite, it is a candiate for a local minima. This means that the function is not curving downwards in any direction.

- **Questions**
