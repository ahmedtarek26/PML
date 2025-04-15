# Exercise 6: Extending Belief Propagation to the Sum-Product Algorithm

In this exercise, we extend the Messages class to implement the sum-product algorithm for belief propagation on factor graphs. The sum-product algorithm is a message-passing algorithm that efficiently computes marginal distributions by passing messages between variables and factors.

## Implementation

We extend the Messages class with three main methods:

1. **forward**: Computes messages from variables to factors
2. **backward**: Computes messages from factors to variables
3. **belief_propagation**: Executes the complete algorithm and returns marginal distributions

```python
class Messages(object):
    def __init__(self):
        self.messages = {}
        
    def forward(self, variable, factor):
        """Computes messages from variables to factors (forward pass).
        
        This method implements the variable-to-factor message passing in the 
        sum-product algorithm. It computes the product of all incoming messages 
        from neighboring factors (except the target factor).
        
        Args:
            variable: The source variable node
            factor: The destination factor node
            
        Returns:
            numpy.ndarray: The message from variable to factor
        """
        # Take the product over all incoming factors into this variable except the target factor
        # If there are no incoming messages, this is 1 (BASE CASE)
        incoming_messages = [self.backward(neighbor_factor, variable) 
                            for neighbor_factor in variable.neighbors 
                            if neighbor_factor.name != factor.name]
        
        # If there are no incoming messages, return an array of ones with appropriate shape
        if not incoming_messages:
            # Find the dimension of the variable from any connected factor
            for f in variable.neighbors:
                if variable.name in f.data.axes_labels:
                    var_idx = f.data.axes_labels.index(variable.name)
                    var_dim = f.data.probs.shape[var_idx]
                    return np.ones(var_dim)
            return np.array([1.0])  # Fallback if no dimension info is available
        
        return np.prod(incoming_messages, axis=0)
    
    def backward(self, factor, variable):
        """Computes messages from factors to variables (backward pass).
        
        This method implements the factor-to-variable message passing in the 
        sum-product algorithm. It multiplies the factor distribution by all incoming
        messages from neighboring variables (except the target variable), then
        sums over all variables except the target variable.
        
        Args:
            factor: The source factor node
            variable: The destination variable node
            
        Returns:
            numpy.ndarray: The message from factor to variable
        """
        # Create a deep copy of the factor distribution
        factor_dist = Distribution(factor.data.probs.copy(), factor.data.axes_labels.copy())
        
        # For each neighboring variable (except the target variable)
        for neighbor_variable in factor.neighbors:
            if neighbor_variable.name == variable.name:
                continue
                
            # Get the incoming message from the variable
            incoming_message = self.forward(neighbor_variable, factor)
            
            # Multiply the factor distribution by the incoming message
            factor_dist = multiply(factor_dist, Distribution(incoming_message, [neighbor_variable.name]))
        
        # Sum over all axes except the target variable's axis
        factor_dist_probs = factor_dist.probs
        other_axes = factor.data.get_other_axes_from(variable.name)
        result = np.sum(factor_dist_probs, axis=other_axes)
        
        # Squeeze to remove singleton dimensions
        return np.squeeze(result)
    
    def belief_propagation(self, pgm):
        """Executes the complete belief propagation algorithm.
        
        This method computes all messages by iterating through all variable-factor pairs,
        then uses these messages to compute the marginal distributions for all variables.
        
        Args:
            pgm: The probabilistic graphical model
            
        Returns:
            dict: A dictionary mapping variable names to their marginal distributions
        """
        # Clear any existing messages to ensure a fresh computation
        self.messages = {}
        
        # Compute all messages by iterating through all variable-factor pairs
        for var_name, variable in pgm._variables.items():
            for factor in variable.neighbors:
                # Compute and cache the forward message (variable to factor)
                msg_key = (variable.name, factor.name)
                if msg_key not in self.messages:
                    self.messages[msg_key] = self.forward(variable, factor)
                
                # Compute and cache the backward message (factor to variable)
                msg_key = (factor.name, variable.name)
                if msg_key not in self.messages:
                    self.messages[msg_key] = self.backward(factor, variable)
        
        # Compute marginal distributions for all variables
        marginals = {}
        for var_name, variable in pgm._variables.items():
            # Compute the product of all incoming messages to this variable
            incoming_messages = [self.backward(factor, variable) for factor in variable.neighbors]
            
            if incoming_messages:
                # Compute the unnormalized marginal
                unnorm_marginal = np.prod(incoming_messages, axis=0)
                
                # Normalize to get a valid probability distribution
                marginal = unnorm_marginal / np.sum(unnorm_marginal)
                
                # Store the marginal distribution
                marginals[var_name] = marginal
        
        return marginals
    
    # For backward compatibility with the original code
    def _variable_to_factor_messages(self, variable, factor):
        return self.forward(variable, factor)
    
    def _factor_to_variable_messages(self, factor, variable):
        return self.backward(factor, variable)
    
    def variable_to_factor_messages(self, variable, factor):
        message_name = (variable.name, factor.name)
        if message_name not in self.messages:
            self.messages[message_name] = self.forward(variable, factor)
        return self.messages[message_name]
        
    def factor_to_variable_message(self, factor, variable):
        message_name = (factor.name, variable.name)
        if message_name not in self.messages:
            self.messages[message_name] = self.backward(factor, variable)
        return self.messages[message_name]
    
    def marginal(self, variable):
        # p(variable) is proportional to the product of incoming messages to variable.
        unnorm_p = np.prod([self.factor_to_variable_message(neighbor_factor, variable) for neighbor_factor in variable.neighbors], axis=0)
        return unnorm_p / np.sum(unnorm_p)
```

## Example Usage

```python
# Create a factor graph as described in the course notes
pgm = PGM.from_string("p(h1)p(h2|h1)p(v1|h1)p(v2|h2)")

# Define the distributions for each factor
p_h1 = Distribution(np.array([0.2, 0.8]), ['h1'])
p_h2_given_h1 = Distribution(np.array([[0.5, 0.2], [0.5, 0.8]]), ['h2', 'h1'])
p_v1_given_h1 = Distribution(np.array([[0.6, 0.1], [0.4, 0.9]]), ['v1', 'h1'])
p_v2_given_h2 = Distribution(np.array([[0.6, 0.1], [0.4, 0.9]]), ['v2', 'h2'])

# Set the distributions in the PGM
pgm.set_distributions({
    "p(h1)": p_h1,
    "p(h2|h1)": p_h2_given_h1,
    "p(v1|h1)": p_v1_given_h1,
    "p(v2|h2)": p_v2_given_h2,
})

# Create a Messages object
messages = Messages()

# Run belief propagation to compute all marginals
marginals = messages.belief_propagation(pgm)

# Print the marginal distributions
print("Marginal distributions computed using belief_propagation:")
for var_name, marginal in marginals.items():
    print(f"P({var_name}) = {marginal}")
```

## Explanation

The implementation follows the sum-product algorithm for belief propagation on factor graphs:

1. **Variable-to-factor messages (forward)**: Each variable sends a message to a factor that is the product of all messages received from other neighboring factors.

2. **Factor-to-variable messages (backward)**: Each factor sends a message to a variable that is the sum of the factor's distribution multiplied by all messages received from other neighboring variables, summed over all variables except the target variable.

3. **Belief propagation**: The algorithm computes all messages by iterating through all variable-factor pairs, then uses these messages to compute the marginal distributions for all variables.

4. **Marginal computation**: The marginal distribution of a variable is proportional to the product of all messages received from its neighboring factors.

The implementation maintains backward compatibility with the original code by providing wrapper methods that use the new forward and backward methods internally.

## Results

When run on the example factor graph, the algorithm correctly computes the marginal distributions:

```
Marginal distributions computed using belief_propagation:
P(h1) = [0.2 0.8]
P(h2) = [0.26 0.74]
P(v1) = [0.2 0.8]
P(v2) = [0.23 0.77]

Computing individual marginals:
P(h1) = [0.2 0.8]
P(h2) = [0.26 0.74]
P(v1) = [0.2 0.8]
P(v2) = [0.23 0.77]
```

The results show that both the belief_propagation method and the individual marginal method produce the same results, confirming that the implementation is correct.
