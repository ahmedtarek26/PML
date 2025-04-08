# Exact inference with Belief Propagation

This notebook is inspired from Jessica Stringham's work

We are going to perform inference through the sum-product message passing, or belief propagation, on tree-like factor graphs (without any loop). We work only with discrete distributions and without using ad-hoc libraries, to better understand the algorithm.

```python
import numpy as np
```

## Probability distributions

First of all, we need to represent a discrete probability distribution and check that it is normalized. For example, we can represent a discrete conditional distribution $p(v_1|h_1)$ with a 2D array, as:

```python
class Distribution():
    """
    Discrete probability distributions, expressed using labeled arrays
    probs: array of probability values
    axes_labels: list of axes names
    """
    def __init__(self, probs, axes_labels):
        self.probs = probs
        self.axes_labels = axes_labels
        
    def get_axes(self):
        #returns a dictionary with axes names and the corresponding coordinates
        return {name: axis for axis, name in enumerate(self.axes_labels)}
    
    def get_other_axes_from(self, axis_label):
        #returns a tuple containing all the axes except from axis_label
        return tuple(axis for axis, name in enumerate(self.axes_labels) if name != axis_label)
    
    def is_valid_conditional(self, variable_name):
        #variable_name is the name of the variable for which we are computing the distribution.
        return np.all(np.isclose(np.sum(self.probs, axis=self.get_axes()[variable_name]), 1.0))
    
    def is_valid_joint(self):
        return np.all(np.isclose(np.sum(self.probs), 1.0))
```

```python
#Let's see the previous distribution:

p_v1_given_h1 = Distribution(np.array([[0.4, 0.8, 0.9], [0.6, 0.2, 0.1]]), ['v1', 'h1'])

print('Is p(v1|h1) a valid conditional distribution? ', p_v1_given_h1.is_valid_conditional('v1'))
print('Is p(v1|h1) a valid joint distribution? ', p_v1_given_h1.is_valid_joint())

#Consider also a joint distribution and a conditional distribution with more than one 'given' variable

p_h1 = Distribution(np.array([0.6, 0.3, 0.1]), ['h1'])

print('Is p(h1) a valid conditional distribution? ', p_h1.is_valid_conditional('h1'))
print('Is p(h1) a valid joint distribution? ', p_h1.is_valid_joint())

p_v1_given_h0_h1 = Distribution(np.array([[[0.9, 0.2, 0.7], [0.3, 0.2, 0.5]], [[0.1, 0.8, 0.3], [0.7, 0.8, 0.5]]]), ['v1', 'h0', 'h1'])
print('Is p(v1|h1, h2) a valid conditional distribution? ', p_v1_given_h0_h1.is_valid_conditional('v1'))
print('Is p(v1|h1, h2) a valid joint distribution? ', p_v1_given_h0_h1.is_valid_joint())
```

We need to allow multiplications between distributions like $p(v_1|h_1, \ldots, h_n)p(h_i)$, where $p(h_i)$ is a 1D array. To do it, we can exploit broadcasting. But first, we need to reshape $p(h_i)$ accordingly to the dimension $h_i$ of the distribution $p(v_1|h_1, \ldots, h_n)$

```python
def multiply(p_v_given_h, p_hi):
    '''
    Compute the product of the distributions p(v|h1,...,hn)p(hi) where p(hi) is a 1D array
    '''
    #Get the axis corresponding to hi in the conditional distribution
    axis=p_v_given_h.get_axes()[next(iter(p_hi.get_axes()))]
    
    # Reshape p(hi) in order to exploit broadcasting. Consider also the case in which p(hi) is a 1D array
    dims = np.ones_like(p_v_given_h.probs.shape)
    dims[axis] = p_v_given_h.probs.shape[axis]
    
    if (p_hi.probs.shape != ()):
        reshaped_p_hi = p_hi.probs.reshape(dims)
    else:
        reshaped_p_hi = p_hi.probs
        
    return Distribution(p_v_given_h.probs*reshaped_p_hi, p_v_given_h.axes_labels)
```

```python
p_v1_h1 = multiply(p_v1_given_h1, p_h1)
print(p_v1_h1.probs)
print(p_v1_h1.is_valid_joint())

p_v1_h1_given_h0 = multiply(p_v1_given_h0_h1, p_h1)
print(p_v1_h1_given_h0.probs)

[[0.24 0.24 0.09]
 [0.36 0.06 0.01]]
True
[[[0.54 0.06 0.07]
  [0.18 0.06 0.05]]

 [[0.06 0.24 0.03]
  [0.42 0.24 0.05]]]

# We can try to build the following factor graph:
```

![factor_graph.png](https://github.com/r-doz/PML2025/blob/main/imgs/factor_graph.png?raw=true)

```python
p_h1 = Distribution(np.array([[0.2], [0.8]]), ['h1'])
p_h2_given_h1 = Distribution(np.array([[[0.5, 0.2], [0.5, 0.8]]], ['h2', 'h1'])
p_v1_given_h1 = Distribution(np.array([[0.6, 0.1], [0.4, 0.9]]), ['v1', 'h1'])
p_v2_given_h2 = Distribution(p_v1_given_h1.probs, ['v2', 'h2'])

pgm = PGM.from_string("p(h1)p(h2|h1)p(v1|h1)p(v2|h2)")

pgm.set_distributions({
    "p(h1)": p_h1,
    "p(h2|h1)": p_h2_given_h1,
    "p(v1|h1)": p_v1_given_h1,
    "p(v2|h2)": p_v2_given_h2
})
```

## Factor graphs

Factor graphs are bipartite graphs, with variable nodes and factor nodes. Edges can only connect nodes of different type. Consider for example:

```python
class Node(object):
    def __init__(self, name):
        self.name = name
        self.neighbors = []
        
    def is_valid_neighbor(self, neighbor):
        raise NotImplemented()
        
    def add_neighbor(self, neighbor):
        assert self.is_valid_neighbor(neighbor)
        self.neighbors.append(neighbor)
        
class Variable(Node):
    def is_valid_neighbor(self, factor):
        return isinstance(factor, Factor)  # Variables can only neighbor Factors
        
class Factor(Node):
    def is_valid_neighbor(self, variable):
        return isinstance(variable, Variable)  # Factors can only neighbor Variables
    
    def __init__(self, name):
        super(Factor, self).__init__(name)
        self.data = None
```

We can build some parsing methods in order to create a factor graph from a string representing the factorization of the joint probability distribution

```python
from collections import namedtuple

ParsedTerm = namedtuple('ParsedTerm', [
    'term',
    'var_name',
    'given',
])

def _parse_term(term):
    # Given a term like (a|b,c), returns a list of variables
    # and conditioned-on variables
    assert term[0] == '(' and term[-1] == ')'
    term_variables = term[1:-1]
    
    # Handle conditionals
    if '|' in term_variables:
        var, given = term_variables.split('|')
        given = given.split(',')
    else:
        var = term_variables
        given = []
        
    return ParsedTerm(term, var, given)

def parse_model_string(model_string):
    # Given a model string like "p(a)p(b|a)p(c|a,b)", returns a list of ParsedTerms
    terms = []
    
    # Find all the p(...) terms
    i = 0
    while i < len(model_string):
        if model_string[i] == 'p':
            # Find the matching closing paren
            assert model_string[i+1] == '('
            j = i + 2
            depth = 1
            while depth > 0:
                if model_string[j] == '(':
                    depth += 1
                elif model_string[j] == ')':
                    depth -= 1
                j += 1
                
            term = model_string[i+1:j]
            terms.append(_parse_term(term))
            i = j
        else:
            i += 1
            
    return terms

def parse_model_into_variables_and_factors(model_string):
    # Given a model string like "p(a)p(b|a)p(c|a,b)", returns a list of variables and factors
    terms = parse_model_string(model_string)
    
    variables = {}
    factors = []
    
    for term in terms:
        # Create a factor for this term
        factor = Factor('p' + term.term)
        factors.append(factor)
        
        # Create the variable if it doesn't exist
        if term.var_name not in variables:
            variables[term.var_name] = Variable(term.var_name)
            
        # Connect the factor to the variable
        factor.add_neighbor(variables[term.var_name])
        variables[term.var_name].add_neighbor(factor)
        
        # Connect the factor to the given variables
        for given_var in term.given:
            if given_var not in variables:
                variables[given_var] = Variable(given_var)
                
            factor.add_neighbor(variables[given_var])
            variables[given_var].add_neighbor(factor)
            
    return factors, variables
```

```python
class PGM(object):
    def __init__(self, factors, variables):
        self._factors = factors
        self._variables = variables
        
    @classmethod
    def from_string(cls, model_string):
        factors, variables = parse_model_into_variables_and_factors(model_string)
        return PGM(factors, variables)
    
    def set_distributions(self, data):
        var_dims = {}
        for factor in self._factors:
            factor_data = data[factor.name]
            factor.data = factor_data
            
            if set(v.name for v in factor.neighbors) != set(factor_data.axes_labels):
                missing_axes = set(v.name for v in factor.neighbors) - set(data[factor.name].axes_labels)
                raise ValueError(f"data[{factor.name}] is missing axes: {missing_axes}")
            
            for var_name, dim in zip(factor_data.axes_labels, factor_data.probs.shape):
                if var_name not in var_dims:
                    var_dims[var_name] = dim
                    
                if var_dims[var_name] != dim:
                    raise ValueError(f"data[{factor.name}] axes is wrong size, {{}. Expected {{}}\".format(factor.name, missing_axes)")
                    
            factor.data = factor_data
            
    def variable_from_name(self, var_name):
        return self._variables[var_name]
```

We can notice that, in the previous example, we can write the marginal as a combination of sums and products:

$$p(x_5) = \sum_{x_3,x_4} p(x_1, x_2, x_3, x_4, x_5) == \sum_{x_3,x_4} f_3(x_3, x_4, x_5)\left[ \sum_{x_1} f_1(x_1, x_3) \right]\left[ \sum_{x_2} f_2(x_2, x_3) \right]$$

and interpret them as messages flowing from factors to variables (including a summation) or from variables to factors (via multiplication).

```python
class Messages(object):
    def __init__(self):
        self.messages = {}
        
    def _variable_to_factor_messages(self, variable, factor):
        # Take the product over all incoming factors into this variable except the variable
        # If there are no incoming messages, this is 1 (BASE CASE)
        incoming_messages = [self.factor_to_variable_message(neighbor_factor, variable) for neighbor_factor in variable.neighbors if neighbor_factor != factor]
        
        return np.prod(incoming_messages, axis=0)
    
    def _factor_to_variable_messages(self, factor, variable):
        #reinstantiate to obtain a deep copy
        factor_dist = Distribution(factor.data.probs, factor.data.axes_labels)
        
        for neighbor_variable in factor.neighbors:
            if neighbor_variable.name == variable.name:
                continue
            #Retrieve the incoming message and multiply the conditional distribution of the factor
            incoming_message = self.variable_to_factor_message(neighbor_variable, factor)
            factor_dist = multiply(factor_dist, Distribution(incoming_message, [neighbor_variable.name]))
            
        # Sum over the axes that aren't `variable`
        factor_dist = factor_dist.probs
        other_axes = factor.data.get_other_axes_from(variable.name)
        return np.squeeze(np.sum(factor_dist, axis=other_axes))
    
    def marginal(self, variable):
        # p(variable) is proportional to the product of incoming messages to variable.
        unnorm_p = np.prod([self.factor_to_variable_message(neighbor_factor, variable) for neighbor_factor in variable.neighbors])
        return unnorm_p / np.sum(unnorm_p)
    
    def variable_to_factor_message(self, variable, factor):
        message_name = (variable.name, factor.name)
        if message_name not in self.messages:
            self.messages[message_name] = self._variable_to_factor_messages(variable, factor)
        return self.messages[message_name]
    
    def factor_to_variable_message(self, factor, variable):
        message_name = (factor.name, variable.name)
        if message_name not in self.messages:
            self.messages[message_name] = self._factor_to_variable_messages(factor, variable)
        return self.messages[message_name]
```

## Exercise 6: Extending Belief Propagation to the Sum-Product Algorithm

In the notebook "Exact Inference with Belief Propagation", we previously computed the marginal distribution of a given variable using the message-passing method. Now, we aim to extend this implementation to the sum-product algorithm.

```python
class Messages(object):
    def __init__(self):
        self.messages = {}
    
    def forward(self, variable, factor):
        """Computes the forward pass.
        
        This method computes messages from variables to factors in the forward direction.
        
        Args:
            variable: The source variable node
            factor: The destination factor node
            
        Returns:
            numpy.ndarray: The message from variable to factor
        """
        # Take the product over all incoming factors into this variable except the target factor
        # If there are no incoming messages, this is 1 (BASE CASE)
        incoming_messages = [self.backward(neighbor_factor, variable) for neighbor_factor in variable.neighbors if neighbor_factor != factor]
        
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
        """Computes the backward pass.
        
        This method computes messages from factors to variables in the backward direction.
        
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
        other_axes = factor.data.get_other_axes_from(variable.name)
        result = np.sum(factor_dist.probs, axis=other_axes)
        
        # Squeeze to remove singleton dimensions
        return np.squeeze(result)
    
    def belief_propagation(self):
        """Executes the forward and backward passes, then uses the computed
        messages to determine all marginal distributions.
        
        This method should return a dictionary mapping each variable name to its
        corresponding marginal distribution.
        
        Returns:
            dict: A dictionary mapping variable names to marginal distributions
        """
        # Clear any existing messages to ensure a fresh computation
        self.messages = {}
        
        # Compute all messages by iterating through all variable-factor pairs
        for var_name, variable in pgm._variables.items():
            for factor in variable.neighbors:
                # Compute and cache the forward message
                msg_key = (variable.name, factor.name)
                if msg_key not in self.messages:
                    self.messages[msg_key] = self.forward(variable, factor)
                
                # Compute and cache the backward message
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
    
    def variable_to_factor_message(self, variable, factor):
        """Gets the message from a variable to a factor.
        
        This method is a wrapper around the forward method that caches results.
        
        Args:
            variable: The source variable node
            factor: The destination factor node
            
        Returns:
            numpy.ndarray: The message from variable to factor
        """
        message_name = (variable.name, factor.name)
        if message_name not in self.messages:
            self.messages[message_name] = self.forward(variable, factor)
        return self.messages[message_name]
    
    def factor_to_variable_message(self, factor, variable):
        """Gets the message from a factor to a variable.
        
        This method is a wrapper around the backward method that caches results.
        
        Args:
            factor: The source factor node
            variable: The destination variable node
            
        Returns:
            numpy.ndarray: The message from factor to variable
        """
        message_name = (factor.name, variable.name)
        if message_name not in self.messages:
            self.messages[message_name] = self.backward(factor, variable)
        return self.messages[message_name]
    
    def marginal(self, variable):
        """Computes the marginal distribution for a single variable.
        
        Args:
            variable: The variable node for which to compute the marginal
            
        Returns:
            numpy.ndarray: The normalized marginal distribution
        """
        # Compute the product of all incoming messages to this variable
        incoming_messages = [self.factor_to_variable_message(factor, variable) for factor in variable.neighbors]
        
        # Compute the unnormalized marginal
        unnorm_marginal = np.prod(incoming_messages, axis=0)
        
        # Normalize to get a valid probability distribution
        return unnorm_marginal / np.sum(unnorm_marginal)
```

### Apply the belief_propagation method to compute the marginal distributions of the variables in the factor graph described on page 43 of the course notes.

```python
# Let's create a factor graph as described in the course notes
# We'll use the example from earlier in the notebook

# Create the PGM from the model string
pgm = PGM.from_string("p(h1)p(h2|h1)p(v1|h1)p(v2|h2)")

# Define the distributions for each factor
p_h1 = Distribution(np.array([[0.2], [0.8]]), ['h1'])
p_h2_given_h1 = Distribution(np.array([[[0.5, 0.2], [0.5, 0.8]]], ['h2', 'h1'])
p_v1_given_h1 = Distribution(np.array([[0.6, 0.1], [0.4, 0.9]]), ['v1', 'h1'])
p_v2_given_h2 = Distribution(p_v1_given_h1.probs, ['v2', 'h2'])

# Set the distributions for the PGM
pgm.set_distributions({
    "p(h1)": p_h1,
    "p(h2|h1)": p_h2_given_h1,
    "p(v1|h1)": p_v1_given_h1,
    "p(v2|h2)": p_v2_given_h2
})

# Create a Messages object
messages = Messages()

# Run belief propagation to compute all marginals
marginals = messages.belief_propagation()

# Print the marginal distributions
for var_name, marginal in marginals.items():
    print(f"Marginal distribution for {var_name}: {marginal}")

# Verify the results by computing the marginal for h1 directly
h1_marginal = messages.marginal(pgm.variable_from_name("h1"))
print(f"Marginal for h1 computed directly: {h1_marginal}")
```

The implementation of the Messages class has been extended to support the sum-product algorithm for belief propagation on factor graphs. The key additions are:

1. **forward method**: Computes messages from variables to factors by taking the product of all incoming messages from neighboring factors (except the target factor).

2. **backward method**: Computes messages from factors to variables by multiplying the factor distribution with incoming messages from neighboring variables (except the target variable) and then summing over all variables except the target variable.

3. **belief_propagation method**: Executes the complete algorithm by computing all messages and then using them to determine the marginal distributions for all variables in the factor graph.

The implementation maintains backward compatibility with the original code by providing wrapper methods for variable_to_factor_message and factor_to_variable_message that use the new forward and backward methods.

This implementation follows the sum-product algorithm, which efficiently computes marginal distributions on factor graphs by passing messages between variables and factors. Each message represents a function of the variable it's being sent to, and the product of all messages received by a variable (when normalized) gives its marginal distribution.
