import numpy as np
from collections import namedtuple

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
        #variable_name is the name of the variable for which we are computing the distribution, e.g. in p(y|x) it is 'y'
        return np.all(np.isclose(np.sum(self.probs, axis=self.get_axes()[variable_name]), 1.0))
    
    def is_valid_joint(self):
        return np.all(np.isclose(np.sum(self.probs), 1.0))

def multiply(p_v_given_h, p_hi):
    ''' 
    Compute the product of the distributions p(v|h1,..,hn)p(hi) where p(hi) is a 1D array
    '''
    #Get the axis corresponding to hi in the conditional distribution
    axis=p_v_given_h.get_axes()[next(iter(p_hi.get_axes()))]

    # Reshape p(hi) in order to exploit broadcasting. Consider also the case in which p(hi) is a scalar.
    dims = np.ones_like(p_v_given_h.probs.shape)
    dims[axis] = p_v_given_h.probs.shape[axis]

    if (p_hi.probs.shape != () ):
        reshaped_p_hi = p_hi.probs.reshape(dims)
    else:
        reshaped_p_hi = p_hi.probs

    return Distribution(p_v_given_h.probs*reshaped_p_hi, p_v_given_h.axes_labels)

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
        var = var.split(',')
        given = given.split(',')
    else:
        var = term_variables
        var = var.split(',')
        given = []

    return var, given


def _parse_model_string_into_terms(model_string):
    return [
        ParsedTerm('p' + term, *_parse_term(term))
        for term in model_string.split('p')
        if term
    ]

def parse_model_into_variables_and_factors(model_string):
    # Takes in a model_string such as p(h1)p(h2∣h1)p(v1∣h1)p(v2∣h2) and returns a
    # dictionary of variable names to variables and a list of factors.
    
    # Split model_string into ParsedTerms
    parsed_terms = _parse_model_string_into_terms(model_string)
    
    # First, extract all of the variables from the model_string (h1, h2, v1, v2). 
    # These each will be a new Variable that are referenced from Factors below.
    variables = {}
    for parsed_term in parsed_terms:
        # if the variable name wasn't seen yet, add it to the variables dict
        for term in parsed_term.var_name:
            if term not in variables:
                variables[term] = Variable(term)

    # Now extract factors from the model. Each term (e.g. "p(v1|h1)") corresponds to 
    # a factor. 
    # Then find all variables in this term ("v1", "h1") and add the corresponding Variables
    # as neighbors to the new Factor, and this Factor to the Variables' neighbors.
    factors = []
    for parsed_term in parsed_terms:
        # This factor will be neighbors with all "variables" (left-hand side variables) and given variables
        
        new_factor = Factor(parsed_term.term)
        all_var_names = parsed_term.var_name + parsed_term.given
        for var_name in all_var_names:
            new_factor.add_neighbor(variables[var_name])
            variables[var_name].add_neighbor(new_factor)
        factors.append(new_factor)

    return factors, variables

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

            if set(factor_data.axes_labels) != set(v.name for v in factor.neighbors):
                missing_axes = set(v.name for v in factor.neighbors) - set(data[factor.name].axes_labels)
                raise ValueError("data[{}] is missing axes: {}".format(factor.name, missing_axes))
                
            for var_name, dim in zip(factor_data.axes_labels, factor_data.probs.shape):
                if var_name not in var_dims:
                    var_dims[var_name] = dim
    
                if var_dims[var_name] != dim:
                    raise ValueError("data[{}] axes is wrong size, {}. Expected {}".format(factor.name, dim, var_dims[var_name]))            
                    
            factor.data = data[factor.name]
            
    def variable_from_name(self, var_name):
        return self._variables[var_name]

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
    
    def _variable_to_factor_messages(self, variable, factor):
        # For backward compatibility, use the forward method
        return self.forward(variable, factor)
    
    def _factor_to_variable_messages(self, factor, variable):
        # For backward compatibility, use the backward method
        return self.backward(factor, variable)
    
    def variable_to_factor_messages(self, variable, factor):
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
        # p(variable) is proportional to the product of incoming messages to variable.
        unnorm_p = np.prod([self.factor_to_variable_message(neighbor_factor, variable) for neighbor_factor in variable.neighbors], axis=0)
        return unnorm_p / np.sum(unnorm_p)

# Example usage
if __name__ == "__main__":
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
    
    # We can also compute individual marginals using the marginal method
    print("\nComputing individual marginals:")
    for var_name, variable in pgm._variables.items():
        marginal = messages.marginal(variable)
        print(f"P({var_name}) = {marginal}")
