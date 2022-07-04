
import sympy as sym
from nn_cno.io.cnograph import CNOGraph
import nn_cno


def transfer_function_norm_hill_fun(parental_var,node):
    """ Use Normalised Hill equation as transfer function
    
    Formulates the Hill equation as a symbolic equation with parameters.

    Parameters
        parental_var: sympy.Symbol
            The parental variable.
        node: sympy.Symbol or str
            The current node name.
        
    Returns 
        list:
            A list of parameters and the symbolic equation.
    """
    k = sym.symbols(str(parental_var) + "_k_" + str(node))
    n = sym.symbols(str(parental_var) + "_n_" + str(node))

    return ([k, n], parental_var**n / (k**n + parental_var**n) * (k**n + 1))


def transfer_function_linear(parental_var,node):
    """ Use a linear expression as transfer function

    Parameters
        parental_var: sympy.Symbol
            The parental variable.
        node: sympy.Symbol or str
            The current node name.
        
    Returns 
        list:
            A list of parameter and the symbolic equation.
    """
    k = sym.symbols(str(parental_var) + "_k_" + str(node))

    return ([k], k*parental_var)

def _ORgate(x,y):
    return x+y-x*y 
    
def _ANDgate(x,y):
    return x*y 

def _simple_reaction_to_sym(node, parent_node, sign, transfer_function):
    """ Convert a simple reaction to a symbolic equation and parameters.
    
    Parameters
        node: str
            The name of the node.
        pred: str
            The name of the predecessor node.
        sign: str
            The sign of the reaction.
        transfer_function: function
            The transfer function to use. normalised_hill_fun by default.
            The transfer function must take the following arguments:
                parental_var: sympy.Symbol
                    The parental variable.
                node: sympy.Symbol or str
                    The current node name.

    Returns
        list:
            A list of parameters and the symbolic equation.
"""

    parent_sym = sym.symbols(parent_node)
    node_sym = sym.symbols(node)
    pars, eqn = transfer_function(parent_sym,node_sym)
    if sign == "-":
        eqn = 1-eqn
    return (pars, eqn)



def _and_reaction_to_sym(node,and_inputs,signs, transfer_function):
    """ Convert an AND reaction to a symbolic equation and parameters.
    
    Parameters
        node: str
            The name of the node.
        and_inputs: list
            The names of the AND inputs.
        signs: list
            The signs of the AND inputs, can be '+' or '-'
    Returns
        list:
            A list of parameters and the symbolic equation.
    """
    eqns = list()
    params = list()
    node_sym = sym.symbols(node)

    for n,sign in zip(and_inputs,signs):
        parent_sym = sym.symbols(n)
        if sign == "+":
            pars, eqn = transfer_function(parent_sym,node_sym)
        elif sign == "-":
            pars, eqn = 1-transfer_function(parent_sym,node_sym)
        else:
            raise Exception("unrecognised sign")
        eqns.append(eqn)
        params.append(pars)
    # combine with ANDgate
    eqn = _ANDgate(eqns[0],eqns[1])

    return(params,eqn)



# generate the right hand side for the node

def _construct_symbolic_rhs(node, G,transfer_function):
    """ Generate the right hand side for the node.
        
        Steps:
        1. get the upstream nodes, 3 cases can happen: 
            1.1. no upstream nodes -> rhs = 0
            1.2. the upstream node is a regular state -> convert the reaction to a symbolic equation 
            1.3. the upstream node is an AND node -> compute the symbolic equation based on the inputs of the AND node    
        2. combine all the reactions with OR gates. 
        3. add the tau parameter and substract the current state

    Parameters
        node: str
            The name of the node.
        G: CNOGraph
            The graph.
    Returns
        list:
            A list of parameters and the symbolic equation.
    """
    # get the upstream nodes
    and_nodes = G._find_and_nodes()
    if node in and_nodes:
        raise Exception("node mustn't be an AND gate node")
    preds = list(G.predecessors(node))
    
    if len(preds) == 0:
        # no input edge: derivative is zero
        sym_eq = 0
        sym_parameters = []
        return (sym_eq, sym_parameters)
    else:
        sym_reactions = list()
        sym_parameters = list()

        for i, pred in enumerate(preds):
            # upstream node is not an AND node: 
            if pred not in and_nodes:
                sign = G.get_edge_data(pred,node)['link']
                p,r = _simple_reaction_to_sym(node=node,
                                            parent_node=pred,
                                            sign=sign,
                                            transfer_function=transfer_function)
                sym_reactions.append(r)
                sym_parameters.append(p)
            else:
                # upstream is an AND node    
                and_inputs = list(G.predecessors(pred))

                signs = [G.get_edge_data(inp,pred)["link"] for inp in and_inputs]

                p,r = _and_reaction_to_sym(node=node,
                                            and_inputs=and_inputs,
                                            signs=signs,
                                            transfer_function=transfer_function)
                sym_reactions.append(r)
                sym_parameters.append(p)
        
        # combine with OR gates
        if len(preds)==1:
            sym_eq = sym.symbols("tau_"+node) * (sym_reactions[0] - sym.symbols(node))
        else:
            aggregated_or = sym_reactions[0]
            for i in range(1,len(sym_reactions)):
                aggregated_or = _ORgate(aggregated_or,sym_reactions[i])
            sym_eq = sym.symbols("tau_"+node) * (aggregated_or - sym.symbols(node))

    return (sym_parameters, sym_eq )


def graph_to_symODE(inp_graph,transfer_function=transfer_function_norm_hill_fun):
    """ Convert a graph to a symbolic ODE.
        
    Parameters
        inp_graph: CNOGraph
            The graph in networkx format.
    Returns
        list:
            A list of parameters and the symbolic equation.

    """
    f_rhs_aut = list()
    pars = list()
    for node in inp_graph.nodes -  inp_graph._find_and_nodes():
        pars, rhs  = _construct_symbolic_rhs(node, inp_graph, transfer_function)
        f_rhs_aut.append(rhs)
        #print(rhs)
        #print(pars)
    return (pars, f_rhs_aut)
