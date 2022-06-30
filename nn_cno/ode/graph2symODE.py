
import sympy as sym
from nn_cno.io.cnograph import CNOGraph
import nn_cno

# normalised Hill equations (for each reactions) ( x^n / (k^n + x^n) * {1/(1/k^n + 1)} )
def norm_hill_fun(parental_var,n,k):
    return parental_var**n / (k**n + parental_var**n) * (k**n + 1)

def ORgate(x,y):
    return x+y-x*y 
    
def ANDgate(x,y):
    return x*y 

def simple_reaction_to_sym(node,pred,sign):
    par_k = sym.symbols(pred + "_k_" + node)
    par_n = sym.symbols(pred + "_n_" + node)
    pred_sym = sym.symbols(pred)
    if sign == "+":
        eqn = norm_hill_fun(pred_sym,par_n,par_k)
    elif sign == "-":
        eqn = 1-norm_hill_fun(pred_sym,par_n,par_k)
    else:
        raise Exception("unrecognised sign")
        
    return ([par_k, par_n], eqn)


# creates the symbolic equations and parameters corresponding to and AND reaction
# node: (str) the name of the node
# and_inputs: vec(str,str) length-2 str vector storing the inputs of the AND gates.   
def and_reaction_to_sym(node,and_inputs,signs):
    eqns = list()
    params = list()
    for n,sign in zip(and_inputs,signs):
        
        par_k = sym.symbols(n + "_k_" + node)
        par_n = sym.symbols(n + "_n_" + node)
        pred_sym = sym.symbols(n)
        if sign == "+":
            eqn = norm_hill_fun(pred_sym,par_n,par_k)
        elif sign == "-":
            eqn = 1-norm_hill_fun(pred_sym,par_n,par_k)
        else:
            raise Exception("unrecognised sign")
        eqns.append(eqn)
        params.append([par_k,par_n])

    return(params,sym.prod(eqns))



# generate the right hand side for the node
# 1. get the upstream nodes, 2 cases can happen: it is another node or it is an AND gate. 
# 1.1 If the upstream is a regular state, convert the reaction into a hill equation and get the parameters
# 1.2 If AND gate, then we have to go up one level, compute the hill equation and apply the AND -rule. 
# 2. combine all the reactions with OR gates. 
# 3. add the tau parameter and substract the current state
def construct_symbolic_rhs(node, G):
    and_nodes = G._find_and_nodes()
    if node in and_nodes:
        raise Exception("node mustn't be an AND gate node")

    preds = list(G.predecessors(node))
    if len(preds) == 0:
        # no input edge: derivative is zero
        sym_eq = 0
        sym_parameters = []
    else:
        sym_reactions = list()
        sym_parameters = list()

        for i, pred in enumerate(preds):
            # upstream node is not an AND node: 
            if pred not in and_nodes:
                sign = G.get_edge_data(pred,node)['link']
                p,r = simple_reaction_to_sym(pred,node,sign)
                sym_reactions.append(r)
                sym_parameters.append(p)
                
            # upstream is an AND node    
            else:
                and_inputs = list(G.predecessors(pred))

                signs = [G.get_edge_data(inp,pred)["link"] for inp in and_inputs]

                p,r = and_reaction_to_sym(node,and_inputs,signs)
                sym_reactions.append(r)
                sym_parameters.append(p)
        
        # combine with OR gates
        if len(preds)==1:
            sym_eq = sym.symbols("tau_"+node) * (sym_reactions[0] - sym.symbols(node))
        else:
            aggregated_or = sym_reactions[0]
            for i in range(1,len(sym_reactions)):
                aggregated_or = ORgate(aggregated_or,sym_reactions[i])
            sym_eq = sym.symbols("tau_"+node) * (aggregated_or - sym.symbols(node))

    return (sym_eq, sym_parameters)


def graph_to_symODE(inp_graph):
    f_rhs_aut = list()
    for node in inp_graph.nodes -  inp_graph._find_and_nodes():
        rhs, pars = construct_symbolic_rhs(node,inp_graph)
        f_rhs_aut.append(rhs)
        #print(rhs)
        #print(pars)
    f_rhs_aut 