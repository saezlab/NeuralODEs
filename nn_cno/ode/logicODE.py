# -*- python -*-
#
# license: GPLv3
# author: Attila Gabor
# date: 2022-07-11
# version: 0.1


from logging import warning
from queue import Empty
import sys
import os

import pandas as pd
import numpy as np
import pylab
import itertools
import jax.numpy as jnp
import jax
import diffrax
import matplotlib.pyplot as plt
import optax

from nn_cno.core import CNOBase, CNORBase
from nn_cno.core.params import  OptionsBase

from nn_cno.core.results import ODEResults
from nn_cno.core import ReportODE
from nn_cno.core.params import ParamsSSM

from nn_cno.ode.graph2symODE import graph_to_symODE, sym2jax
import nn_cno.ode.graph2symODE as graph2symODE
# from biokit.rtools import bool2R

from nn_cno.core.params import params_to_update
from easydev import Logging

__all__ = ["logicODE"]


class logicODE(CNOBase):
    """JAX/Diffrax based ODE modeling 

    Attributes:
        jax_model : JAX/Diffrax ODE model
             JAX compatible right hand side of the ODE model
        ODE: dictionary
            dictionary containing the parameters and initial conditions of the ODE model
        symbolicODE: dictionary
            dictionary containing the symbolic representation of the ODE model(parameters, states and equations)
        conditions: list
            list of dictionaries containing the conditions for each experimental condition. What are the nodes
            stimulated, inhibited, measured, time of the simulation and the initial conditions(y0).
        transfer_function: function
            function that maps the state of the ODE model to the state of the network.
        transfer_function_type: string
            type of the transfer function. It can be "normalised_hill", "linear" or "custom"

    TODOs:
        - add logging: https://www.loggly.com/use-cases/6-python-logging-best-practices-you-should-be-aware-of/
        - add a way to specify the initial conditions for the ODE model 
        - add a way to specify the parameters for the ODE model
        - what does use_cnodata mean?
        
    """
    def __init__(self, model=None, data=None, tag=None, verbose=True,
                 config=None, use_cnodata=False, transfer_function_type = "normalised_hill",custom_transfer_function=None):
        
        # the CNORbase class takes care of importing PKN and MIDAS files. 
        CNOBase.__init__(self,model, data, verbose=verbose, tag=tag, 
                config=config, use_cnodata=use_cnodata)
        
        self.config.General.pknmodel.value = self.pknmodel.filename
        self.config.General.data.value = self.data.filename

        # process the transfer function
        self._set_transfer_function(transfer_function_type, custom_transfer_function)

        # convert the network to a symbolic logicODE
        self._update_ODE_model()

        # generate conditions from the MIDAS 
        self._initialize_conditions()

    def _initialize_conditions(self):
        """ initialize the conditions for the ODE model simulation

        The conditions are a list of dictionaries. They contains information about the 
        stimulated, inhibited and measured nodes, the initial values and the time range.
        Must be called if MIDAS(self.midas) or network (self._model) changes to adjust initial conditions.
        """
        
        # convert the conditions from the pandas dataframe to a list of dictionaries
        stimuli = self.midas.experiments.Stimuli
        inhibitors = self.midas.experiments.Inhibitors
        names_stimuli = self.midas.names_stimuli
        names_inhibitors = self.midas.names_inhibitors
        
        # measurements_index is the index of the measured nodes in the state variabels. Needed to subset the simulation results fast. 
        self.measurements_index = [ self.states.index(x) if x in self.states else None for x in self.midas.names_signals]

        conditions = list()
        for iexp in range(self.midas.nExps):
            stimulated = list(itertools.compress(names_stimuli,stimuli.iloc[iexp,:]==1))
            inhibited = list(itertools.compress(names_inhibitors,inhibitors.iloc[iexp,:]==1))
            measured = list(self.midas.names_signals)
            time = self.midas.times
            y0 = self._initialize_y0(self._model.species, stimulated, inhibited)
            conditions.append(dict(stimulated=stimulated, inhibited=inhibited, measured=measured, time=time, y0=y0))
        self.conditions = conditions
        
        self.measurements_df = jnp.array(self.midas.df.to_numpy(),dtype=jnp.float32)
      

    def _initialize_y0(self,species, stimulated, inhibited,default_value=0.0):
        """Get the initial conditions for a single experimental condition, considering the perturbations

        Parameters
            species: list
                list of species names
            stimulated: list
                list of stimulated species names
            inhibited: list
                list of inhibited species names
            default_value: float
                default value for the initial conditions
        Returns
            y0: numpy array
        """
        y0 = dict()
        for s in species:
            if s in stimulated:
                y0[s] = 1.0
            elif s in inhibited:
                y0[s] = 0.0
            else:
                y0[s] = default_value
        return y0

    def _update_ODE_model(self):
        """Generates the ODE model from the current network. Need to be called after changing the network."""
        
        pars, states, eqns = graph2symODE.graph_to_symODE(self._model, self.transfer_function)

        # store states in fixed order: 
        self.states = tuple([str(s) for s in states])      
        
        self.symbolicODE = dict(pars=pars, states=states, eqns=eqns)

        # generate the numerical parameter vector from the symbolic parameters
        self._initialize_ODEparameters()

        # convert the symbolic logicODE to a JAX/Diffrax ODE
        self.jax_model = graph2symODE.sym2jax(sym_eqs = self.symbolicODE["eqns"], sym_params=[*self.symbolicODE["pars"], *self.symbolicODE["states"]])

    def _initialize_ODEparameters(self):
        parNames = [str(p) for p in self.symbolicODE['pars']]
        parameters = dict()
        lb = dict()
        ub = dict()
        dz = dict()
        for p in parNames:
            if 'tau' in p:
                parameters[p] = 0.1
                lb[p] = 1e-3
                ub[p] = 10.0
                dz[p] = 1e-3
            elif '_n_'  in p:
                parameters[p] = 2.0
                lb[p] = 1
                ub[p] = 5
                dz[p] = 0.1
            elif '_k_' in p:
                parameters[p] = 1.0
                lb[p] = 1e-3 # DO NOT use 0.0 -> leads to division by zero
                ub[p] = 10
                dz[p] = 0.1
        
        self.ODEparameters = parameters
        self.ODEparameters_lb = jnp.array(list(lb.values()))
        self.ODEparameters_ub = jnp.array(list(ub.values()))
        self.ODEparameters_dz = jnp.array(list(dz.values()))
    
    def get_ODEparameters(self):
        return self.ODEparameters
    
    def set_ODEparameters(self, parameters):
        self.ODEparameters = parameters

    def preprocessing(self, expansion=False, compression=True, cutnonc=True, maxInputsPerGate=2):
        """ Preprocessing of the network. Same as in CNORBase, but expansion is False by default. Also updates the ODE model."""
        super().preprocessing(expansion, compression, cutnonc, maxInputsPerGate)
        # after preprocessing, we need to update the ODE model and the conditions (initial values)
        self._update_ODE_model()
        self._initialize_conditions()
        self.setup_simulation()
        self.setup_optimization()
        
    def _set_transfer_function(self, transfer_function_type, custom_transfer_function):
        """Set the transfer function to use for the ODE model.
        
        Attributes:
        transfer_function_type: str
            The type of transfer function to use: "normalised_hill", "lienar", "custom"
            custom_transfer_function: function
                Make sure, the transfer function takes the following arguments:
                    parental_var: sympy.Symbol
                        The parental variable.
                    node: sympy.Symbol or str
                        The current node name.
                and returns:
                    list:   A list of parameters and the symbolic equation.
                when the transfer function is evaluated with numerical values, it should return numeric values in the [0, 1] interval.
                """

        valid_transfer_function_types = ["normalised_hill", "linear", "custom"]
        if transfer_function_type not in valid_transfer_function_types:
            raise ValueError("transfer_function_type must be one of %s" % valid_transfer_function_types)
        self.transfer_function_type = transfer_function_type
        
        if transfer_function_type == "custom": 
            if custom_transfer_function is None:
                raise ValueError("custom_transfer_function is missing")
            self.transfer_function = custom_transfer_function
        elif transfer_function_type == "normalised_hill":
            self.transfer_function = graph2symODE.transfer_function_norm_hill_fun
        elif transfer_function_type == "linear":
            self.transfer_function = graph2symODE.transfer_function_linear_fun
        

    # get the y0 from the conditions as a numpy array
    def get_y0(self):
        y0 = jnp.array([jnp.array(list(c['y0'].values())) for c in self.conditions])
        return y0

    def get_inhibited_species(self):
        """ Returns an indicator array for the inhibited species. 
        
        returns:
            numpy array of shape (n_conditions, n_species), where the entry i,j is 1 if the species j is inhibited in condition i, 0 otherwise.

        """
        inhibited = np.zeros((len(self.conditions),len(self.states)))
        for i in range(len(self.conditions)):            
            for j in range(len(self.conditions[i]["inhibited"])):
                inhibited[i,self.states.index(self.conditions[i]["inhibited"][j])] = 1

        return(inhibited)

    def get_rhs_function(self):
        """ Generate a right hand side function compatible with Diffrax 
        
            This is a slower version of the simulation, not used for the optimization. 
        """
        
        f_jax = self.jax_model[0]
        f_jax_p = self.jax_model[1]
        states = self.states


        def f(t, y, args):
            # get users data containinf the jax objects and model parameters: 
            #f_jax = args["jax_eqns"]
            #f_jax_p = args["jax_parameters"]
            parameter_vals = args["model_parameters"]
            condition = args["condition"]

            #jax_pars = jnp.concatenate((parameter_vals,y))
            jax_pars = [*parameter_vals,*y]
            fy = list()

            for i in range(len(y)):
                
                if states[i] in condition["inhibited"]:
                    fy.append(jnp.array([0.0,])[0])
                elif f_jax[i] == 0:
                    fy.append(jnp.array([0.0,])[0])
                else:
                    fy.append(f_jax[i](jnp.array([jax_pars,]),f_jax_p[i])[0])
               
            return jnp.asarray(fy)
                
        return f

    def setup_simulation(self):
        # this is when jax generates the graph, takes some time. 
        self.sim_function = self.get_rhs_function()
        self._diffrax_ODEterm = diffrax.ODETerm(self.sim_function)
        
        

    def _get_vectorized_simulation(self):
        """ Get a simulation function that is JIT-ted"""
        f_jax = self.jax_model[0]
        f_jax_p = self.jax_model[1]

        jax.jit
        def simulate(pars,y0,inhibited_states):
    
            @jax.jit
            def rhs_function( t, y, args):  
                X = [*pars,*y]
                dy = list()
                for i in range(len(y)):
                    if f_jax[i] == 0:
                        dy.append(jnp.array([0.0,])[0])
                    else:
                        dy.append((1.0-inhibited_states[i])*f_jax[i](jnp.array([X,]),f_jax_p[i])[0])

                return jnp.asarray(dy)
            
            ts = self.midas.times

            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(rhs_function),
                diffrax.Heun(),
                ts[0],
                ts[-1],
                dt0=0.1,
                y0 = y0,
                stepsize_controller =diffrax.ConstantStepSize(),
                saveat=diffrax.SaveAt(ts=ts),
            )
            return(sol.ys)
        return simulate

    def simulate(self, ODEparameters=None, timepoints = None, stepsize_controller =diffrax.ConstantStepSize(), plot_simulation = False):
        """Simulate the model with the given parameters and conditions
        
        steps: 
            1. gets the parameters
            2. gets the conditions from the midas attributes
            3. runs the simulation by calling the diffrax 

        Attributes:
        ODEparameters: list or jax.numpy.array of parameters
            The parameters to use for the simulation. If None, the current parameters are used.
        timepoints: array_like
            The timepoints to use for the simulation in each condition. Useful to obtain smooth, high-res curves
        stepsize_controller: diffrax.PIDController or diffrax.ConstantStepSize
            Error control for the simulation of ODE systems. If None, the default 
             diffrax.PIDController(atol=1e-3,rtol=1e-3) PIDController is used with Atol=1e-3 and Rtol=1e-3.
        plot_simulation: bool
            If True, the simulation is plotted.

        Returns:
        simulation_results: list
            A list containing the simulation results for each condition.
        """
        

        # obtain numeric model parameters either from the user or from the object
        if(ODEparameters is None):
            parameters = jnp.array(list(self.get_ODEparameters().values()))
        elif(isinstance(ODEparameters, dict)):
            parameters = jnp.array(list(ODEparameters.values()))
        elif(isinstance(ODEparameters, jax.numpy.ndarray)):
            parameters = ODEparameters
        else: 
            raise TypeError("ODEparameters must be a dictionary or a jax.numpy.ndarray")

        if timepoints is not None:
            for c in self.conditions:
                c["time"] = timepoints
            
        # run the simulation for each condition
        simulation_data = list()
        ODEsolver = diffrax.Heun()
        ODEterm = self._diffrax_ODEterm 


        for c in self.conditions:
            # run the simulation in a specific condition
            user_data = dict(model_parameters = parameters,
                            condition = c
                            )
            t0 = c["time"][0]
            t1 = c["time"][-1]
            y0 = jnp.array(list(c["y0"].values()))
            ts = c["time"]
            
            sim_condition = diffrax.diffeqsolve(ODEterm, ODEsolver,
                        t0=t0, t1=t1, dt0=0.1,
                        y0=y0,
                        args = user_data, 
                        saveat=diffrax.SaveAt(ts=ts),
                        stepsize_controller = stepsize_controller)

            # add the simulation data to the list
            simulation_data.append(sim_condition)
        
        if plot_simulation is True:
            self.plot_simulation(simulation_data)

        return simulation_data


    def plot_simulation(self, simulation_data):
        """Plot the simulation data"""
        states = self.states

        fig, axs= plt.subplots(len(states), len(simulation_data), figsize=(12,6))
        for istate in range(len(states)):
            for icond in range(len(simulation_data)):
                axs[istate,icond].plot(simulation_data[icond].ts, simulation_data[icond].ys[:,istate])
                
                axs[istate,icond].set_xlabel("time")
                axs[istate,icond].set_ylim(0,1)
                if icond == 0:
                    axs[istate,icond].set_ylabel(states[istate])
                if istate == 0:
                    axs[istate,icond].set_title("condition " + str(icond))

    def fit(self, params=None, optimizer=optax.adam(learning_rate=1e-2), max_iter=100, boundary_handling = "penalty",  verbose=False, plot_convergence=False):

        
        if(params is None):
            params = jnp.array(list(self.get_ODEparameters().values()))
        elif(isinstance(params, jax.numpy.ndarray)):
            params = params
        elif(isinstance(params, dict)):
            params = jnp.array(list(params.values()))
        else: 
            raise TypeError("ODEparameters must be either a dictionary or a jax.numpy.ndarray")

        if(plot_convergence is True):
            from IPython.display import display, clear_output
            fig = plt.figure()
            axs = fig.add_subplot(1,1,1)    

        opt_state = optimizer.init(params)
        
        def step(params, opt_state):
            loss_value, grads = jax.value_and_grad(self.loss_function)(params)

            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value, grads

        convergence = list()
        steps = list()
        opt_params = params
        opt_loss = np.Inf

        for i in range(max_iter):
            params, opt_state, loss_value, grad_value = step(params, opt_state)
            convergence.append(loss_value)
            steps.append(i)
            
            if loss_value < opt_loss:
                opt_params = params
                opt_loss = loss_value
            
            if(plot_convergence is True):
                axs.set_xlim(0,i)
                axs.cla()
                axs.plot(steps,convergence, marker = "o")
                axs.plot(i,opt_loss, marker = "o", color = "red")
                display(fig)
                clear_output(wait=True)
                
            if i % 100 == 0:
                print('  {:<5} \t {:<9} \t {:<9}\t {:<9}'.format( "iter", 	 "current loss", 	 "min. loss", 	 "grad norm"))
            if i % 10 == 0:
                print('  {:<5d} \t {:<9.4e} \t {:<9.4e} \t {:<9.4e}'.format(i, loss_value, opt_loss, jnp.linalg.norm(grad_value)))
                

        return opt_params

    def setup_optimization(self):
        """Setup the optimization problem
        
        This initializes the self.loss_function and self.loss_function_grad
        
        """
        f_jax = self.jax_model[0]
        f_jax_p = self.jax_model[1]
        lb = self.ODEparameters_lb
        ub = self.ODEparameters_ub
        dz = self.ODEparameters_dz
        
        self.y0_array = self.get_y0()
        self.inhibited_array = self.get_inhibited_species()

        @jax.jit
        def simulate(pars,y0,inhibited_states):
            
            @jax.jit
            def rhs_function( t, y, args):  
                X = [*pars,*y]
                dy = list()
                for i in range(len(y)):
                    if f_jax[i] == 0:
                        dy.append(jnp.array([0.0,])[0])
                    else:
                        dy.append((1.0-inhibited_states[i])*f_jax[i](jnp.array([X,]),f_jax_p[i])[0])

                return jnp.asarray(dy)
            
            ts = self.midas.times

            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(rhs_function),
                diffrax.Heun(),
                ts[0],
                ts[-1],
                dt0=0.1,
                y0 = y0,
                stepsize_controller =diffrax.ConstantStepSize(),
                saveat=diffrax.SaveAt(ts=ts),
            )
            return(sol.ys)

        # map the simulation across the conditions
        simulate_all_map = jax.vmap(simulate,in_axes=(None,0,0))

        @jax.jit
        def bound_penalty(params):
            p = jnp.where(params < lb+dz, 1, 0) * jnp.power((params-lb)/dz,2)*1e3
            p += jnp.where(params > ub-dz, 1, 0) * jnp.power((ub-params)/dz,2)*1e3
            return(p.sum())

        @jax.jit
        def fit_penalty(params):
            params = jnp.clip(params, lb, ub)

            #v_sim = jax.vmap(simulate,in_axes=(None,0))
            simulation_data = simulate_all_map(params,self.y0_array,self.inhibited_array)
            
            # extract the simulation data from the output of diffrax solution and subset it to the measured nodes. 
            simulation_df = jnp.concatenate([s[:,self.measurements_index] for s in simulation_data])

            sq = jnp.power(simulation_df-self.measurements_df,2)
            return jnp.sum(sq)/jnp.size(sq)
        
        @jax.jit
        def loss(params):
            
            # assure that parameters are within bounds for the simulation
            mse_penalty = bound_penalty(params) 

            mse =  fit_penalty(params)  + mse_penalty

            return mse

        self.loss_function = loss
        self.loss_function_grad = jax.grad(loss)


        












