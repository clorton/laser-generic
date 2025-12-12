# Simulation structure

To fully leverage the power and flexibility of the LASER framework, it is important to understand how the various components work together to produce simulations, and ultimately, simulation results. You can think of a simulation as having two main pieces: the scenario and the model.

For a simple working example of building a simulation, see [Build a model](../get-started/index.md).

## Scenarios

The scenario is information that describes the environment in which your model will operate. Scenarios include information on:

- The number of nodes or patches
- Population sizes
- Node coordinates (latitude and longitude) for spatial models
- The initial disease state counts


The scenario is the static information that tells LASER what exists in your model world prior to the first timestep. Scenario information will be contained in a DataFrame, either using the `grid()` function or created manually. Scenarios DataFrames should be formatted with one row per node, and should not be modified during the running of simulations.

!!! note "Scenarios for testing"

    As tests require different initial conditions, it is best practice to build new scenario files for each test instead of reusing or modifying the same scenario. Modification or reuse across test runs can cause the GeoDataFrame to become corrupted.


An example using the `grid()` function:

```python
from laser.generic.newutils import grid

scenario = grid(M=1, N=1, population_fn=lambda i, j: 100000)
```

will produdce a GeoDataFrame with columns:

- nodeid 0 ... N-1, which uniquely identifies each node
- population, which is the initial population at that node
- geometry, a polygon representing the node boundary
- x,y, the centroid coordinates

To create a multi-node scenario with a 5x5 grid of nodes each with a defined population,

```python
scenario = grid(
    M=5,
    N=5,
    node_size_km=10,
    population_fn=lambda i, j: 5000 + (i+j)*100
)
```

To give each node or patch its own initial counts for various disease states, assign those values as columns in the scenario. For example, if you want to configure:

- Node 0: 999 S, 1 I, 0 R
- Node 1: 990 S, 10 I, 0 R
- Node 2: 950 S, 50 I, 0 R
- All other nodes: 1000 S, 0 I, O R

The code would be written as:

```python
scenario["S"] = [999, 990, 950, 1000, 1000, 1000, 1000, 1000, 1000]
scenario["I"] = [1, 10, 50, 0, 0, 0, 0, 0, 0]
scenario["R"] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
```


## Models

While the scenario is static information, the model is the behavior of the simulation. Models include a `PropertySet` custom object containing model parameters, a `LaserFrame` class to hold simulation data, and as a collection of LASER components which combine to operate on the scenario over time.

Information contained in the `PropertySet` object includes:

- The transmission rate, Beta
- Infection durations
- The number of timesteps, `nticks`
- The random seed generator
- Importation information for seeding infections
- Seasonality information (if relevant)
- Migration information (if relevant)


!!! note "Setting a global random seed"

    To ensure reproducibility across draws in your simulation, including initial infection seeding, birth and death events, agent sampling, and timers and durations, it is important to set a global random seed. To do so, use the `laser.core.random.seed()` function, which will set the NumPy global PRNG and all Numba thread-local seeds.

    mkdFor example,
    ```python

    from laser.core.random import seed
    from laser.generic import Model
    from laser.generic.utils import get_default_parameters

    seed(42)

    params = get_default_parameters() | {"nticks": 365, "beta": 0.2}
    model = Model(scenario, params)
    ```



Information contained in the `LaserFrame` class (which acts as a DataFrame to hold dynamically allocated data) includes:

- Agent-level data such as state, timers, age, and node ID
- Node-level time series data such as updated disease states or incidence
- Any other agent or node properties with are updated over time

!!! note "Seeding infections"

    Disease components (SI, SIR, SEIR, and others) initialize agent state internally based on the state columns in the scenario table. Do not manually assign states in `LaserFrame` as these will be overwritten during component initialization. Instead, the correct way to seed an outbreak is to modify the scenario:

    ```
    scenario["S"] = initial_susceptible
    scenario["I"] = initial_infected
    scenario["R"] = initial_recovered
    ```

    LASER will create the `agent-level_state` and `_itimer` values automatically.

Example components include:

- Transmission
- Exposure
- Infection
- Mortality estimators
- Birth rates

Components act to update the data stored in `LaserFrame` over each timestep. When using components, it is important to remember that you should not manually modify agent-level properties directly, modify the scenario table as LASER will update agent properties automatically.

It is important to add the desired transmission components prior to running simulations. The components must be passed as a list into `model.components` to avoid error messages from LASER.

For example, for a model with SI dynamics,

```python
model = Model(scenario, params)
model.components = [
    SI.Susceptible(model),
    SI.TransmissionSIX(model),
    SI.InfectiousSI(model),
]
```
