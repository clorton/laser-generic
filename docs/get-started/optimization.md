# Optimize LASER models

As an agent-based model, research using LASER will require thousands of simulation replicates. While the model is designed to perform well with large agent populations, there is still a need to utilize high compute power and to optimize model performance.

When creating LASER models, knowing how to identify and fix performance bottlenecks can save compute time and speed results.

## Identify bottlenecks

Typically, we do not recommend running the code through a profiler, at least not initially. Instead, we take advantage of LASER’s highly modular structure and AI-driven optimization.

The first step is to add simple timing code that tracks the total time spent in each component over a realistic simulation. Then, plot a pie chart at the end to visualize where the most time is spent. A simple way to track execution time is using the `time package`.

/// details | Code example: Identify bottlenecks

```
def run(self):
    self.component_times = {component.__class__.__name__: 0 for component in self.components}
    self.component_times["reporting"] = 0
    for tick in tqdm(range(self.pars.timesteps)):
        for component in self.components:
            start_time = time.time()  # Start timing for the component
            component.step()
            end_time = time.time()  # End timing for the component

            # Accumulate the time taken for this component
            elapsed_time = end_time - start_time
            component_name = component.__class__.__name__
```
///

This often reveals the top one to three performance bottlenecks. Focus first on the biggest offender—it provides the most opportunity for speedup. Often, the largest bottleneck is not what you might instinctively expect. Avoid optimizing a component only to find out it contributes just a small percentage of the total runtime. A modest improvement in the runtime of an “expensive” component is often more effective than spending a lot of time on highly optimizing a component which only accounts for a small fraction of runtime. Also, make sure that your reporting code is being measured and reported, ideally in its own ‘bucket’. This may be easier or harder depending on how you are doing reporting. Since reporting usually involves counting over the entire population, it usually shows up as a hotspot sooner or later. Fortunately, it’s usually fairly easy to speed up. Or even eliminate.

## Squash, save, and load

As the number agents in your LASER population model grows (e.g., 1e8), it can become computationally expensive and unnecessary to repeatedly run the same initialization routine every sim. In many cases — particularly during model calibration — it is far more efficient to initialize the population once, save it, and then reload the initialized state for subsequent runs.

This approach is especially useful when working with EULAs – Epidemiologically Uninteresting Light Agents. For example, it can be a very powerful optimization to compress all the agents who are already (permanently) recovered or immune in a measles or polio model into a number/bucket. In such models, the majority of the initial population may be in the “Recovered” state, potentially comprising 90% or more of all agents. If you are simulating 100 million agents, storing all of them can result in punitive memory usage.

To address this, LASER supports a **squashing** process. Squashing involves defragmenting the data frame such that all epidemiologically active or “interesting” agents (e.g., Susceptible or Infectious) are moved to the beginning of the array or table, and less relevant agents (e.g., Recovered) are moved to the end. Though please note that you should assume that squashed agent data is overwritten.

Some notes about squashing:

- The population count is adjusted so that all for loops and step functions iterate only over the active population.
- This not only reduces memory usage but also improves performance by avoiding unnecessary computation over inactive agents.


Some notes about using saved populations:

- You will want to be confident that the saved population is sufficiently randomized and representative of your overall population.
- If you are calibrating parameters used to create the initial population in the first place, you’ll need to recreate those parts of the population after loading, diminishing the benefit of the save/load approach.
- When saving a **snapshot**, note that only the active (unsquashed) portion of the population is saved. Upon reloading:

    - Only this subset is allocated in memory.
    - This prevents the performance penalty of managing large volumes of unused agent data.

!!! note
    Before squashing, you should count and record the number of recovered (or otherwise squashed) agents. This count should be stored in a summary variable —- typically the R column of the results data frame. This ensures your model retains a complete epidemiological record even though the agents themselves are no longer instantiated.

### Procedure

1. Add squashing:
    - Add a `squash_recovered()` function. This should call `LaserFrame.squash(…)` with a boolean mask that includes non-recovered agents (disease_state != 2). You may choose a different criterion, such as age-based squashing.
    - Count your “squashed away” agents first. You must compute and store all statistics related to agents being squashed before the `squash()` call. After squashing, only the left-hand portion of the arrays (up to .count) remains valid.
    - Seed infections after squashing. If your model seeds new infections (disease_state == 1), this must happen after squashing. Otherwise, infected agents may be inadvertently removed.
    - Store the squashed-away totals by node. Before squashing, compute and record node-wise totals (e.g., recovered counts) in `results.R[0, :]` so this pre-squash information persists.
    - (Optionally) simulate EULA effects once and save. If modeling aging or death among squashed agents, simulate this up front and store the full `[time, node]`` matrix (e.g., `results.R[:, :]`). This avoids recomputation at runtime.

2. Save function: implement a `save(path)` method:
    - Use `LaserFrame.save_snapshot(path, results_r=..., pars=...)`
    - Include:
        - The squashed population (active agents only)
        - The `results.R` matrix containing both pre-squash and live simulation values
        - The full parameter set in a `PropertySet`

3. Load function: implement a `load(path)` class method:
    - Call `LaserFrame.load_snapshot(path)` to retrieve:
        - Population frame
        - Results matrix
        - Parameters
    - Set `.capacity = .count` if not doing births, else set capacity based on projected population growth from count.
    - Reconstruct all components using `init_from_file()`

    !!! warning
        When modeling vital dynamics, especially births, there is an additional step needed to ensure consistency after loading:

        Property initialization for unborn individuals must be repeated if your model pre-assigns properties up to `.capacity`. For example, if timers or demographic attributes (like `date_of_birth`) are pre-initialized at `t=0`, you must ensure this initialization is re-applied after loading, because only the `.count` population is reloaded, not the future `.capacity`.

        Failing to do so may result in improperly initialized agents being birthed after the snapshot load, which can lead to subtle or catastrophic model errors.

4. Preserve EULA'd results:

    Use "+=" to track new recoveries alongside pre-squash R values. In `run()`, use additive updates so that pre-saved recovered agents are preserved:

    ```
    self.results.R[t, nid] += ((self.population.node_id == nid) &
                           (self.population.disease_state == 2)).sum()
    ```

    This ensures your output accounts for both squashed-away immunity and recoveries during the live simulation.




/// details | Code example: Add squashing and snapshot support to SIR models

```
import numpy as np
import click
import matplotlib.pyplot as plt
from pathlib import Path

from laser.core import LaserFrame, PropertySet

class Transmission:
    """
    A simple transmission component that spreads infection within each node.
    """
    def __init__(self, population, pars):
        self.population = population
        self.pars = pars

    def step(self):
    """
    For each node in the population, calculate the number of new infections as a function of:
    - the number of infected individuals,
    - the number of susceptibles,
    - adjustments for migration and seasonality,
    - and individual-level heterogeneity.

    Then, select new infections at random from among the susceptible individuals in each node,
    and initiate infection in those individuals.
    """
    pass  # Implementation omitted for documentation purposes

    @classmethod
    def init_from_file(cls, population, pars):
        return cls(population, pars)

class Progression:
    """
    A simple progression component that recovers infected individuals probabilistically.
    """
    def __init__(self, population, pars):
        self.population = population
        self.pars = pars

    def step(self):
    """
    At each time step, update the disease state of infected individuals based on the model's
    progression logic. This may be driven by probabilities, timers, or other intrahost dynamics.
    """
    pass  # Implementation omitted for documentation

    @classmethod
    def init_from_file(cls, population, pars):
        return cls(population, pars)

class RecoveredSquashModel:
    """
    A simple multi-node SIR model demonstrating use of LASER's squash and snapshot mechanisms.
    """
    def __init__(self, num_agents=100000, num_nodes=20, timesteps=365):
        self.num_agents = num_agents
        self.num_nodes = num_nodes
        self.timesteps = timesteps
        self.population = LaserFrame(capacity=num_agents, initial_count=num_agents)
        self.population.add_scalar_property("node_id", dtype=np.int32)
        self.population.add_scalar_property("disease_state", dtype=np.int8)  # 0=S, 1=I, 2=R

        self.results = LaserFrame(capacity=self.num_nodes)
        self.results.add_vector_property("S", length=timesteps, dtype=np.int32)
        self.results.add_vector_property("I", length=timesteps, dtype=np.int32)
        self.results.add_vector_property("R", length=timesteps, dtype=np.int32)

        self.pars = PropertySet({
            "r0": 2.5,
            "migration_k": 0.1,
            "seasonal_factor": 0.8,
            "transmission_prob": 0.2,
            "recovery_days": 14
        })

        self.components = [
            Transmission(self.population, self.pars),
            Progression(self.population, self.pars)
            # could add other components like vaccination
        ]

    def initialize(self):
        np.random.seed(42)
        self.population.node_id[:] = np.random.randint(0, self.num_nodes, size=self.num_agents)
        recovered = np.random.rand(self.num_agents) < 0.6
        self.population.disease_state[:] = np.where(recovered, 2, 0)

    def seed_infections(self):
        susceptible = self.population.disease_state == 0
        num_seed = max(1, int(0.001 * self.population.count))
        seed_indices = np.random.choice(np.where(susceptible)[0], size=num_seed, replace=False)
        self.population.disease_state[seed_indices] = 1

    def squash_recovered(self):
        """
        Removes all agents who are recovered (state 2).
        This reduces memory footprint and speeds up simulation.
        """
        keep = self.population.disease_state[:self.population.count] != 2
        self.population.squash(keep)

    def populate_results(self):
        """
        Populate initial R values before squashing to reflect the pre-squash immunity landscape.
        """
        for nid in range(self.num_nodes):
            initial_r = ((self.population.disease_state == 2) & (self.population.node_id == nid)).sum()
            decay = np.linspace(initial_r, initial_r * 0.9, self.timesteps, dtype=int)
            self.results.R[:, nid] = decay
        print("Initial R counts per node:", self.results.R[0, :])
        print("Total initial R (summed):", self.results.R[0, :].sum())

    def run(self):
        for t in range(self.timesteps):
            for component in self.components:
                component.step()
            for nid in range(self.num_nodes):
                self.results.S[t, nid] = ((self.population.node_id == nid) & (self.population.disease_state == 0)).sum()
                self.results.I[t, nid] = ((self.population.node_id == nid) & (self.population.disease_state == 1)).sum()
                self.results.R[t, nid] += ((self.population.node_id == nid) & (self.population.disease_state == 2)).sum()

    def save(self, path):
        """
        Save the current model state to an HDF5 file, including population frame,
        pre-squash results, and simulation parameters.
        """
        self.population.save_snapshot(path, results_r=self.results.R, pars=self.pars)

    @classmethod
    def load(cls, path):
        """
        Reload a model from an HDF5 snapshot. Note: reloaded population will have
        only post-squash agents (e.g., susceptibles and infected).
        """
        pop, results_r, pars = LaserFrame.load_snapshot(path)
        model = cls(num_agents=pop.capacity, num_nodes=results_r.shape[1], timesteps=results_r.shape[0])
        model.population = pop
        model.results.R[:, :] = results_r
        model.pars = PropertySet(pars)
        model.pars["transmission_prob"] /= 10  # example modification after reload
        model.components = [
            Transmission.init_from_file(model.population, model.pars),
            Progression.init_from_file(model.population, model.pars)
        ]
        return model

    def plot(self):
        """
        Plot the time series of total S, I, and R across all nodes.
        """
        # details omitted

@click.command()
@click.option("--init-pop-file", type=click.Path(), default=None, help="Path to snapshot to resume from.")
@click.option("--output", type=click.Path(), default="model_output.h5")
def main(init_pop_file, output):
    if init_pop_file:
        model = RecoveredSquashModel.load(init_pop_file)
        model.run()
        model.plot()
    else:
        model = RecoveredSquashModel()
        model.initialize()
        model.seed_infections()
        model.populate_results()
        model.squash_recovered()
        model.save(output)
        print(f"Initial population saved to {output}")

if __name__ == "__main__":
    main()
```
///

## Leverage AI

Once you have identified the slowest component, the easiest way to improve performance is by using ChatGPT. Try prompting with:

`"This code is much too slow. (My arrays are all about 1e6 or 1e7 in size.)"`

If your code consists mainly of for-loops without much NumPy, you can add:

`"Is there anything we can vectorize better with NumPy?"`

This approach can often transform a naive implementation into a highly optimized one.



## Implement unit tests

Instead of testing performance within the full simulation, consider building unit tests. This ensures correctness while optimizing for speed.

- Use AI to generate unit tests that validate output against a known correct (but slower) version.
- Include performance benchmarks in the tests.
- Ensure large array sizes (e.g., 1 million+ elements) to get meaningful speed comparisons.

<!-- would be nice if we have unit tests already built? Things that are more concrete to share? -->


## Optimize with NumPy and Numba

After achieving good performance with NumPy, consider trying Numba for further improvements.

Even if you’re new to Numba, ChatGPT can generate optimized solutions easily. Keep in mind:

- Numba moves back to explicit for-loops (unlike NumPy, which uses vectorization syntax).
- GPT’s first solution may use `range` instead of `prange`. Prompt it with:
    `"Can we parallelize this with prange?"`
- If your code involves common counters, atomic operations may become a bottleneck. Ask GPT about:
    `"Can we use thread-local storage to avoid atomic operations?"`
- Numba may be slower than NumPy for small arrays (e.g., thousands or tens of thousands of elements). Test with at least 1 million elements.


## C and OpenMP

If the best Numba solution still isn’t fast enough, consider compiled C.

- Use ctypes to call C functions from Python.
- Mention “use OpenMP” in AI prompts if parallelization is possible.
- Ask: `"Can you generate an OpenMP solution with the best pragmas?"`
- The more CPU cores available, the greater the potential speedup. That said, it’s usually a case of diminishing returns as one goes from 8 cores to 16 and to 32. Our research shows that often you’re better off running 4 sims across 8 cores each than running 1 sim on all 32 cores available. Also be aware that with both Numba and OpenMP you can constrain the number of cores used to less than the number available by setting the appropriate environment variable. (Numba environment variable = NUMBA_NUM_THREADS; OpenMP environment variable = OMP_NUM_THREADS)

## Additional advice

- Don’t duplicate. Sometimes reporting will duplicate transmission code and need to be combined.

- Never append. There may be cases where you are collecting information as it happens without knowing ahead of time how many rows/entries/elements you’ll need. This is easy in Python using list appending, for example, but that’s a performance killer. Really try to find a way to figure out ahead of time how many entries there will be, and then allocate memory for that, and insert into the existing row.

- Some components have long timescales, like mortality. By default you are probably going to end up doing most component steps every timestep. You can probably get away with doing mortality updates, for example, far less often. You can experiment with weekly, fortnightly or monthly updates, depending on the timescale of the component you’re optimizing. Just be sure to move everything forward by a week if you’re only doing the update every week. And expect “blocky” plots. Note that there are fancier solutions like ‘strided sharding’ (details omitted).

- When prompting AI, use questions rather than directives. For example:

    `"Do you think it might be better to...?"`

    This prevents oversteering the AI into sub-optimal solutions.
