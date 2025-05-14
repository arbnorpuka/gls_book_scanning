<table border="0">
 <tr>
    <td><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/University_of_Prishtina_logo.svg/1200px-University_of_Prishtina_logo.svg.png" width="150" alt="University Logo" /></td>
    <td>
      <p>University of Prishtina</p>
      <p>Faculty of Electronic and Computer Engineering</p>
      <p>Master's Program</p>
      <p>Professor: Prof. Dr. Kadri Sylejmani</p>
      <p>Topic: Algorithms Inspired by Nature</p>
    </td>
 </tr>
</table>

# Guided Local Search for Book Scanning Optimization

This repository includes an advanced implementation of **Guided Local Search (GLS)** applied to the **Google Hash Code Book Scanning problem**. The algorithm builds on classic local search by dynamically modifying the cost landscape with feature penalties, making it easier to escape local optima and find better solutions over time.

---

## Problem Summary

The problem involves scheduling libraries to sign up and scan books within a limited number of days. Each library has a:

* **Sign-up time** (number of days before it starts scanning)
* **Scanning rate** (number of books it can scan per day)
* **Book collection** with individual scores

The goal is to **maximize the total score** of distinct books scanned before the deadline.

---

## Algorithm: Guided Local Search (GLS)

GLS is a metaheuristic that augments the standard objective function with a penalty term, aiming to escape local optima by discouraging overused components ("features"). It follows the logic of **Algorithm 113: Guided Local Search** from *Essentials of Metaheuristics* by Sean Luke (2014).

### Key Concepts

1. **Augmented Objective Function**:
   GLS modifies the original objective `f(s)` with penalties:

   $f'(s) = f(s) + \lambda \sum_{i \in Features(s)} p_i \cdot c_i$

   Where:

    * `\lambda` is the penalty weight
    * `p_i` is the penalty for feature `i`
    * `c_i` is the cost contribution of feature `i`

2. **Feature Utility**:
   Features are evaluated by how much they contribute to the cost:

   $utility(i) = \frac{c_i}{1 + p_i}$

   The features with the highest utility are penalized.

3. **Search Flow**:

    * Initialize solution greedily (e.g., GRASP)
    * Repeat:

        * Apply local search on the augmented cost function
        * Penalize high-utility features
        * Optionally restart if stuck (stagnation)

---

## GLS Applied to Book Scanning

### Feature Definition

* Features: Library IDs in the solution
* Cost: Based on library score contribution and usage
* Penalties: Applied to overused libraries

### Penalization Strategy

* Track penalties `p[i]` per library
* Compute utility:

  ```
  utility[i] = component_score[i] / (1 + penalty[i])
  ```
* Penalize libraries with highest utility after each local search iteration

### Objective Function Augmentation

Our implementation *minimizes* a modified version:

```python
penalized_score = original_score - lambda * sum(penalty[i] for i in solution)
```

This discourages reusing libraries that were overexploited in earlier solutions.

---

## Local Search

### Neighborhood Operators

We implement lightweight tweak operators including:

* `swap_last_book_light` — swaps the last book in a library
* `swap_signed_light` — switches the order of two signed libraries
* `insert_library_light` — inserts an unsigned library
* `swap_signed_with_unsigned_light` — exchanges signed/unsigned libraries
* `swap_same_books_light` — swaps libraries with similar book sets
* `swap_neighbor_libraries_light` — swaps adjacent libraries

Operators are selected probabilistically, weighted by historical effectiveness.

### Adaptive Operator Selection

* Tracks the success rate of each tweak
* Weights updated using reward signals (improvement score)

---

## Stagnation and Restarts

To prevent search stagnation:

* Penalization acts as a soft diversification method
* If no improvement after N rounds:

    * Reset penalties
    * Reinitialize with a new greedy solution

---

## Implementation Details

### Function: `guided_local_search(data, max_time, max_iterations)`

* `data`: Input problem instance
* `max_time`: Wall-clock runtime limit in seconds
* `max_iterations`: Number of GLS meta-iterations

### Core Structures

* `Solution` object: Signed/unsigned libraries, scanned books
* `penalty vector (p)`: `library_id → penalty count`
* `component_utilities`: Caches utility per feature

### Execution Loop

```python
for iteration in range(max_iterations):
    local_search()
    penalize_high_utility_features()
    update_best_solution()
```

---

## How to Run

```python
from solver import Solver

solver = Solver()
result = solver.guided_local_search(data, max_time=600)
print("Final Score:", result.fitness_score)
```

To process all input instances:

```bash
python app.py
```

---

## Performance Considerations

* Penalization adds mild overhead, but greatly improves diversity
* Operators avoid full recomputation (delta-based scoring)
* Penalization and tweak selection scale well with instance size

---

## References

* Luke, S. (2014). *Essentials of Metaheuristics (2nd ed.)*, Chapter 8.4

    * Algorithm 113: Guided Local Search
    * [http://cs.gmu.edu/\~sean/book/metaheuristics/](http://cs.gmu.edu/~sean/book/metaheuristics/)

* Algorithms Inspired by Nature 2025 repository:

    * [https://github.com/ArianitHalimi/AIN\_25](https://github.com/ArianitHalimi/AIN_25)

