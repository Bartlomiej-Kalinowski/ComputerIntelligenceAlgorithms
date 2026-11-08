import matplotlib.pyplot as plt
import ast
import random
import math
from random import randint

N_ITER = 10000
N_RUNS = 100

def draw(raw_data: list[str], objective_val, output_file: str = "average_local_search_progress.png"):
    runs = []
    for file in raw_data:
        values = []
        with open(file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                decoded = ast.literal_eval(line)
                values.append(objective_val(decoded))
        if values:
            runs.append(values)

    if not runs:
        return []

    avg_progress = [sum(step_vals) / len(step_vals) for step_vals in zip(*runs)]
    plt.semilogy(range(1, len(avg_progress) + 1), avg_progress)  # Logarytmiczna os Y
    plt.xlabel("Iteration")
    plt.ylabel("Average objective value")
    plt.title("Average Local Search Progress")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    return avg_progress



class GeneticAlgorithm:
    def __init__(self, dimensions=10, bits_per_dimension=16,
                 lower_bound=-3, upper_bound=3, start_pop_size = 100, representation="real"):
        self.dimensions = dimensions
        self.bits_per_dimension = bits_per_dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.representation = representation
        self.start_pop_size = start_pop_size
        if self.representation not in {"gray", "real"}:
            raise ValueError("representation must be 'gray' or 'real'")

    def make_random_population(self):
        if self.representation == "gray":
            return [self.make_individual_gray() for _ in range(self.start_pop_size)]
        return [self.make_individual_real() for _ in range(self.start_pop_size)]

    def make_individual_gray(self):
        x = [random.randint(0, 1) for _ in range(self.dimensions * self.bits_per_dimension)]
        sigma = [self.initial_sigma() for _ in range(self.dimensions)]
        return {"x": x, "sigma": sigma}

    def make_individual_real(self):
        bits =  [random.randint(0, 1) for _ in range(self.dimensions * self.bits_per_dimension)]
        x = self.decode_solution(bits)
        sigma = [self.initial_sigma() for _ in range(self.dimensions)]
        return {"x": x, "sigma": sigma}

    def initial_sigma(self):
        return (self.upper_bound - self.lower_bound) / 6

    @staticmethod
    def gray_bits_to_int(bits):
        value = 0
        binary_bit = 0
        for bit in bits:
            binary_bit ^= bit
            value = (value << 1) | binary_bit
        return value

    def int_to_gray_bits(self, value):
        gray_value = value ^ (value >> 1)
        bits = []
        for index in range(self.bits_per_dimension - 1, -1, -1):
            bits.append((gray_value >> index) & 1)
        return bits

    def encode_solution(self, values):
        bits = []
        maximum = 2 ** self.bits_per_dimension - 1
        for value in values:
            ratio = (value - self.lower_bound) / (self.upper_bound - self.lower_bound)
            integer_value = round(ratio * maximum)
            bits.extend(self.int_to_gray_bits(integer_value))
        return bits

    def decode_solution(self, x):
        values = []
        maximum = 2 ** self.bits_per_dimension - 1
        for i in range(self.dimensions):
            start = i * self.bits_per_dimension
            end = start + self.bits_per_dimension
            number = self.gray_bits_to_int(x[start:end])
            value = self.lower_bound + number / maximum * (self.upper_bound - self.lower_bound)
            values.append(value)
        return values

    def objective_value(self, individual):
        if self.representation == "gray":
            values = self.decode_solution(individual["x"])
        else:
            values = individual["x"]
        return sum(value * value for value in values)

    def selection(self, pop, scores, k=3):
        selection_ix = random.randint(0, len(pop) - 1)

        for _ in range(k - 1):
            ix = random.randint(0, len(pop) - 1)
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix

        return pop[selection_ix]

    def select_parents(self, pop, scores):
        selected = [self.selection(pop, scores) for _ in range(len(pop))]
        return selected

    def clone_individual(self, individual):
        return {"x": individual["x"].copy(), "sigma": individual["sigma"].copy()}

    def crossover(self, p1, p2, r_cross=0.9):
        c1 = self.clone_individual(p1)
        c2 = self.clone_individual(p2)
        if random.random() < r_cross:
            # one point crossover
            if len(p1["x"]) > 2:
                pt = randint(1, len(p1["x"]) - 2)
                c1["x"] = p1["x"][:pt] + p2["x"][pt:]
                c2["x"] = p2["x"][:pt] + p1["x"][pt:]
            if len(p1["sigma"]) > 1:
                spt = randint(1, len(p1["sigma"]) - 1)
                c1["sigma"] = p1["sigma"][:spt] + p2["sigma"][spt:]
                c2["sigma"] = p2["sigma"][:spt] + p1["sigma"][spt:]
        return [c1, c2]

    def weighted_recombination_real(self, parents):
        if not parents:
            raise ValueError("weighted_recombination_real requires at least one parent")

        ranked_parents = sorted(parents, key=self.objective_value)
        parent_count = len(ranked_parents)
        weights = [math.log(parent_count + 1) - math.log(j) for j in range(1, parent_count + 1)]
        weight_sum = sum(weights)
        normalized_weights = [weight / weight_sum for weight in weights]
        mu_eff = 1.0 / sum(weight * weight for weight in normalized_weights)

        # Global and local learning rates for sigma self-adaptation in ES.
        new_learning_rate = 1 / math.sqrt(2 * mu_eff * self.dimensions)
        old_learning_rate = 1 / math.sqrt(2 * math.sqrt(mu_eff * self.dimensions))

        min_sigma = 1e-8
        max_sigma = self.upper_bound - self.lower_bound
        global_noise = random.gauss(0, 1)

        child_x = []
        child_sigma = []
        for i in range(self.dimensions):
            sigma_weighted = (
                sum(weights[j] * ranked_parents[j]["sigma"][i] for j in range(parent_count))
                / weight_sum
            )
            local_sigma_noise = random.gauss(0, 1)
            sigma_i = sigma_weighted * math.exp(
                new_learning_rate * global_noise + old_learning_rate * local_sigma_noise
            )
            if sigma_i < min_sigma:
                sigma_i = min_sigma
            if sigma_i > max_sigma:
                sigma_i = max_sigma

            x_weighted = (
                sum(weights[j] * ranked_parents[j]["x"][i] for j in range(parent_count))
                / weight_sum
            )
            x_i = x_weighted + sigma_i * random.gauss(0, 1)
            if x_i < self.lower_bound:
                x_i = self.lower_bound
            if x_i > self.upper_bound:
                x_i = self.upper_bound

            child_sigma.append(sigma_i)
            child_x.append(x_i)

        return {"x": child_x, "sigma": child_sigma}

    def mutation(self, individual, r_mut):
        tau = 1 / (self.dimensions ** 0.5)
        min_sigma = 1e-8
        max_sigma = self.upper_bound - self.lower_bound
        if self.representation == "gray":
            x_values = self.decode_solution(individual["x"])
            changed = False
            for i in range(self.dimensions):
                if random.random() < r_mut:
                    changed = True
                    individual["sigma"][i] = individual["sigma"][i] * math.exp(tau * random.gauss(0, 1))
                    if individual["sigma"][i] < min_sigma:
                        individual["sigma"][i] = min_sigma
                    if individual["sigma"][i] > max_sigma:
                        individual["sigma"][i] = max_sigma

                    x_values[i] += random.gauss(0, individual["sigma"][i])
                    if x_values[i] < self.lower_bound:
                        x_values[i] = self.lower_bound
                    if x_values[i] > self.upper_bound:
                        x_values[i] = self.upper_bound
            if changed:
                individual["x"] = self.encode_solution(x_values)
            return

        for i in range(self.dimensions):
            if random.random() < r_mut:
                individual["sigma"][i] = individual["sigma"][i] * math.exp(tau * random.gauss(0, 1))
                if individual["sigma"][i] < min_sigma:
                    individual["sigma"][i] = min_sigma
                if individual["sigma"][i] > max_sigma:
                    individual["sigma"][i] = max_sigma

                individual["x"][i] += random.gauss(0, individual["sigma"][i])
                if individual["x"][i] < self.lower_bound:
                    individual["x"][i] = self.lower_bound
                if individual["x"][i] > self.upper_bound:
                    individual["x"][i] = self.upper_bound

    def run(self, n_iter=100, r_cross=0.9, r_mut=None, verbose=False, print_every=10):
        population = self.make_random_population()
        if r_mut is None:
            r_mut = 1 / self.dimensions

        best = self.clone_individual(population[0])
        best_score = self.objective_value(best)
        history = []

        for generation in range(1, n_iter + 1):
            scores = [self.objective_value(individual) for individual in population]
            generation_best_ix = min(range(len(scores)), key=lambda ix: scores[ix])
            generation_best = population[generation_best_ix]
            generation_best_score = scores[generation_best_ix]

            if generation_best_score < best_score:
                best = self.clone_individual(generation_best)
                best_score = generation_best_score

            history.append(best_score)

            if verbose and (generation == 1 or generation % print_every == 0 or generation == n_iter):
                avg_score = sum(scores) / len(scores)
                print(
                    f"Generacja {generation:4d} | najlepszy: {best_score:.8f} | "
                    f"najlepszy_w_generacji: {generation_best_score:.8f} | sredni: {avg_score:.8f}"
                )

            selected = self.select_parents(population, scores)
            children = []
            if self.representation == "real":
                for _ in range(self.start_pop_size):
                    children.append(self.weighted_recombination_real(selected))
            else:
                for i in range(0, len(selected), 2):
                    p1 = selected[i]
                    p2 = selected[(i + 1) % len(selected)]
                    for child in self.crossover(p1, p2, r_cross):
                        self.mutation(child, r_mut)
                        children.append(child)
            population = children[:self.start_pop_size]

        if self.representation == "gray":
            best_values = self.decode_solution(best["x"])
        else:
            best_values = best["x"]

        return {
            "best_individual": best,
            "best_values": best_values,
            "best_sigma": best["sigma"],
            "best_score": best_score,
            "history": history,
        }

    def test(self, n_iter=80):
        print("=" * 72)
        print("TEST ALGORYTMU GENETYCZNEGO")
        print(
            f"reprezentacja={self.representation}, wymiary={self.dimensions}, "
            f"bity_na_wymiar={self.bits_per_dimension}, populacja={self.start_pop_size}"
        )
        sample_bits = self.make_individual_gray()["x"]
        sample_decoded = self.decode_solution(sample_bits)
        sample_encoded = self.encode_solution(sample_decoded)
        print(f"test kodowanie/dekodowanie gray: {'OK' if sample_bits == sample_encoded else 'NIEZGODNE'}")

        result = self.run(
            n_iter=n_iter,
            r_cross=0.9,
            r_mut=None,
            verbose=True,
            print_every=max(1, n_iter // 8),
        )
        print(f"koniec testu | najlepszy wynik={result['best_score']:.8f}")
        preview = [round(value, 6) for value in result["best_values"][:min(5, len(result["best_values"]))]]
        sigma_preview = [round(value, 6) for value in result["best_sigma"][:min(5, len(result["best_sigma"]))]]
        print(f"najlepszy wektor x (pierwsze {len(preview)}): {preview}")
        print(f"najlepszy wektor sigma (pierwsze {len(sigma_preview)}): {sigma_preview}")
        return result

def test_current_implementation():
    print("START TESTU IMPLEMENTACJI")
    for representation in ("gray", "real"):
        ga = GeneticAlgorithm(representation=representation)
        ga.test(n_iter=80)

def main():
    test_current_implementation()

if __name__ == '__main__':
    main()
