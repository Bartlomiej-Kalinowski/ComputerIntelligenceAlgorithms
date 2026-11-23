import matplotlib.pyplot as plt
import ast
import random
import math
from random import randint

DIMENSIONS=3
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
    def __init__(self, dimensions=DIMENSIONS, bits_per_dimension=16,
                 lower_bound=-3, upper_bound=3, start_pop_size = 100, representation="real"):
        self.dimensions = dimensions
        self.bits_per_dimension = bits_per_dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.representation = representation
        self.start_pop_size = start_pop_size
        self.beta_theta = math.radians(5.0)
        self.theta_pairs = [(i, j) for i in range(self.dimensions - 1) for j in range(i + 1, self.dimensions)]
        self.theta_count = len(self.theta_pairs)
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
        theta = [0.0 for _ in range(self.theta_count)]
        return {"x": x, "sigma": sigma, "theta": theta}

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
        cloned = {"x": individual["x"].copy(), "sigma": individual["sigma"].copy()}
        if "theta" in individual:
            cloned["theta"] = individual["theta"].copy()
        return cloned

    @staticmethod
    def _wrap_angle(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _build_rotation_matrix(self, theta_values):
        if len(theta_values) != self.theta_count:
            raise ValueError("theta vector length does not match dimensions")
        matrix = [
            [1.0 if row == col else 0.0 for col in range(self.dimensions)]
            for row in range(self.dimensions)
        ]
        for angle, (i, j) in zip(theta_values, self.theta_pairs):
            cosine = math.cos(angle)
            sine = math.sin(angle)
            row_i = matrix[i][:]
            row_j = matrix[j][:]
            for col in range(self.dimensions):
                matrix[i][col] = cosine * row_i[col] - sine * row_j[col]
                matrix[j][col] = sine * row_i[col] + cosine * row_j[col]
        return matrix

    def _apply_matrix_to_vector(self, matrix, vector):
        return [
            sum(matrix[row][col] * vector[col] for col in range(self.dimensions))
            for row in range(self.dimensions)
        ]

    def _mutate_real_with_rotations(self, individual):
        tau_prime = 1 / math.sqrt(2 * self.dimensions)
        tau = 1 / math.sqrt(2 * math.sqrt(self.dimensions))
        min_sigma = 1e-8
        max_sigma = self.upper_bound - self.lower_bound

        global_noise = random.gauss(0, 1)
        sigma_prime = []
        for sigma_i in individual["sigma"]:
            local_noise = random.gauss(0, 1)
            sigma_new = sigma_i * math.exp(tau_prime * global_noise + tau * local_noise)
            if sigma_new < min_sigma:
                sigma_new = min_sigma
            if sigma_new > max_sigma:
                sigma_new = max_sigma
            sigma_prime.append(sigma_new)

        previous_theta = individual.get("theta", [0.0 for _ in range(self.theta_count)])
        theta_prime = []
        for theta_j in previous_theta:
            theta_new = self._wrap_angle(theta_j + self.beta_theta * random.gauss(0, 1))
            theta_prime.append(theta_new)

        rotation_matrix = self._build_rotation_matrix(theta_prime)
        sigma_correlated = self._apply_matrix_to_vector(rotation_matrix, sigma_prime)
        gaussian_step = [random.gauss(0, 1) for _ in range(self.dimensions)]

        x_next = []
        for i in range(self.dimensions):
            x_i = individual["x"][i] + gaussian_step[i] * sigma_correlated[i]
            if x_i < self.lower_bound:
                x_i = self.lower_bound
            if x_i > self.upper_bound:
                x_i = self.upper_bound
            x_next.append(x_i)

        individual["x"] = x_next
        individual["sigma"] = sigma_prime
        individual["theta"] = theta_prime

    def print_rotation_matrix_example_10d(self):
        if self.representation != "real" or self.dimensions != 10:
            return
        theta_example = [self._wrap_angle(self.beta_theta * random.gauss(0, 1)) for _ in range(self.theta_count)]
        rotation_matrix = self._build_rotation_matrix(theta_example)
        print("Przykladowa macierz R(theta) dla 10 wymiarow:")
        for row in rotation_matrix:
            print(" ".join(f"{value: .5f}" for value in row))

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

        child_x = []
        child_sigma = []
        for i in range(self.dimensions):
            child_sigma.append(
                sum(normalized_weights[j] * ranked_parents[j]["sigma"][i] for j in range(parent_count))
            )
            child_x.append(
                sum(normalized_weights[j] * ranked_parents[j]["x"][i] for j in range(parent_count))
            )

        child_theta = []
        for theta_index in range(self.theta_count):
            weighted_sin = sum(
                normalized_weights[j] * math.sin(ranked_parents[j].get("theta", [0.0] * self.theta_count)[theta_index])
                for j in range(parent_count)
            )
            weighted_cos = sum(
                normalized_weights[j] * math.cos(ranked_parents[j].get("theta", [0.0] * self.theta_count)[theta_index])
                for j in range(parent_count)
            )
            child_theta.append(math.atan2(weighted_sin, weighted_cos))

        child = {"x": child_x, "sigma": child_sigma, "theta": child_theta}
        self._mutate_real_with_rotations(child)
        return child

    def mutation(self, individual, r_mut):
        if self.representation == "gray":
            tau = 1 / (self.dimensions ** 0.5)
            min_sigma = 1e-8
            max_sigma = self.upper_bound - self.lower_bound
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

        if random.random() < r_mut:
            self._mutate_real_with_rotations(individual)

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
        if self.representation == "real" and self.dimensions == 10:
            self.print_rotation_matrix_example_10d()
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
        if self.representation == "real":
            theta_preview = [
                round(value, 6)
                for value in result["best_individual"].get("theta", [])[:min(5, len(result["best_individual"].get("theta", [])))]
            ]
            print(f"najlepszy wektor theta (pierwsze {len(theta_preview)}): {theta_preview}")
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
