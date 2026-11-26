import ast
import math
import random
from random import randint
import matplotlib.pyplot as plt

DIMENSIONS = 30
N_ITER = 10000*DIMENSIONS
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
    plt.semilogy(range(1, len(avg_progress) + 1), avg_progress)
    plt.xlabel("Iteration")
    plt.ylabel("Average objective value")
    plt.title("Average Local Search Progress")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    return avg_progress


class GeneticAlgorithm:
    def __init__(
        self,
        dimensions=DIMENSIONS,
        bits_per_dimension=16,
        lower_bound=-3.0,
        upper_bound=3.0,
        start_pop_size=100,
        representation="real",
    ):
        self.dimensions = dimensions
        self.bits_per_dimension = bits_per_dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.representation = representation
        self.start_pop_size = start_pop_size

        # Wspolny krok mutacji na poziomie algorytmu (brak sigm i thet w osobniku)
        self.base_sigma = (self.upper_bound - self.lower_bound) / 6.0
        self.sigma_decay = 0.99

        if self.representation not in {"gray", "real"}:
            raise ValueError("representation must be 'gray' or 'real'")

    def make_random_population(self):
        if self.representation == "gray":
            return [self.make_individual_gray() for _ in range(self.start_pop_size)]
        return [self.make_individual_real() for _ in range(self.start_pop_size)]

    def make_individual_gray(self):
        x = [random.randint(0, 1) for _ in range(self.dimensions * self.bits_per_dimension)]
        return {"x": x}

    def make_individual_real(self):
        span = self.upper_bound - self.lower_bound
        x = [self.lower_bound + random.random() * span for _ in range(self.dimensions)]
        return {"x": x}

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
        maximum = (1 << self.bits_per_dimension) - 1
        span = self.upper_bound - self.lower_bound
        for value in values:
            ratio = (value - self.lower_bound) / span
            integer_value = max(0, min(maximum, round(ratio * maximum)))
            bits.extend(self.int_to_gray_bits(integer_value))
        return bits

    def decode_solution(self, x):
        values = []
        maximum = (1 << self.bits_per_dimension) - 1
        span = self.upper_bound - self.lower_bound
        for i in range(self.dimensions):
            start = i * self.bits_per_dimension
            end = start + self.bits_per_dimension
            number = self.gray_bits_to_int(x[start:end])
            value = self.lower_bound + (number / maximum) * span
            values.append(value)
        return values

    def objective_value(self, individual):
        if self.representation == "gray":
            values = self.decode_solution(individual["x"])
        else:
            values = individual["x"]
        return sum(value * value for value in values)

    def selection(self, pop, scores, k=3):
        pop_len = len(pop)
        best_ix = random.randint(0, pop_len - 1)
        best_score = scores[best_ix]
        for _ in range(k - 1):
            ix = random.randint(0, pop_len - 1)
            score = scores[ix]
            if score < best_score:
                best_ix = ix
                best_score = score
        return pop[best_ix]

    def select_parents(self, pop, scores):
        return [self.selection(pop, scores) for _ in range(len(pop))]

    def clone_individual(self, individual):
        return {"x": individual["x"].copy()}

    def crossover(self, p1, p2, r_cross=0.9):
        c1 = self.clone_individual(p1)
        c2 = self.clone_individual(p2)
        if random.random() < r_cross and len(p1["x"]) > 2:
            pt = randint(1, len(p1["x"]) - 2)
            c1["x"] = p1["x"][:pt] + p2["x"][pt:]
            c2["x"] = p2["x"][:pt] + p1["x"][pt:]
        return [c1, c2]

    @staticmethod
    def _log_weights(count: int) -> list[float]:
        raw = [math.log(count + 1) - math.log(j) for j in range(1, count + 1)]
        total = sum(raw)
        return [w / total for w in raw]

    def weighted_recombination_real(self, sorted_parents, weights, current_sigma):
        # sorted_parents jest juz posortowana po ocenie - brak sortowania w petli po dzieciach
        parent_count = len(sorted_parents)
        child_x = []
        for d in range(self.dimensions):
            val = sum(weights[j] * sorted_parents[j]["x"][d] for j in range(parent_count))
            val += random.gauss(0.0, current_sigma)
            if val < self.lower_bound:
                val = self.lower_bound
            elif val > self.upper_bound:
                val = self.upper_bound
            child_x.append(val)
        return {"x": child_x}

    def mutation(self, individual, r_mut, current_sigma=None):
        if self.representation == "gray":
            # Szybka mutacja bitowa bez kodowania/dekodowania floatow
            bits = individual["x"]
            for idx in range(len(bits)):
                if random.random() < r_mut:
                    bits[idx] ^= 1
            return

        sigma = self.base_sigma if current_sigma is None else current_sigma
        x_next = []
        for val in individual["x"]:
            if random.random() < r_mut:
                val = val + random.gauss(0.0, sigma)
                if val < self.lower_bound:
                    val = self.lower_bound
                elif val > self.upper_bound:
                    val = self.upper_bound
            x_next.append(val)
        individual["x"] = x_next

    def run(self, n_iter=100, r_cross=0.9, r_mut=None, verbose=False, print_every=10):
        population = self.make_random_population()
        if r_mut is None:
            r_mut = (1.0 / len(population[0]["x"])) if self.representation == "gray" else (1.0 / self.dimensions)

        scores = [self.objective_value(ind) for ind in population]
        best_ix = min(range(len(scores)), key=scores.__getitem__)
        best = self.clone_individual(population[best_ix])
        best_score = scores[best_ix]
        history = []

        weights_cache = {}
        current_sigma = self.base_sigma

        for generation in range(1, n_iter + 1):
            gen_best_ix = min(range(len(scores)), key=scores.__getitem__)
            gen_best_score = scores[gen_best_ix]

            if gen_best_score < best_score:
                best = self.clone_individual(population[gen_best_ix])
                best_score = gen_best_score

            history.append(best_score)

            if verbose and (generation == 1 or generation % print_every == 0 or generation == n_iter):
                avg_score = sum(scores) / len(scores)
                print(
                    f"Generacja {generation:4d} | najlepszy: {best_score:.8f} | "
                    f"najlepszy_w_generacji: {gen_best_score:.8f} | sredni: {avg_score:.8f}"
                )

            if self.representation == "real":
                # Sortujemy wybrane jednostki raz na generacje + elita z najlepszym wynikiem
                selected = self.select_parents(population, scores)
                ranked_pairs = sorted(
                    zip(selected, [self.objective_value(ind) for ind in selected]),
                    key=lambda pair: pair[1],
                )
                # Bierzemy najlepsza polowe wybranych rodzicow dla lepszej zbieznosci i szybkosci
                top_k = max(2, len(ranked_pairs) // 2)
                ranked_parents = [pair[0] for pair in ranked_pairs[:top_k]]
                if top_k not in weights_cache:
                    weights_cache[top_k] = self._log_weights(top_k)
                weights = weights_cache[top_k]

                children = [self.clone_individual(best)]
                for _ in range(self.start_pop_size - 1):
                    children.append(self.weighted_recombination_real(ranked_parents, weights, current_sigma))
                current_sigma *= self.sigma_decay
            else:
                selected = self.select_parents(population, scores)
                children = [self.clone_individual(best)]
                for i in range(0, len(selected) - 1, 2):
                    for child in self.crossover(selected[i], selected[i + 1], r_cross):
                        self.mutation(child, r_mut)
                        children.append(child)
                        if len(children) >= self.start_pop_size:
                            break
                    if len(children) >= self.start_pop_size:
                        break

            population = children[: self.start_pop_size]
            scores = [self.objective_value(ind) for ind in population]

        best_values = self.decode_solution(best["x"]) if self.representation == "gray" else best["x"]
        return {
            "best_individual": best,
            "best_values": best_values,
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
        preview = [round(value, 6) for value in result["best_values"][: min(5, len(result["best_values"]))]]
        print(f"najlepszy wektor x (pierwsze {len(preview)}): {preview}")
        return result


def test_current_implementation():
    print("START TESTU IMPLEMENTACJI")
    for representation in ("gray", "real"):
        ga = GeneticAlgorithm(representation=representation)
        ga.test(n_iter=80)


def main():
    test_current_implementation()


if __name__ == "__main__":
    main()