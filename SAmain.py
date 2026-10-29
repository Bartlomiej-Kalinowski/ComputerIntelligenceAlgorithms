## history do przygotowania wykresow i danych w excel

import math
import random
import matplotlib.pyplot as plt


def function_1(x):
    suma = 0
    for value in x:
        suma += value * value

    e = math.exp(-5 / (1 + suma))
    cot = 1 / math.tan(e)
    return -5 / (1 + suma) + math.sin(cot)


def function_2(x):
    suma_kwadratow = 0
    suma_cos = 0

    for value in x:
        suma_kwadratow += value * value
        suma_cos += math.cos(2 * math.pi * value)

    n = len(x)
    return (-20 * math.exp(-0.2 * math.sqrt(suma_kwadratow / n))
            - math.exp(suma_cos / n) + 20 + math.e)

def gray_coding_neighbor(x):
    bit_index = random.randint(0, len(x) - 1)
    neighbor = x.copy()
    neighbor[bit_index] = 1 - neighbor[bit_index]
    return neighbor


# def gray_coding_neighbor(x, m):
#     neighbor = x.copy()
#     for j in range(len(x)):
#         if random.uniform(0,1) < m/16:
#             neighbor[j] = 1 - neighbor[j]
#     return neighbor


def real_neighbor(x, lower_bound, upper_bound, step):
    neighbor = x.copy()
    index = random.randint(0, len(x) - 1)
    neighbor[index] += random.gauss(0, step)

    if neighbor[index] < lower_bound:
        neighbor[index] = lower_bound
    if neighbor[index] > upper_bound:
        neighbor[index] = upper_bound

    return neighbor


class SimulatedAnnealing:
    def __init__(self, dimensions=10, bits_per_dimension=16,
                 lower_bound=-3, upper_bound=3, representation="real"):
        self.dimensions = dimensions
        self.bits_per_dimension = bits_per_dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.representation = representation

    def gray_bits_to_int(self, bits):
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

    def simulated_annealing(self, objective_value, limit=10000,
                            starting_temperature=10, final_temperature=0.01,
                            step=1):
        x = [self.upper_bound for i in range(self.dimensions)]


        if self.representation == "gray":
            value = objective_value(x)
            x = self.encode_solution(x)
        else:
            value = objective_value(x)

        best_value = value
        best_x = x.copy()
        history = [best_value]

        for i in range(1, limit):
            temperature = starting_temperature * (final_temperature / starting_temperature) ** (i / (limit - 1))

            if self.representation == "gray":
                x_prim = gray_coding_neighbor(x)
                new_value = objective_value(self.decode_solution(x_prim))
            else:
                x_prim = real_neighbor(x, self.lower_bound, self.upper_bound, step)
                new_value = objective_value(x_prim)

            difference = new_value - value

            if difference <= 0 or random.uniform(0, 1) < math.exp(-difference / temperature):
                x = x_prim
                value = new_value

            if value < best_value:
                best_value = value
                best_x = x.copy()

            history.append(best_value)

        if self.representation == "gray":
            best_x = self.decode_solution(best_x)

        return history, best_value, best_x


def experiment(objective_value, lower_bound, upper_bound, representation,
               repetitions=100, counter=10000, step=1):
    sa = SimulatedAnnealing(10, 16, lower_bound, upper_bound, representation)
    all_runs = []
    best_results = []
    best_points = []
    averages = [0] * counter

    for run in range(repetitions):
        history, best_value, best_x = sa.simulated_annealing(
            objective_value, limit=counter, step=step)
        all_runs.append(history)
        best_results.append(best_value)
        best_points.append(best_x)

        for i in range(counter):
            averages[i] += history[i]

        print("Powtorzenie", run + 1, "/", repetitions,
              "- najlepsza wartosc:", round(best_value, 7))

    for i in range(counter):
        averages[i] /= repetitions

    return all_runs, averages, best_results, best_points


def draw(averages, title):
    plt.figure()
    plt.plot(range(1, len(averages) + 1), averages)
    plt.title(title)
    plt.xlabel("Numer wywolania funkcji oceny")
    plt.ylabel("Srednia najlepsza wartosc funkcji")
    plt.grid(True)
    plt.show()



def main():
    results = {}

    tests = [
        ("F1", function_1, -3, 3, 0.3),
        ("F2", function_2, -32.768, 32.768, 2.0)
    ]

    for name, function, lower, upper, step in tests:
        for representation in ["real", "gray"]:
            print("\nEksperyment:", name, representation)
            all_runs, averages, best_results, best_points = experiment(
                function, lower, upper, representation,
                repetitions=100, counter=10000, step=step)

            results[(name, representation)] = {
                "raw": all_runs,           # 100 list po 10000 wynikow
                "average": averages,       # 10000 usrednionych wynikow
                "best_values": best_results,
                "best_points": best_points
            }

            print("Srednia wartosc koncowa:", sum(best_results) / len(best_results))
            print("Najlepsza wartosc:", min(best_results))
            draw(averages, name + " - " + representation)

    return results


if __name__ == "__main__":
    results = main()
