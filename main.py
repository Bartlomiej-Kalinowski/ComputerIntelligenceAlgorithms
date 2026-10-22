import datetime
import ast
import random
import time

import matplotlib.pyplot as plt
from tqdm import tqdm
from time import asctime

def gray_coding_neighbor_simple(x):
    bit_index = random.randint(0, len(x) - 1)
    neighbor = x.copy()
    neighbor[bit_index] = 1 - neighbor[bit_index]
    return neighbor

def gray_coding_neighbor_random(x, m):
    neighbor = x.copy()
    for j in range(len(x)):
        if random.uniform(0,1) < m/16:
            neighbor[j] = 1 - neighbor[j]
    return neighbor

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






class BestImprovementLocalSearch:
    def __init__(
            self,
            dimensions: int,
            bits_per_dimension_: int = 16,
            lower_bound: float = -10.0,
            upper_bound: float = 10.0,
    ):
        self.dimensions = dimensions
        self.bits_per_dimension = bits_per_dimension_
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.x = [0] * self.dimensions * self.bits_per_dimension

    def gray_bits_to_int(self, bits: list[int]) -> int:
        value = 0
        binary_bit = 0
        for bit in bits:
            binary_bit ^= bit
            value = (value << 1) | binary_bit
        return value

    def int_to_gray_bits(self, value: int) -> list[int]:
        gray_value = value ^ (value >> 1)
        return [
            (gray_value >> bit_index) & 1
            for bit_index in range(self.bits_per_dimension - 1, -1, -1)
        ]

    def encode_dimension(self, value: float) -> list[int]:
        if value < self.lower_bound or value > self.upper_bound:
            raise ValueError(
                f"value {value} is outside bounds [{self.lower_bound}, {self.upper_bound}]"
            )

        max_integer = (1 << self.bits_per_dimension) - 1
        if max_integer == 0:
            return []

        ratio = (value - self.lower_bound) / (self.upper_bound - self.lower_bound)
        integer_value = round(ratio * max_integer)
        return self.int_to_gray_bits(integer_value)

    def encode_solution(self, values: list[float]) -> list[int]:
        if len(values) != self.dimensions:
            raise ValueError("starting_point length does not match dimensions")

        bits = []
        for value in values:
            bits.extend(self.encode_dimension(value))
        return bits

    def decode_dimension(self, bits: list[int]) -> float:
        integer_value = self.gray_bits_to_int(bits)
        max_integer = (1 << self.bits_per_dimension) - 1

        if max_integer == 0:
            return self.lower_bound

        ratio = integer_value / max_integer
        return self.lower_bound + ratio * (self.upper_bound - self.lower_bound)

    def decode_solution(self, x: list[int]) -> list[float]:
        if len(x) != self.dimensions * self.bits_per_dimension:
            print(len(x), self.dimensions * self.bits_per_dimension)
            raise ValueError("solution length does not match dimensions * bits_per_dimension")

        values = []
        for dimension in range(self.dimensions):
            start = dimension * self.bits_per_dimension
            end = start + self.bits_per_dimension
            values.append(self.decode_dimension(x[start:end]))
        return values

    def objective_value(self, x: list[int]) -> float:
        decoded_x = self.decode_solution(x)
        return sum(value * value for value in decoded_x)

    def objective_function(self, x: list[int], x_prim: list[int]) -> bool:
        """returns True if x_prim is better than x else False"""
        return self.objective_value(x_prim) < self.objective_value(x)

    def neighbor_generator(self, x: list, gen_type):
        """generates a neighbor of x suing gen_type neighborhood generator.
         If there is no more neighbor, returns None"""
        return gen_type(x, 1)

    def local_search(
            self,
            counter: int,
            starting_point: list = None,
            no_improvement_limit: int = 100,
            show_progress: bool = True,
    ):
        if starting_point is not None:
            x = self.encode_solution(starting_point)
        else:
            x = self.x.copy()

        if len(x) != self.dimensions * self.bits_per_dimension:
            raise ValueError("solution length does not match dimensions * bits_per_dimension")

        i = 0
        no_improvement_calls = 0

        pbar = tqdm(total=counter, desc="Local search", disable=not show_progress)
        try:
            while i < counter and no_improvement_calls < no_improvement_limit:
                x_prim = self.neighbor_generator(x, gray_coding_neighbor_random)
                i += 1
                pbar.update(1)

                if self.objective_function(x, x_prim):
                    x = x_prim
                    self.x = x
                    no_improvement_calls = 0
                else:
                    no_improvement_calls += 1
                yield x
        finally:
            pbar.close()


def main():
    ls_2d = BestImprovementLocalSearch(10)
    filename = "raw_"
    name = []
    for i in tqdm(range(100), desc="Generating runs"):
        name.append(filename + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{i}.txt")
        with open(name[i], "w") as f:
            for x_prim in ls_2d.local_search(
                    10000,
                    [10.0 for i in range(10)],
                    no_improvement_limit=10000,
                    show_progress=False,
            ):
                f.write(str(x_prim))
                f.write("\n")

    draw(name, ls_2d.objective_value)

if __name__ == '__main__':
    main()

# import datetime
# import ast
# import random
# import time

# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from time import asctime

# def gray_coding_neighbor_simple(x):
#     bit_index = random.randint(0, len(x) - 1)
#     neighbor = x.copy()
#     neighbor[bit_index] = 1 - neighbor[bit_index]
#     return neighbor

# def gray_coding_neighbor_random(x, m):
#     neighbor = x.copy()
#     for j in range(len(x)):
#         if random.uniform(0,1) < m/16:
#             neighbor[j] = 1 - neighbor[j]
#     return neighbor

# def draw(raw_data: list[str], objective_val, output_file: str = "average_local_search_progress.png"):
#     runs = []
#     for file in raw_data:
#         values = []
#         with open(file, "r") as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 decoded = ast.literal_eval(line)
#                 values.append(objective_val(decoded))
#         if values:
#             runs.append(values)

#     if not runs:
#         return []

#     avg_progress = [sum(step_vals) / len(step_vals) for step_vals in zip(*runs)]
#     plt.plot(range(1, len(avg_progress) + 1), avg_progress)
#     plt.xlabel("Iteration")
#     plt.ylabel("Average objective value")
#     plt.title("Average Local Search Progress")
#     plt.grid(True, alpha=0.3)
#     plt.savefig(output_file, dpi=150, bbox_inches="tight")
#     plt.close()
#     return avg_progress






# class BestImprovementLocalSearch:
#     def __init__(
#             self,
#             dimensions: int,
#             bits_per_dimension_: int = 16,
#             lower_bound: float = -10.0,
#             upper_bound: float = 10.0,
#     ):
#         self.dimensions = dimensions
#         self.bits_per_dimension = bits_per_dimension_
#         self.lower_bound = lower_bound
#         self.upper_bound = upper_bound
#         self.x = [0] * self.dimensions * self.bits_per_dimension

#     def gray_bits_to_int(self, bits: list[int]) -> int:
#         value = 0
#         binary_bit = 0
#         for bit in bits:
#             binary_bit ^= bit
#             value = (value << 1) | binary_bit
#         return value

#     def int_to_gray_bits(self, value: int) -> list[int]:
#         gray_value = value ^ (value >> 1)
#         return [
#             (gray_value >> bit_index) & 1
#             for bit_index in range(self.bits_per_dimension - 1, -1, -1)
#         ]

#     def encode_dimension(self, value: float) -> list[int]:
#         if value < self.lower_bound or value > self.upper_bound:
#             raise ValueError(
#                 f"value {value} is outside bounds [{self.lower_bound}, {self.upper_bound}]"
#             )

#         max_integer = (1 << self.bits_per_dimension) - 1
#         if max_integer == 0:
#             return []

#         ratio = (value - self.lower_bound) / (self.upper_bound - self.lower_bound)
#         integer_value = round(ratio * max_integer)
#         return self.int_to_gray_bits(integer_value)

#     def encode_solution(self, values: list[float]) -> list[int]:
#         if len(values) != self.dimensions:
#             raise ValueError("starting_point length does not match dimensions")

#         bits = []
#         for value in values:
#             bits.extend(self.encode_dimension(value))
#         return bits

#     def decode_dimension(self, bits: list[int]) -> float:
#         integer_value = self.gray_bits_to_int(bits)
#         max_integer = (1 << self.bits_per_dimension) - 1

#         if max_integer == 0:
#             return self.lower_bound

#         ratio = integer_value / max_integer
#         return self.lower_bound + ratio * (self.upper_bound - self.lower_bound)

#     def decode_solution(self, x: list[int]) -> list[float]:
#         if len(x) != self.dimensions * self.bits_per_dimension:
#             print(len(x), self.dimensions * self.bits_per_dimension)
#             raise ValueError("solution length does not match dimensions * bits_per_dimension")

#         values = []
#         for dimension in range(self.dimensions):
#             start = dimension * self.bits_per_dimension
#             end = start + self.bits_per_dimension
#             values.append(self.decode_dimension(x[start:end]))
#         return values

#     def objective_value(self, x: list[int]) -> float:
#         decoded_x = self.decode_solution(x)
#         return sum(value * value for value in decoded_x)

#     def objective_function(self, x: list[int], x_prim: list[int]) -> bool:
#         """returns True if x_prim is better than x else False"""
#         return self.objective_value(x_prim) < self.objective_value(x)

#     def neighbor_generator(self, x: list, gen_type):
#         """generates a neighbor of x suing gen_type neighborhood generator.
#          If there is no more neighbor, returns None"""
#         return gen_type(x, 1)

#     def local_search(
#             self,
#             counter: int,
#             starting_point: list = None,
#             no_improvement_limit: int = 100,
#             show_progress: bool = True,
#     ):
#         if starting_point is not None:
#             x = self.encode_solution(starting_point)
#         else:
#             x = self.x.copy()

#         if len(x) != self.dimensions * self.bits_per_dimension:
#             raise ValueError("solution length does not match dimensions * bits_per_dimension")

#         i = 0
#         no_improvement_calls = 0

#         pbar = tqdm(total=counter, desc="Local search", disable=not show_progress)
#         try:
#             while i < counter and no_improvement_calls < no_improvement_limit:
#                 x_prim = self.neighbor_generator(x, gray_coding_neighbor_random)
#                 i += 1
#                 pbar.update(1)

#                 if self.objective_function(x, x_prim):
#                     x = x_prim
#                     self.x = x
#                     no_improvement_calls = 0
#                 else:
#                     no_improvement_calls += 1
#                 yield x
#         finally:
#             pbar.close()


# def main():
#     ls_2d = BestImprovementLocalSearch(10)
#     filename = "raw_"
#     name = []
#     for i in tqdm(range(100), desc="Generating runs"):
#         name.append(filename + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{i}.txt")
#         with open(name[i], "w") as f:
#             for x_prim in ls_2d.local_search(
#                     10000,
#                     [10.0 for i in range(10)],
#                     no_improvement_limit=10000,
#                     show_progress=False,
#             ):
#                 f.write(str(x_prim))
#                 f.write("\n")

#     draw(name, ls_2d.objective_value)

# if __name__ == '__main__':
#     main()
