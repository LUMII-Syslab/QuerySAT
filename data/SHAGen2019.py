#!/usr/bin/env python3

import os
import platform
import random
import subprocess

from utils.sat import remove_unused_vars

try:
    from data.k_sat import KSAT
    from pysat.solvers import Solver
except ImportError as err:
    if __name__ != "__main__":
        print("import error; run the main script outside the data folder")
        raise err
    else:
        print("import ok")


        class KSAT:
            pass

TEST_MODE = False  # True
BENCHMARK_MODE = False  # True


def random_binary_string(n):
    return "".join([str(random.randint(0, 1)) for _ in range(n)])


class SHAGen2019(KSAT):
    """ Dataset with random SAT instances based on the SHA1 algorithm. We use cgen with the parameters similar to SAT Competition 2019.
    """

    CGEN_EXECUTABLE = "./cgen"
    if platform.system() == "Linux":
        CGEN_EXECUTABLE = "./data/cgen_linux64"
        if not os.path.exists(CGEN_EXECUTABLE):
            CGEN_EXECUTABLE = "./cgen_linux64"

    if platform.system() == "Darwin":
        CGEN_EXECUTABLE = "./cgen_mac"
        if not os.path.exists(CGEN_EXECUTABLE):
            CGEN_EXECUTABLE = "./data/cgen_mac"

    TMP_FILE_NAME = "data.tmp"

    def __init__(self, data_dir,
                 min_vars=4, max_vars=100020,
                 force_data_gen=False, **kwargs) -> None:
        super(SHAGen2019, self).__init__(data_dir, min_vars=min_vars,
                                         max_vars=max_vars, force_data_gen=force_data_gen, **kwargs)
        # maximum number of samples; if there are less, we will stop earlier
        self.train_size = 100000
        self.test_size = 5000
        if TEST_MODE:
            self.test_size = 1

        #### constraints ####
        # how many free bits; max 512 free bits 
        self.bits_from = 2
        self.bits_to = 20

        self.generate_hard_instances = True  # If True, for #rounds < 6 the set of clauses will be empty.

        # the number of rounds (max==80 by SHA-1 specs)
        self.sha_rounds_from = 17
        self.sha_rounds_to = 17
        #### the desired number of variables ####
        self.min_vars = min_vars
        self.max_vars = max_vars
        # how many times we need to check the number of variables to be within the given range before we stop the generator
        self.max_attempts = 100

    def train_generator(self) -> tuple:
        return self.__generator(self.train_size)

    def test_generator(self) -> tuple:
        return self.__generator(self.test_size)

    def __generator(self, size) -> tuple:
        samplesSoFar = 0

        while samplesSoFar < size:
            attempts = 0
            while attempts < self.max_attempts:
                n_bits = random.randint(self.bits_from, self.bits_to)

                sha_rounds = random.randint(self.sha_rounds_from, max(
                    self.sha_rounds_from, self.sha_rounds_to))
                if sha_rounds < 1:
                    sha_rounds = 1
                if sha_rounds > 80:
                    sha_rounds = 80

                bits_position = 0

                bitsstr = random_binary_string(512)
                hashstr = random_binary_string(160)
                bitsstr = "0b" + bitsstr

                if TEST_MODE:
                    # bitsstr = "string:a pad:sha1"
                    n_bits = 512
                    # n_bits = 75#145
                    sha_rounds = 5
                    self.max_vars = 100000

                if self.generate_hard_instances:
                    cmd = SHAGen2019.CGEN_EXECUTABLE + " encode SHA1 -vM " + bitsstr + " except:1.." + str(
                        n_bits) + " -vH compute -r " + str(
                        sha_rounds) + " " + SHAGen2019.TMP_FILE_NAME
                else:
                    cmd = SHAGen2019.CGEN_EXECUTABLE + " encode SHA1 -vM " + bitsstr + " except:1.." + str(
                        n_bits) + " -r " + str(
                        sha_rounds) + " " + SHAGen2019.TMP_FILE_NAME

                # Launching the process and reading its output
                if os.path.exists(SHAGen2019.TMP_FILE_NAME):
                    os.remove(SHAGen2019.TMP_FILE_NAME)
                ok = False
                out = ""
                try:
                    out = subprocess.check_output(
                        cmd, shell=True, universal_newlines=True)
                except:
                    out = ""  # an unsatisfiable formula or an execution error
                # print(cmd)
                # print(cmd,"["+out+"]") # -- debug

                # Searching for the "CNF: <nvars> var" substring;
                # ok will be true iff <nvars> is between MIN_VARS and MAX_VARS;
                # if not ok, we will delete the file.
                j1 = out.find("CNF:")
                j2 = out.find("var", j1 + 1)
                if j1 >= 0 and j2 >= 0:
                    nvars = int(out[j1 + 4:j2].strip())
                    ok = nvars >= self.min_vars and nvars <= self.max_vars

                if ok:
                    # if TEST_MODE:
                    # copyfile("sha1r17m75a_p.cnf", SHAGen2019.TMP_FILE_NAME)
                    f = open(SHAGen2019.TMP_FILE_NAME, 'r')
                    lines = f.readlines()
                    f.close()
                    os.remove(SHAGen2019.TMP_FILE_NAME)
                    clauses = []
                    for line in lines:
                        line = line.strip()
                        if len(line) == 0 or line[0].isalpha():
                            continue
                        clause = []
                        for s in line.split():
                            i = int(s)
                            if i == 0:
                                break  # end of clause
                            clause.append(i)
                        clauses.append(clause)

                    # try Cadical and Glucose3/4
                    if BENCHMARK_MODE:
                        with Solver(name="Cadical", bootstrap_with=clauses, use_timer=True) as solver:
                            is_sat = solver.solve()
                            print("Cadical result: ", is_sat, '{0:.4f}s'.format(solver.time()), nvars, " vars", len(clauses), " clauses")
                        with Solver(name="Glucose3", bootstrap_with=clauses, use_timer=True) as solver:
                            is_sat = solver.solve()
                            print("Gluecose3 result: ", is_sat, '{0:.4f}s'.format(solver.time()), nvars, " vars", len(clauses), " clauses")
                        with Solver(name="Glucose4", bootstrap_with=clauses, use_timer=True) as solver:
                            is_sat = solver.solve()
                            print("Gluecose4 result: ", is_sat, '{0:.4f}s'.format(solver.time()), nvars, " vars", len(clauses), " clauses")

                    if len(clauses) == 0:
                        ok = False

                    if TEST_MODE:
                        print("Cgen vars: ", nvars)
                    nvars, clauses = remove_unused_vars(nvars, clauses)
                    if TEST_MODE:
                        print("Fixed vars: ", nvars)
                    ok = self.min_vars <= nvars <= self.max_vars  # checking once again after the removal of unused vars

                if ok:
                    yield nvars, clauses
                    samplesSoFar += 1
                    break  # while attempts

                if os.path.exists(SHAGen2019.TMP_FILE_NAME):
                    os.remove(SHAGen2019.TMP_FILE_NAME)
                # if break haven't occurred, try the next attempt:
                attempts += 1

            # after while ended, let's check if we reached the attempt limit
            if attempts == self.max_attempts:
                break  # stop the iterator, too many attempts; perhaps, we are not able to generate the desired number of variables according to the given constraints


if __name__ == "__main__":
    nvars, clauses = remove_unused_vars(10, [[-1, 4, 9], [-4, 1, -10], [10]])
    # nvars, clauses = remove_unused_vars(3, [[-1,3]])
    # nvars, clauses = remove_unused_vars(3, [[]])
    print(nvars, clauses)
