import scipy.io
import numpy
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt

__author__ = "Tom Sandmann (s4330048) & Abdullah Rasool (s4350693)"

sbox = dict([
    (hex(0), hex(12)),
    (hex(1), hex(5)),
    (hex(2), hex(6)),
    (hex(3), hex(11)),
    (hex(4), hex(9)),
    (hex(5), hex(0)),
    (hex(6), hex(10)),
    (hex(7), hex(13)),
    (hex(8), hex(3)),
    (hex(9), hex(14)),
    (hex(10), hex(15)),
    (hex(11), hex(8)),
    (hex(12), hex(4)),
    (hex(13), hex(7)),
    (hex(14), hex(1)),
    (hex(15), hex(2)),
])

rows = 14900 # invalues
columns = 16 # keys


# Read the input from the in.mat file
def read_inputs_file():
    file = scipy.io.loadmat('in.mat')
    return file['in']


# Read the input from the traces.mat file
def read_traces_file():
    file = scipy.io.loadmat('traces.mat')
    return file['traces']


# Generates all 2^4 possibilities for k, from 1 to 16
def create_all_keys():
    keys = []
    for i in range(0, 16):
        keys.append(i)
    return keys


# Use the values of in, the key and the sbox to craft y
def create_value_prediction_matrix(in_values, keys):
    matrix = numpy.zeros((rows, columns))
    row = 0
    for i in in_values:
        for k in keys:
            i_xor_k = hex(i[0] ^ k)
            y = sbox[i_xor_k]
            matrix[row][k] = int(y, 16)       # Access matrix by column row
        row += 1

    return matrix


# Converts the value prediction matrix into the power predication matrix, using the hamming weigth
# of the values
def create_power_prediction_matrix(value_prediction_matrix):
    matrix = numpy.zeros((rows, columns))
    for row in range(rows):
        for column in range(columns):
            value_in_bin = bin(int(value_prediction_matrix[row][column]))
            matrix[row][column] = value_in_bin.count("1")

    return matrix


# Compute the pearson correlation coefficient for every column in the power prediction matrix
# with every column of the traces matrix
def create_column_wise_correlation(traces, power_predication_matrix):
    candidates = []

    for candidate in range(columns):
        time_samples = []
        coefficients = []
        for time_sample in range(6990):
            # pearsonnr() returns tuple, first element is correlation value, second is p-value.
            corcoef = abs(pearsonr(power_predication_matrix[:, candidate], traces[:, time_sample])[0])
            time_samples.append(time_sample)
            coefficients.append(corcoef)
        candidates.append((time_samples, coefficients, candidate, max(coefficients)))

    sorted_candidates = sorted(candidates, key=lambda tup: tup[3], reverse=True)
    print("Sorted candidates: ")
    print("Candidate:\t\tCorrelation Value:")
    for c in sorted_candidates:
        print(str(c[2]), str(c[3]), sep="\t\t\t")

    return sorted_candidates


# Create a plot of the correlations
def create_candidate_plot(candidates):
    # candidates[0]: First candidate
    # candidates[0][0]: time data of first candidate
    # candidates[0][1]: coefficient data of first candidate
    # candidates[0][2]: 'name' of first candidate

    # Get the candidate with the highest correlation coefficient
    highest_correlated_candidate = candidates[0][2]
    print('Candidate with highest correlation : ' + str(candidates[0][2]) + ' with correlation value ' +
          str(candidates[0][3]))

    for p in candidates:
        if p[2] != highest_correlated_candidate:
            plt.plot(p[0], p[1], 'r', label=p[2])
        else:
            plt.plot(p[0], p[1], 'g', label=p[2])

    plt.ylabel('Correlation')
    plt.xlabel('Time')
    plt.legend()
    plt.show()

    return highest_correlated_candidate


# Calculate the correlation for a different amount (ie. rows) of traces with each of the columns of the
# power prediction matrix. We need to plot the ranking of the highest correlated candidate from the previous
# step
def create_stepped_power_traces_graph(traces, power_prediction_matrix, highest_correlated_candidate):
    nr_of_traces = [500, 1000, 2000, 4000, 8000, 12000]
    ranking = []

    for attack_size in nr_of_traces:
        attack_ranking = []
        for candidate in range(columns):
            coefficients = []
            for time_sample in range(6990):
                corcoeff = abs(pearsonr(traces[0:attack_size, time_sample], power_prediction_matrix[0:attack_size, candidate])[0])
                coefficients.append(corcoeff)

            attack_ranking.append((candidate, max(coefficients)))

        sorted_attack_ranking = sorted(attack_ranking, key=lambda tup: tup[1], reverse=True)
        ranking.append((attack_size, sorted_attack_ranking))

    rankings_of_highest_correlated_candidate = []
    for attack in ranking:
        ranking_of_best_correlation = attack[1]
        rank = [y[0] for y in ranking_of_best_correlation].index(highest_correlated_candidate)
        rankings_of_highest_correlated_candidate.append(rank+1)

    plt.plot(nr_of_traces, rankings_of_highest_correlated_candidate)
    plt.xlabel('Nr of traces')
    plt.ylabel('Ranking of candidate')
    plt.show()


in_values = read_inputs_file()
keys = create_all_keys()
vpm = create_value_prediction_matrix(in_values, keys)
ppm = create_power_prediction_matrix(vpm)
traces = read_traces_file()
# print(len(traces[0:500, 0])) # To get the first n from a column do this
candidates = create_column_wise_correlation(traces, ppm)
hcc = create_candidate_plot(candidates)
create_stepped_power_traces_graph(traces, ppm, hcc)
