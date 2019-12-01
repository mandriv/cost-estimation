import random
import operator

import numpy
import networkx as nx
import matplotlib.pyplot as plt

from deap import algorithms, base, creator, tools, gp

def run(config, inputs, train_data, test_data):
    # Define new functions
    pset = gp.PrimitiveSet("MAIN", len(inputs))
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addEphemeralConstant("rand101int", lambda: random.randint(-1,1))
    pset.addEphemeralConstant("rand101float", lambda: round(random.uniform(-1,1), 3))
    # rename arguments to inputs
    arg_num = 0
    for input in inputs:
        eval_str = 'pset.renameArguments(ARG' + str(arg_num) + '=\'' + input + '\')'
        arg_num += 1
        eval(eval_str)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def evalIndividual(individual, compare_data):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        # Evaluate the Mean Magnitude of Relative Error
        mre_sum = 0
        for data in compare_data:
            result = func(*data['inputs'])
            mre = abs(data['output'] - result) / data['output']
            mre_sum += mre
        return [mre_sum / len(compare_data)]

    toolbox.register("evaluate", evalIndividual, compare_data=train_data)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config['depth_limit']))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config['depth_limit']))

    pop = toolbox.population(n=config['population_size'])
    hof = tools.HallOfFame(1)
    # fitness statistics
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_fit.register("avg", numpy.mean)
    # stats_fit.register("std", numpy.std)
    stats_fit.register("min", numpy.min)
    stats_fit.register("max", numpy.max)
    # size statistics
    stats_size = tools.Statistics(len)
    stats_size.register("avg", numpy.mean)
    # stats_size.register("std", numpy.std)
    stats_size.register("min", numpy.min)
    stats_size.register("max", numpy.max)
    # comparison stats
    def get_fitness_against_test_data(pop):
        fit = []
        for ind in pop:
            fit.append(evalIndividual(ind, test_data)[0])
        return fit

    def avg_test_data(pop):
        return numpy.mean(get_fitness_against_test_data(pop))

    def min_test_data(pop):
        return numpy.min(get_fitness_against_test_data(pop))

    def max_test_data(pop):
        return numpy.max(get_fitness_against_test_data(pop))

    def std_test_data(pop):
        return numpy.std(get_fitness_against_test_data(pop))
    stats_test_comparison = tools.Statistics(lambda ind: ind)
    stats_test_comparison.register("avg", avg_test_data)
    stats_test_comparison.register("std", std_test_data)
    stats_test_comparison.register("min", min_test_data)
    stats_test_comparison.register("max", max_test_data)

    mstats = tools.MultiStatistics(fitness=stats_fit,
        test_data_comparison=stats_test_comparison, size=stats_size)

    mate_chance = config['mating_chance']
    mutation_chance = config['mutation_chance']
    number_of_generations = config['number_of_generations']

    _, log = algorithms.eaSimple(pop, toolbox,
        mate_chance, mutation_chance, number_of_generations,
        stats=mstats,
        halloffame=hof,
        verbose=True)

    return (hof[0], log)

def draw_graph(individual):
    plt.figure()
    nodes, edges, labels = gp.graph(individual)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels)
    plt.draw()

def draw_statistics(logbook):
    gen = logbook.select("gen")
    fit_mins = logbook.chapters["fitness"].select("min")
    fit_mins_test_data = logbook.chapters["test_data_comparison"].select("min")
    size_avgs = logbook.chapters["size"].select("avg")
    # min fitness to average size
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness (Mean Magnitude of Relative Error)")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="best")
    plt.draw()
    # min fitness of train data to test data
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Min Fitness (MMRE) over train data")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, fit_mins_test_data, "r-", label="Min Fitness (MMRE) over test data")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="best")
    plt.draw()
