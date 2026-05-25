import random
from deap import base, creator, tools, algorithms
from model import HealthcareModel

class EvolutionaryOptimizer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.fitness_history = []
    
    def evaluate(self, individual):
        depth = individual[0]
        model = HealthcareModel(max_depth=depth)
        model.train(self.X_train, self.y_train)
        metrics = model.evaluate(self.X_test, self.y_test)
        accuracy = metrics["Accuracy"]
        return (accuracy,)
    
    def optimize(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_int", random.randint, 2, 10)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutUniformInt, low=2, up=10, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=10)
        algorithms.eaSimple(population, toolbox, 0.5, 0.2, 5, verbose=False)

        best = tools.selBest(population, 1)[0]
        return best[0]