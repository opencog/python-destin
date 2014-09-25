class junky:
    def initLearningAlgorithm(self, AlgorithmChoice, AlgParams, InitNodeBelief, InitNodeLearnedFeatures):
        self.AlgorithmChoice = AlgorithmChoice
        if AlgorithmChoice == 'SparseAutoEncoder':
            InitialWeightMatrix = InitNodeLearnedFeatures  #
            self.LearningAlgorithm = SparseAutoEncoder(AlgParams, InitNodeLearnedFeatures)
        elif AlgorithmChoice == 'KMeansClustering':
        # InitialCentroids = InitNodeLearnedFeatures #
        #self.LearningAlgorithm = KMClustering(AlgParams,InitialCentroids)
        elif AlgorithmChoice == 'LogRegression':
            self.AlgorithmChoice = AlgorithmChoice  # Name of the Algorithm
            self.Belief = InitNodeBelief
            self.LearnedFeatures = InitNodeLearnedFeatures
            #Attrbutes For the learning Algorithm Class
            self.LearningAlgoritm = LearningAlgorithm(AlgParams)
            self.LearningAlgoritm.D = AlgParams['D']
            self.LearningAlgoritm.N = AlgParams['N']
            self.LearningAlgoritm.training_steps = AlgParams['training_steps']
            self.LearningAlgoritm.feats = AlgParams['feats']
            self.LearningAlgoritm.w = AlgParams['w']
        #InitialCentroids = InitNodeLearnedFeatures #
        #self.LearningAlgorithm = KMClustering(AlgParams,InitialCentroids)
        elif AlgorithmChoice == 'EvolutionaryLearning':
        # God Knows what to do here :D
        elif AlgorithmChoice == 'MiscLearning':
        # We'll write new stuff here, like combining two learning algorithms
        else:
            print('make sure that you are choosing an available learning algorithm')
            print('python is exitting')
            exit(0)

