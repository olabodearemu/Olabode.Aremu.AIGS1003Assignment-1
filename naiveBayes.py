import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    self.features = list(trainingData[0].keys()) # this could be useful for your code later...
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
 
    from collections import Counter
import math

class NaiveBayes:
    def __init__(self, k=1):
        self.k = k
        self.word_counts = []
        self.class_counts = Counter()
        self.vocab = set()

    def train(self, training_data, labels):
        for label in labels:
            self.class_counts[label] += 1

        for data in training_data:
            counts = Counter(data)
            self.word_counts.append(counts)
            self.vocab.update(counts.keys())

        for word in self.vocab:
            for label in self.class_counts:
                count = sum([self.word_counts[i][word] for i in range(len(self.word_counts)) if labels[i] == label])
                smoothed_count = count + self.k
                total_count = sum([sum(self.word_counts[i].values()) for i in range(len(self.word_counts)) if labels[i] == label])
                smoothed_total_count = total_count + len(self.vocab) * self.k
                probability = math.log(smoothed_count / smoothed_total_count)
                self.class_word_probabilities[label][word] = probability

    def predict(self, data):
        results = []
        for datum in data:
            class_scores = {label: math.log(self.class_counts[label]) for label in self.class_counts}
            words = set(datum)
            for word in words:
                if word not in self.vocab: continue
                for label in self.class_counts:
                    class_scores[label] += self.class_word_probabilities[label][word]
            results.append(max(class_scores, key=class_scores.get))
        return results

    util.raiseNotDefined()
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    """
    logJoint = util.Counter()
    
    "*** from collections import Counter
import math

class NaiveBayes:
    def __init__(self, k=1):
        self.k = k
        self.word_counts = []
        self.class_counts = Counter()
        self.vocab = set()

    def train(self, training_data, labels):
        for label in labels:
            self.class_counts[label] += 1

        for data in training_data:
            counts = Counter(data)
            self.word_counts.append(counts)
            self.vocab.update(counts.keys())

        for word in self.vocab:
            for label in self.class_counts:
                count = sum([self.word_counts[i][word] for i in range ***"
    util.raiseNotDefined()
    
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    """
    featuresOdds = []
        
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
    

    
      
