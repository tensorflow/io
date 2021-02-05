import abc
import bisect
import logging
import math
import sys
from random import expovariate, gauss, randint, random, shuffle

from pkg_resources import resource_filename

from tests.test_benchmark.generator.constraint_parser import \
    parse_lambda_constraint_into_intervals

WARN_PROB_TH = 0.01  # Warning threshold for probabilities


class DataGenerator:
    """
    A data generator has a next method that provides the next element. If we have a random process then next elements
    follow a distribution.

    If drawing enough next() values all these distributions must match their characteristic parameters. For instance,
    for a normal distribution the empirical mean and standard deviation must match those parameters of the distribution.

    To fix the random seed for all these distributions use the method random.seed(my_seed).
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        """
        Create an instance of this data generator by setting the name.

        :param name: The name for this data generator.
        """
        self.name = name

    @abc.abstractmethod
    def _next(self):
        """
        Draws the next value.

        :return: Returns a sample or list of samples.
        """
        pass

    def next(self, num_data=1):
        """
        Draws the next value.

        :param num_data: Number of samples.
        :return: Returns a sample or list of samples.
        :raise ValueError if num_samples < 1.
        """
        if num_data < 1:
            raise ValueError("Number of samples is {} but must be > {}".format(num_data, 0))

        if num_data > 1:
            return [self._next() for _ in range(num_data)]
        else:
            return self._next()

    @abc.abstractmethod
    def __str__(self):
        pass


class BernoulliDataGenerator(DataGenerator):
    """
    Creates a Bernoulli distribution.
    """

    def __init__(self, prob_true):
        """
        Define the Bernoulli distribution through the probability for true.

        :param prob_true: Probability for true event.
        """
        super(BernoulliDataGenerator, self).__init__(self.__class__.__name__)
        self.prob_true = prob_true

    def _next(self):
        """
        Draw the next boolean value.

        :return: Next boolean value.
        """
        return random() <= self.prob_true

    def __str__(self):
        """
        A string with name and probability for true.

        :return: String.
        """
        return "{0} with prob_true = {1}".format(self.name, self.prob_true)


class UniformIntegerDataGenerator(DataGenerator):
    """
    Uniform integer distribution in range [min_val, max_val].
    """

    def __init__(self, min_val, max_val):
        """
        Creates an uniform integer distribution.

        :param min_val: Minimum value (inclusive).
        :param max_val: Maximum value (Inclusive).
        """
        super(UniformIntegerDataGenerator, self).__init__(self.__class__.__name__)
        self.min_val = min_val
        self.max_val = max_val

    def _next(self):
        """
        Draw next value.

        :return: Random value in range [min_value, max_value].
        """
        return randint(a=self.min_val, b=self.max_val)

    def __str__(self):
        """
        A string with name and range for this uniform distribution.

        :return: String.
        """
        return "{0} has min = {1} and max = {2}".format(self.name, self.min_val, self.max_val)


class GaussianFloatDataGenerator(DataGenerator):
    """
    Create a normal distribution for floating point values.
    """

    def __init__(self, mu, sigma):
        """
        Creates a normal distribution with parameter mean (mu) and standard deviation (sigma)

        :param mu: Mean value.
        :param sigma: Standard deviation.
        """
        super(GaussianFloatDataGenerator, self).__init__(self.__class__.__name__)
        self.mu = mu
        self.sigma = sigma

    def _next(self):
        """
        Draw next random value.

        :return: Random float value in the range (-inf, +inf).
        """
        return gauss(mu=self.mu, sigma=self.sigma)

    def __str__(self):
        """
        A string with name, mean, and standard deviation for this distribution.

        :return: String.
        """
        return "{0} with mean = {1} and standard deviation = {2}".format(self.name, self.mu, self.sigma)


class ExponentialIntegerDataGenerator(DataGenerator):
    """
    This is a power distribution with the parameters lambda and max_value.

    The distribution is f(x; lambda) = lambda exp(-lambda x) for x >= 0 (otherwise 0).

    The cumulative distribution is F(x; lambda) = 1 - exp(-lambda x) for x >= 0 (otherwise 0).

    The expected mean is beta = 1/lambda.
    """

    def __init__(self, beta, max_val):
        """
        Initialize an exponential integer distribution.

        :param beta: The expected mean.
        :param max_val: The maximum value.
        """
        super(ExponentialIntegerDataGenerator, self).__init__(self.__class__.__name__)
        self.lambd = 1.0 / beta
        self.max_val = max_val
        # The probability mass beyond max_val is F(inf; lambda) - F(max_val; lambda) = exp(-max_val lambda)
        prob_mass = math.exp(-max_val * self.lambd)
        if prob_mass > WARN_PROB_TH:
            logging.warn("Probability mass beyond max_val is {0} but should be {1}".format(prob_mass, WARN_PROB_TH))

    def _next(self):
        """
        Draw the next value in the distribution.

        :return: A value in the range [0, max_val)
        """
        val = self.max_val
        while val >= self.max_val:
            val = int(round(expovariate(lambd=self.lambd)))
        return val

    def __str__(self):
        """
        A string with name, mean, and max of this distribution.

        :return: String.
        """
        return "{0} has mean = {1} and max = {2}".format(self.name, 1.0 / self.lambd, self.max_val)


class StringDataGenerator(DataGenerator):
    """
    Returns strings from a vocabulary in a random order.

    The random sequence repeats after all words have been traversed.
    """

    def __init__(self, words):
        """
        Creates a random distribution over words.

        :param words: The list of words for the distribution.
        """
        super(StringDataGenerator, self).__init__(self.__class__.__name__)
        if len(words) == 0:
            raise ValueError("Empty list of words given! Can't have a distribution over no words")
        self.words = words
        shuffle(self.words)
        self.i_words = -1

    def _next(self):
        """
        Draw the next randomly chosen word.

        :return: Randomly chosen word.
        """
        self.i_words += 1
        # We made it once through the entire word list, then shuffle the words and reset the counter
        if self.i_words == len(self.words):
            shuffle(self.words)
            self.i_words = 0
        return self.words[self.i_words]

    def __str__(self):
        """
        A string with name and number of words.

        :return: String.
        """
        return "{0} with {1} words.".format(self.name, len(self.words))

    @staticmethod
    def create_from_file(filename=resource_filename("tests.test_benchmark.resources", "wordlist")):
        """
        Loads the vocabulary from a given file. As default loads a dictionary of about 70k English words.

        :param filename: The filename for the vocabulary: Each line contains one word.

        :return: A random sequence over the words in the file.
        """
        with open(filename) as text_file:
            lines = text_file.readlines()
            lines = [line.strip() for line in lines]
        return StringDataGenerator(lines)


class BytesDataGenerator(DataGenerator):
    """
    Returns bytes from a vocabulary in a random order.

    The random sequence repeats after all words have been traversed.
    """

    def __init__(self, words):
        """
        Creates a random distribution over words.

        :param words: The list of words for the distribution.
        """
        super(BytesDataGenerator, self).__init__(self.__class__.__name__)
        if len(words) == 0:
            raise ValueError("Empty list of words given! Can't have a distribution over no words")
        self.words = words
        shuffle(self.words)
        self.i_words = -1

    def _next(self):
        """
        Draw the next randomly chosen word.

        :return: Randomly chosen word as bytes.
        """
        self.i_words += 1
        # We made it once through the entire word list, then shuffle the words and reset the counter
        if self.i_words == len(self.words):
            shuffle(self.words)
            self.i_words = 0
        return self.words[self.i_words]

    def __str__(self):
        """
        A string with name and number of words.

        :return: String.
        """
        return "{0} with {1} words.".format(self.name, len(self.words))

    @staticmethod
    def create_from_file(filename=resource_filename("tests.test_benchmark.resources", "wordlist")):
        """
        Loads the vocabulary from a given file. As default loads a dictionary of about 70k English words.

        :param filename: The filename for the vocabulary: Each line contains one word.

        :return: A random sequence over the words in the file.
        """
        with open(filename, "rb") as fh:
            lines = fh.readlines()
            lines = [line.strip() for line in lines]
        return BytesDataGenerator(lines)


class CountDataGenerator(DataGenerator):
    """
    Creates a data generator that counts.
    """

    def __init__(self, initial_count=0):
        """
        Constructs a count data generator.

        :param initial_count: The initial count value. Default is '0'.
        """
        super(CountDataGenerator, self).__init__(self.__class__.__name__)
        self.count = initial_count - 1  # Decrease by one because we increment before returning the value

    def _next(self):
        """
        Returns the next integer count value starting from initial_count param.

        :return: Integer value.
        """
        self.count += 1
        return self.count

    def __str__(self):
        """
        Returns a string representation of this data generator.

        :return: String.
        """
        # Increment the counter by one because we increment before returning the value
        return "{0} counts from: {1}".format(self.name, self.count + 1)


class ConstantDataGenerator(DataGenerator):
    """
    A data generator that returns a fixed constant.
    """

    def __init__(self, constant_value=1):
        """
        Constructs a constant data generator.

        :param constant_value: The constant value.
        """
        super(ConstantDataGenerator, self).__init__(self.__class__.__name__)
        self.constant_value = constant_value

    def _next(self):
        """
        Returns the constant value.

        :return: Constant value.
        """
        return self.constant_value

    def __str__(self):
        """
        Returns a string representation of this data generator.

        :return: String.
        """
        # Increment the counter by one because we increment before returning the value
        return "{} with constant {}".format(self.name, self.constant_value)


class RepeatDataGenerator(DataGenerator):
    """
    A data generator that repeats the same value.

    Use this generator to chain with another generator.
    """

    def __init__(self, data_generator, repeat_num=1):
        """
        Constructs a repeat data generator

        :param data_generator: The input data generator
        :param repeat_num:
        """
        super(RepeatDataGenerator, self).__init__(self.__class__.__name__)
        self.repeat_count = 0
        self.repeat_num = repeat_num
        self.data_generator = data_generator
        self.data = self.data_generator.next()

    def _next(self):
        """
        Returns the next value while repeating for repeat_num times

        :return: The next value
        """
        if self.repeat_count >= self.repeat_num:
            self.data = self.data_generator.next()
            self.repeat_count = 0
        self.repeat_count += 1
        return self.data

    def __str__(self):
        """
        Returns a string representation of this data generator.

        :return: String.
        """
        return "{} with repeat {}".format(self.name, self.repeat_num)


class BoundsDataGenerator(DataGenerator):
    """
    A data generator that puts bounds of min and max on the produced values.

    Use this generator to chain with another generator.
    """

    def __init__(self, data_generator, min_value, max_value):
        """
        Constructs a bounds generator.

        :param data_generator: The input data generator
        :param min_value: The minimum value bound.
        :param max_value: The maximum value bound.
        """
        super(BoundsDataGenerator, self).__init__(self.__class__.__name__)
        assert min_value <= max_value, f"{min_value} must be smaller or equal to {max_value}"
        self.data_generator = data_generator
        self.min_value = min_value
        self.max_value = max_value

    def _next(self):
        """
        Returns the next value with minimum and maximum bound applied.

        :return: The next value.
        """
        return max(min(self.data_generator.next(), self.max_value), self.min_value)

    def __str__(self):
        """
        Returns a string representation of this data generator.

        :return: String.
        """
        return "{} with min {} and max {}".format(self.name, self.min_value, self.max_value)


class StringCountDataGenerator(DataGenerator):
    """
    Creates a data generator that counts, but returns a string value.

    Useful for creating UUID fields
    """

    def __init__(self, initial_count=0):
        """
        Constructs a count data generator.

        :param initial_count: The initial count value. Default is '0'.
        """
        super(StringCountDataGenerator, self).__init__(self.__class__.__name__)
        self.count = initial_count - 1  # Decrease by one because we increment before returning the value

    def _next(self):
        """
        Returns the next integer count value starting from initial_count param.

        :return: String value.
        """
        self.count += 1
        return str(self.count)

    def __str__(self):
        """
        Returns a string representation of this data generator.

        :return: String.
        """
        # Increment the counter by one because we increment before returning the value
        return "{0} counts from: {1}".format(self.name, self.count + 1)


class EnumerationDataGenerator(DataGenerator):
    """
    Creates a data generator for an enumeration of values.
    """

    def __init__(self, values):
        """
        Constructs the data generator.

        :param values: The enumeration of values.
        """
        super(EnumerationDataGenerator, self).__init__(name=self.__class__.__name__)
        self.values = values
        self.count = -1
        shuffle(self.values)

    def _next(self):
        """
        Draw the next element from the enumeration, randomly.

        :return: The next element.
        """
        self.count = self.count + 1
        if self.count >= len(self.values):
            shuffle(self.values)
            self.count = 0
        return self.values[self.count]

    def __str__(self):
        return "Enumeration with {} values.".format(len(self.values))


class ConstrainedIntegerDataGenerator(DataGenerator):
    """
    A constrained integer data generator.
    """

    def __init__(self, constraint):
        """
        Constructs the data generator with constraints.

        :param constraint: A lambda function with the constraints.
        """
        super(ConstrainedIntegerDataGenerator, self).__init__(name=self.__class__.__name__)

        def _compute_cum_ratios(merged_intervals):
            lengths = [interval[1] - interval[0] for interval in merged_intervals]
            total = sum(lengths)
            # Map these into the range of 0...1
            ratios = [float(length) / total for length in lengths]
            cum = 0
            for i_ratio, ratio in enumerate(ratios):
                cum += ratio
                ratios[i_ratio] = cum
            return ratios

        def _sort_and_merge_intervals(intervals):
            # Assumes that intervals[i][0] <= intervals[i][1] for all i

            # Sort by lower boundary and the upper boundary
            intervals = sorted(intervals, key=lambda interval: interval[0])
            # Find overlap, then merge from front to back sequentially
            n_intervals = len(intervals)
            i_start = 0
            merged_intervals = []
            while i_start < n_intervals:
                lower = intervals[i_start][0]
                upper = intervals[i_start][1]

                i_merge = i_start + 1
                max_upper = upper

                while i_merge < n_intervals and intervals[i_merge][0] < upper:
                    max_upper = max(max_upper, intervals[i_merge][1])
                    i_merge += 1

                merged_intervals.append([lower, max_upper])
                i_start = i_merge

            return merged_intervals

        self.min_val = -sys.maxsize - 1
        self.max_val = sys.maxsize
        self.intervals = _sort_and_merge_intervals(parse_lambda_constraint_into_intervals(constraint))
        self.ratios = _compute_cum_ratios(self.intervals)

    def _next(self):
        # Use binary search to find the interval to draw from (since ratios monotonically increase we don't have
        # problems with the intervals being skipped)
        interval = self.intervals[bisect.bisect(self.ratios, random())]
        return randint(a=interval[0], b=interval[1])

    def __str__(self):
        return "Intervals are {}".format(self.intervals)
