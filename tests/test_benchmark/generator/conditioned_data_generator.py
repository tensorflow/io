import abc
from random import shuffle


class ConditionedDataGenerator:
    """
    A conditioned data generator generates values using a conditional. Thus, the next method that provides the next
    generated value depends on a conditional. Often, conditional data generators are defined through a map:
      conditional -> list of possible values.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        """
        Create an instance of this conditioned data generator by setting the name.

        :param name: The name for this conditional data generator.
        """
        self.name = name

    @abc.abstractmethod
    def next(self, conditional, num_samples=1):
        """
        Draws the next value according to the conditioned data generator underneath.

        :param conditional: Conditional part that decides which data to pick.
        :param num_samples: The number of samples that his method returns.
        :return: Next sample or list of num_sample next samples.
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        """
        An informative, human-readable string for this instance.
        :return: String.
        """
        pass


class KeyConditionedValueGenerator(ConditionedDataGenerator):
    def __init__(self, name, conditionals):
        """
        Define a keyed conditional data generator.

        :param conditionals: A map where the key is the condition and the value the list of words for this key.
        """
        super(KeyConditionedValueGenerator, self).__init__(name)
        self.conditionals = conditionals
        self.counters = {key: 0 for key in self.conditionals.keys()}

    def _next(self, conditional, count, values):
        """
        Generates the next sample for the conditional, with count and values.

        :param conditional: The conditional.
        :param count: The count.
        :param values: The values for the conditional.
        :return:
        """
        num_values = len(values)
        if count >= num_values:
            shuffle(values)
            self.conditionals[conditional] = values
            count = count % num_values
        next_value = values[count]
        self.counters[conditional] = count + 1
        return next_value

    def next(self, conditional, num_samples=1):
        """
        Draws the next value for the conditional data generator.

        :param conditional: The conditional key.
        :param num_samples: The number of samples.
        :return: A sample or list of samples.
        :raise ValueError if num_samples < 1.
        """
        if num_samples < 1:
            raise ValueError(
                "Number of samples is {} but must be > {}".format(num_samples, 0)
            )

        if conditional not in self.conditionals:
            raise ValueError(
                "Conditional {0} not present in conditionals {1}".format(
                    conditional, self.conditionals.keys()
                )
            )
        count = self.counters[conditional]
        values = self.conditionals[conditional]

        if num_samples > 1:
            return [self._next(conditional, count, values) for _ in range(num_samples)]
        else:
            return self._next(conditional, count, values)

    @abc.abstractmethod
    def __str__(self):
        pass


class StringConditionedStringGenerator(KeyConditionedValueGenerator):
    def __init__(self, conditionals):
        super(StringConditionedStringGenerator, self).__init__(
            name=self.__class__.__name__, conditionals=conditionals
        )

    def __str__(self):
        """
        :return: A descriptive string for this conditional data generator.
        """
        return "The conditional string generator is {}.".format(self.conditionals)


class StringConditionedIntegerGenerator(KeyConditionedValueGenerator):
    def __init__(self, conditionals):
        super(StringConditionedIntegerGenerator, self).__init__(
            name=self.__class__.__name__, conditionals=conditionals
        )

    def __str__(self):
        """
        :return: A descriptive string for this conditional data generator.
        """
        return "The conditional integer generator is {}.".format(self.conditionals)


class StringConditionedFloatGenerator(KeyConditionedValueGenerator):
    def __init__(self, conditionals):
        super(StringConditionedFloatGenerator, self).__init__(
            name=self.__class__.__name__, conditionals=conditionals
        )

    def __str__(self):
        """
        :return: A descriptive string for this conditional data generator.
        """
        return "The conditional floating point generator is {}.".format(
            self.conditionals
        )


class BooleanConditionedStringGenerator(KeyConditionedValueGenerator):
    def __init__(self, conditionals):
        super(BooleanConditionedStringGenerator, self).__init__(
            name=self.__class__.__name__, conditionals=conditionals
        )

    def __str__(self):
        """
        :return: A descriptive string for this conditional data generator.
        """
        return "The conditional string generator is {}.".format(self.conditionals)


class StringConditionedBooleanGenerator(KeyConditionedValueGenerator):
    def __init__(self, conditionals):
        super(StringConditionedBooleanGenerator, self).__init__(
            name=self.__class__.__name__, conditionals=conditionals
        )

    def __str__(self):
        """
        :return: A descriptive string for this conditional data generator.
        """
        return "The string conditioned boolean generator is {}".format(
            self.conditionals
        )


class BooleanConditionedFloatGenerator(KeyConditionedValueGenerator):
    def __init__(self, conditionals):
        super(BooleanConditionedFloatGenerator, self).__init__(
            name=self.__class__.__name__, conditionals=conditionals
        )

    def __str__(self):
        """
        :return: A descriptive string for this conditional data generator.
        """
        return "THe boolean conditioned float generator is {}".format(self.conditionals)
