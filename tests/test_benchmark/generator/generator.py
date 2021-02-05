import logging
import os.path

from avro.datafile import DataFileWriter
from avro.io import DatumWriter
from avro.schema import ArraySchema, MapSchema, Parse, PrimitiveSchema, RecordSchema, UnionSchema

from tests.test_benchmark.generator.conditioned_data_generator import ConditionedDataGenerator
from tests.test_benchmark.generator.dag import Dag
from tests.test_benchmark.generator.data_generator import (
    BernoulliDataGenerator,
    BytesDataGenerator,
    ExponentialIntegerDataGenerator,
    GaussianFloatDataGenerator,
    StringDataGenerator,
    UniformIntegerDataGenerator,
)


class Generator:
    """
    Generates mock-up data based on any avro schema. It supports all primitive types, nested types, maps, arrays,
    records, and unions.

    We use the custom classes for data_generators of all
     * boolean values
     * integer values
     * long values
     * float values
     * double values
     * bytes values
     * string values
     * number of array items
     * number of map pairs
     * value for the keys in a map

    Uses avro-python3 to parse a schema into python and to write avro data.
    """

    namespace_separator = "."  # Separator for namespaces
    type_separator = ":"  # Separator between type and names

    array_item_start = "["  # Start of array item
    array_item_end = "]"  # End of array item

    map_value_start = "["  # Start of map value
    map_value_end = "]"  # End of map value

    array_name = "array"  # Used to identify an array
    map_name = "map"  # Used to identify a map

    def __init__(self, json_schema, default_namespace="null"):
        """
        Create a data generator specific to the json_schema.

        :param json_schema: The avro schema in json format.
        :param default_namespace: The default namespace which is assumed if NO namespace was defined in the schema.
        """
        logging.info("Creating generator with schema: {}".format(json_schema))

        # Parse the json schema into an avro schema using avro and save the json schema for error reporting
        self.schema = Parse(json_schema)
        self.json_schema = json_schema

        # Set the namespace with the one from the schema or the default if none was defined in the schema
        if not self.schema.namespace:
            self.namespace = default_namespace
        else:
            self.namespace = self.schema.namespace
        logging.info("Set namespace to '{}'.".format(self.namespace))

        # We maintain lists of name:types to set default data_generator
        self.name_type_booleans = []
        self.name_type_integers = []
        self.name_type_longs = []
        self.name_type_floats = []
        self.name_type_doubles = []
        self.name_type_bytes = []
        self.name_type_strings = []
        self.name_type_nulls = []
        self.name_type_arrays = []
        self.name_type_maps = []
        self.name_type_unions = []

        # Parse all name types and store them per type
        for full_name_type in self._get_name_types(self.schema, namespace=self.namespace):
            this_type = full_name_type.split(Generator.type_separator)[1]
            if this_type == "boolean":
                self.name_type_booleans.append(full_name_type)
            elif this_type == "int":
                self.name_type_integers.append(full_name_type)
            elif this_type == "long":
                self.name_type_longs.append(full_name_type)
            elif this_type == "float":
                self.name_type_floats.append(full_name_type)
            elif this_type == "double":
                self.name_type_doubles.append(full_name_type)
            elif this_type == "bytes":
                self.name_type_bytes.append(full_name_type)
            elif this_type == "string":
                self.name_type_strings.append(full_name_type)
            elif this_type == "null":
                self.name_type_nulls.append(full_name_type)  # need to use full name type to avoid ambiguity
            elif this_type == "array":
                self.name_type_arrays.append(full_name_type)
            elif this_type == "map":
                self.name_type_maps.append(full_name_type)
            elif this_type == "union":
                self.name_type_unions.append(full_name_type)  # need to use full name type to avoid ambiguity
            else:
                logging.warn("Unknown type string '{}'".format(this_type))

        # Holds conditional and unconditional data generators for primitive types, maps, arrays, and unions with key
        # being the name type and the value being the data generator
        self.data_generator_for_name_type = {}

        # Holds a condition for name type; where the key is the name type and the value the conditional
        self.condition_for_name_type = {}

        # Holds all name type for unconditioned data generators
        self.unconditioned_name_types = set()

        # This dag holds the dependency information for the conditionals
        self.dag = Dag()

        # Set default data generators for all primitive types
        self.set_data_generator_for_all_boolean_types()
        self.set_data_generator_for_all_integer_types()
        self.set_data_generator_for_all_long_types()
        self.set_data_generator_for_all_float_types()
        self.set_data_generator_for_all_double_types()
        self.set_data_generator_for_all_bytes_types()
        self.set_data_generator_for_all_string_types()

        # Distribution for the number of array items, the number of map pairs, and the map keys
        self.data_generator_for_array = {}
        self.map_key_num_gen = None

        # TODO: Set data_generator per array and map
        self.set_data_generator_for_all_array_num()
        self.set_data_generator_for_all_map_key_num()

        # TODO: Set a data_generator per union
        self._set_uniform_data_generator_for_all_union()

    def set_data_generator_for_name_type(self, name_type, data_generator):
        """
        Set the data_generator for name:string_for_type if it exists in the schema.

        :param name_type: This is the name type string, e.g. 'com.linkedin.browsemap.memberId:long'.
        :param data_generator: This is the data generator to be set.

        :raises: NameError if the name type is not present in the schema that this generator was created for.
        """
        if name_type not in self.data_generator_for_name_type:
            raise NameError(
                "'{}' is not part of the schema '{}'.\nYou may have to add the '{}' namespace or append "
                "the ':type' value.".format(name_type, self.schema, "null")
            )
        self._set_data_generator_for_name_type_no_check(name_type, data_generator)

    def _set_data_generator_for_name_type_no_check(self, name_type, data_generator):
        """
        This sets full_name:string_for_type initially without checking that an entry exists.

        :param name_type: The name and type.
        :param data_generator: The data generator to be used for this full name and type.
        """
        if name_type in self.data_generator_for_name_type:
            logging.info("Overwriting '{}' with data_generator: '{}'".format(name_type, data_generator))
        self.data_generator_for_name_type[name_type] = data_generator
        self.unconditioned_name_types.add(name_type)

    def _set_data_generator_for_full_name_type_list(self, name_type_list, data_generator):
        """
        Sets the same data_generator for all the name types in the list.

        :param name_type_list: The list with the name types.
        :param data_generator: The data generator.
        """
        for name_type in name_type_list:
            self._set_data_generator_for_name_type_no_check(name_type, data_generator)

    @staticmethod
    def _check_name_and_conditional(name_type, conditional_name_type, same_type):
        """
        Check that the path to the last 'same type' as given by the conditional name type matches that in the name type.

        :param name_type: The name type.
        :param conditional_name_type: The conditional name type.
        :param same_type: The type that should be compared, e.g. 'array' or 'map'.
        :raises ValueError is thrown if the path to the 'same type' is not the same.
        """
        last = conditional_name_type.rfind(same_type)  # last = -1 if not found
        if last > -1 and name_type[:last] != conditional_name_type[:last]:
            raise ValueError("Condition {} is not in the same {} as the value {}".format(conditional_name_type, same_type, name_type))

    @staticmethod
    def _check_has_same_array(name_type, conditional_name_type):
        """
        Check that the name type and the conditional name type have arrays in the same position.

        Further details see '_check_name_and_conditional'.
        """
        Generator._check_name_and_conditional(name_type=name_type, conditional_name_type=conditional_name_type, same_type=Generator.array_name)

    @staticmethod
    def _check_has_same_map(name_type, conditional_name_type):
        """
        Check that the name type and the conditional name type have maps in the same position.

        Further details see '_check_name_and_conditional'.
        """
        Generator._check_name_and_conditional(name_type=name_type, conditional_name_type=conditional_name_type, same_type=Generator.map_name)

    def set_conditioned_data_generator_for_name_type(self, name_type, conditional_name_type, data_generator):
        """
        Set a conditioned data_generator for a name type together with a conditional name type.

        :param name_type: The name type.
        :param conditional_name_type: The conditional name type.
        :param data_generator: The conditioned data generator.
        :raises ValueError if the data_generator is not of the type 'ConditionedDataGenerator'.
        """

        # Check the type of the data_generator
        if not isinstance(data_generator, ConditionedDataGenerator):
            raise ValueError("Distribution {} must be of type {}.".format(data_generator, type(ConditionedDataGenerator)))

        # Write or overwrite the data_generator
        self.set_data_generator_for_name_type(name_type, data_generator)

        # The name type might have been added to unconditioned name types, check and possibly remove
        if name_type in self.unconditioned_name_types:
            self.unconditioned_name_types.remove(name_type)

        # The conditional name type might have been added to unconditioned name types, check and possibly remove
        if conditional_name_type in self.unconditioned_name_types:
            self.unconditioned_name_types.remove(conditional_name_type)

        # Check that conditionals are in the same array, maps
        # If they appear in different arrays we might have a different number of items for each and, thus, cannot
        # condition correctly; hence, this restriction
        # If they appear in different maps it is not guaranteed that the conditioned map has the same key
        Generator._check_has_same_array(name_type=name_type, conditional_name_type=conditional_name_type)
        Generator._check_has_same_map(name_type=name_type, conditional_name_type=conditional_name_type)

        # If the this conditional data_generator was set before remove the previous dependency
        if name_type in self.condition_for_name_type:
            conditional = self.condition_for_name_type[name_type]
            # If they match the edge exists before, if not we got a cycle but we let the add_edge method handle that
            if conditional_name_type == conditional:
                self.dag.remove_edge(name_type, conditional)

        # Add the new dependency
        self.dag.add_edge(name_type, conditional_name_type)

        # Update the conditional name
        self.condition_for_name_type[name_type] = conditional_name_type

    def set_data_generator_for_all_boolean_types(self, data_generator=BernoulliDataGenerator(prob_true=0.7)):
        """
        Set the data_generator for boolean values. The default is a Bernoulli data_generator.

        :param data_generator: The data generator.
        """
        self._set_data_generator_for_full_name_type_list(name_type_list=self.name_type_booleans, data_generator=data_generator)

    def set_data_generator_for_all_integer_types(self, data_generator=UniformIntegerDataGenerator(min_val=0, max_val=100)):
        """
        Set the data_generator for integer values. The default is a uniform integer data_generator.

        :param data_generator: The data generator.
        """
        self._set_data_generator_for_full_name_type_list(name_type_list=self.name_type_integers, data_generator=data_generator)

    def set_data_generator_for_all_long_types(self, data_generator=UniformIntegerDataGenerator(min_val=0, max_val=1000)):
        """
        Set the data_generator for long values. The default is a uniform integer data_generator. If you need to especially
        test the entire range of long values you may write your own data_generator class and use it here.

        :param data_generator: The data generator.
        """
        self._set_data_generator_for_full_name_type_list(name_type_list=self.name_type_longs, data_generator=data_generator)

    def set_data_generator_for_all_float_types(self, data_generator=GaussianFloatDataGenerator(mu=0, sigma=50.5)):
        """
        Set the data_generator for float values. The default is a normal data_generator.

        :param data_generator: The data generator.
        """
        self._set_data_generator_for_full_name_type_list(name_type_list=self.name_type_floats, data_generator=data_generator)

    def set_data_generator_for_all_double_types(self, data_generator=GaussianFloatDataGenerator(mu=0, sigma=1005.23)):
        """
        Set the data_generator for double values. The default is a normal data_generator. Underneath this normal
        data_generator uses random.gauss, check the documentation for that function to see if it fits your range
        requirements.

        :param data_generator: The data generator.
        """
        self._set_data_generator_for_full_name_type_list(name_type_list=self.name_type_doubles, data_generator=data_generator)

    def set_data_generator_for_all_bytes_types(self, data_generator=BytesDataGenerator.create_from_file()):
        """
        Set the data_generator for bytes values. The default is a string data_generator.

        :param data_generator: The data generator.
        """
        self._set_data_generator_for_full_name_type_list(name_type_list=self.name_type_bytes, data_generator=data_generator)

    def set_data_generator_for_all_string_types(self, data_generator=StringDataGenerator.create_from_file()):
        """
        Set the data_generator for string values. The default is a string data_generator.

        :param data_generator: The data generator.
        """
        self._set_data_generator_for_full_name_type_list(name_type_list=self.name_type_strings, data_generator=data_generator)

    def set_data_generator_for_all_array_num(self, data_generator=ExponentialIntegerDataGenerator(beta=5, max_val=25)):
        """
        Set the data_generator for the number of array items. The default is an exponential integer data_generator.

        :param data_generator: The data generator.
        """
        logging.info("Set the data_generator for the number of array items to: {0}".format(str(data_generator)))
        for name_type_array in self.name_type_arrays:
            self.data_generator_for_array[name_type_array] = data_generator

    def set_data_generator_for_arrays_num(self, name_type_arrays, data_generator):
        """
        Set the data_generator for a specific number of items in an array. No default generator.
        :param name_type_arrays: The array of identifiers for the array in the schema.
        :param data_generator: The data generator.
        """
        for name_type_array in name_type_arrays:
            if name_type_array not in self.data_generator_for_array:
                raise NameError(
                    "'{}' is not part of the schema '{}'.\nYou may have to add the '{}' "
                    "namespace or append the ':array' value.".format(name_type_array, self.schema, "null")
                )
            self.data_generator_for_array[name_type_array] = data_generator

    def set_data_generator_for_all_map_key_num(self, data_generator=ExponentialIntegerDataGenerator(beta=2, max_val=10)):
        """
        Set the data_generator for the number of key pairs. The default is an exponential integer data_generator.

        :param data_generator: The data generator.
        """
        logging.info("Set the data_generator for the number of keys in maps to: {0}".format(str(data_generator)))
        self.map_key_num_gen = data_generator

    @staticmethod
    def _get_num_of_branches_for_union(namespace_with_types):
        """
        Get the number of branches for a union.

        :param namespace_with_types: The namespace path with type and digit that describes the number of branches. E.g.
                                     com.linkedin.relevance.id:union:2. In this example the number is 2.
        :return: The number of branches for a union.
        """

        # The last item separated out by a colon is the number for unions
        return int(namespace_with_types.split(Generator.type_separator)[-1])

    def _set_uniform_data_generator_for_all_union(self):
        """
        Sets a uniform data_generator for all union types in their given range.
        """
        logging.info("Set uniform data_generator for all union types in their range.")
        for name_type in self.name_type_unions:
            self.data_generator_for_name_type[name_type] = UniformIntegerDataGenerator(
                min_val=0, max_val=Generator._get_num_of_branches_for_union(name_type) - 1
            )

    def _get_update_order(self):
        """
        Get the update order that respects the constraints.

        :return: A list of name types in an order that conditioned name types come first.
        """
        return self.dag.flatten() + list(self.unconditioned_name_types)

    def _get_dependent_names(self):
        """
        Get dependent names without the type ending.

        :return: A set of dependent names without type.
        """
        return {dependent.split(Generator.type_separator)[0] for dependent in self.dag.dependents()}

    def write(self, output_path, n_data, n_part, codec="deflate", name_format="part-{:05d}.avro"):
        """
        Use the generator to write data points (rows) in parts to the output path using a codec and name format.

        :param output_path: The output path. If it does not exist we create it.
        :param n_data: The number of data points.
        :param n_part: The number of parts.
        :param codec: The codec used for writing. Default is 'deflate'.
        :param name_format: The format string to create the filename. Default is 'part-{:05}.avro'. This follows the
                            format of the 'format' command in python.
        :raise ValueError: If the number of parts is larger than the number of data points or either < 0.
        """

        # Check the input values
        if n_part > n_data:
            raise ValueError("Number of parts '{}' < number of data '{}'".format(n_part, n_data))
        if n_part < 0:
            raise ValueError("Number of parts is '{}' but must be > '{}'".format(n_part, 0))
        if n_data < 0:
            raise ValueError("Number of data points is '{}' but must be > '{}'".format(n_data, 0))

        # If the output directory does not exist, create it
        if not os.path.exists(output_path):
            logging.info("Creating output directory '{}'.".format(output_path))
            os.makedirs(output_path)

        # Compute the number of rows per part and the update order for the value, which respects conditionals
        logging.info("Writing '{}' data points in '{}' parts to '{}'.".format(n_data, n_part, output_path))
        n_row = n_data // n_part
        update_order = self._get_update_order()
        dependent_names = self._get_dependent_names()

        # Write parts 0...n_part-2
        for i_part in range(n_part - 1):
            self._write_part(
                out_filename=os.path.join(output_path, name_format.format(i_part)),
                codec=codec,
                n_row=n_row,
                update_order=update_order,
                dependent_names=dependent_names,
            )

        # Write part n_part-1
        self._write_part(
            out_filename=os.path.join(output_path, name_format.format(n_part - 1)),
            codec=codec,
            n_row=n_data - (n_part - 1) * n_row,
            update_order=update_order,
            dependent_names=dependent_names,
        )

    def generate(self, n_data):
        """
        Generate data in memory.

        :param n_data: The number of data rows.

        :return: A list of data rows.
        """
        update_order = self._get_update_order()
        dependent_names = self._get_dependent_names()
        logging.info("The update order is '{}'".format(update_order))
        data = [None] * n_data
        for i_data in range(n_data):
            data[i_data] = self._create_datum(update_order=update_order, dependent_names=dependent_names)
        return data

    def _get_name_types(self, next_schema, namespace):
        """
        Get name type strings for all types within the schema.

        :param next_schema: The initial schema.
        :param namespace: The initial namespace.

        :return: A list of of strings that are fully qualified name:string_of_type. For instance, if we are using the
                 namespace: 'com.linkedin.browsemap' and have a field 'memberId' of type 'long' the string is
                    'com.linkedin.browsemap.memberId:long'.
        """
        name_type_tuples = []

        # If this schema is a primitive type find it and return a tuple of the name and type
        if isinstance(next_schema, PrimitiveSchema):
            logging.debug("Found type '{}'".format(next_schema.type))
            if (
                next_schema.type == "null"
                or next_schema.type == "boolean"
                or next_schema.type == "string"
                or next_schema.type == "bytes"
                or next_schema.type == "int"
                or next_schema.type == "long"
                or next_schema.type == "float"
                or next_schema.type == "double"
            ):
                name_type_tuples.append(namespace + Generator.type_separator + next_schema.type)
            else:
                logging.error("Found unknown type {0}".format(next_schema.type))

        # If this schema is a record parse its fields separately and add the name to the namespace
        elif isinstance(next_schema, RecordSchema):
            for field in next_schema.props["fields"]:
                logging.debug("Found property '{}'".format(field.name))
                name_type_tuples.extend(self._get_name_types(field.props["type"], namespace + Generator.namespace_separator + field.name))
            return name_type_tuples

        # If this schema is an array type follow the items
        elif isinstance(next_schema, ArraySchema):
            logging.debug("Found array")
            name_type_tuples.append(namespace + Generator.type_separator + Generator.array_name)
            name_type_tuples.extend(self._get_name_types(next_schema.props["items"], namespace + Generator.namespace_separator + next_schema.type))

        # If this schema is a union type we add tuples with all possible types
        elif isinstance(next_schema, UnionSchema):
            logging.debug("Found union")
            name_type_tuples.append(namespace + Generator.type_separator + "union" + Generator.type_separator + str(len(next_schema.schemas)))
            for schema in next_schema.schemas:
                name_type_tuples.extend(self._get_name_types(schema, namespace))

        # If this schema is a map type we add the type of the map with this namespace
        elif isinstance(next_schema, MapSchema):
            logging.debug("Found map")
            name_type_tuples.append(namespace + Generator.type_separator + Generator.map_name)
            name_type_tuples.extend(self._get_name_types(next_schema.props["values"], namespace + Generator.namespace_separator + next_schema.type))
        # None of the known schema types
        else:
            logging.error("Unknown type in schema {0}".format(next_schema))

        return name_type_tuples

    @staticmethod
    def _get_name_type_list(namespace, name_type):
        """
        Get the names and type for the name_type.

        :param name_type: The name type in one string separated by ':'. Names are separated by '.'.
        :return: A tuple. The first component is a list of names. The second component is a the type.
        """
        name_type_list = name_type[len(namespace) + 1 :].split(Generator.namespace_separator)  # Split on separator
        name_type_list += name_type_list.pop().split(Generator.type_separator)  # Split name type
        return name_type_list

    @staticmethod
    def _add_datum(data, datum):
        """
        Adds the datum to data map-list structure.

        :param data: The overall data.
        :param datum: The specific datum.
        :return: The overall data with datum added.
        """
        if isinstance(datum, dict):
            # If this dictionary is empty, return it
            if len(datum) == 0:
                return datum
            # If we don't have a map yet create one
            if data is None:
                data = {}
            for key in datum.keys():
                # If the key does not exist in the map, create it
                if key not in data:
                    data[key] = None
                # Assign the new data to the key
                data[key] = Generator._add_datum(data[key], datum[key])
                # Done
                return data
        elif isinstance(datum, list):
            # If we already have a list add to the existing items
            if isinstance(data, list):
                return [Generator._add_datum(item, new_item) for item, new_item in zip(data, datum)]
            # If we don't have a list create one and add new items
            else:
                return [Generator._add_datum(None, new_item) for new_item in datum]
        else:
            # If we don't have a list nor a map then this is the item
            return datum

    def _create_datum(self, update_order, dependent_names):
        """
        Creates a datum for a record/row.

        :param update_order: The update order for the attributes in this row that respects conditional data_generators.
        :param dependent_names: Dependent names that need to be stored to produce other values.
        :return: The datum for this record/row.
        """
        # Caches the lastly created value per full name type; full name types contain array indices and map keys
        datum_for_dependent_full_name_type = {}

        # Defines the number of items in an array per array present in the schema
        num_array_items = {name_type: generator.next() for name_type, generator in self.data_generator_for_array.items()}
        # num_array_items = {name_type: self.array_num_gen.next() for name_type in self.name_type_arrays}

        # Defines the number of entries in a map per map present in the schema
        num_map_values = {name_type: self.map_key_num_gen.next() for name_type in self.name_type_maps}

        # Number branches in union per union in the schema
        num_union_branches = {name_type: self.data_generator_for_name_type[name_type].next() for name_type in self.name_type_unions}

        # The initial data container for this avro record, empty map
        data = {}

        def _resolve_condition_full_name_type(name_type, full_name_type):
            """
            Resolving conditionals for a full name type means to add the indices and keys from arrays to uniquely
            reference a datum.

            :param name_type: The name type.
            :param full_name_type: The full name type which contains indices and keys.
            :return: The condition full name type.
            """
            condition_name_type = self.condition_for_name_type[name_type]
            condition_name_type_list = Generator._get_name_type_list(self.namespace, name_type=condition_name_type)
            condition_name_list = condition_name_type_list[:-1]  # Extract name
            condition_type = condition_name_type_list[-1]  # Extract type
            # Parse out the indices for arrays and values for maps and inject them properly into the condition name
            full_name_type_list = Generator._get_name_type_list(self.namespace, name_type=full_name_type)
            condition_full_name_type = self.namespace
            num_min = min(len(condition_name_list), len(full_name_type_list) - 1)
            for condition, name in zip(condition_name_list[:num_min], full_name_type_list[:num_min]):
                if condition == name:
                    condition_full_name_type += Generator.namespace_separator + condition
                else:
                    if name.startswith(Generator.array_name) or name.startswith(Generator.map_name):
                        condition_full_name_type += Generator.namespace_separator + name
                    else:
                        condition_full_name_type += Generator.namespace_separator + condition
            # For any remaining items in conditionals
            for condition in condition_name_list[num_min:]:
                condition_full_name_type += Generator.namespace_separator + condition
            # Add the type
            condition_full_name_type += Generator.type_separator + condition_type
            logging.debug("Resolved conditional for full name type {} is {}.".format(full_name_type, condition_full_name_type))
            return condition_full_name_type

        def _create_value(namespace, current_schema, name_type_list, full_name_type, i_names=0):
            """
            Creates a value within a datum.

            :param namespace: The namespace for all names when creating this value.
            :param current_schema: The current schema; will change with recursion.
            :param name_type_list: The name type list; won't change with recursion.
            :param full_name_type: The full name type which includes indices of arrays and keys of maps.
            :param i_names: The index into names.
            :return: The newly generated value.
            """

            logging.debug("Create data with namespace: '{}', full name type: '{}'.".format(namespace, full_name_type))

            if isinstance(current_schema, PrimitiveSchema):
                # Construct the name and name type
                name = self.namespace + Generator.namespace_separator + Generator.namespace_separator.join(name_type_list[:-1])
                name_type = name + Generator.type_separator + current_schema.type

                # Construct the full name type, has array indices and map key values in addition to the name type
                full_name_type += Generator.type_separator + name_type_list[-1]

                # Handle null
                if current_schema.type == "null":
                    # Store this datum only if it is a dependent
                    if name in dependent_names:
                        datum_for_dependent_full_name_type[full_name_type] = None
                    return None
                # Handle primitive avro types other than null
                elif (
                    current_schema.type == "boolean"
                    or current_schema.type == "string"
                    or current_schema.type == "bytes"
                    or current_schema.type == "int"
                    or current_schema.type == "long"
                    or current_schema.type == "float"
                    or current_schema.type == "double"
                ):

                    # If this data is conditioned on some other data, find the other data and generate the new data
                    if name_type in self.condition_for_name_type:
                        condition_full_name_type = _resolve_condition_full_name_type(name_type=name_type, full_name_type=full_name_type)

                        if condition_full_name_type not in datum_for_dependent_full_name_type:
                            raise RuntimeError(
                                "When generating {0} could not find conditional {1} in already "
                                "generated data. Check dependencies.".format(name_type, condition_full_name_type)
                            )
                        data_generator = self.data_generator_for_name_type[name_type]
                        if not isinstance(data_generator, ConditionedDataGenerator):
                            raise RuntimeError("Distribution type is {0} but expected {1}.".format(type(data_generator), ConditionedDataGenerator))
                        # The other data
                        conditional = datum_for_dependent_full_name_type[condition_full_name_type]
                        # Handle the null case here; occurs if we have different branches in a union where one is null
                        # the other branch is not
                        if conditional is None:
                            value = None
                        else:
                            value = data_generator.next(datum_for_dependent_full_name_type[condition_full_name_type])
                    else:
                        value = self.data_generator_for_name_type[name_type].next()
                    # Store this datum only if it is a dependent
                    if name in dependent_names:
                        datum_for_dependent_full_name_type[full_name_type] = value
                    return value
                else:
                    logging.error("Found unknown type {0}".format(current_schema.type))

            # Get the name up to this part
            name = namespace + Generator.namespace_separator + Generator.namespace_separator.join(name_type_list[:i_names])

            # In case of a union just follow the branch given, keep the names the same (don't prepend nor inc i_names)
            if isinstance(current_schema, UnionSchema):
                name_type = name + Generator.type_separator + current_schema.type + Generator.type_separator + str(len(current_schema.schemas))
                # Select a branch according to the randomization above
                i_schemas = num_union_branches[name_type]
                return _create_value(
                    namespace=namespace,
                    current_schema=current_schema.schemas[i_schemas],
                    name_type_list=name_type_list,
                    full_name_type=full_name_type,
                    i_names=i_names,
                )

            # In all other cases with a name for the current field and the condition
            field_name = name_type_list[i_names]

            if isinstance(current_schema, RecordSchema):
                # Find the field/attribute in the record with the corresponding name
                fields = current_schema.props["fields"]
                i_field_name = -1
                for i_field, field in enumerate(fields):
                    if field.name == field_name:
                        i_field_name = i_field
                        break
                if i_field_name == -1:
                    raise ValueError("Could not find field '{}' in '{}'.".format(field_name, fields))
                # Recurse down that found field/attribute
                return {
                    field_name: _create_value(
                        namespace=namespace,
                        current_schema=fields[i_field_name].props["type"],
                        name_type_list=name_type_list,
                        full_name_type=full_name_type + Generator.namespace_separator + field_name,
                        i_names=i_names + 1,
                    )
                }

            if isinstance(current_schema, ArraySchema):
                # Define the name type
                name_type = name + Generator.type_separator + current_schema.type
                # Get the number of items from the above map
                num_items = num_array_items[name_type]
                # Create a empty of None items for the number of items
                items = [None] * num_items
                for i_items in range(num_items):
                    items[i_items] = _create_value(
                        namespace=namespace,
                        current_schema=current_schema.props["items"],
                        name_type_list=name_type_list,
                        full_name_type=(
                            full_name_type + Generator.namespace_separator + field_name + Generator.array_item_start + str(i_items) + Generator.array_item_end
                        ),
                        i_names=i_names + 1,
                    )
                return items

            if isinstance(current_schema, MapSchema):
                # Construct the full name type
                name_type = name + Generator.type_separator + current_schema.type
                # Get the number of key,value pairs for the map
                num_values = num_map_values[name_type]
                # The map values
                values = {}
                # TODO: Add keys in the map that are drawn from a data_generator too; I've fixed them to be indices
                for i_values in range(num_values):
                    values[str(i_values)] = _create_value(
                        namespace=namespace,
                        current_schema=current_schema.props["values"],
                        name_type_list=name_type_list,
                        full_name_type=(
                            full_name_type + Generator.namespace_separator + field_name + Generator.map_value_start + str(i_values) + Generator.map_value_end
                        ),
                        i_names=i_names + 1,
                    )
                return values

        # Here is where the main data generation loop starts
        for name_type in update_order:
            # Construct the name type list from the name type value
            name_type_list = Generator._get_name_type_list(namespace=self.namespace, name_type=name_type)

            # Create the new datum, internally this method will use the same number of items in arrays for all
            # attributes in the item type; also it will use the same number of map keys for all attributes of a map
            # value; assuming that both the array items and map values are avro record types
            new_datum = _create_value(namespace=self.namespace, current_schema=self.schema, name_type_list=name_type_list, full_name_type=self.namespace)
            # Some debug information
            logging.debug("Datum for name-type '{}' is '{}'.".format(name_type, new_datum))
            data = Generator._add_datum(data, new_datum)
            logging.debug("Data is '{}'.".format(data))

        return data

    def _write_part(self, out_filename, codec, n_row, update_order, dependent_names):
        """
        Writes a part to local disk. This does not support direct writing to hdfs -- on purpose. We expect this
        generator to be used for local mock-up data only. Do not put such data onto HDFS.

        :param out_filename: The output filename.
        :param codec: The compression codec when writing avro files.
        :param n_row: The number of rows for this part.
        :param update_order: The order in which data is updated in a row to respect dependencies.
        :param dependent_names: The dependent names -- some other datum depends on this one.
        """
        with open(out_filename, "wb") as out:
            # Open the writer
            writer = DataFileWriter(out, DatumWriter(), writer_schema=Parse(self.json_schema), codec=codec)
            for i_row in range(n_row):
                writer.append(self._create_datum(update_order=update_order, dependent_names=dependent_names))
            writer.close()

    @staticmethod
    def create(schema_file):
        """
        Create an avro data generator for a given schema file.

        :param schema_file: The given schema file.

        :return: AvroDataGenerator.
        """
        json_schema = open(schema_file, "rb").read()
        return Generator(json_schema)

    @staticmethod
    def _write_data_part(data, out_filename, json_schema, codec, i_start_row, i_end_row):
        """
        Write data part using avro-python3.

        :param data: The data.
        :param out_filename: The output file name for this part.
        :param json_schema: The json schema that the data is formatted in.
        :param codec: The codec used to write the avro file.
        :param i_start_row: The first row (inclusive) where to start writing data from.
        :param i_end_row: The last row (exclusive) where to stop writing data from.
        """
        with open(out_filename, "wb") as out:
            writer = DataFileWriter(out, DatumWriter(), writer_schema=Parse(json_schema), codec=codec)
            for i_row in range(i_start_row, i_end_row):
                writer.append(data[i_row])
            writer.close()

    @staticmethod
    def write_data(data, output_path, n_part, json_schema, codec="deflate", name_format="part-{:05d}.avro"):
        """
        Write the data to the output path in n_part parts.

        :param data: The data.
        :param output_path: The output path.
        :param n_part: The number of parts.
        :param json_schema: The json schema string for the avro data.
        :param codec: The codec to write the avro. Default is 'deflate'.
        :param name_format: The name format for the avro files. Default is 'part-{:05d}.avro'. The format follows the
                            'format' command in python.

        :raise ValueError: If the number of parts is larger than the number of data points or either < 0.
        """
        n_data = len(data)

        # Check the input values
        if n_part > n_data:
            raise ValueError("Number of parts '{}' < number of data '{}'".format(n_part, n_data))
        if n_part < 0:
            raise ValueError("Number of parts is '{}' but must be > '{}'".format(n_part, 0))
        if n_data < 0:
            raise ValueError("Number of data points is '{}' but must be > '{}'".format(n_data, 0))

        if not os.path.exists(output_path):
            logging.info("Creating output directory '{}'.".format(output_path))
            os.makedirs(output_path)

        logging.info("Writing '{}' data points in '{}' parts to '{}'.".format(n_data, n_part, output_path))
        n_row = n_data // n_part  # Number of rows for parts 0...n_part-1
        for i_part in range(n_part - 1):
            Generator._write_data_part(
                data,
                out_filename=os.path.join(output_path, name_format.format(i_part)),
                json_schema=json_schema,
                codec=codec,
                i_start_row=i_part * n_row,
                i_end_row=(i_part + 1) * n_row,
            )
        # Write part n_part-1
        Generator._write_data_part(
            data,
            out_filename=os.path.join(output_path, name_format.format(n_part - 1)),
            json_schema=json_schema,
            codec=codec,
            i_start_row=(n_part - 1) * n_row,
            i_end_row=min(n_part * n_row, n_data),
        )
