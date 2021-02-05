import inspect


def parse_lambda_constraint_into_intervals(lambda_constraint, dtype=int):
    """
    Parses a lambda constraint function into its intervals.

    Supported lambda formats are:

    lambda x: a < x < b
    lambda x: b > x > a
    lambda x: a <= x <= b
    lambda x: b >= x >= a
    lambda x: a <= x <= b or c < x < d
    ...

    :param lambda_constraint: The lambda constraint from the source code.
    :param dtype: Data type

    :return: Intervals from all parsed parts. E.g. 'lambda x: a <= x <= b or c < x < d' returns [[a, b], [c, d]].

    :raise ValueError whenever the above format is not followed.
    """

    source = inspect.getsourcelines(lambda_constraint)
    source_line_number = source[1]
    source_code_str = source[0][0]
    source_file = inspect.getsourcefile(lambda_constraint)

    # Parse the source code into intervals
    lambda_name = "lambda"
    lt_name = "<"
    gt_name = ">"
    let_name = "<="
    get_name = ">="
    or_name = "or"
    and_name = "and"
    closing_parenth_name = ")"
    function_separator_name = ":"
    supported_dtypes = [int]

    def check_supported_dtype(dtype):
        if dtype not in supported_dtypes:
            raise ValueError("'{}' is not supported for constraint {} in line {} of file '{}'.".format(dtype, source_code_str, source_line_number, source_file))

    def check_no_and(function_body_str):
        # If we find and throw an error
        and_index = function_body_str.find(and_name)
        if and_index != -1:
            raise ValueError("'{}' not supported in constraint {} in line {} of file '{}'.".format("and", source_code_str, source_line_number, source_file))

    def extract_var_name_and_function_body(lambda_function_str):
        # Parse variable name
        function_index = lambda_function_str.find(function_separator_name)
        if function_index == -1:
            raise ValueError("Missing '{}' in constraint '{}' in line {} of file '{}'.".format(":", source_code_str, source_line_number, source_file))
        return lambda_function_str[:function_index], lambda_function_str[function_index + 1 :]

    def extract_lambda_function_str(source_code_str):
        # Strip off anything before the lambda
        lambda_index = source_code_str.rfind(lambda_name)
        if lambda_index == -1:
            raise ValueError(
                "Missing keyword '{}' in constraint '{}' in line {} of file '{}'.".format("lambda", source_code_str, source_line_number, source_file)
            )
        lambda_index += len(lambda_name) + 1

        # Strip off any trailing characters after the closing parenthesis
        lambda_function_str = source_code_str[lambda_index:]
        closing_parenth_index = lambda_function_str.find(closing_parenth_name)
        if closing_parenth_index >= 0:
            lambda_function_str = lambda_function_str[:closing_parenth_index]

        # Remove any leading or trailing blank spaces
        lambda_function_str = lambda_function_str.strip()

        # Remove closing parenthesis from definition
        # lambda_function_str = lambda_function_str[:-1]

        return lambda_function_str

    def find_lhs_constant(lhs_str, constraint_name):
        index = lhs_str.find(constraint_name)
        if index != -1:
            return lhs_str[:index].strip()
        else:
            return None

    def find_rhs_constant(rhs_str, constraint_name):
        index = rhs_str.find(constraint_name)
        if index != -1:
            return rhs_str[index + len(constraint_name) :].strip()
        else:
            return None

    def parse_lhs_into_interval(lhs_str, interval):
        # TODO: Handle types other than integer

        lhs_constant_str = find_lhs_constant(lhs_str=lhs_str, constraint_name=let_name)  # x <=
        if lhs_constant_str is not None and interval[0] is None:
            return [int(lhs_constant_str), interval[1]]

        lhs_constant_str = find_lhs_constant(lhs_str=lhs_str, constraint_name=lt_name)  # x <
        if lhs_constant_str is not None and interval[0] is None:
            return [int(lhs_constant_str) + 1, interval[1]]

        lhs_constant_str = find_lhs_constant(lhs_str=lhs_str, constraint_name=get_name)  # x >=
        if lhs_constant_str is not None and interval[1] is None:
            return [interval[0], int(lhs_constant_str)]

        lhs_constant_str = find_lhs_constant(lhs_str=lhs_str, constraint_name=gt_name)  # x >
        if lhs_constant_str is not None and interval[1] is None:
            return [interval[0], int(lhs_constant_str) - 1]

        raise ValueError("Could not parse lhs constraint '{}' in line {} of file '{}'.".format(lhs_str, source_line_number, source_file))

    def parse_rhs_into_interval(rhs_str, interval):
        rhs_constant_str = find_rhs_constant(rhs_str=rhs_str, constraint_name=let_name)  # <= x
        if rhs_constant_str is not None and interval[1] is None:
            return [interval[0], int(rhs_constant_str)]

        rhs_constant_str = find_rhs_constant(rhs_str=rhs_str, constraint_name=lt_name)  # < x
        if rhs_constant_str is not None and interval[1] is None:
            return [interval[0], int(rhs_constant_str) - 1]

        rhs_constant_str = find_rhs_constant(rhs_str=rhs_str, constraint_name=get_name)  # >= x
        if rhs_constant_str is not None and interval[0] is None:
            return [int(rhs_constant_str), interval[1]]

        rhs_constant_str = find_rhs_constant(rhs_str=rhs_str, constraint_name=gt_name)  # > x
        if rhs_constant_str is not None and interval[0] is None:
            return [int(rhs_constant_str) + 1, interval[1]]
        raise ValueError("Could not parse rhs constraint '{}' in line {} of file '{}'.".format(rhs_str, source_line_number, source_file))

    def constraint_str_to_interval(constraint_str, var_name):
        equals = constraint_str.split(var_name)
        if len(equals[0]) == 0 or len(equals[1]) == 0:
            raise ValueError(
                "Found constraint '{}' in line {} of file '{}'. Only support for format {}.".format(
                    constraint_str, source_line_number, source_file, "a < x < b"
                )
            )
        interval = parse_lhs_into_interval(lhs_str=equals[0], interval=[None, None])
        interval = parse_rhs_into_interval(rhs_str=equals[1], interval=interval)
        if interval[0] is None or interval[1] is None or interval[1] < interval[0]:
            # Check interval
            raise ValueError("Interval [{}, {}] is either missing a boundary or is empty.", interval[0], interval[1])
        return interval

    # Check that we work with only supported data types
    check_supported_dtype(dtype=dtype)

    # Extract the lambda function string from the source code string
    lambda_function_str = extract_lambda_function_str(source_code_str=source_code_str)

    # Extract the variable name and the function body from the lambda function string
    var_name, function_body_str = extract_var_name_and_function_body(lambda_function_str=lambda_function_str)

    # Check that there are no 'and's used
    check_no_and(function_body_str=function_body_str)

    # Parse logical constraints (split on or's)
    constraints_str = function_body_str.split(or_name)

    # Parse all constraint strings into intervals
    return [constraint_str_to_interval(constraint_str.strip(), var_name) for constraint_str in constraints_str]
