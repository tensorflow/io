import logging


class Dag:
    """
    Representation of a DAG which is optimized for:
    i) Dynamic add of edges and detection of multiples and cycles.
    ii) Flattening the DAG where dependent nodes come first.
    """

    def __init__(self):
        """
        Initialize a DAG which we represent through a map of: target -> {source1, source2,...}.
        """
        self.depends_on = {}

    def get_cycle(self, source, target):
        """
        Returns a cycle as list if adding the edge source to target would create one; otherwise returns an empty list.

        :param source: The source node of the edge.
        :param target: The target node of the edge.
        :return: A list with the cycle, if one exists; otherwise an empty list.
        """
        # Trick: Run the problem backwards (since that is the direction we have)
        cycle = []
        if source in self.depends_on:
            for next_source in self.depends_on[source]:
                if next_source == target:
                    return [source, next_source]
                cycle = self.get_cycle(next_source, target)
                if len(cycle) > 0:
                    cycle.insert(0, source)
                    return cycle
        return cycle

    def add_edge(self, from_vertex, to_vertex):
        """
        Add a directed edge from from_vertex to to_vertex.
        Throws ValueError if
        * the edge already exists OR
        * by adding the edge a cycle would be created.

        :param from_vertex: From vertex of the edge.
        :param to_vertex: To vertex of the edge.
        :raise ValueError: When adding the edge would create a cycle.
        """
        cycle = self.get_cycle(source=to_vertex, target=from_vertex)
        if len(cycle) > 0:
            raise ValueError("Cyclic dependency between {0} and {1} through {2}.".format(from_vertex, to_vertex, cycle))
        if from_vertex in self.depends_on:
            dependencies = self.depends_on[from_vertex]
            if to_vertex in dependencies:
                logging.warning("Edge from {0} to {1} already exists.".format(from_vertex, to_vertex))
            self.depends_on[from_vertex].add(to_vertex)
        else:
            self.depends_on[from_vertex] = {to_vertex}

    def remove_edge(self, from_vertex, to_vertex):
        """
        Removes the edge from 'from vertex' to 'to vertex'.

        :param from_vertex: From vertex of the edge.
        :param to_vertex: To vertex of the edge.
        :raise Raises a ValueError if the edge does not exist.
        """
        if from_vertex in self.depends_on and to_vertex in self.depends_on[from_vertex]:
            self.depends_on[from_vertex].remove(to_vertex)
            if len(self.depends_on[from_vertex]) == 0:
                del self.depends_on[from_vertex]
        else:
            raise ValueError("Edge from {0} to {1} does not exist.".format(from_vertex, to_vertex))

    def dependents(self):
        """
        Get vertices that are dependents to other vertices in a set.

        :return: Set of dependents.
        """
        dependent_vertices = set()
        for neighbors in self.depends_on.values():
            dependent_vertices |= neighbors
        return dependent_vertices

    def flatten(self):
        """
        We define the flattened version of a DAG through its topological sort. For more details see here:
        https://en.wikipedia.org/wiki/Topological_sorting.

        Notice, this still leaves some ambiguities. To resolve these we use the order that the python set uses.

        :return: A flattened list representation of the DAG where dependencies come first.
        """
        # Idea: Count the dependencies for each node in the DAG and use an inverse mapping to decrease the count
        # Inverse mapping
        is_dependency_of = {}
        for node, dependencies in self.depends_on.items():
            for dependency in dependencies:
                if dependency not in is_dependency_of:
                    is_dependency_of[dependency] = [node]
                else:
                    is_dependency_of[dependency].append(node)
        num_dependents = {name_type: len(conditionals) for name_type, conditionals in self.depends_on.items()}
        is_never_dependency = set(self.depends_on.keys()) - set(is_dependency_of.keys())
        has_no_dependents = set(is_dependency_of.keys()) - set(self.depends_on.keys())

        # Add name_type's that depend on nothing (end points)
        for node in has_no_dependents:
            num_dependents[node] = 0

        # Add name_type's that have no dependency (start points)
        for node in is_never_dependency:
            is_dependency_of[node] = []

        # Create the update order until all name_type's have been added
        update_order = []
        while len(num_dependents) > 0:
            no_dependencies = []
            for node, n_depends in num_dependents.items():
                if n_depends == 0:
                    no_dependencies.append(node)

            for node in no_dependencies:
                del num_dependents[node]
                for dependent in is_dependency_of[node]:
                    num_dependents[dependent] -= 1

            # sort to guarantee deterministic update order
            update_order += sorted(no_dependencies)

        return update_order
