/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_DATA_PREFIX_TREE_H_
#define TENSORFLOW_DATA_PREFIX_TREE_H_

#include <memory>
#include <string>
#include <vector>

namespace tensorflow {
namespace data {

class PrefixTreeNode;
using PrefixTreeNodeSharedPtr = std::shared_ptr<PrefixTreeNode>;

// A prefix tree node
class PrefixTreeNode {
 public:
  // Constructs a prefix tree node for the given prefix and father
  PrefixTreeNode(const std::string& prefix = "",
                 PrefixTreeNode* father = nullptr);

  // Get the children of this tree node
  // TODO(fraudies): For better performance convert this into an iterator
  inline std::vector<PrefixTreeNodeSharedPtr> GetChildren() const {
    return children_;
  }

  // Get the prefix for this tree node
  inline std::string GetPrefix() const { return prefix_; };

  // Get's the full name using the separator between prefix names of tree nodes
  // Note, that this excludes the root node which is an auxiliary node
  std::string GetName(char separator) const;

  // Is terminal if this node has no children
  inline bool IsTerminal() const { return children_.size() == 0; }

  // Find the prefix tree for the given child prefix
  PrefixTreeNodeSharedPtr Find(const std::string& child_prefix) const;

  // Find the prefix tree node and if it does not exist, add it
  PrefixTreeNodeSharedPtr FindOrAddChild(const std::string& child_prefix);

  // Return a string representation of this tree node in human readable format
  std::string ToString(int level) const;

 private:
  // The prefix for this tree node
  std::string prefix_;

  // The father of this tree node -- always exists in memory
  PrefixTreeNode* father_;

  // The children of this prefix tree node
  std::vector<PrefixTreeNodeSharedPtr> children_;
};

// An ordered prefix tree maintains the order of it's children
// This order is defined by
class OrderedPrefixTree {
 public:
  // Construct a prefix tree with the optional root_name
  OrderedPrefixTree();

  // Get the root of the prefix tree
  inline PrefixTreeNodeSharedPtr GetRoot() const { return root_; }

  // Tries to insert these prefixes; no changes are made to the tree if they
  // don't exist yet
  void Insert(const std::vector<std::string>& prefixes);

  // Builds an ordered prefix tree for the provided list of prefixes.
  static void Build(OrderedPrefixTree* tree,
                    const std::vector<std::vector<std::string>>& prefixes_list);

  // Will return the node as far as the prefixes could be matched and put the
  // unmatched part in remaining
  PrefixTreeNodeSharedPtr FindNearest(
      std::vector<std::string>* remaining,
      const std::vector<std::string>& prefixes) const;

  // Returns tree node if found otherwise nullptr
  PrefixTreeNodeSharedPtr Find(const std::vector<std::string>& prefixes) const;

  // Returns a string representation of the ordered prefix tree in a human
  // readable format
  std::string ToString() const;

 private:
  // The root node for this prefix tree
  PrefixTreeNodeSharedPtr root_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_DATA_PREFIX_TREE_H_
