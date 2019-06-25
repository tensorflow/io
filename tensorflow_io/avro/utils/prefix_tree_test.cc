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
#include "gtest/gtest.h"
#include "tensorflow_io/avro/utils/prefix_tree.h"

namespace tensorflow {
namespace data {

// ------------------------------------------------------------
// Tests for a tree node
// ------------------------------------------------------------
TEST(PrefixTreeNodeTest, IsTerminal) {
  PrefixTreeNode node("father");
  EXPECT_TRUE(node.IsTerminal());
  node.FindOrAddChild("child");
  EXPECT_TRUE(!node.IsTerminal());
}

TEST(PrefixTreeNodeTest, HasPrefix) {
  PrefixTreeNode wout;
  EXPECT_TRUE(!wout.HasPrefix());
  PrefixTreeNode with("name");
  EXPECT_TRUE(with.HasPrefix());
}

TEST(PrefixTreeNodeTest, GetPrefix) {
  PrefixTreeNode node("name");
  EXPECT_EQ(node.GetPrefix(), "name");
}

// Tests: Find, FindOrAdd, GetPrefix
TEST(PrefixTreeNodeTest, SingleChild) {
  PrefixTreeNode node("father");
  // Expect the child does not exist
  EXPECT_TRUE(!node.Find("child"));
  // Insert the child
  node.FindOrAddChild("child");
  // Child must be present now
  PrefixTreeNodeSharedPtr child(node.Find("child"));
  EXPECT_TRUE(child != nullptr);
  EXPECT_EQ((*child).GetPrefix(), "child");
  // Check the name
  EXPECT_EQ((*child).GetName('.'), "father.child");
}

TEST(PrefixTreeNodeTest, GetChildren) {
  PrefixTreeNode node("father");
  node.FindOrAddChild("child1");
  node.FindOrAddChild("child2");
  node.FindOrAddChild("child3");
  std::vector<PrefixTreeNodeSharedPtr> children(node.GetChildren());
  int n_child = 3;
  EXPECT_EQ(children.size(), n_child);
  std::vector<std::string > names{"child1", "child2", "child3"};
  for (int i_child = 0; i_child < n_child; ++i_child) {
    EXPECT_EQ((*children[i_child]).GetPrefix(), names[i_child]);
  }
}


// ------------------------------------------------------------
// Tests for an ordered prefix tree
// ------------------------------------------------------------
TEST(OrderedPrefixTree, GetRootPrefix) {
  OrderedPrefixTree wout;
  EXPECT_EQ(wout.GetRootPrefix(), "");

  OrderedPrefixTree with("namespace");
  EXPECT_EQ(with.GetRootPrefix(), "namespace");
}

TEST(OrderedPrefixTree, BuildEmpty) {
  std::vector< std::vector<std::string> > prefixes_list;
  OrderedPrefixTree tree;
  OrderedPrefixTree::Build(&tree, prefixes_list);
}

TEST(OrderedPrefixTree, BuildSmall) {
  std::vector< std::vector<std::string> > prefixes_list{{"com"}};
  std::vector< std::string > present{"com"};
  std::vector< std::string > absent{"nothing"};
  OrderedPrefixTree tree;
  OrderedPrefixTree::Build(&tree, prefixes_list);

  // Check for present prefixes
  PrefixTreeNodeSharedPtr node(tree.Find(present));

  EXPECT_TRUE(node);
  EXPECT_EQ((*node).GetPrefix(), "com");

  // Check for absent prefixes
  EXPECT_TRUE(!tree.Find(absent));
}

TEST(OrderedPrefixTree, BuildLarge) {
  std::vector< std::vector<std::string> > prefixes_list{{"com", "google", "search"},
    {"com", "linkedin", "jobs"}, {"com", "linkedin", "members"}};
  std::vector< std::string > present_with_remaining{"com", "google", "search", "cloud"};
  std::vector< std::string > present_partial_match{"com", "google"};
  std::vector< std::string > present_full_match{"com", "linkedin", "members"};
  std::vector< std::string > absent{"com", "linkedin", "members", "us"};
  OrderedPrefixTree tree;
  OrderedPrefixTree::Build(&tree, prefixes_list);

  // Check for present prefixes
  EXPECT_TRUE(tree.Find(present_partial_match));
  EXPECT_TRUE(tree.Find(present_full_match));

  // Check for absent prefixes
  EXPECT_TRUE(!tree.Find(absent));

  // Check that the partial match returns the right remaining
  std::vector<std::string> remaining;
  // A partial match returns false and the remaining part matches cloud
  EXPECT_TRUE(tree.FindNearest(&remaining, present_with_remaining));
  EXPECT_EQ(remaining.size(), 1);
  EXPECT_EQ(remaining.front(), "cloud");
}

TEST(OrderedPrefixTree, ToString) {
  std::vector< std::vector<std::string> > prefixes_list{{"com", "google", "search"},
    {"com", "linkedin", "jobs"}, {"com", "linkedin", "members"}};
  OrderedPrefixTree tree("namespace");
  OrderedPrefixTree::Build(&tree, prefixes_list);
  const std::string expected =
    "|---namespace\n"
    "|   |---com\n"
    "|   |   |---google\n"
    "|   |   |   |---search\n"
    "|   |   |---linkedin\n"
    "|   |   |   |---jobs\n"
    "|   |   |   |---members\n";

  EXPECT_EQ(tree.ToString(), expected);
}

}
}