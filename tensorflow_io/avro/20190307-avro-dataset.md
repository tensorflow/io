# Avro Dataset for TensorFlow IO

| Status        | Accepted                                                          |
| :------------ | :---------------------------------------------------------------- |
| **Author(s)** | florian.raudies@gmail.com                                         |
| **Sponsor**   | N/A                                                               |
| **Updated**   | 2019-03-07                                                        |
| **Obsoletes** | N/A                                                               |

## Overview

Load [Avro](https://avro.apache.org/docs/current/) formatted data natively into  [TensorFlow](https://www.tensorflow.org/) from file systems that are supported by TensorFlow, e.g. the Hadoop Distributed File System ([HDFS](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html)).

## Motivation

Loading Avro formatted data will enable the data science community to leverage TensorFlow more easily. Currently, loading Avro formatted data into TensorFlow is not supported. Thus, ways of getting data into TensorFlow requires a conversion to other data formats, e.g. protobuf, CSV, or proprietary formats. Such custom solutions are difficult to maintain and do not generalize well for other uses.

## Design Proposal

The Avro dataset shall support

- Transform primitive and nested Avro types into TensorFlow sparse and dense tensors
- The creation of dimensions by (see also examples in Tables 2 and 3)
  - Combining of several rows -- also called batching
  - Combining several Avro fields to define indices and values for sparse tensors
  - Combining and reshaping of elements of an Avro non-nested array into a multi-dimensional tensor using row-major-order
  - Nestings of Avro typed arrays
- Filling in scalars or tensors for missing values whenever the number of provided elements does not match the desired number of elements by using default values -- if no defaults or the wrong defaults are provided this results in a runtime error surfaced through an exception in python and erroneous status returned in c/c++
- Full support for N-dimensional dense tensors and N-dimensional sparse tensors
- A mapping of Avro primitive types to TensorFlow primitive types (see Table 1)
- For Avro&#39;s null type the default shall be assumed if provided; otherwise fail
- An API/syntax that follows the one in Spark/Avro tooling that allows to resolve
  - Elements in an array
  - Elements in a map
  - Nested records
  - Namespaces
  - Branches in unions -- will resolve them but will use strict typing which will render them useless whenever an attribute has multiple types or whenever a null is encountered. Note, TensorFlow uses strict typing for it&#39;s tensors.
- An API for filtering in arrays
- The mapping of Avro enumerations to TensorFlow strings carrying the symbol name
- The mapping of Avro fixed types to TensorFlow strings
- The same operating systems as supported by TensorFlow -- Note, that this only accounts for the development of the avro dataset and not it&#39;s dependencies
- Python 2.x and 3.x following TensorFlow&#39;s requirements
- Thread safety; Note that TensorFlow&#39;s  dataset API may use multiple threads

| **Avro primitive type** | **TensorFlow primitive type** |
| --- | --- |
| null: no value | If provided default is assumed otherwise fail |
| boolean: a binary value | tf.bool |
| int: 32-bit signed integer | tf.int32 |
| long: 64-bit signed integer | tf.in64 |
| float: single precision 32-bit IEEE floating point number | tf.float32 |
| double: double precision 64-bit IEEE floating point number | tf.float64 |
| fixed: a fixed length byte sequence | tf.string |
| bytes: a sequence of 8-bit unsigned bytes | tf.string |
| string: unicode character sequence | tf.string |
| enum: enumeration | tf.string |
**Table 1** lists the mapping of Avro primitive types to TensorFlow primitive types.

The Avro dataset shall NOT support

- --Type promotion or type coercion when mapping from Avro types to TensorFlow types, instead we use strict typing here
- --Recursive type definitions in avro
- --The compile and distribution of TensorFlow external tools that are necessary; e.g. libavro.so. It assumed that the user will compile and distribute this library separately.

## Design Details

The design shall leverage existing software as follows

- For the API leverage the tf.dataset natively and corresponding python classes
- Leverage the TensorFlow filesystem natively. Note, that currently meta information for file systems are mostly provided through environment variables (e.g. HDFS). For more information follow the special interest group (SIG) tensorflow.io and these links: [Hadoop](https://www.tensorflow.org/deploy/hadoop) and [gfile](https://www.tensorflow.org/api_docs/python/tf/gfile) .
- Leverage Avro&#39;s c API. Note, the c++ API for Avro depends on boost which the TensorFlow author&#39;s do not want to depend on nor maintain
- Leverage TensorFlow&#39;s build system for dependency management and build
- Leverages Google&#39;s regex re2 package to parse user-provided strings to resolve record nestings, namespaces, array and map indexing
- Follow google&#39;s c++ style guide and google&#39;s python style guide
- We develop against Avro 1.8.2 with bugfixes for jansson build -- which is the latest stable release; see [avroc](https://github.com/apache/avro/tree/release-1.8.2/lang/c)

### API Design

To showcase the API we define an example record of a person that has a name, address, friends, jobs and coworkers at jobs, and cars, where a car is defined through its color and engine (see Listing 1). Note, that we formally defined the record recursively to keep the example short--however, recursive type definitions are not supported by the avro dataset.

```json
{  
   "doc":"Person with friends and coworkers",
   "namespace":"com.test",
   "type":"record",
   "name":"person",
   "fields":[  
      {  
         "name":"name",
         "type":"record",
         "fields":[  
            {  
               "name":"first",
               "type":"string"
            },
            {  
               "name":"initial",
               "type":"string"
            },
            {  
               "name":"second",
               "type":"string"
            }
         ]
      },
      {  
         "name":"address",
         "type":"record",
         "fields":[  
            {  
               "name":"street",
               "type":"string"
            }
         ]
      },
      {  
         "name":"gender",
         "type":"string"
      },
      {  
         "name":"friends",
         "type":"array",
         "items":"com.test.person"
      },
      {  
         "name":"jobs",
         "type":"array",
         "items":{  
            "name":"coworkers",
            "type":"array",
            "items":"com.test.person"
         }
      },
      {  
         "name":"car",
         "type":"record",
         "fields":[  
            {  
               "name":"color",
               "type":"string"
            },
            {  
               "name":"engine",
               "type":"record",
               "fields":[  
                  {  
                     "name":"id",
                     "type":"int"
                  },
                  {  
                     "name":"power",
                     "type":"float"
                  },
                  {  
                     "name":"cylinders",
                     "type":"int"
                  },
                  {  
                     "name":"sparkplugs",
                     "type":"int"
                  }
               ]
            }
         ]
      },
      {  
         "name":"cars",
         "type":"map",
         "values":"com.test.car"
      }
   ]
}
```
**Listing 1** defines a person with name, address, friends, jobs, and cars.  Note, that the definition of the nested person would require a union with a nested type to be valid--but to simplify we left this out here.

We start of the discussion of the API through examples (see Table 2 and Table 3). These examples list the key expressions together with an explanation. The status indicates whether this is a new, or the same as in the existing API and if such a change would be backward incompatible.

| **Key Expressions** | **Brief Explanation** |
| --- | --- |
| name.first | Get the first name of the name record. |
| friends[2].name.first | Get the 3rd friend&#39;s first name. |
| friends[\*].name.first | Get all friend&#39;s first names. |
| friends[\*].address[\*].street | Get all friend&#39;s addresses street name. In this example we have two dimensions. |
| friends[\*].jobs[\*].coworkers[\*].name.first | Get the first names of all coworkers from all jobs that your friends have had. In this example we have three dimensions. Each \* adds a dimension. |
| cars[&#39;nickname&#39;].color | Get the color of your car with that nickname from a map. |
| friends[gender=&#39;unknown&#39;].name.first | Get the first name of all friends whose gender is unknown. |
| friends[name.first=name.last].name.initial | Get initials from all the friends where their first name matches their last name. |
| friends[name.first=@name.first].name.initial | Get initials from all friends that have the same first name as the person itself. Need some identifier to indicate that we would like to start searching from the root--here @. |

**Table 2** describes the key expressions and lists their status. New means that this was not supported before. Same means it already exists. Break means that this would introduce a backward incompatible change.

| **Key Expression** | **Feature Mapping** | **Shape** |
| --- | --- | --- |
| name.first | FixedLenFeature(shape=[], dtype=tf.string) | 1 |
| friends[2].name.first | FixedLenFeature(shape=[], dtype=tf.string) | 1 |
| friends[\*].name.first | VarLenFeature(dtype=tf.string) | N |
| friends[\*].address[\*].street | FixedLenFeature(shape=[1, 2], dtype=tf.string, default\_value= [[&#39;35 Park Street&#39;, &#39;275 California Street&#39;], [&#39;950 Maude Ave&#39;, &#39;1000 Moore Parkway&#39;]]) | 1 x 2 |
| friends[\*].address[\*].street | VarLenFeature(dtype=tf.string) | Sparse 2D |
| friends[\*].address[\*].street | FixedLenSequenceFeature(shape=[1, 2], dtype=tf.string, default\_value=&quot;Park Street 35&quot;, allow\_missing=True) | 1 x 2 |
| friends[\*].jobs[\*].coworkers[\*].name.first | FixedLenFeature(shape=[1, 2, 3], dtype=tf.string, default\_value=[[[&#39;August&#39;, &#39;Astro&#39;, &#39;Anan&#39;],[&#39;Ben&#39;, &#39;Bob&#39;, &#39;Bart&#39;]], [[&#39;Carl&#39;, &#39;Claus&#39;, &#39;Castor&#39;],[&#39;Dan&#39;, &#39;Dave&#39;, &#39;Donald&#39;]]) | 1 x 2 x 3 |
| engine | SparseFeature(index\_key=&#39;id&#39;, value\_key=&#39;power&#39;, dtype=tf.float32, size=10000) | Sparse 1D |
| engine | SparseFeature(index\_key=[&#39;@car.serial&#39;, &#39;id&#39;], value\_key= &#39;power&#39;, dtype=tf.float32, size=[12, 10000]) | Sparse 2D |
| tensorName | SparseFeature(index\_key=[&#39;@cars[\*].engine.cylinders&#39;, &#39;@cars[\*].engine.id&#39;], value\_key=&#39;@cars[\*].engine.power&#39;, dtype=tf.float32, size=[12, 10000]) | Sparse 2D |
| friends[\*].cars[\*].engine | SparseFeature(index\_key=[&#39;cylinders&#39;, &#39;id&#39;], value\_key=&#39;power&#39;, dtype=tf.float32, size=[12, 10000]) | Sparse 2D |
| friends[\*].cars[\*].engine | SparseFeature(index\_key=[&#39;clinders&#39;, &#39;id&#39;], value\_key=&#39;power&#39;, dtype=tf.float32, size=[100, 50, 12, 10000]) | Sparse 4D |
| engine | SparseFeature(index\_key=[&#39;cylinders&#39;,sparkplugs&#39;, &#39;id&#39;], value\_key=&#39;power&#39;, dtype=tf.flaot32, size[12, 24, 10000]) | Sparse 3D |

**Table 3** lists the key expression alongside the feature value definition. Note, that the feature values FixedLenFeature, VarLenFeature, FixedLenSequenceFeature, SparseFeature are defined in TensorFlow and use the same semantics here as used for the &quot;Example&quot; format in TensorFlow.

To simplify the reading of the key expressions and feature mappings, we elaborate the examples from Table 3 for dense tensors and sparse tensors.

#### Dense Tensors

The key expression &#39;name.first&#39; with the feature mapping &#39;FixedLenFeature(shape=[], dtype=tf.string)&#39; resolves all first names of persons into a Tensor with one dimension of 1 of string type. Note, if we batch, e.g. 128 items, then the tensor dimensions become 128 x 1.

The key expression &#39;friends[2].name.first&#39; with the feature mapping &#39;FixedLenFeature(shape=[],

dtype=tf.string)&#39; uses a filter expression, resolving the 3rd item (we start counting from 0) in the array of friends and there the friend&#39;s first name. These first names are returned as Tensor with one dimension of 1 of string type.

The key expression &#39;friends[\*].name.first&#39; with the feature mapping &#39;VarLenFeature(dtype=tf.string)&#39; resolves all friends first names into a sparse tensor. It is sparse because each row can have a different number of friends. The sparse tensor has the dense shape N -- where N is the maximum number of friends in a batch. Note, when batching is added to this tensor the first index of the sparse tensor indicates the row in the batch and the second dimension in the tensor indicates the friends index. The values of the sparse tensor are the friends&#39; first names and are of type string.

The key expression &#39;friends[\*].address[\*].street&#39; with the feature mapping &#39;FixedLenFeature( shape=[1, 2], dtype=tf.string, default\_value= [[&#39;35 Park Street&#39;, &#39;275 California Street&#39;], [&#39;950 Maude Ave&#39;, &#39;1000 Moore Parkway&#39;]])&#39; assumes that there are at most 1 friend which has at most 2 addresses. If for any record one such friend is absent the following street addresses &#39;35 Park Street&#39;, &#39;275 California Street&#39;, &#39;950 Maude Ave&#39;, and &#39;1000 Moore Parkway&#39; are filled in.

The key expression &#39;friends[\*].address[\*].street&#39; with the feature mapping &#39;VarLenFeature(dtype=tf.string)&#39; maps all friends&#39; street addresses into a variable length feature, or sparse tensor with a dense shape of maximum number of friends and addresses that appear in a batch. If the batch size is one, then the dense shape is the number of friends x the maximum number of addresses for all friends. The first dimension in the sparse tensor defines the friends index. The second index in the sparse tensor defines the address. The value of the sparse tensor is the street address and of type string. Each record may have a variable number of friends and addresses for those friends. If batching the batch index will become the first dimension.

The key expression &#39;friends[\*].address[\*].street&#39; with the feature mapping &#39;FixedLenSequenceFeature (shape=[1, 2], dtype=tf.string, default\_value=&quot;Park Street 35&quot;, allow\_missing=True)&#39; maps the first and only friends&#39; two street addresses into a tensor with the shape 1 x 2 and the type string. If a record does not have a friend or addresses for a friend the following &quot;Park Street 35&quot; is filled in for each entry.

The key expression &#39;friends[\*].jobs[\*].coworkers[\*].name.first&#39; with the feature mapping &#39;FixedLenFeature(shape=[1, 2, 3], dtype=tf.string, default\_value=[[[&#39;August&#39;, &#39;Astro&#39;, &#39;Anan&#39;],[&#39;Ben&#39;, &#39;Bob&#39;, &#39;Bart&#39;]], [[&#39;Carl&#39;, &#39;Claus&#39;, &#39;Castor&#39;],[&#39;Dan&#39;, &#39;Dave&#39;, &#39;Donald&#39;]])&#39; maps the friends&#39; jobs&#39; coworkers first names to a tensor with the shape 1 x 2 x 3 and the type string. This assumes that each record has: (i) one friend, (ii) each friend has exactly two jobs, (iii) at each of these jobs there are exactly three coworkers.

#### Sparse Tensors

The key expression &#39;engine&#39; with the feature mapping &#39;SparseFeature(index\_key=&#39;id&#39;, value\_key=&#39;power&#39;, dtype=tf.float32,  size=10000)&#39; maps the attributes &#39;id&#39; and &#39;power&#39; of an engine into a sparse tensor with the dense shape 10000 and the type float. Note, in this case we do not have to use a sparse vector since each engine has only one id and one power attribute. We chose this example to start with the simplest case for a sparse feature.

The key expression &#39;engine&#39; with the feature mapping &#39;SparseFeature(index\_key=[&#39;@car.serial&#39;, &#39;id&#39;], value\_key= &#39;power&#39;, dtype=tf.float32, size=[12, 10000])&#39; maps the attributes &#39;@car.serial&#39;, &#39;id&#39; and &#39;power&#39; of an engine and the car it appears in into a sparse tensor with the dense shape 12 x 10000 and the type float. Notice, that we use the &#39;@&#39; syntax to reference from the root of the avro value; like we have done in case of filters. This allows users to pull in indices or values from another data structure within the avro value. Note, all index entries and the value entry must have the same number of elements. This will be enforced and error is surfaced if there is a miss-match in the number of elements.

The key expression &#39;tensorName&#39; with the feature mapping &#39;SparseFeature(index\_key =[&#39;@cars[\*].engine.cylinders&#39;, &#39;@cars[\*].engine.id&#39;], value\_key=&#39;@cars[\*].engine.power&#39;, dtype=tf.float32, size=[12, 10000])&#39; maps the attributes &#39;@cars[\*].engine.cylinders&#39;, &#39;@cars[\*].engine.id&#39;, and &#39;@cars[\*].engine.power&#39; into a sparse tensor with the dense shape 12 x 10000 and the type float. As before the &#39;@&#39; syntax references from the root of the avro value. In this example all index entries and the value entry for the sparse tensor reference from the root. In this special case, the key expression is not used to resolve any part of the avro value. As always though, the key expression is used as tensor name.

The key expression &#39;friends[\*].cars[\*].engine.&#39; with the feature mapping &#39;SparseFeature( index\_key=[&#39;cylinders&#39;, &#39;id&#39;], value\_key=&#39;power&#39;, dtype=tf.float32, size=[12, 10000])&#39; maps cylinders and ids for each power into a sparse tensor with the dense shape 12 x 10000 and the type float.  This example shows how to use two keys for the index together with the \* notation for arrays. Note, that each entry in these arrays will create a separate entry in the sparse tensor with the corresponding indices. The first index is the number of cylinders. The second index is the id. This assumes that you have only one power value for each cylinder / id pair (uniqueness).

The key expression &#39;engine&#39; with the feature mapping &#39;SparseFeature(index\_key=[&#39;cylinders&#39;, &#39;sparkplugs&#39;, &#39;id&#39;], value\_key=&#39;power&#39;, dtype=tf.flaot32, size[12, 24, 10000])&#39; maps cylinders, sparkplugs, and ids for each power into a sparse tensor with the dense shape 12 x 24 x 10000 and the type float.

### Code design c++

The c++ implementation of the Avro dataset is split into the following parts: the Avro memory reader, the Avro parser, the Value Buffer, and the Avro dataset. Testing will leverage the avroc library to stage data and check data for equality. Whenever necessary we add wrapper methods, helper methods, and helper classes to simplify the staging process.

#### Avro memory reader (thread safe)

Reads and parses avro data into memory using avroc and provides avro values through a read method; This reader assumes that the memory region includes header information; in particular schema information at the start; this encapsulates the memory management and interfacing between low level c concepts and managed memory in c++ 11.

To guarantee thread safety we will use locks around mutually exclusive resources such as files and memory mapped files.

#### Avro parser (single instance per thread)

Handles the resolution of nestings and provides the next primitive type from the parsing process; resolves unions, nestings, nulls,...

We can design this thread safe w/out locks by creating a new parser instance for each step of work; if we like to share context information that is valid across multiple calls, we will pass that as argument.

For efficiency and feature completeness this parser leverages an ordered prefix tree where the prefixes are defined through common ancestors in the avro data structure. This prefix tree must be ordered to support filter expressions. The parts of a filter expression have to be evaluated before being able to evaluate filter itself.

#### Value buffer (single instance per thread)

Assembles avro parsed data into a TensorFlow tensor.

Handles filling in of missing values using default scalar and tensor data.

Handles adding of a dimension for batching.

Not used concurrently, e.g. no reading/writing occurs concurrently. Each thread holds its own instance exclusively. We can divide work across multiple tensors or slices of a tensor. Slices can be generated in a map and then an assembler thread can pick up these slices to create the entire tensor.

#### Avro dataset

Maps tensors to TensorFlow&#39;s dataset interface.

Provides the bridge to the python code.

### Code Design Python

Supports Python 2.x and 3.x.

The code is organized into two parts: 1) the Avro dataset and 2) a make method that creates the dataset. For testing we leverage the avro python library and the avro python-3 library with some thin wrappers to support Python 2.x and 3.x transparently.  Warning: Python 2.x and Python 3.x represents strings differently. Here, we treat Python 3.x as first-class citizen and provide backward compatible support to python 2.x whenever necessary.

#### Avro dataset

The bridge to the native implementation following examples of other datasets such as the Example dataset that leverages protobuf or the csv dataset.

#### Make Avro dataset

A helper method that fills in defaults for most parameters of the dataset. In addition, the make method takes care of batching, shuffling, and prefetching.

## Alternative designs
- Rather than providing a langauge for mapping all avro into TensorFlow; once could have defined a fixed schema of TensorFlow types in avro. An example for this is the Example format in TensorFlow that leverages protobuf. The above design provides more flexibility and was chosen for that reason.
- Could we have mapped any avro type into TensorFlow?  This would have been limited to mapping arrays to dense tensors and some form of fixed record structures into sparse tensors. The implicit assumptions of what is supported vs. not supported and how the mapping is done would have been too confusing.


## Questions and Discussion Topics
- Should we include building avroc into scope?  Additional effort is transforming a cmake script into bazel build config. This would enable running tests and guards against code quality digression.
- Should we focus more on performance?  I've taken some considerations in the design to allow for performance optimizations.
- Should we include writing support?  This has been a general topic and will become more relevant if TensorFlow is used for data transformation.
