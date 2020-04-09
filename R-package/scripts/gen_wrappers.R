library(reticulate)
library(scaffolder)

# Instructions:
# * Ensure you have the following R libraries installed via the following:
#     `install.packages(c("reticulate", "scaffolder", "roxygen2", "pkgdown"))`
# * Update `tensorflow_io_repo_path` with the local path to TensorFlow IO repo
# * Update `python_binary_paths` with your Python binary
# * Execute this R script
tensorflow_io_repo_path <- "~/repos/io"
python_binary_path <- "/opt/miniconda3/bin/python"

setwd(tensorflow_io_repo_path)
use_python(python_binary_path)
tf_io <- import("tensorflow_io")
r_package_path <- file.path(tensorflow_io_repo_path, "R-package")
output_dir <- file.path(r_package_path, "R")

if (file.exists(output_dir)) {
  for (file in list.files(output_dir)) {
    if (!file %in% c("package.R", "dataset_utils.R", "reexports.R"))
      file.remove(file.path(output_dir, file))
  }
} else {
  dir.create(output_dir)
}

for (module in names(tf_io)) {
  if (module %in% c("genome", "image", "arrow", "IODataset", "IOTensor", "core", "experimental")) {
    print(paste0("Generating code for module: ", module))
    for (func in names(tf_io[[module]])) {
      # TODO: manually check all errors and filter out the unsuccessful ones
      tryCatch({
        scaffolder::scaffold_py_function_wrapper(sprintf("tf_io$%s$%s", module, func), file_name = file.path(output_dir, paste0(module, "_wrappers.R")))
      }, error = function(e) print(sprintf("Error when generating code for %s.%s:%s", module, func, e$message)))
    }
  }
}

# TODO: This file was not generated successfully due to some errors
# in auto-generated roxygen docs.
roxygen2::roxygenize(r_package_path)
pkgdown::build_site(r_package_path)
