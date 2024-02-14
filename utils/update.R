library(mlr3mbo)
library(mlr3)
library(mlr3learners)
library(bbotk)
library(data.table)
library(tibble)
library(R.utils)

source("mlr3mbo-demo/utils/processing.R") # for colab, adjust this if run locally

update_experiment <- function(data, metadata) {
    set.seed(metadata$seed)
    result = load_archive(metadata)
    full_data = as.data.table(data)
    # print(data)
    data <- tail(full_data, n=metadata$num_random_lines)
    print(data)
        
    archive = result[[1]]
    acq_function = result[[2]]
    acq_optimizer = result[[3]]
    
    # Check if metadata$output_column_names is NULL or empty
    if (is.null(metadata$output_column_names) || length(metadata$output_column_names) == 0) {
        stop("metadata$output_column_names is NULL or empty")
    }

    # Check if all output_column_names exist in data
    if (!all(metadata$output_column_names %in% names(data))) {
        stop("Some names in metadata$output_column_names do not exist in data")
    }

    # print(class(archive))
    # print(methods(class=class(archive)))


    # Now you can safely call the add_evals method
    archive$add_evals(xdt = data[, names(metadata$parameter_info), with=FALSE], ydt = data[, metadata$output_column_names, with=FALSE])
    print("Model archive so far: ")
    print(archive)
    q = metadata$num_random_lines
    result = add_evals_to_archive(archive, acq_function, acq_optimizer, data, q, metadata)

    candidate = result[[1]]
    archive = result[[2]]
    acq_function = result[[3]]

    print(result)

    x2 <- candidate[, names(metadata$parameter_info), with=FALSE]
    print("New candidates: ")
    print(x2)
    print("New archive: ")
    print(archive)

    x2_dt <- as.data.table(x2)
    full_data <- rbindlist(list(full_data, x2_dt), fill = TRUE)
    # Recalculate the ignored column
    full_data[is.na(get(metadata$calculated_column)), (metadata$calculated_column) := 100 - Reduce(`+`, .SD), .SDcols = setdiff(names(data), c(metadata$calculated_column, metadata$output_column_names))]
    # full_data[is.na(metadata$calculated_column), Cu := 100 - Reduce(`+`, .SD), .SDcols = setdiff(names(data), c("Cu", "DSC_Af"))]
    # Reorder the columns to move Cu to the left-most position
    setcolorder(full_data, c(metadata$calculated_column, setdiff(names(full_data), metadata$calculated_column)))
    # setcolorder(full_data, c("Cu", setdiff(names(full_data), "Cu")))
    print(full_data)

    # Save the data table as a CSV file in the same directory
    dir_path = paste0(metadata$bucket_name, "/", metadata$user_id, "/", metadata$table_name, "/", metadata$batch_number)
    data.table::fwrite(full_data, paste0(dir_path, "/output.csv"))
    print(paste0("mlr3mbo update is finished. Results file is saved at: ", dir_path))

    return(full_data)

    }