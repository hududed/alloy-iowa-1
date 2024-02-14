# %%
library(mlr3mbo)
library(mlr3)
library(mlr3learners)
library(bbotk)
library(data.table)
library(tibble)
source("alloy-iowa-1/utils/propose.R") # for colab, adjust this if run locally
#%%
file <- 'AlCeMg-data.csv'
data <- as.data.table(read.csv(file))
data

# %%
# clean col names
 # remove everything after the first two dots
names(data) <- gsub("\\.{2}.*", "", names(data)) 
# remove non-alphanumeric characters except dots and underscores
names(data) <- gsub("[^[:alnum:]_.]", "", names(data))  
names(data) <- gsub("^_", "", names(data))  # remove leading underscores
names(data) <- gsub("_$", "", names(data))  # remove trailing underscores
names(data) <- gsub("\\.", "_", names(data))  # replace dots with underscores
# Capitalize the first letter and make the rest lowercase for each column name
names(data) <- paste0(toupper(substr(names(data), 1, 1)),
                      tolower(substr(names(data),
                                     2, nchar(names(data)))))
#%%
column_names <- names(data)
print(column_names)
#%%
selected_columns <- c("Ce_wt", "Mg_wt", "Average")  # replace with your column names
dt <- data[, ..selected_columns]
dt

# %%

metadata <- list(
  seed = 42,  # The seed for reproducibility
  bucket_name = "my_bucket",  # The name of the bucket where the archive will be saved
  user_id = "my_id",  # The user ID
  table_name = "AlCeMg",  # The name of the table
  batch_number = "1",  # The batch number
  parameter_info = list(
    Ce_wt = "float",  # The type of the Mn parameter
    Mg_wt = "float"  # The type of the Ni parameter
    # Add more parameters as needed
  ),
  parameter_ranges = list(
    Ce_wt = "(0,20)",  # The range of the Mn parameter
    Mg_wt = "(0,20)"  # The range of the Ni parameter
    # Add more ranges as needed
  ),
  output_column_names = c("Average"),  # The names of the output columns
  calculated_column = "Al_wt", # This column is ignored from the search space, but calculated in post-processing
  direction = "maximize",  # The direction of the optimization ("minimize" or "maximize")
  num_random_lines = 30,  # The number of random lines to generate
  to_nearest = 0.1  # The value to round to
)

#%%
# Run the experiment function
result <- propose_experiment(dt, metadata)

# %%
metadata <- list(
  seed = 42, # For reproducibility of results
  bucket_name = "my_bucket",  # The name of the bucket where the archive will be saved
  user_id = "my_id",  # The user ID
  table_name = "AlCeMg",  # The name of the table
  batch_number = "2",  # The batch number
  parameter_info = list(
    Ce_wt = "float",  # The type of the Mn parameter
    Mg_wt = "float"  # The type of the Ni parameter
    # Add more parameters as needed
  ),
  parameter_ranges = list(
    Ce_wt = "(0, 20)",  # The range of the Mn parameter
    Mg_wt = "(0, 20)"  # The range of the Ni parameter
    # Add more ranges as needed
  ),
  output_column_names = c("Average"),  # The names of the output columns
  calculated_column = "Al_wt", # This column is ignored from the search space, but calculated in post-processing
  direction = "maximize",  # Direction of the optimization ("minimize" / "maximize")
  num_random_lines = 16,  # The number of random lines to generate
  to_nearest = 0.1  # The value to round to
)
# %%
# Inspect trained model
result <- load_archive(metadata)
archive <- result[[1]]
acq_function <- result[[2]]
acq_optimizer <- result[[3]]

# Extract our trained model
model <- acq_function$surrogate$model$model

# Preprocessing
raw_data <- na.omit(data)

# Select the predictor columns
predictor_data <- raw_data[, c("Al_wt", "Ce_wt", "Mg_wt")]

# Select the target column
hardness <- raw_data$Average
# %%
library(iml)
# Create a Predictor object with the model and the data frame
predictor <- Predictor$new(model, data = predictor_data, y = hardness)

# Compute feature importance
importance <- FeatureImp$new(predictor, loss = "mse", n.repetitions = 100)

# Plot the feature importance
importance$plot()
# %%
# Generate 1-way partial dependence plot
mg_effect <- FeatureEffect$new(predictor, feature = "Mg_wt",
                               method = "pdp+ice")
mg_effect$plot()
# %%
# Generate 1-way partial dependence plot
ce_effect <- FeatureEffect$new(predictor, feature = "Ce_wt",
                               method = "pdp+ice")
ce_effect$plot()
# %%
library(viridis)
# R
# Generate the 2-way partial dependence plot
pd <- FeatureEffect$new(predictor, c("Mg_wt", "Ce_wt"), method = "pdp")
p <- pd$plot() +
  scale_fill_viridis(option = "D")

# Change the color bar title
p <- p + labs(fill = "Hardness")

# Print the plot
print(p)
#%%
library(ggplot2)
# Perform PCA
pca_result <- prcomp(predictor_data, scale. = TRUE)

# Create a data frame of the first two principal components
pca_data <- data.frame(pca_result$x[,1:2])

# Add the 'Average' column to the PCA data
pca_data$Average <- y


# Install and load the akima package
if (!require(akima)) {
  install.packages("akima")
}
library(akima)

# Perform interpolation
interp_data <- with(pca_data, interp(x = PC1, y = PC2, z = Average))

# Convert the interpolated data to a data frame
interp_df <- data.frame(
  expand.grid(PC1 = interp_data$x, PC2 = interp_data$y),
  Average = c(interp_data$z)
)

# Create the plot
ggplot(interp_df, aes(x = PC1, y = PC2, fill = Average)) +
  geom_raster(interpolate = TRUE) +
  scale_fill_gradient(low = "blue", high = "red") +
  geom_point(data = pca_data, color = "white", size = 3) +
  xlab("First Principal Component") +
  ylab("Second Principal Component") +
  ggtitle("Interpolated PCA Plot with White Scatter Points")

# %%
# Import functions (see https://github.com/hududed/mlr3mbo-demo.git for the source files)
# FOR UPDATES MAKE SURE THIS IS SOURCED, NOT mlr3mbo-demo/utils/batch.R!
source("mlr3mbo-demo/utils/update.R") # for colab, adjust this if run locally

# %%
# Please upload the new updated file in your session (See Folder icon on the left pane)
file = 'updated.csv'
data <- as.data.table(read.csv(file))
data
# %%
#metadata for updated batch 2 is already defined above
# %%



# %%
# Run the experiment (FOR UPDATES MAKE SURE mlr3mbo-demo/utils/update.R is sourced, not batch.R)
new_result <- update_experiment(data, metadata)
# %%
