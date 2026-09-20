# AGENTS.md

## Project Goal

Build a simple proof-of-concept customer segmentation web application.

The application allows a user to upload a CSV file and automatically:

1. Validate and clean the customer transaction data.
2. Create RFM features:

   * Recency
   * Frequency
   * Monetary
3. Prepare and scale the features for clustering.
4. Find a reasonable number of clusters (`K`).
5. Train a K-Means clustering model.
6. Analyze the characteristics of each cluster.
7. Assign understandable names to the clusters.
8. Visualize the customer segments.
9. Present the results in a Streamlit application.

The first priority is to prove that this complete pipeline works.

---

# Technology

Use:

* Python
* Streamlit for the frontend
* pandas for data processing
* scikit-learn for preprocessing and K-Means
* matplotlib and/or other already-approved visualization libraries
* SQLite3 only when persistent storage is actually needed

Avoid adding new technologies, frameworks, databases, or dependencies unless there is a clear reason.

---

# POC Scope

The first version is NOT a generic CSV segmentation platform.

Use one known CSV structure with clearly defined required columns.

Do not attempt to automatically understand arbitrary CSV schemas yet.

Generic CSV support may be added after the basic POC works.

Do NOT implement the following unless explicitly requested:

* churn prediction
* LTV prediction
* authentication
* user accounts
* cloud deployment
* APIs
* background workers
* microservices
* complex configuration systems
* unnecessary database functionality

The initial goal is RFM-based customer segmentation.

---

# Intended Pipeline

The application should conceptually follow this pipeline:

CSV Upload
→ Validation
→ Data Cleaning
→ RFM Feature Engineering
→ Feature Scaling
→ K Selection
→ K-Means Training
→ Cluster Analysis
→ Cluster Naming
→ Visualization
→ Streamlit Presentation

Keep this flow easy to understand in the code.

---

# RFM

RFM means:

* Recency: how recently the customer purchased.
* Frequency: how often the customer purchased.
* Monetary: how much the customer spent.

RFM should be calculated before clustering.

Only features intentionally selected for clustering should be passed to K-Means.

Do not silently include unrelated numeric columns.

---

# K-Means

Before training the final K-Means model:

1. Prepare the RFM features.
2. Scale the relevant features.
3. Evaluate reasonable candidate values for K.
4. Select a reasonable K using an understandable method such as silhouette score and/or elbow analysis.
5. Train the final model.

Keep the approach simple and explainable.

Do not introduce advanced clustering algorithms unless explicitly requested.

---

# Cluster Interpretation

K-Means initially produces meaningless identifiers such as:

* Cluster 0
* Cluster 1
* Cluster 2

After clustering, calculate summary statistics for each cluster.

Use those characteristics to understand what each cluster represents.

Only then assign human-readable names where appropriate.

Example:

Cluster 0
→ low recency, high frequency, high monetary
→ "High Value / Loyal"

The naming logic must be understandable.

Do not invent misleading business meanings that are not supported by the cluster characteristics.

---

# Visualization

The POC should eventually contain two main visualizations:

1. A scatterplot showing the customer segments.
2. A heatmap showing the characteristics of the segments.

The visualizations should help a user understand both:

* which customers belong to each segment
* why the segments differ

Prefer understandable visualizations over visually impressive but difficult-to-interpret charts.

---

# Streamlit

Streamlit is the user interface.

The intended user journey is:

1. User opens the application.
2. User uploads the expected CSV file.
3. Application validates the file.
4. Application processes and cleans the data.
5. RFM features are calculated.
6. Segmentation is performed.
7. Results are displayed.
8. User can inspect segment characteristics and customer data.

Keep the Streamlit interface simple for the POC.

Business/data-processing logic should not become unnecessarily mixed with UI code.

---

# Architecture Philosophy

The human developer owns the architecture.

The agent helps implement it.

The architecture should remain as small and understandable as possible.

Prefer:

* simple functions
* clear modules
* descriptive names
* explicit data flow
* small incremental changes

Avoid:

* premature abstraction
* unnecessary classes
* unnecessary design patterns
* unnecessary service layers
* repository patterns
* factories
* managers
* interfaces
* excessive helper modules
* deeply nested folders
* duplicated functionality

Do not create abstractions simply because they might become useful later.

Build for the current POC.

Refactor when there is an actual reason.

---

# File Creation Rules

Do NOT create many files automatically.

Before creating a new file, consider whether the functionality belongs in an existing file.

Every file should have a clear responsibility that the developer can explain.

For significant new files, explain:

1. Why the file is needed.
2. What responsibility it will have.
3. Why the functionality should not live in an existing file.

Prefer a small number of understandable files during the POC.

---

# Agent Workflow

For non-trivial tasks, use this workflow:

1. Inspect the relevant existing code.
2. Understand the current architecture.
3. Explain the proposed change.
4. List the files that need to change.
5. Identify any new files or dependencies.
6. Prefer the smallest reasonable implementation.
7. Implement the change.
8. Run relevant tests/checks.
9. Review the result for unnecessary complexity.
10. Summarize what changed and why.

If a task would significantly change the architecture, stop after the planning stage and request approval before implementing it.

---

# Small Changes

Prefer small, focused changes.

For example, instead of implementing:

"Build the entire segmentation platform"

prefer tasks such as:

* validate the CSV
* clean transaction data
* calculate RFM
* scale RFM features
* evaluate K
* train K-Means
* summarize clusters
* name clusters
* create scatterplot
* create heatmap
* display results in Streamlit

Each step should work before unnecessary complexity is added.

---

# Code Quality

Code should be understandable by a junior Python developer.

Prefer readability over cleverness.

Use descriptive names.

Avoid large functions when they clearly contain multiple responsibilities, but do not split functions purely to make them smaller.

Add comments when they explain WHY something is done.

Do not add comments that merely repeat what obvious code does.

Avoid unnecessary duplication.

---

# Dependencies

Do not install a new dependency without a clear reason.

Before adding one:

1. Check whether the existing dependencies or Python standard library can solve the problem.
2. Explain why the dependency is useful.
3. Prefer well-established libraries.

Do not introduce dependencies for trivial functionality.

---

# Testing and Verification

Do not assume generated code works.

After implementation:

* run relevant tests if they exist
* run relevant lint/type checks if configured
* verify imports
* check for obvious runtime errors
* verify the affected pipeline step

When fixing an error, understand the cause before changing code.

Do not repeatedly make random changes until an error disappears.

---

# Git Safety

Do not:

* force push
* rewrite Git history
* delete branches
* delete unrelated files
* commit secrets
* modify unrelated code

Keep changes focused on the requested task.

Do not automatically commit or push changes unless explicitly requested.

The developer should be able to inspect the Git diff before committing.

---

# Important Principle

The goal is NOT to generate as much code as possible.

The goal is to build the smallest understandable system that correctly demonstrates the complete customer-segmentation workflow.

When choosing between:

A complex solution that might be useful later

and

A simple solution that solves the current POC

prefer the simple solution.
