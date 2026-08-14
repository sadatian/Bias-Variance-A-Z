---
trigger: model_decision
description: When generating or modifying any `.py` file for this project.
---

1. **Preamble:** The file must start with a `# %% [markdown]` block containing a title and a brief introduction. Include a mermaid diagram conceptualizing the lesson **ONLY IF** it is absolutely necessary and crucial to explaining a complex architecture.  
2. **Cell Structure:** Every logical step (Imports, Data Gen, Modeling, Plotting) must be separated by `# %%`.  
3. **Explanatory Text:** Every code cell MUST be immediately preceded by a `# %% [markdown]` cell explaining the advanced math, statistical theory, or engineering logic of what the next code cell will do. Assume an advanced ML Engineer audience.  
4. **Standalone Verification:** Before outputting the code, internally verify: "Can this script run independently without importing functions from another custom `.py` file?" If no, refactor.