.PHONY: install run-basic run-ml run-collab run-multi run-scenario test clean clean-results

# Install dependencies
install:
	pip install -r requirements.txt

# Run basic examples
run-basic:
	python examples/01_basic_egt_simulation.py

# Run ML integration example
run-ml:
	python examples/02_simple_ml_integration.py

# Run collaborative ML example
run-collab:
	python examples/03_collaborative_ml.py

# Run multi-distribution simulation with default parameters
run-multi:
	@echo "Available scenarios in configs directory:"
	@ls -1 examples/configs/*.json | sed 's/.*\/\(.*\)\.json/\1/'
	@echo "\nRun with: make run-multi SCENARIO=<scenario_name> [POINTS=20] [EPOCHS=5]"
	@if [ -z "$(SCENARIO)" ]; then \
		echo "\nRunning with default scenario: ch5_s1_baseline_config"; \
		python examples/multi_starting_distribution.py --scenario ch5_s1_baseline_config --points $(or $(POINTS),20) --epochs $(or $(EPOCHS),5); \
	else \
		echo "\nRunning with scenario: $(SCENARIO)"; \
		python examples/multi_starting_distribution.py --scenario $(SCENARIO) --points $(or $(POINTS),20) --epochs $(or $(EPOCHS),5); \
	fi

# Run a specific scenario with all parameters
run-scenario:
	python examples/multi_starting_distribution.py --scenario $(SCENARIO) --points $(POINTS) --epochs $(EPOCHS) --processes $(PROCS)

# Test if import paths are working correctly
test:
	python -c "import sys; sys.path.insert(0, '.'); from registry.experiment import ExperimentBuilder; print('Import test successful!')"

# Clean up cache and data files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete 

# Clean up results from previous runs
clean-results:
	@echo "Removing all results from ./examples/results"
	@if [ -d "./examples/results" ]; then \
		rm -rf ./examples/results/*; \
		echo "Results directory cleaned."; \
	else \
		echo "Results directory does not exist."; \
	fi 