run_evals:
	@echo "Running all evaluation scripts..."
	@find evals -name "eval_*.py" -exec python {} \;
