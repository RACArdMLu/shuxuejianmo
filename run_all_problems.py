#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to run all NIPT mathematical modeling problems

Author: Generated for NIPT Mathematical Modeling Competition
Usage: python3 run_all_problems.py
"""

import sys
import os
import time

def run_problem(problem_file, problem_name):
    """Run a specific problem script"""
    print(f"\n{'='*80}")
    print(f"Running {problem_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Import and run the problem
        if problem_file == "problem1_correlation_analysis.py":
            import problem1_correlation_analysis
            problem1_correlation_analysis.main()
        elif problem_file == "problem2_bmi_timing.py":
            import problem2_bmi_timing
            problem2_bmi_timing.main()
        elif problem_file == "problem3_multifactor.py":
            import problem3_multifactor
            problem3_multifactor.main()
        elif problem_file == "problem4_female_abnormality.py":
            import problem4_female_abnormality
            problem4_female_abnormality.main()
        
        end_time = time.time()
        print(f"\n{problem_name} completed successfully in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error running {problem_name}: {str(e)}")
        return False
    
    return True

def main():
    """Main function to run all problems"""
    print("NIPT Mathematical Modeling Competition - Complete Solution")
    print("2025年高教社杯全国大学生数学建模竞赛 C题")
    print("NIPT的时点选择与胎儿的异常判定")
    print("="*80)
    
    # Check if data file exists
    if not os.path.exists("附件.xlsx"):
        print("Error: Data file '附件.xlsx' not found!")
        print("Please ensure the Excel data file is in the current directory.")
        return
    
    problems = [
        ("problem1_correlation_analysis.py", "Problem 1: Y Chromosome Concentration Correlation Analysis"),
        ("problem2_bmi_timing.py", "Problem 2: BMI Grouping and Optimal NIPT Timing"),
        ("problem3_multifactor.py", "Problem 3: Multi-Factor Analysis"),
        ("problem4_female_abnormality.py", "Problem 4: Fetal Abnormality Detection")
    ]
    
    total_start_time = time.time()
    successful_runs = 0
    
    for problem_file, problem_name in problems:
        if os.path.exists(problem_file):
            if run_problem(problem_file, problem_name):
                successful_runs += 1
            time.sleep(2)  # Brief pause between problems
        else:
            print(f"Warning: {problem_file} not found, skipping...")
    
    total_end_time = time.time()
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully completed {successful_runs}/{len(problems)} problems")
    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds")
    
    print(f"\nGenerated outputs:")
    output_files = [
        "correlation_heatmap.png",
        "scatter_plots_y_chromosome.png", 
        "regression_diagnostics.png",
        "gestational_age_analysis.png",
        "problem2_results.png",
        "error_analysis.png",
        "problem3_comprehensive_results.png",
        "problem4_female_abnormality_detection.png"
    ]
    
    for file in output_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (not generated)")
    
    print(f"\nSolution documentation: README_SOLUTION.md")
    print("\nAll analysis complete! Check the generated plots and results.")

if __name__ == "__main__":
    main()