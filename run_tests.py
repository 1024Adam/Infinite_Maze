#!/usr/bin/env python3
"""
Test runner script for Infinite Maze test suite.

This script provides convenient commands to run different categories
of tests with appropriate configurations.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


class TestRunner:
    """Test runner for Infinite Maze test suite."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / "tests"
    
    def run_unit_tests(self, verbose=False, coverage=False):
        """Run unit tests."""
        cmd = ["python", "-m", "pytest", "-m", "unit"]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend(["--cov=infinite_maze", "--cov-report=html", "--cov-report=term-missing"])
        
        cmd.append(str(self.tests_dir / "unit"))
        
        print("🧪 Running unit tests...")
        return subprocess.run(cmd, cwd=self.project_root)
    
    def run_integration_tests(self, verbose=False):
        """Run integration tests."""
        cmd = ["python", "-m", "pytest", "-m", "integration"]
        
        if verbose:
            cmd.append("-v")
        
        cmd.append(str(self.tests_dir / "integration"))
        
        print("🔗 Running integration tests...")
        return subprocess.run(cmd, cwd=self.project_root)
    
    def run_functional_tests(self, verbose=False):
        """Run functional tests."""
        cmd = ["python", "-m", "pytest", "-m", "functional"]
        
        if verbose:
            cmd.append("-v")
        
        cmd.append(str(self.tests_dir / "functional"))
        
        print("🎮 Running functional tests...")
        return subprocess.run(cmd, cwd=self.project_root)
    
    def run_performance_tests(self, verbose=False, benchmark=False):
        """Run performance tests."""
        cmd = ["python", "-m", "pytest", "-m", "performance", "--run-performance"]
        
        if verbose:
            cmd.append("-v")
        
        if benchmark:
            cmd.extend(["--benchmark-only", "--benchmark-sort=mean"])
        
        cmd.append(str(self.tests_dir / "performance"))
        
        print("⚡ Running performance tests...")
        return subprocess.run(cmd, cwd=self.project_root)
    
    def run_all_tests(self, verbose=False, coverage=False, skip_slow=False):
        """Run all tests."""
        cmd = ["python", "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend(["--cov=infinite_maze", "--cov-report=html", "--cov-report=term-missing"])
        
        if skip_slow:
            cmd.append("--skip-slow")
        else:
            cmd.append("--run-performance")
        
        cmd.append(str(self.tests_dir))
        
        print("🚀 Running all tests...")
        return subprocess.run(cmd, cwd=self.project_root)
    
    def run_quick_tests(self, verbose=False):
        """Run quick tests (unit + integration, skip slow tests)."""
        cmd = ["python", "-m", "pytest", "-m", "unit or integration", "--skip-slow"]
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([str(self.tests_dir / "unit"), str(self.tests_dir / "integration")])
        
        print("⚡ Running quick tests...")
        return subprocess.run(cmd, cwd=self.project_root)
    
    def run_specific_test(self, test_pattern, verbose=False):
        """Run specific test(s) by pattern."""
        cmd = ["python", "-m", "pytest", "-k", test_pattern]
        
        if verbose:
            cmd.append("-v")
        
        cmd.append(str(self.tests_dir))
        
        print(f"🎯 Running tests matching: {test_pattern}")
        return subprocess.run(cmd, cwd=self.project_root)
    
    def check_test_environment(self):
        """Check if test environment is properly set up."""
        print("🔍 Checking test environment...")
        
        # Check if pytest is installed
        try:
            import pytest
            print(f"✅ pytest version: {pytest.__version__}")
        except ImportError:
            print("❌ pytest not installed. Run: pip install pytest")
            return False
        
        # Check if infinite_maze package is importable
        try:
            import infinite_maze
            print("✅ infinite_maze package found")
        except ImportError:
            print("❌ infinite_maze package not found. Check PYTHONPATH.")
            return False
        
        # Check test directories
        required_dirs = ["unit", "integration", "functional", "performance", "fixtures"]
        for dir_name in required_dirs:
            test_dir = self.tests_dir / dir_name
            if test_dir.exists():
                print(f"✅ {dir_name} tests directory found")
            else:
                print(f"⚠️  {dir_name} tests directory not found")
        
        print("✅ Test environment check complete")
        return True
    
    def generate_test_report(self, output_format="html"):
        """Generate comprehensive test report."""
        cmd = [
            "python", "-m", "pytest",
            "--cov=infinite_maze",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--html=test_report.html",
            "--self-contained-html",
            "--run-performance",
            str(self.tests_dir)
        ]
        
        print("📊 Generating comprehensive test report...")
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode == 0:
            print("✅ Test report generated:")
            print(f"   📄 HTML report: {self.project_root}/test_report.html")
            print(f"   📊 Coverage report: {self.project_root}/htmlcov/index.html")
        
        return result


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Infinite Maze Test Runner")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow-running tests")
    
    subparsers = parser.add_subparsers(dest="command", help="Test commands")
    
    # Unit tests
    unit_parser = subparsers.add_parser("unit", help="Run unit tests")
    unit_parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    
    # Integration tests  
    subparsers.add_parser("integration", help="Run integration tests")
    
    # Functional tests
    subparsers.add_parser("functional", help="Run functional tests")
    
    # Performance tests
    perf_parser = subparsers.add_parser("performance", help="Run performance tests")
    perf_parser.add_argument("--benchmark", action="store_true", help="Run as benchmarks")
    
    # All tests
    all_parser = subparsers.add_parser("all", help="Run all tests")
    all_parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    all_parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")
    
    # Quick tests
    subparsers.add_parser("quick", help="Run quick tests (unit + integration)")
    
    # Specific test
    specific_parser = subparsers.add_parser("run", help="Run specific test(s)")
    specific_parser.add_argument("pattern", help="Test pattern to match")
    
    # Environment check
    subparsers.add_parser("check", help="Check test environment")
    
    # Generate report
    report_parser = subparsers.add_parser("report", help="Generate test report")
    report_parser.add_argument("--format", choices=["html", "xml"], default="html", help="Report format")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.command == "unit":
        result = runner.run_unit_tests(verbose=args.verbose, coverage=getattr(args, 'coverage', False))
    elif args.command == "integration":
        result = runner.run_integration_tests(verbose=args.verbose)
    elif args.command == "functional":
        result = runner.run_functional_tests(verbose=args.verbose)
    elif args.command == "performance":
        result = runner.run_performance_tests(verbose=args.verbose, benchmark=getattr(args, 'benchmark', False))
    elif args.command == "all":
        result = runner.run_all_tests(
            verbose=args.verbose, 
            coverage=getattr(args, 'coverage', False),
            skip_slow=getattr(args, 'skip_slow', False)
        )
    elif args.command == "quick":
        result = runner.run_quick_tests(verbose=args.verbose)
    elif args.command == "run":
        result = runner.run_specific_test(args.pattern, verbose=args.verbose)
    elif args.command == "check":
        result = runner.check_test_environment()
        sys.exit(0 if result else 1)
    elif args.command == "report":
        result = runner.generate_test_report(output_format=args.format)
    else:
        # Default: run quick tests
        result = runner.run_quick_tests(verbose=args.verbose)
    
    sys.exit(result.returncode if hasattr(result, 'returncode') else 0)


if __name__ == "__main__":
    main()
