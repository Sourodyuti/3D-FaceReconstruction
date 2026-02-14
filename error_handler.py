#!/usr/bin/env python3
"""
Error Handler Module
Comprehensive error handling, logging, and recovery system
"""

import logging
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Any
from functools import wraps
import json


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter with color output for terminal
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format the message
        result = super().format(record)
        
        return result


class ErrorHandler:
    """
    Centralized error handling and logging system
    """
    
    def __init__(self, 
                 log_dir: str = './logs',
                 log_level: int = logging.INFO,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 colored_output: bool = True):
        """
        Initialize error handler
        
        Args:
            log_dir: Directory for log files
            log_level: Logging level
            enable_console: Enable console logging
            enable_file: Enable file logging
            colored_output: Use colored console output
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('3D-FaceReconstruction')
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()  # Clear any existing handlers
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            
            if colored_output:
                console_format = '%(levelname)s | %(name)s | %(message)s'
                console_handler.setFormatter(ColoredFormatter(console_format))
            else:
                console_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
                console_handler.setFormatter(logging.Formatter(console_format))
            
            self.logger.addHandler(console_handler)
        
        # File handler
        if enable_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = self.log_dir / f'reconstruction_{timestamp}.log'
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            file_format = '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s'
            file_handler.setFormatter(logging.Formatter(file_format))
            
            self.logger.addHandler(file_handler)
            self.logger.info(f"Log file created: {log_file}")
        
        # Error log for critical issues
        self.error_log_file = self.log_dir / 'errors.json'
        self.errors = []
    
    def log_error(self, 
                  error: Exception, 
                  context: Optional[str] = None,
                  severity: str = 'ERROR',
                  save_to_file: bool = True):
        """
        Log an error with context
        
        Args:
            error: Exception object
            context: Additional context information
            severity: Error severity level
            save_to_file: Save to error log file
        """
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'type': type(error).__name__,
            'message': str(error),
            'context': context,
            'severity': severity,
            'traceback': traceback.format_exc()
        }
        
        # Log to logger
        log_message = f"{error_info['type']}: {error_info['message']}"
        if context:
            log_message += f" | Context: {context}"
        
        if severity == 'CRITICAL':
            self.logger.critical(log_message)
        else:
            self.logger.error(log_message)
        
        # Save to error log
        if save_to_file:
            self.errors.append(error_info)
            self._save_error_log()
    
    def _save_error_log(self):
        """Save error log to JSON file"""
        try:
            with open(self.error_log_file, 'w') as f:
                json.dump(self.errors, f, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to save error log: {e}")
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Get a logger instance
        
        Args:
            name: Logger name (optional)
        
        Returns:
            Logger instance
        """
        if name:
            return logging.getLogger(f'3D-FaceReconstruction.{name}')
        return self.logger


def handle_errors(error_handler: Optional[ErrorHandler] = None,
                  reraise: bool = False,
                  default_return: Any = None,
                  context: Optional[str] = None):
    """
    Decorator for automatic error handling
    
    Args:
        error_handler: ErrorHandler instance
        reraise: Whether to reraise the exception
        default_return: Default return value on error
        context: Context description
    
    Example:
        @handle_errors(error_handler, context="Processing frame")
        def process_frame(frame):
            # Your code here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log error
                if error_handler:
                    ctx = context or f"Function: {func.__name__}"
                    error_handler.log_error(e, context=ctx)
                else:
                    # Fallback to basic logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error in {func.__name__}: {e}")
                    logger.debug(traceback.format_exc())
                
                # Reraise or return default
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


class PerformanceMonitor:
    """
    Monitor and log performance metrics
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize performance monitor
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, metric_name: str):
        """
        Start timing a metric
        
        Args:
            metric_name: Name of the metric
        """
        import time
        self.start_times[metric_name] = time.time()
    
    def end_timer(self, metric_name: str, log: bool = True):
        """
        End timing a metric
        
        Args:
            metric_name: Name of the metric
            log: Whether to log the result
        
        Returns:
            Elapsed time in seconds
        """
        import time
        if metric_name not in self.start_times:
            self.logger.warning(f"Timer '{metric_name}' was not started")
            return 0.0
        
        elapsed = time.time() - self.start_times[metric_name]
        
        # Store metric
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(elapsed)
        
        # Log if requested
        if log:
            self.logger.debug(f"{metric_name}: {elapsed:.4f}s")
        
        return elapsed
    
    def get_average(self, metric_name: str) -> float:
        """
        Get average time for a metric
        
        Args:
            metric_name: Name of the metric
        
        Returns:
            Average time in seconds
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return 0.0
        return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
    
    def get_stats(self, metric_name: str) -> dict:
        """
        Get statistics for a metric
        
        Args:
            metric_name: Name of the metric
        
        Returns:
            Dictionary with min, max, avg, count
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {'min': 0, 'max': 0, 'avg': 0, 'count': 0}
        
        values = self.metrics[metric_name]
        return {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'count': len(values)
        }
    
    def report(self) -> str:
        """
        Generate performance report
        
        Returns:
            Formatted report string
        """
        report = "\n" + "=" * 70 + "\n"
        report += " PERFORMANCE REPORT\n"
        report += "=" * 70 + "\n\n"
        
        for metric_name in sorted(self.metrics.keys()):
            stats = self.get_stats(metric_name)
            report += f"  {metric_name}:\n"
            report += f"    Count: {stats['count']}\n"
            report += f"    Avg:   {stats['avg']:.4f}s\n"
            report += f"    Min:   {stats['min']:.4f}s\n"
            report += f"    Max:   {stats['max']:.4f}s\n\n"
        
        report += "=" * 70
        return report
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.start_times.clear()


class ResourceMonitor:
    """
    Monitor system resources (CPU, Memory, GPU)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize resource monitor
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Try to import psutil for system monitoring
        try:
            import psutil
            self.psutil = psutil
            self.has_psutil = True
        except ImportError:
            self.logger.warning("psutil not available. Resource monitoring disabled.")
            self.has_psutil = False
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        if not self.has_psutil:
            return 0.0
        return self.psutil.cpu_percent(interval=0.1)
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage"""
        if not self.has_psutil:
            return {'percent': 0.0, 'used_mb': 0, 'total_mb': 0}
        
        mem = self.psutil.virtual_memory()
        return {
            'percent': mem.percent,
            'used_mb': mem.used / (1024 ** 2),
            'total_mb': mem.total / (1024 ** 2)
        }
    
    def check_resources(self, 
                       cpu_threshold: float = 90.0,
                       memory_threshold: float = 90.0) -> dict:
        """
        Check if resources are within acceptable limits
        
        Args:
            cpu_threshold: CPU usage threshold (percent)
            memory_threshold: Memory usage threshold (percent)
        
        Returns:
            Dictionary with status and warnings
        """
        if not self.has_psutil:
            return {'status': 'unknown', 'warnings': []}
        
        warnings = []
        cpu = self.get_cpu_usage()
        memory = self.get_memory_usage()
        
        if cpu > cpu_threshold:
            warnings.append(f"High CPU usage: {cpu:.1f}%")
            self.logger.warning(f"High CPU usage: {cpu:.1f}%")
        
        if memory['percent'] > memory_threshold:
            warnings.append(f"High memory usage: {memory['percent']:.1f}%")
            self.logger.warning(f"High memory usage: {memory['percent']:.1f}%")
        
        status = 'ok' if not warnings else 'warning'
        
        return {
            'status': status,
            'cpu_percent': cpu,
            'memory_percent': memory['percent'],
            'memory_used_mb': memory['used_mb'],
            'warnings': warnings
        }
    
    def log_resource_status(self):
        """Log current resource status"""
        if not self.has_psutil:
            return
        
        cpu = self.get_cpu_usage()
        memory = self.get_memory_usage()
        
        self.logger.info(
            f"Resources - CPU: {cpu:.1f}%, "
            f"Memory: {memory['percent']:.1f}% ({memory['used_mb']:.0f}/{memory['total_mb']:.0f} MB)"
        )


# Global error handler instance
_global_error_handler = None


def setup_global_error_handler(**kwargs) -> ErrorHandler:
    """
    Setup global error handler
    
    Args:
        **kwargs: Arguments for ErrorHandler
    
    Returns:
        ErrorHandler instance
    """
    global _global_error_handler
    _global_error_handler = ErrorHandler(**kwargs)
    return _global_error_handler


def get_global_error_handler() -> Optional[ErrorHandler]:
    """
    Get global error handler instance
    
    Returns:
        ErrorHandler instance or None
    """
    return _global_error_handler


if __name__ == "__main__":
    # Test error handler
    print("Testing Error Handler...\n")
    
    # Setup error handler
    handler = setup_global_error_handler(
        log_dir='./logs',
        log_level=logging.DEBUG,
        colored_output=True
    )
    
    logger = handler.get_logger('test')
    
    # Test logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test error logging
    try:
        raise ValueError("This is a test error")
    except Exception as e:
        handler.log_error(e, context="Testing error handler")
    
    # Test performance monitor
    print("\nTesting Performance Monitor...\n")
    perf = PerformanceMonitor(logger)
    
    perf.start_timer('test_operation')
    import time
    time.sleep(0.1)
    perf.end_timer('test_operation')
    
    perf.start_timer('test_operation')
    time.sleep(0.2)
    perf.end_timer('test_operation')
    
    print(perf.report())
    
    # Test resource monitor
    print("\nTesting Resource Monitor...\n")
    resource = ResourceMonitor(logger)
    resource.log_resource_status()
    status = resource.check_resources()
    print(f"Resource status: {status}")
    
    print("\nError handler test complete!")
