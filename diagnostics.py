#!/usr/bin/env python3
"""
Error handling and diagnostics for Diffusion-Pipe Training GUI
"""

import os
import sys
import traceback
import datetime
import signal
import faulthandler
import gc
import platform
import psutil
import logging
from threading import Thread

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler = logging.FileHandler('diffusion_gui_error.log')
log_handler.setFormatter(log_formatter)

logger = logging.getLogger('DiffusionGUI')
logger.setLevel(logging.DEBUG)
logger.addHandler(log_handler)

def setup_crash_handlers():
    """Set up handlers for crashes and signals"""
    # Create a log file for faulthandler
    fault_log = open('diffusion_gui_fault.log', 'w')
    
    # Enable Python's built-in fault handler
    faulthandler.enable(file=fault_log)
    
    # Register other signals that aren't automatically handled
    # Note: Do NOT register SIGSEGV as faulthandler.enable() already handles it
    faulthandler.register(signal.SIGABRT, file=fault_log, all_threads=True, chain=True)
    
    # Set up custom signal handlers
    signal.signal(signal.SIGABRT, handle_crash)
    
    # Return the file handle so it doesn't get garbage collected
    return fault_log

def handle_crash(sig, frame):
    """Handle crash signals and log diagnostic information"""
    logger.critical(f"Received signal {sig}")
    
    # Log stack trace
    logger.critical("Stack trace:")
    for line in traceback.format_stack():
        logger.critical(line.strip())
    
    # Log system information
    logger.critical(f"Python version: {sys.version}")
    logger.critical(f"Platform: {platform.platform()}")
    
    # Log memory information
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.critical(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
    
    # Try to log open files
    try:
        open_files = process.open_files()
        logger.critical(f"Open files: {len(open_files)}")
        for file in open_files[:10]:  # Log first 10 files
            logger.critical(f"  {file.path}")
    except:
        logger.critical("Could not get open files")
    
    # Try to log threads
    try:
        threads = process.threads()
        logger.critical(f"Threads: {len(threads)}")
        for thread in threads:
            logger.critical(f"  Thread ID: {thread.id}, CPU: {thread.user_time}")
    except:
        logger.critical("Could not get thread information")
    
    # Force garbage collection and log info
    logger.critical("Forcing garbage collection...")
    gc.collect()
    
    # Log some garbage collection stats
    logger.critical(f"Garbage collector stats: {gc.get_stats()}")
    
    # Only use default handler for SIGABRT, not SIGSEGV
    if sig == signal.SIGABRT:
        signal.default_int_handler(sig, frame)

def log_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler to log uncaught exceptions"""
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't log keyboard interrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def safe_process_start(command, cwd=None, env=None):
    """
    Start a process safely with additional error handling
    
    Args:
        command: Command string to execute
        cwd: Working directory
        env: Environment variables
        
    Returns:
        subprocess.Popen: Process object or None if failed
    """
    import subprocess
    
    logger.info(f"Starting process: {command}")
    logger.info(f"Working directory: {cwd}")
    
    try:
        # Create process with additional buffers and safeguards
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=cwd,
            env=env,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )
        
        logger.info(f"Process started with PID: {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Failed to start process: {str(e)}")
        logger.exception("Process start exception")
        return None

def check_system_resources():
    """
    Check system resources and log diagnostics
    
    Returns:
        dict: System resource information
    """
    resources = {}
    
    # Get CPU info
    resources['cpu_percent'] = psutil.cpu_percent(interval=0.1)
    resources['cpu_count'] = psutil.cpu_count()
    
    # Get memory info
    memory = psutil.virtual_memory()
    resources['total_memory'] = memory.total / (1024 * 1024 * 1024)  # GB
    resources['available_memory'] = memory.available / (1024 * 1024 * 1024)  # GB
    resources['memory_percent'] = memory.percent
    
    # Get disk info
    try:
        disk = psutil.disk_usage('/')
        resources['disk_total'] = disk.total / (1024 * 1024 * 1024)  # GB
        resources['disk_free'] = disk.free / (1024 * 1024 * 1024)  # GB
        resources['disk_percent'] = disk.percent
    except:
        resources['disk_error'] = "Could not get disk info"
    
    # Log all the info
    logger.info("System resources:")
    for key, value in resources.items():
        logger.info(f"  {key}: {value}")
    
    return resources

def init_diagnostics():
    """Initialize all diagnostic tools and error handlers"""
    # Configure basic logging first
    try:
        # Set up global exception hook
        sys.excepthook = log_exception
        
        # Set up crash handlers
        fault_log = open('diffusion_gui_fault.log', 'w')
        faulthandler.enable(file=fault_log)
        
        # Check system resources
        resources = check_system_resources()
        
        # Log initialization
        logger.info("Diagnostics initialized")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {platform.platform()}")
        
        # Return the fault log handle to prevent garbage collection
        return fault_log, resources
    except Exception as e:
        logger.error(f"Error during diagnostics initialization: {str(e)}")
        # Return minimal information
        return None, {}