# src/gui_progress_handler.py
"""
Progress handling utilities for GUI operations.
"""

import tkinter as tk
import ttkbootstrap as ttk
from threading import Thread
from queue import Queue, Empty
import time


class ProgressDialog:
    """
    A modal progress dialog with cancel support.
    """
    
    def __init__(self, parent, title="Processing...", can_cancel=True):
        """
        Initialize progress dialog.
        
        Args:
            parent: Parent tkinter window
            title: Dialog title
            can_cancel: Whether user can cancel the operation
        """
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("500x250")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (parent.winfo_screenwidth() // 2) - (500 // 2)
        y = (parent.winfo_screenheight() // 2) - (150 // 2)
        self.dialog.geometry(f"+{x}+{y}")
        
        self.cancelled = False
        self.can_cancel = can_cancel
        
        # Status label
        self.status_label = ttk.Label(
            self.dialog, 
            text="Initializing...", 
            font=("Helvetica", 10)
        )
        self.status_label.pack(pady=(20, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.dialog,
            variable=self.progress_var,
            maximum=100,
            length=400,
            mode='determinate'
        )
        self.progress_bar.pack(pady=10)
        
        # Detail label
        self.detail_label = ttk.Label(
            self.dialog,
            text="",
            font=("Helvetica", 9),
            foreground="gray"
        )
        self.detail_label.pack(pady=(0, 10))
        
        # Cancel button
        if can_cancel:
            self.cancel_button = ttk.Button(
                self.dialog,
                text="Cancel",
                command=self._on_cancel,
                bootstyle="danger-outline"
            )
            self.cancel_button.pack(pady=10)
        
        # Handle window close
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)
    
    def update(self, current, total, message="", detail=""):
        """
        Update progress dialog.
        
        Args:
            current: Current progress value
            total: Total progress value
            message: Main status message
            detail: Additional detail message
        """
        if self.cancelled:
            return
        
        if total > 0:
            percentage = (current / total) * 100
            self.progress_var.set(percentage)
        
        if message:
            self.status_label.config(text=message)
        
        if detail:
            self.detail_label.config(text=detail)
        
        self.dialog.update()
    
    def _on_cancel(self):
        """Handle cancel button click."""
        if self.can_cancel:
            self.cancelled = True
            self.status_label.config(text="Cancelling...")
            if hasattr(self, 'cancel_button'):
                self.cancel_button.config(state="disabled")
    
    def close(self):
        """Close the dialog."""
        self.dialog.destroy()
    
    def is_cancelled(self):
        """Check if operation was cancelled."""
        return self.cancelled


class CancelledError(Exception):
    """Exception raised when operation is cancelled."""
    pass


class BackgroundTask:
    """
    Runs a task in a background thread with progress updates.
    """
    
    def __init__(self, parent, task_func, callback, error_callback=None, cancel_callback=None):
        """
        Initialize background task.
        
        Args:
            parent: Parent window
            task_func: Function to run in background (receives progress_callback)
            callback: Function to call on completion (receives result)
            error_callback: Function to call on error (receives exception)
            cancel_callback: Function to call on cancellation
        """
        self.parent = parent
        self.task_func = task_func
        self.callback = callback
        self.error_callback = error_callback
        self.cancel_callback = cancel_callback
        self.queue = Queue()
        self.progress_dialog = None
        self.thread = None
        self.result = None
        self.error = None
    
    def start(self, dialog_title="Processing...", can_cancel=False):
        """
        Start the background task.
        
        Args:
            dialog_title: Title for progress dialog
            can_cancel: Whether task can be cancelled
        """
        # Create progress dialog
        self.progress_dialog = ProgressDialog(
            self.parent, 
            title=dialog_title,
            can_cancel=can_cancel
        )
        
        # Start background thread
        self.thread = Thread(target=self._run_task, daemon=True)
        self.thread.start()
        
        # Start checking for updates
        self._check_queue()
    
    def _run_task(self):
        """Run the task in background thread."""
        try:
            # Create progress callback
            def progress_callback(current, total, message):
                if self.progress_dialog and self.progress_dialog.is_cancelled():
                    raise CancelledError("Operation cancelled")
                self.queue.put(('progress', (current, total, message)))
            
            # Run the task
            self.result = self.task_func(progress_callback)
            self.queue.put(('done', self.result))
            
        except CancelledError:
            self.queue.put(('cancelled', None))
        except Exception as e:
            self.error = e
            self.queue.put(('error', e))
    
    def _check_queue(self):
        """Check for messages from background thread."""
        try:
            while True:
                msg_type, data = self.queue.get_nowait()
                
                if msg_type == 'progress':
                    current, total, message = data
                    self.progress_dialog.update(current, total, message)
                    
                elif msg_type == 'done':
                    self.progress_dialog.close()
                    if self.callback:
                        self.callback(data)
                    return
                    
                elif msg_type == 'cancelled':
                    self.progress_dialog.close()
                    if self.cancel_callback:
                        self.cancel_callback()
                    return
                    
                elif msg_type == 'error':
                    self.progress_dialog.close()
                    if self.error_callback:
                        self.error_callback(data)
                    return
                    
        except Empty:
            pass
        
        # Check if cancelled via dialog button (backup check if thread isn't reporting back yet)
        if self.progress_dialog.is_cancelled():
            # Wait for thread to acknowledge cancellation via queue
            # but allow GUI to remain responsive
            pass
        
        # Schedule next check
        self.parent.after(100, self._check_queue)


def run_with_progress(parent, task_func, on_complete, on_error=None, on_cancel=None,
                      dialog_title="Processing...", can_cancel=False):
    """
    Convenience function to run a task with progress dialog.
    
    Args:
        parent: Parent window
        task_func: Function to run (receives progress_callback argument)
        on_complete: Callback for successful completion (receives result)
        on_error: Callback for errors (receives exception)
        on_cancel: Callback for cancellation
        dialog_title: Title for progress dialog
        can_cancel: Whether task can be cancelled
    
    Example:
        def my_task(progress_callback):
            for i in range(100):
                progress_callback(i, 100, f"Processing item {i}")
                time.sleep(0.1)
            return "Done!"
        
        def on_done(result):
            print(f"Task completed: {result}")
        
        run_with_progress(root, my_task, on_done, dialog_title="My Task", can_cancel=True)
    """
    task = BackgroundTask(parent, task_func, on_complete, on_error, on_cancel)
    task.start(dialog_title, can_cancel)


class ProgressBar:
    """
    A simple standalone progress bar for embedding in frames.
    """
    
    def __init__(self, parent, length=300):
        """
        Initialize progress bar.
        
        Args:
            parent: Parent frame
            length: Length of progress bar in pixels
        """
        self.frame = ttk.Frame(parent)
        
        self.label = ttk.Label(self.frame, text="Ready")
        self.label.pack()
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.frame,
            variable=self.progress_var,
            maximum=100,
            length=length,
            mode='determinate'
        )
        self.progress_bar.pack(pady=5)
    
    def update(self, current, total, message=""):
        """Update progress bar."""
        if total > 0:
            percentage = (current / total) * 100
            self.progress_var.set(percentage)
        
        if message:
            self.label.config(text=message)
        
        self.frame.update()
    
    def reset(self):
        """Reset progress bar."""
        self.progress_var.set(0)
        self.label.config(text="Ready")
    
    def pack(self, **kwargs):
        """Pack the progress bar frame."""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the progress bar frame."""
        self.frame.grid(**kwargs)