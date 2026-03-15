from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

class TrainingProgress:
    """
    A class to display training progress using Rich
    
    Example:
        Training Epoch 1/100 ━━━━━━━━━━━━━━━━━━━━ 50% 50/100 • 0:02:30 remaining
        Epoch 1 | Batch 1/100 | Loss: 2.3456 | Acc: 0.1234
        Epoch 1 | Batch 10/100 | Loss: 1.8765 | Acc: 0.3456

        ┌─────────────── Epoch 1/100 ───────────────┐
        │ Metric          │ Value                   │
        ├─────────────────┼─────────────────────────┤
        │ Loss            │ 1.5432                  │
        │ Accuracy        │ 0.4567                  │
        │ Learning Rate   │ 1.00e-03                │
        └─────────────────┴─────────────────────────┘
    
    """
    def __init__(self, total_epochs: int, console: Console = None):
        self.console = console or Console()
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.epoch_loss = 0.0
        self.epoch_acc = 0.0
        self.batch_count = 0
        self.recent_losses = []
        self.recent_accs = []
    
    def create_progress_bar(self, description: str = "Training"):
        """Create a Rich progress bar for the current epoch"""
        return Progress(
            TextColumn("[bold blue]{task.description}", justify="right"),
            # control the length of the bar
            BarColumn(bar_width=80),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            MofNCompleteColumn(),
            "•",
            TimeRemainingColumn(),
            "•",
            SpinnerColumn(),
            console=self.console,
        )
        
    def create_metrics_table(self, epoch: int, loss: float, acc: float, lr: float):
        """Create a Rich table for displaying metrics"""
        table = Table(title=f"Epoch {epoch}/{self.total_epochs}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # The color changes dynamically according to the value
        loss_color = "green" if loss < 1.0 else "yellow" if loss < 2.0 else "red"
        acc_color = "green" if acc > 0.8 else "yellow" if acc > 0.6 else "red"
        
        table.add_row("Loss", f"[{loss_color}]{loss:.4f}")
        table.add_row("Accuracy", f"[{acc_color}]{acc:.4f}")
        table.add_row("Learning Rate", f"{lr:.2e}")
        
        return table
    
    def print_batch_info(self, epoch: int, batch: int, loss: float, acc: float, 
                        total_batches: int = None, refresh: bool = False):
        """
        print every batch info in the format:
        
        Example:
            Epoch 5 | Batch 20/100 | Loss: 1.2345 | Acc: 0.7891
            ============================================================
        """
        if total_batches:
            self.console.print(
                f"[bold cyan]Epoch {epoch}[/] | "
                f"[bold yellow]Batch {batch}/{total_batches}[/] | "
                f"Loss: [red]{loss:.4f}[/] | "
                f"Acc: [green]{acc:.4f}[/]"
            )
        else:
            self.console.print(
                f"[bold cyan]Epoch {epoch}[/] | "
                f"[bold yellow]Batch {batch}[/] | "
                f"Loss: [red]{loss:.4f}[/] | "
                f"Acc: [green]{acc:.4f}[/]"
            )
        
        if refresh:
            self.console.print("[dim]" + "="*60 + "[/]")

    def print_epoch_summary(self, epoch: int, avg_loss: float, avg_acc: float, lr: float):
        """打印 epoch 汇总信息"""
        table = self.create_metrics_table(epoch, avg_loss, avg_acc, lr)
        self.console.print(table)
        self.console.print("\n")
    
    