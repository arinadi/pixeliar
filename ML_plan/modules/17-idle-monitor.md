# Module 17: IdleMonitor — Auto-Shutdown

| | |
|---|---|
| **Estimated Complexity** | S |
| **Estimated Files** | 1 |
| **Key Risks** | False idle detection; premature shutdown |

## Requirements
- Track last activity timestamp
- Auto-shutdown Colab runtime after idle timeout (default 5 min)
- Reset timer on each image processed
- Graceful shutdown: notify → unassign

## Data & API
```python
class IdleMonitor:
    def __init__(self, idle_minutes=5):
        self.idle_minutes = idle_minutes
        self.last_activity = datetime.now()

    def reset(self):
        self.last_activity = datetime.now()

    async def watch(self):
        while True:
            await asyncio.sleep(60)
            elapsed = datetime.now() - self.last_activity
            if elapsed > timedelta(minutes=self.idle_minutes):
                print(f"⏰ Idle {self.idle_minutes}min — shutting down")
                from google.colab import runtime
                runtime.unassign()
                break
```

## Technical Implementation
- Adapted from TTB/bot_classes.py IdleMonitor
- Runs as background async task
- 60s check interval
- Colab `runtime.unassign()` for clean shutdown

## Testing
- [ ] Timer resets on activity
- [ ] Triggers shutdown after idle_minutes
- [ ] No false shutdown during active processing
- [ ] Graceful message before shutdown
