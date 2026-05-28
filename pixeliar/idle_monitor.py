"""
pixeliar — IdleMonitor (Module 17)
Auto-shutdown Colab after idle period.
Adapted from TTB/bot_classes.py IdleMonitor.
"""

from datetime import datetime, timedelta
import asyncio


class IdleMonitor:
    def __init__(self, idle_minutes=5, logger=None):
        self.idle_minutes = idle_minutes
        self.last_activity = datetime.now()
        self.logger = logger
        self._shutdown = False

    def reset(self):
        self.last_activity = datetime.now()

    async def watch(self):
        while True:
            await asyncio.sleep(60)
            elapsed = datetime.now() - self.last_activity
            if elapsed > timedelta(minutes=self.idle_minutes):
                msg = (f"⏰ Idle for {self.idle_minutes} min — "
                       f"shutting down Colab runtime")
                if self.logger:
                    self.logger.idle(msg)
                else:
                    print(msg, flush=True)
                try:
                    from google.colab import runtime
                    runtime.unassign()
                except Exception:
                    print("⚠️  Could not auto-shutdown — shutdown manually", flush=True)
                self._shutdown = True
                break

    def is_shutdown(self):
        return self._shutdown
