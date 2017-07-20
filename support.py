from __future__ import print_function, division, unicode_literals
import time


# ==============================================================================
#                                                                    PRETTY_TIME
# ==============================================================================
def pretty_time(t):
    """ Given an elapsed time in seconds, it returns the time as a string
        formatted as: "HH:MM:SS"
    """
    hours = int(t // 3600)
    mins = int((t % 3600) // 60)
    secs = int((t % 60) // 1)
    return "{:02d}:{:02d}:{:02d}".format(hours, mins, secs)


# ==============================================================================
#                                                                          TIMER
# ==============================================================================
class Timer(object):
    def __init__(self, start=True):
        """ Creates a convenient stopwatch-like timer.
            By default it starts the timer automatically as soon as it is
            created. Set start=False if you do not want this.
        """
        self.start_time = 0
        if start:
            self.start()
    
    def start(self):
        """ Start the timer """
        self.start_time = time.time()
    
    def elapsed(self):
        """ Return the number of seconds since the timer was started. """
        now = time.time()
        return (now - self.start_time)
    
    def elapsed_string(self):
        """ Return the amount of elapsed time since the timer was started as a
            formatted string in format:  "HH:MM:SS"
        """
        return pretty_time(self.elapsed())



