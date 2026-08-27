from datetime import datetime


class TimeFormatHelper:

    @staticmethod
    def format_epoch(epoch_time):
        dt = datetime.fromtimestamp(epoch_time)
        return dt.strftime("%H:%M:%S")

    @staticmethod
    def format_datetime(dt):
        return dt.strftime("%H:%M:%S")

    @staticmethod
    def format_elapsed(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"