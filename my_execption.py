class PrintColors:
    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def warning(info):
    print(f"{PrintColors.FAIL}{info}{PrintColors.END}")


def header(info):
    print(f"{PrintColors.HEADER}{info}{PrintColors.END}")


def success(info):
    print(f"{PrintColors.OK_GREEN}{info}{PrintColors.END}")


class MyException(Exception):
    def __init__(self, info, e=None):
        self.info = info
        self.e = e

    def cast_error(self):
        if self.e is not None:
            return Exception(f"\n{PrintColors.FAIL}"
                             f"{str(self.e)}"
                             f"\n"
                             f"{self.info}"
                             f"{PrintColors.END}")
        else:
            return Exception(f"{PrintColors.FAIL}"
                             f"{self.info}"
                             f"{PrintColors.END}")