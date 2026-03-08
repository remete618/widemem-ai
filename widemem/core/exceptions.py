class WideMemError(Exception):
    pass


class ProviderError(WideMemError):
    pass


class ExtractionError(WideMemError):
    pass


class ConflictResolutionError(WideMemError):
    pass


class StorageError(WideMemError):
    pass
