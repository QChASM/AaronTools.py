class classproperty:
    def __init__(self, f):
        self._f = f
    
    def __get__(self, obj, owner):
        return self._f(owner)
